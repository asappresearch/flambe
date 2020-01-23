from abc import ABC, abstractmethod


from typing import Dict

import numpy as np

from flambe.search.distribution import Distribution


class Space(object):
    """Search space object.

    Manages the various dimensions that the hyperparameters are in.
    This object is used internally by the search aglrothms and should
    not be used directly.

    """

    def __init__(self, distributions: Dict[str, Distribution]):
        '''
        Initialize the search space with a list of search distributions.

        search_dists: List of search distributions, e.g.

        '''
        self.distributions = distributions
        self.var_names = list(distributions.keys())
        self.dists = [distributions[name] for name in self.var_names]
        self.n_vars = len(self.dists)

        self.var_types = [dist.var_type for dist in self.dists]
        self.var_bounds = [(dist.var_min, dist.var_max) if dist.is_numerical  # type: ignore
                           else (np.nan, np.nan) for dist in self.dists]

        self.name2idx = {name: i for i, name in enumerate(self.var_names)}

    def sample(self):
        '''
        Randomly sample from the space.
        '''
        samp = {}
        for name, dist in self.distributions.items():
            samp[name] = dist.sample()
        return samp

    def sample_raw_dist(self):
        '''
        Randomly sample from the raw distributions
        without applying transforms.
        '''
        samp = {}
        for name, dist in self.distributions.items():
            if dist.is_numerical:
                s = dist.sample_raw_dist()
            else:
                s = dist.sample()
            samp[name] = s
        return samp

    def apply_transform(self, samp):
        '''
        Apply transforms to a sample drawn from the space
        using sample_dist.

        samp: Dictionary of hyperparameter values, e.g.
                    samp = {'hp1': 7, 'hp2': 3}
        '''
        transformed = {}
        for var_name in samp.keys():
            dist = self.dists[self.name2idx[var_name]]
            if dist.is_numerical:
                t = dist.transform_fn(samp[var_name])
            else:
                t = samp[var_name]
            transformed[var_name] = t
        return transformed

    def round_to_space(self, hp_dict):
        '''
        Round hyperparameters to possible choices for ordered variables.

        hp_dict: Dictionary of hyperparameter values.
        '''
        rounded = {}
        for name in hp_dict.keys():
            dist = self.dists[self.name2idx[name]]
            val = hp_dict[name]

            if dist.var_type == 'discrete':
                val = dist.round_to_options(val)

            rounded[name] = val
        return rounded

    def normalize_to_space(self, hp_dict, return_as_list=False):
        '''
        Normalize hyperparameters to [0, 1] based on range for
        numerical variables and convert them to integers for
        categorical variables.

        hp_dict: Dictionary of hyperparameter values.
        return_as_list: Whether or not to return the
            hyperparameters as a list.
        '''
        normalized = {}
        for name in hp_dict.keys():
            dist = self.dists[self.name2idx[name]]
            val = hp_dict[name]

            if dist.is_numerical:
                val = dist.normalize_to_range(val)
            else:
                val = dist.option_to_int(val)
            normalized[name] = val

        if return_as_list:
            lst = []
            for name in self.var_names:
                if name in normalized.keys():
                    lst.append(normalized[name])
            return lst
        else:
            return normalized

    def unnormalize(self, hp_dict):
        '''
        Reverse the normalize_to_space function.

        hp_dict: Dictionary of normalized hyperparameter values.
        '''
        unnormalized = {}
        for name in hp_dict.keys():
            dist = self.dists[self.name2idx[name]]
            val = hp_dict[name]

            if dist.is_numerical:
                val = dist.unnormalize(val)
            else:
                val = dist.int_to_option(val)
            unnormalized[name] = val
        return unnormalized


class Searcher(ABC):
    '''
    Base searcher class.  Searchers handle hyperparameter
    configuration selection and only  have access to a) the
    hyperparameters that need to be tuned and b) the performance of
    these hyperparameters.
    '''

    def __init__(self):
        self.space = None

    def assign_space(self, space):
        space = Space(space)
        self._check_space(space)
        self.space = space
        self.n_configs_proposed = 0

    @classmethod
    def _check_space(cls, space):
        pass

    def propose_new_params(self, **kwargs):
        '''
        Proposes a new hyperparameter configuration based on
        internal searching algorithm.  All
        subclasses must implement this method.
        '''
        if self.space is None:
            raise ValueError('Space has not been assigned to searcher.')
        params_id = self.n_configs_proposed
        params = self._propose_new_params(**kwargs)
        if params is None:
            return None, None
        else:
            self.n_configs_proposed += 1
            return params_id, params

    @abstractmethod
    def _propose_new_params(self, **kwargs):
        pass

    @abstractmethod
    def register_results(self, results, **kwargs):
        pass


class ModelBasedSearcher(Searcher):

    def __init__(self, min_configs_in_model):
        '''
        space: A hypertune.space.Space object.
        min_points_in_model: Minimum number of points
        to collect before model building.
        '''
        super().__init__()
        self.min_configs_in_model = min_configs_in_model
        self.data = {}

    @property
    def n_configs_in_model(self):
        return sum([1 for datum in self.data.values() if datum['result'] is not None])

    @property
    def has_enough_configs_for_model(self):
        '''
        Check if the searcher has enough points to build a model.
        '''
        return self.n_configs_in_model >= self.min_configs_in_model

    def _propose_new_params(self, **kwargs):
        '''
        Propose a new hyperparameter configuration.
        Calls internal self._propose_new_model_hp
        function and applies the transforms from the space.
        Note that model building happens
        in the distribution space (not the transformed space).
        '''
        params_in_model_space = self._propose_new_params_in_model_space(**kwargs)
        params_id = self.n_configs_proposed
        self.data[params_id] = {'params_in_model_space': params_in_model_space,
                                'result': None}

        params = self._apply_transform(params_in_model_space)
        return params

    @abstractmethod
    def _propose_new_params_in_model_space(self, **kwargs):
        '''
        Subclasses must override this method that
        generates samples in the distribution space.
        '''
        if not self.has_enough_configs_for_model:
            return self.space.sample_raw_dist()

    @abstractmethod
    def _apply_transform(self, params_in_model_space):
        pass

    def register_results(self, results, **kwargs):
        '''
        Record results.

        hp_results: List of hyperparameters and associated results.
        '''
        for params_id, result in results.items():
            self.data[params_id]['result'] = result
