import numpy as np
from typing import Dict, Any, Optional, Set, Type

from flambe.search.distribution import Distribution
from flambe.search.searcher.searcher import ModelBasedSearcher
from flambe.search.searcher.searcher import Space


class MultiFidSearcher(ModelBasedSearcher):
    """A searcher that allows for multiple fidelities.

    Meant to be paired with HyperBand scheduler.

    """

    def __init__(self,
                 space: Dict[str, Distribution],
                 searcher_class: Type[ModelBasedSearcher],
                 search_kwargs: Dict[str, Any]):
        """Creates a searcher comprised of several sub-searchers.

        Parameters
        ----------
        space: Dict[str, Distribution]
            Dictionary mapping variable names to their initial
            distributions.
        searcher_class: ModelBasedSearcher
            A subclass of ModelBasedSearcher.
        search_kwargs: Dict[str, Any]
            The parameters of the ModelBasedSearcher class used for
            instantiation.

        """
        super().__init__(space, min_configs_in_model=0)

        self.searcher_class = searcher_class
        self.search_kwargs = search_kwargs
        self.max_fid = -np.inf
        self.max_fid_searcher = self.searcher_class(space, **self.search_kwargs)
        self.fids_dict: Dict[str, int] = dict()
        self.searchers: Dict[int, ModelBasedSearcher] = dict()

    def check_space(self, space: Space):
        """Check that a particular space is valid for the searcher.

        Parameters
        ----------
        space: Space
            A Space object that holds the distributions to search over.

        Raises
        ----------
        ValueError
            If invalid space object is passed to searcher.

        """
        self.max_fid_searcher.check_space(space)

    def _propose_new_params_in_model_space(self) -> Optional[Dict[str, Any]]:
        """Propose new parameters in model space using its current
        maximum fidelity searcher.

        Returns
        ----------
        Dict[str, Any], optional
            A hyperparameter configuration in model space.

        """
        return self.max_fid_searcher._propose_new_params_in_model_space()

    def _apply_transform(self, params_in_model_space: Dict[str, Any]) -> Dict[str, Any]:
        """Applies transform to configuration in model space.

        Parameters
        ----------
        params_in_model_space: Dict[str, Any]
            A hyperparameter configuration in model space.

        Returns
        ----------
        Dict[str, Any]
            A hyperparameter configuration in transformed space.

        """
        return self.max_fid_searcher._apply_transform(params_in_model_space)

    def register_results(self, results: Dict[str, float]):
        """Records results of hyperparameter configurations.

        Based on fidelities.  The appropriate subsearcher is called to
        register results.  Note that the subsearcher is only called if
        the fidelity of the configuration is the current maximum.

        Parameters
        ----------
        results: Dict[int, float]
            A dictionary mapping parameter id to result.

        """
        # Update fidelities
        fids_dict = dict()
        for key, value in results.items():
            if key in self.fids_dict:
                self.fids_dict[key] += 1
            else:
                self.fids_dict[key] = 1
            fids_dict[key] = self.fids_dict[key]

        # Group param_ids by fidelity
        fids_groups: Dict[int, Set[str]] = dict()
        for param_id, fid in fids_dict.items():
            if fid in fids_groups:
                fids_groups[fid].add(param_id)
            else:
                fids_groups[fid] = set([param_id])

        for fid, params_ids in fids_groups.items():

            # No need to train the model if fidelity is less than max
            if fid < self.max_fid:
                continue

            if fid in self.searchers:
                searcher = self.searchers[fid]
            else:
                searcher = self.searcher_class(self.space.dists, **self.search_kwargs)
                self.searchers[fid] = searcher

            # Transfer data from multifidelity searcher to subsearcher
            for params_id in params_ids:
                params_in_model_space = self.params[params_id]
                searcher.params[params_id] = params_in_model_space

            filtered_results = {p_id: res for p_id, res in results.items() if p_id in params_ids}
            searcher.register_results(filtered_results)

            # Check if max fidelity searcher should be updated
            if fid > self.max_fid and searcher.has_enough_configs_for_model:
                self.max_fid = fid
                self.max_fid_searcher = searcher
