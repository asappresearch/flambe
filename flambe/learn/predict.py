from typing import Optional

import torch

from flambe.compile import Component
from flambe.dataset import Dataset
from flambe.nn import Module
from flambe.sampler import Sampler
from flambe.logging import log
from flambe.logging import get_trial_dir


class Predictor(Component):
    """Implement a Predictor block.

    A `Predictor` takes as input data, and a model and computes
    its predictions. This is a single step `Component` object.

    """

    def __init__(self,
                 dataset: Dataset,
                 eval_sampler: Sampler,
                 model: Module,
                 eval_data: str = 'test',
                 device: Optional[str] = None) -> None:
        """Initialize the evaluator.

        Parameters
        ----------
        dataset : Dataset
            The dataset to run evaluation on
        eval_sampler : Sampler
            The sampler to use over validation examples
        model : Module
            The model to train
        eval_data: str
            The data split to evaluate on: one of train, val or test
        device: str, optional
            The device to use in the computation.

        """
        self.eval_sampler = eval_sampler
        self.model = model
        self.eval_metric = None
        self.dataset = dataset

        # Select right device
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        data = getattr(dataset, eval_data)
        self._eval_iterator = self.eval_sampler.sample(data)
        self.save_dir = get_trial_dir() 

    def run(self, block_name: str = None) -> bool:
        """Run the evaluation.

        Returns
        ------
        bool
            Whether the component should continue running.

        """
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = []

            for batch in self._eval_iterator:
                pred = self.model(*[t.to(self.device) for t in batch])
                preds.append(pred.cpu())

            preds = torch.cat(preds, dim=0)  # type: ignore

        with open(os.path.join(self.save_dir, 'predictions.pt')) as f:
            torch.save(preds, f)  # type: ignore

        continue_ = False  # Single step so don't continue
        return continue_
