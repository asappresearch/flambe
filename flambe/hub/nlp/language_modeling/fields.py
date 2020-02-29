from typing import Tuple
import torch

from flambe.field import TextField


class LMField(TextField):
    """Language Model field.

    Generates the original tensor alongside its shifted version.

    """

    def __init__(self,
                 **kwargs) -> None:
        super().__init__(**kwargs)

    def process(self, example: str) -> Tuple[torch.Tensor, ...]:  # type: ignore
        """Process an example and create 2 Tensors.

        Parameters
        ----------
        example: str
            The example to process, as a single string

        Returns
        -------
        Tuple[torch.Tensor, ...]
            The processed example, tokenized and numericalized

        """
        ret = super().process(example)
        return ret[:-1], ret[1:]
