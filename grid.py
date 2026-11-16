from typing import Tuple

import torch


def ndgrid(*tensors) -> Tuple[torch.Tensor, ...]:
    try:
        return torch.meshgrid(*tensors, indexing='ij')
    except TypeError:
        return torch.meshgrid(*tensors)


def meshgrid(*tensors) -> Tuple[torch.Tensor, ...]:

    return torch.meshgrid(*tensors, indexing='xy')

