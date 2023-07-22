import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DiceLoss(nn.Module):
    def __init__(self, average: str = "micro", eps: float = 1e-8) -> None:
        super().__init__()
        self.average = average
        self.eps = eps

    def forward(self, input: Tensor,
                target: Tensor) -> Tensor:
        return dice_loss(input, target, self.average, self.eps)


def dice_loss(input: Tensor,
              target: Tensor,
              average: str = "micro",
              eps: float = 1e-8) -> Tensor:
    if not isinstance(input, Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxNxHxW. Got: {input.shape}")

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError(
            f"input and target shapes must be the same. Got: {input.shape} and {target.shape}")

    if not input.device == target.device:
        raise ValueError(
            f"input and target must be in the same device. Got: {input.device} and {target.device}")

    # compute softmax over the classes axis
    input_soft: Tensor = F.softmax(input, dim=1)

    # compute the actual dice score
    possible_average = {"micro", "macro"}
    if average == 'micro':
        dims = (1, 2, 3)
    elif average == 'macro':
        dims = (2, 3)
    else:
        assert f"The `average` has to be one of {possible_average}. Got: {average}"
    
    intersection = torch.sum(input_soft * target, dims)
    cardinality = torch.sum(input_soft + target, dims)

    dice_score = 2.0 * intersection / (cardinality + eps)
    dice_loss = -dice_score + 1.0
    dice_loss = torch.mean(dice_loss)

    return dice_loss