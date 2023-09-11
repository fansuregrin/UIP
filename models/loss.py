import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import vgg19_bn, VGG19_BN_Weights
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor


class SemanticContentLoss(nn.Module):
    """Semantic Content Loss.
    """
    def __init__(self, weights_k=(1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0)):
        """Initialize the loss.

        Args:
            weights_k: Five weights for summing.
        """
        super(SemanticContentLoss, self).__init__()
        self.weights_k = weights_k
        self.weights = VGG19_BN_Weights.DEFAULT
        model_ = vgg19_bn(weights=self.weights)
        _, self.eval_nodes = get_graph_node_names(model_)
        self.return_nodes = {
            'features.36': 'feature_1',
            'features.40': 'feature_2',
            'features.43': 'feature_3',
            'features.46': 'feature_4',
            'features.49': 'feature_5',
        }
        self.model = create_feature_extractor(model_, return_nodes=self.return_nodes)
        self.preprocess = self.weights.transforms()

    def forward(self, x, y):
        self.model.eval()
        with torch.no_grad():
            x_features = self.model(x)
            y_features = self.model(y)
        loss = 0.0
        for i in range(5):
            loss += (self.weights_k[i] * F.l1_loss(x_features[f'feature_{i+1}'], y_features[f'feature_{i+1}']))
        return loss
    

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


class L1CharbonnierLoss(nn.Module):
    """L1 Charbonnier loss."""
    def __init__(self):
        super(L1CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = X - Y
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class FourDomainLoss(nn.Module):
    """"""
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        y_hat = torch.fft.rfft2(y_hat)
        y = torch.fft.rfft2(y)
        amp_y_hat = torch.abs(y_hat)
        amp_y = torch.abs(y)
        phase_y_hat = torch.angle(y_hat)
        phase_y = torch.angle(y)

        return amp_y - amp_y_hat + phase_y - phase_y_hat