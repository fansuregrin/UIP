import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch import Tensor
from torchvision.models import vgg19_bn, VGG19_BN_Weights
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from kornia.filters import canny


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

        return F.l1_loss(amp_y_hat, amp_y) + F.l1_loss(phase_y_hat, phase_y)
    

class FourDomainLoss2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        y_hat = torch.fft.rfft2(y_hat)
        y = torch.fft.rfft2(y)

        return F.l1_loss(y_hat, y)
    

class FourDomainLoss3(nn.Module):
    """"""
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        y_hat = torch.fft.rfft2(y_hat)
        y = torch.fft.rfft2(y)
        amp_y_hat = torch.abs(y_hat)
        amp_y = torch.abs(y)

        return F.l1_loss(amp_y_hat, amp_y)
    

class EdgeLoss(nn.Module):
    """Loss function based on edge detection."""
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        _, y_hat_edges = canny(y_hat)
        _, y_edges = canny(y)
        return F.l1_loss(y_hat_edges, y_edges)
    

class ContrastLoss(nn.Module):
    def __init__(self, win_size=3) -> None:
        super().__init__()
        assert win_size%2 == 1, 'Please use odd kernel size'
        self.win_size = win_size
        if torch.cuda.is_available():
            device_ = torch.device('cuda')
        else:
            device_ = torch.device('cpu')
        gamma_ = torch.zeros((1,), dtype=torch.float, device=device_)
        self.gamma = torch.nn.Parameter(gamma_, requires_grad=True)

    def _calc_contrast(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # RGB to GrayScale
        lumi = tvF.rgb_to_grayscale(x)
        
        # calculate local contrast
        unfolded_ = F.unfold(lumi, self.win_size, stride=1, padding=(self.win_size-1)//2)\
            .reshape((B, lumi.shape[1], -1, H, W))
        mean_vals = unfolded_.mean(dim=2, keepdim=True)
        deviations_ = unfolded_ - mean_vals
        std_devs_ = deviations_.std(dim=2, keepdim=True)
        local_contrast = std_devs_ / (mean_vals + 1e-5)
        local_contrast = local_contrast.squeeze(2).reshape((B, -1, H, W))
        
        return torch.mean(local_contrast)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: predicted
            y: reference
        """
        assert x.shape == y.shape, 'x and y must have the same shape'
        dist = torch.abs(self._calc_contrast(x) - self._calc_contrast(y))
        loss = (-self._calc_contrast(x) + dist) * self.gamma

        return loss