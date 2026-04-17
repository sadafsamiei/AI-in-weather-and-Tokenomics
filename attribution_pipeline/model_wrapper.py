import copy
from typing import Tuple

import torch

from utils import scale_sigma


class WeatherModelWrapper:
    """
    Wrapper for e2s weather models.
    Enables GradSmoother to manipulate it.
    """
    def __init__(self, model, device):
        self.model = model.to(device).eval()
        self.device = device
        self.name = model.name

    def save_original_weights(self):
        self.original_weights = copy.deepcopy(self.model.state_dict())

    def reset_weights(self):
        self.model.load_state_dict(self.original_weights, strict=True)

    def perturbe_weights(self, noise_level):
        def _sample_from_dist(layer):
            min_, max_ = layer.data.min(), layer.data.max()
            sigma = scale_sigma(noise_level, min_, max_)
            return torch.distributions.normal.Normal(
                        loc=0, scale=sigma
                    ).sample(layer.size()).to(layer.device)
        with torch.no_grad():
            for layer in self.model.parameters():
                try:
                    noise = _sample_from_dist(layer)
                    layer.add_(noise)
                except:
                    continue
                    
    def __call__(self, x, coords):
        return self.model._forward(x, coords)
    
    def backward(self, z: torch.Tensor, coords, target_var_lat_lon: Tuple[int, int, int]):
        """
        Computes gradients of the target_var_lat_lon in the predicted next state,
        with respect to the input state.
        """
        z = torch.tensor(z, dtype=torch.float32, device=self.device, requires_grad=True)
        z = z.unsqueeze(1)  # add time dimension for e2s models
        # NOTE: remove the squeeze(2) in fcn3 to make it equal everywhere
        # print(f"z in backward bebore forward: {z.shape}")
        # print("z.shape", z.shape)
        next_state = self.model._forward(z, coords)  
        # print("next_state.shape",next_state.shape)
        next_state = next_state.squeeze(1) # time dimension in second place  
        # print("next_state.shape",next_state.shape)
        target_pix = next_state[:, *target_var_lat_lon] 
        # print("target_pix.shape",target_pix.shape)
        grad_trick = target_pix.sum(0)
        # print("grad_trick.shape",grad_trick.shape)
        grads = torch.autograd.grad(grad_trick, z)[0]
        grads = grads.squeeze(1)  # autograd returns grads in shape of z
        # print("grads.shape", grads.shape)
        grads = grads.detach().cpu().numpy()  
        return grads