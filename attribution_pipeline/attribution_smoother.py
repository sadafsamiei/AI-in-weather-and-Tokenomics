from typing import Tuple

import numpy as np

from attribution_methods import AttributionMethod
from model_wrapper import WeatherModelWrapper

from utils import scale_sigma


class AttributionSmoother():
    """
    With this class one can perform either one of the following smoothings:
    - no_smoothing: m=1, n=1
    - noise_grad:   m>1, n=1
    - smooth_grad:  m=1, n>1
    - fusion_grad:  m>1, n>1
    code ref: https://github.com/understandable-machine-intelligence-lab/NoiseGrad/blob/master/examples/example-noisegrad-pytorch.ipynb
    """
    def __init__(
            self, model: WeatherModelWrapper,
            weights_noise_level, input_noise_level, m, n
        ):
        assert m >= 1 and n >= 1, f"Values for m and n must be >= 1, got {m} and {n}."
        self.model = model
        self.m = m
        self.n = n
        self.weights_noise_level = weights_noise_level
        self.input_noise_level = input_noise_level
        # previously if m > 1, now for 
        self.model.save_original_weights()    

    def _preprocess(self, array_3d):
        """
        Return batched (4D) version of input.
        If smoothgrad is applied (n>1) create noisy batch.
        """
        if self.n > 1:
            max_, min_ = np.max(array_3d, axis=(1,2)), np.min(array_3d, axis=(1,2))
            sigma = scale_sigma(self.input_noise_level, min_, max_)
            sigma = sigma[None, ..., None, None]
            sigma = np.repeat(sigma, self.n, axis=0)    
            x = array_3d[None, ...]
            x = np.repeat(x, self.n, axis=0)                             # (k, C, H, W)
            noise = sigma * np.random.randn(*x.shape).astype(x.dtype)    # same dtype as x
            x = x + noise                                                 # noise ~ N(0, σ)
            return x
        else: 
            return np.expand_dims(array_3d, axis=0)
        
    def _perturbe_weights(self):
        if self.m > 1:
            self.model.reset_weights()
            self.model.perturbe_weights(self.weights_noise_level)

    def _iter_tensor(self, x, i):
        return x[i]

    def _accumulate(self, acc_attr, attr, grad_accumulation_strategy: str):
        """
        Sums the new attribution to the accumulated one.
        """
        if grad_accumulation_strategy == "absolute":
            acc_ = acc_attr + np.absolute(attr)
        elif grad_accumulation_strategy == "directional":
            acc_ = acc_attr + attr
        else:
            raise ValueError(f"{grad_accumulation_strategy} Not a valid grad_acc_strategy.")
        return acc_

    def generate_attribution(
            self, array_3d,
            coords,  # needed by e2s model
            target_var_lat_lon: Tuple[int, int, int],  
            explainer: AttributionMethod, 
            grad_accumulation_strategy: str,
        ) -> np.ndarray:
        """
        Generates 3D attribution:
        - perturbe input array if smoothgrad setting (n>1)
        - perturbe the weights if noisegrad setting (m>1)
        - for each input in the batch compute attribution and accumulate
        """
        x = self._preprocess(array_3d) 
        k = self.m * self.n    
        acc_attr = np.zeros(array_3d.shape)
        step = 1
        for _ in range(self.m):
            self._perturbe_weights()
            for i in range(self.n):
                # print(f"{step}/{k}")
                x_ = self._iter_tensor(x, i)
                attr = explainer(self.model, x_, coords, target_var_lat_lon)
                acc_attr = self._accumulate(acc_attr, attr, grad_accumulation_strategy)
                step += 1
        self.model.reset_weights()
        return acc_attr  # TODO: .unsqueeze(0) if still batch 1 removes, ow ok