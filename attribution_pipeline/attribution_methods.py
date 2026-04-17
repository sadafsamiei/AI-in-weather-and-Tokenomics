from typing import Protocol

import numpy as np
from scipy.ndimage import gaussian_filter1d

from assets import create_stats_registry
from model_wrapper import WeatherModelWrapper

# TODO
# - fix expected grad and integrated grad
# - set K to 8/16/32 on cluster
# interpolation steps
K = 4  # try 8-16

STATS_REG = create_stats_registry()

def get_grads(model: WeatherModelWrapper, z, coords, target_var_lat_lon):
    grads = model.backward(z, coords, target_var_lat_lon)
    # print("grads in get_grads", grads.shape)
    return grads


class AttributionMethod(Protocol):
    """
    Protocol: enforce signature while preserving functional implementation of explainers.
    --> no need for abstract class.
    """
    def __call__(self, model, x: np.ndarray, coords, target_var_lat_lon: int) -> np.ndarray:
        # x: (H, W, C) -> input tensor which prediction we want to explain.
        # target_idx: (c,h,w) -> take grad wrt to this position in next state
        # z: modified input according to attribution_method
        ...

def vanilla_grad(model, x, coords, target_var_lat_lon):
    z = np.expand_dims(x, axis=0)  # (k,C,H,W) with k=1
    vg = get_grads(model, z, coords, target_var_lat_lon).squeeze(0)  
    # print("vg.shape",vg.shape)
    return vg

def gradient_times_input(model, x, coords, target_var_lat_lon):
    gti = x * vanilla_grad(model, x, coords, target_var_lat_lon)  # (C,H,W)
    # print("gti.shape",gti.shape)
    return gti  

def integrated_grad(model, x, coords, target_var_lat_lon):
    # TODO: extract historical means! 
    # TODO: the noise should have a location and mean that are meaningful maybe
    # what if alpha is 1 and we only use the noise,
    # but the model has never seen numbers in that range?
    # We should probably pick a max and a min for each layer or even pixel for generating the noise.
    # print(x.min(), x.max(), x.mean(), x.std())
    means = STATS_REG[model.name]["means"] # baseline is literally the mean tensor --> probably not optimal for this use case
    # print(means.min(), means.max(), means.mean(), means.std())
    x_hat = means
    delta = x - x_hat
    alphas = np.linspace(1/K, 1.0, num=K, dtype=x.dtype).reshape(K, 1, 1, 1)  # (k,1,1,1)

    # interpolate
    z = x_hat[None, ...] + alphas * delta[None, ...]  # (k, H, W, C)

    grads = get_grads(model, z, coords, target_var_lat_lon) 
    avg_grads = grads.mean(axis=0)  # (k,C,H,W) -> (C,H,W)
    ig = delta * avg_grads         
    # print("ig.shape",ig.shape)
    return ig  

def expected_grad(model, x, coords, target_var_lat_lon):
    # TODO: check that data is on same scale as stats!!!!!!!!!!!!!!!!!!! 
    # print(x.min(), x.max(), x.mean(), x.std())

    means = STATS_REG[model.name]["means"] 
    stds = STATS_REG[model.name]["stds"]
    # print(means.min(), means.max(), means.mean(), means.std())
    # old: x_hat = np.random.rand(K, *x.shape)  # baseline
    x_hat = np.random.randn(K, *means.shape) * stds + means # sample from gaussian for each point in 3d tensor using the 3d means and 3d stds

    delta = x[None, ...] - x_hat  # (k, H, W, C)
    alphas = np.random.rand(K).reshape(K, 1, 1, 1)  
    
    # interpolate
    z = x_hat + alphas * delta

    grads = get_grads(model, z, coords, target_var_lat_lon)
    eg = (delta * grads).mean(axis=0)  
    # print("eg.shape",eg.shape)
    return eg  

def blur_integrated_grad(model, x, coords, target_var_lat_lon):
    def _blur(img, sigma, kernel_size):
        # cap kernel like torchvision
        truncate = max((kernel_size - 1) / (2.0 * sigma), 0.5)
        # blur only H and W; leave channels untouched
        y = gaussian_filter1d(img, sigma=sigma, axis=0, truncate=truncate)
        y = gaussian_filter1d(y,   sigma=sigma, axis=1, truncate=truncate)
        return y

    sigma_max = 150.0
    sigma_min = 0.01
    # TODO: check the blurring with different params
    kernel_size = 13

    sigmas = np.linspace(sigma_max, sigma_min, num=K).astype(x.dtype)  # (k,)

    z = np.stack([_blur(x, sigma, kernel_size) for sigma in sigmas])

    grads = get_grads(model, z, coords, target_var_lat_lon)

    x_blurred_max = _blur(x, sigma_max, kernel_size)
    blur_ig = (x - x_blurred_max) * (grads.sum(axis=0) / K) 
    # print("blur_ig.shape",blur_ig.shape)
    return blur_ig  