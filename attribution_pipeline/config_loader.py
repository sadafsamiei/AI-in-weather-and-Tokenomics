import itertools
from typing import Dict, List, Callable, Iterator
import yaml

from earth2studio.models.px import FCN, SFNO
from attribution_methods import *

from assets import Files

# --- registries  ---
MODEL_REG: Dict[str, Callable] = {
    "fcn": FCN,
    "sfno": SFNO
}
EXPLAINER_REG: Dict[str, Callable] = {
    "vanilla_grad": vanilla_grad,
    "gradient_times_input": gradient_times_input,
    "integrated_grad": integrated_grad,
    "expected_grad": expected_grad,
    "blur_integrated_grad": blur_integrated_grad
}

def load_config() -> dict:
    with open(Files.yaml_config, "r") as f:
        return yaml.safe_load(f)

def _generate_list_of_smoother_dicts(smoothers_dict: Dict) -> List[Dict]:
    """Expands smoother config to list of smoothers."""
    expanded_grid: List[Dict] = []
    for name, cfg in smoothers_dict.items():
        for m, n, wnl, inl in itertools.product(
            cfg["m"], cfg["n"], cfg["weights_noise_level"], cfg["input_noise_level"]
        ):
            expanded_grid.append(
                {
                    "smoother": name,
                    "m": m,
                    "n": n,
                    "weights_noise_level": wnl,
                    "input_noise_level": inl,
                }
            )
    return expanded_grid

def param_grid_from_config(cfg: Dict) -> Dict[str, List]:
    """Returns param grid which can be iterated over in the experiment."""
    if "model" in cfg:
        grid = {
            "attribution_smoother": _generate_list_of_smoother_dicts(cfg["attribution_smoother"]), # list of dicts
            "attribution_method": [(name, EXPLAINER_REG[name]) for name in cfg["attribution_method"]], # list of tuples                            
            "model": [(name, MODEL_REG[name]) for name in cfg["model"]], # list of tuples
            "target_var": cfg["target_var"], # list of strs                                     
            "target_lat_lon": cfg["target_lat_lon"], # list of strs 
            "grad_acc_strategy": cfg["grad_acc_strategy"], # list of strs 
            "date_time": cfg["date_time"], # list of strs 
        }
    else:
        grid = {
            "attribution_smoother": _generate_list_of_smoother_dicts(cfg["attribution_smoother"]), # list of dicts
            "attribution_method": [(name, EXPLAINER_REG[name]) for name in cfg["attribution_method"]], # list of tuples                            
            "target_var": cfg["target_var"], # list of strs                                     
            "target_lat_lon": cfg["target_lat_lon"], # list of strs 
            "grad_acc_strategy": cfg["grad_acc_strategy"], # list of strs 
        }
    return grid

def iter_experiments(param_grid: Dict[str, List]) -> Iterator[Dict[str, List]]:
    """Iterates through cartesian product of all possible combinations of parameters."""
    keys = list(param_grid.keys())
    vals = [param_grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))
