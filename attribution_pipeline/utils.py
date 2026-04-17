import json
import operator
from typing import Dict, Any
from functools import reduce
from collections import defaultdict
from datetime import datetime

import torch
import numpy as np
from earth2studio.utils.type import TimeArray
from earth2studio.utils.time import to_time_array

from assets import Files, Folders


def scale_sigma(noise_level, min_, max_):
    """Used to correctly scale noise for model inputs and model layers."""
    return noise_level * (max_ - min_)

def get_time_array(date_time: str) -> TimeArray:
    """Converts string to datetime and feeds to earth2studio function."""
    date_time = [datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S")]
    date_time = to_time_array(date_time)
    return date_time

def log_params_to_fig_map(key: str, experiment_name: str):
    """Appends new experiment's map key:fig to json."""
    with open(Files.param_to_png_json, "a", encoding="utf-8") as f:
        experiment_name += ".png"
        f.write(json.dumps({key: experiment_name}) + "\n")

def log_params_to_array_map(mode:str, key: str, iter_: str):
    """Appends new experiment's map key:fig to json."""
    if mode == "attribution":
        file = Files.param_to_npy_attr_json
    elif mode == "x_gt_y":
        file = Files.param_to_npy_x_gt_y_json
    else:
        raise Exception(f"Mode {mode} not valid")
    with open(file, "a", encoding="utf-8") as f:
        iter_ += ".npy"
        f.write(json.dumps({key: iter_}) + "\n")

def get_device(verbose=True) -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"found device {device}")
    return device

def count_experiments(grid: dict) -> int:
    """Used for reporting in main."""
    lengths = [len(grid["attribution_smoother"])] + [
        len(grid[k]) for k in grid.keys() if k != "attribution_smoother"
    ]
    total = reduce(operator.mul, lengths, 1)
    return total

def _transform_param_grid(
    param_grid
) -> Dict[str, Any]:
    """Used internally by save_params_registry for converting main-param-grid to structure used in dashboard."""
    def _name_only(x):
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], str):
            return x[0]
        return str(x)
    def _uniq(seq):
        seen, out = set(), []
        for item in seq:
            key = tuple(item) if isinstance(item, list) else item
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out
    smoother_param_dict = defaultdict(lambda: defaultdict(set))
    for sp in param_grid.get("attribution_smoother", []):
        name = sp.get("smoother")
        if not name:
            continue
        for k, v in sp.items():
            if k != "smoother":
                smoother_param_dict[name][k].add(v)
    return {
        "date_time": _uniq(param_grid.get("date_time", [])),
        "model": _uniq(map(_name_only, param_grid.get("model", []))),
        "target_var": _uniq(param_grid.get("target_var", [])),
        "target_lat_lon": _uniq(param_grid.get("target_lat_lon", [])),
        "gradient_accumulation_strategy": _uniq(param_grid.get("grad_acc_strategy", [])),
        "attribution_method": _uniq(map(_name_only, param_grid.get("attribution_method", []))),
        "attribution_smoother": {
            smoother: {param: sorted(values) for param, values in params.items()}
            for smoother, params in smoother_param_dict.items()
        },
    }

def save_params_registry(param_grid):
    """Saves converted param-grid for dashboard."""
    param_grid = _transform_param_grid(param_grid)
    with open(Files.params_registry_json, 'w') as f:
        json.dump(param_grid, f, indent=4)

def post_process_params_map(): 
    """Run at end of main to convert list of key:fig maps to proper json."""
    # transforms list of maps {params: result} to single merged dict
    files = [
        Files.param_to_png_json, 
        Files.param_to_npy_attr_json,
        Files.param_to_npy_x_gt_y_json
    ]
    for file in files: 
        merged_dict = {}
        with open(file, "r") as f:
            for line in f:
                if line.strip():
                    merged_dict.update(json.loads(line))
        # save merged dict
        with open(file, "w") as f:
            json.dump(merged_dict, f, indent=4)

def save_arrays(arrays, iter):
    names = {
            "attribution": Files.attribution_npy,
            "input_state": Files.input_state,
            "ground_truth": Files.ground_truth,
            "prediction": Files.prediction
        }
    for key in arrays.keys():
        formatted = names[key].format(iter)
        np.save(formatted, arrays[key])
    
    
    
    
    
    