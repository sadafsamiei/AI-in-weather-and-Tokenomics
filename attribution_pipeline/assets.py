import os
from pathlib import Path
from typing import ClassVar

from numpy import load as np_load


RAND4 = str(random.randint(1000, 9999))

class Folders:
    import random
    _base: ClassVar[str] = str(Path(__file__).parent.parent.resolve())
    _you: ClassVar[str] = str(Path(__file__).parent.parent.parent.parent.parent.resolve())
    _scratch: ClassVar[str] = _you + "scratch/alajos"
    results: ClassVar[str] = _scratch + f"/results_{RAND4}"
    src: ClassVar[str] = _base + "/src"
    logs: ClassVar[str] = _base + "/logs"
    experiment_assets: ClassVar[str] = _base + "/experiment_assets"
    analysis_assets: ClassVar[str] = _base + "/analysis_assets"
    attributions_png: ClassVar[str] = results + "/attributions_png"
    attributions_npy: ClassVar[str] = results + "/attributions_npy"
    input_states: ClassVar[str] = results + "/input_states"
    ground_truths: ClassVar[str] = results + "/ground_truths"
    predictions: ClassVar[str] = results + "/predictions"

class Files:
    yaml_config: ClassVar[str] = Folders.experiment_assets + "/config.yaml"
    # logs
    experiment_log: ClassVar[str] = Folders.logs + "/experiment.log"
    params_registry_json: ClassVar[str] = Folders.analysis_assets + "/params_registry.json"
    param_to_png_json: ClassVar[str] = Folders.analysis_assets + "/param_to_png.json"
    param_to_npy_attr_json: ClassVar[str] = Folders.analysis_assets + "/param_to_npy_attr.json"
    param_to_npy_x_gt_y_json: ClassVar[str] = Folders.analysis_assets + "/param_to_npy_x_gt_y.json"
    attribution_variables: ClassVar[str] = Folders.analysis_assets + "/attribution_variables.json"
    fcn_means: ClassVar[str] = Folders.experiment_assets + "/fcn_means.npy"
    fcn_stds: ClassVar[str] = Folders.experiment_assets + "/fcn_stds.npy"
    sfno_means: ClassVar[str] = Folders.experiment_assets + "/sfno_means.npy"
    sfno_stds: ClassVar[str] = Folders.experiment_assets + "/sfno_stds.npy"
    attribution_png: ClassVar[str] = Folders.attributions_png + "/{}.png"
    attribution_npy: ClassVar[str] = Folders.attributions_npy + "/{}.npy"
    input_state: ClassVar[str] = Folders.input_states + "/{}.npy"
    ground_truth: ClassVar[str] = Folders.ground_truths + "/{}.npy"
    prediction: ClassVar[str] =  Folders.predictions + "/{}.npy"

def create_stats_registry():
    # load stats once and access registry
    fcn_means = np_load(Files.fcn_means).squeeze(0).squeeze(0)
    fcn_stds = np_load(Files.fcn_stds).squeeze(0).squeeze(0)
    sfno_means = np_load(Files.sfno_means).squeeze(0).squeeze(0)
    sfno_stds = np_load(Files.sfno_stds).squeeze(0).squeeze(0)
    STATS_REG = {
        "fcn": {
            "means": fcn_means,
            "stds": fcn_stds
        },
        "sfno": {
            "means": sfno_means,
            "stds": sfno_stds
        }
    }
    return STATS_REG
    
def ensure_files_and_folders():
    folders = [  # does not contain assets and dashboard_assets -> must already exist
        Folders.results,
        Folders.logs,
        Folders.attributions_png,
        Folders.attributions_npy,
        Folders.input_states,
        Folders.ground_truths,
        Folders.predictions
    ]
    for path in folders:
        os.makedirs(path, exist_ok=True)
    if not os.path.isfile(Files.yaml_config):
        raise FileNotFoundError("Define a config for experiments!")
    if not os.path.isfile(Files.experiment_log):
        open(Files.experiment_log, 'w+')
    files = [
        Files.params_registry_json,
        Files.param_to_png_json,
        Files.param_to_npy_attr_json,
        Files.param_to_npy_x_gt_y_json
        ]
    for path in files:
        with open(path, "w") as f:
            f.write("")