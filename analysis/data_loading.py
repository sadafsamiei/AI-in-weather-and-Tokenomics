import json

import numpy as np

from paths import Folders, Files


def load_parameters():
    with open(Files.parameters, "r") as file:
        return json.load(file)


def extract_parameters():
    parameters = load_parameters()
    smoothers = list(parameters["attribution_smoother"].keys())
    methods = parameters["attribution_method"]
    date_times = parameters["date_time"]
    models = parameters["model"]

    return smoothers, methods, date_times, models


SMOOTHERS, METHODS, DATETIMES, MODELS = extract_parameters()


def load_variables():
    with open(Files.variables, "r") as file:
        return json.load(file)


def compute_matching_indexes():
    variables = load_variables()
    fcn_variables = variables["fcn"]
    fcn_var_to_idx = {name: i for i, name in enumerate(fcn_variables)}
    sfno_variables = variables["sfno"]
    sfno_var_to_idx = {name: i for i, name in enumerate(sfno_variables)}
    common_variables = [v for v in sfno_variables if v in fcn_variables]

    fcn_indexes = []
    sfno_indexes = []
    for v in common_variables:
        fcn_indexes.append(fcn_var_to_idx[v])
        sfno_indexes.append(sfno_var_to_idx[v])

    return fcn_indexes, sfno_indexes, common_variables


FCN_INDEXES, SFNO_INDEXES, COMMON_VARIABLES = compute_matching_indexes()


def load_attribution(model: str, smoother: str, attribution_method: str, time_stamp: str):
    file_name = f"model={model}, smoother={smoother}, attribution_method={attribution_method}, time_stamp={time_stamp}.npy"
    path = Folders.attributions / file_name
    attribution = np.load(path)
    return attribution


def format_attribution(attribution, model):
    attribution = attribution[:, :720, :]

    if model.lower() == "fcn":
        return attribution[FCN_INDEXES, ...]
    elif model.lower() == "sfno":
        return attribution[SFNO_INDEXES, ...]
    else:
        raise ValueError(f"Unknown model: {model}")


def standardize(data: np.ndarray):
    return (data - data.min()) / (data.max() - data.min())


def get_attribution(model: str, smoother: str, attribution_method: str, time_stamp: str):
    attribution = load_attribution(model, smoother, attribution_method, time_stamp)
    attribution = format_attribution(attribution, model)
    attribution = standardize(attribution)
    return attribution


def create_gaussian_data(size=(24, 720, 1440)):
    rng = np.random.default_rng()
    data = rng.standard_normal(size=size)
    return data


def create_exponential_data(size=(24, 720, 1440)):
    rng = np.random.default_rng()
    data = rng.exponential(scale=1.0, size=size)
    return data


def load_curve(method: str, smoother: str, threshold_type="lin", model=None):
    if model is None:
        file = f"{method}_{smoother}_{threshold_type}_threshold_curve.npy"
    else:
        file = f"{method}_{smoother}_{model}_{threshold_type}_threshold_curve.npy"
    return np.load(Folders.output / file)


def load_fake_curve():
    return np.load(Folders.output / "fake_curve.npy")


def load_thresholds(threshold_type="lin"):
    file = f"{threshold_type}_thresholds.npy"
    return np.load(Folders.output / file)


def load_average_attribution(model: str):
    file = f"{model}_average_attribution.npy".lower()
    return np.load(Folders.output / file)


def load_per_method_average_attribution(method: str, smoother: str, model: str):
    file = f"{method}_{smoother}_{model}_average_attribution.npy".lower()
    return np.load(Folders.output / "average_attributions" / file)


def load_similarities(model: str, method=None, smoother=None):
    if method is None:
        file = f"{model}_{smoother}_method_similarities.npy"
    elif smoother is None:
        file = f"{model}_{method}_smoother_similarities.npy"
    else:
        raise ValueError("Either provide a method or a smoother.")
    return np.load(Folders.output / "similarities" / file)
