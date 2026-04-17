import itertools

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

import data_loading
from paths import Folders


def compute_threshold_agreements(arrays, thresholds):
    """Computes the average agreement between n arrays across several thresholds."""
    stacked = np.stack(arrays, axis=0)
    n = len(arrays)
    flat = stacked.reshape(n, -1)
    i, j = np.triu_indices(n, k=1)  # upper triangular indexes to avoid double-counting agreements

    threshold_agreements = []
    for t in thresholds:
        thresholded = flat >= t
        agreements = (thresholded[:, None, :] & thresholded[None, :, :]).mean(axis=2)
        agreements = agreements[i, j].mean()
        threshold_agreements.append(agreements)

    return np.array(threshold_agreements)


def compute_average_difference(arrays):
    stacked = np.stack(arrays, axis=0)
    n = len(arrays)
    flat = stacked.reshape(n, -1)
    i, j = np.triu_indices(n, k=1)  # upper triangular indexes to avoid double-counting differences

    differences = np.abs((flat[:, None, :] - flat[None, :, :])).mean(axis=2)
    differences = differences[i, j].mean()

    return differences


def compute_agreement_curve_between_models(method, smoother, thresholds):
    """Computes agreement at timestamp t between the two models. Then averages over all timestamps."""
    fcn = data_loading.MODELS[0]
    sfno = data_loading.MODELS[1]

    threshold_agreements = []

    for time in data_loading.DATETIMES:
        fcn_attribution = data_loading.get_attribution(model=fcn, smoother=smoother, attribution_method=method,
                                                       time_stamp=time)
        sfno_attribution = data_loading.get_attribution(model=sfno, smoother=smoother, attribution_method=method,
                                                        time_stamp=time)
        threshold_agreements.append(compute_threshold_agreements([fcn_attribution, sfno_attribution], thresholds))

    return np.stack(threshold_agreements, axis=0)


def compute_agreement_curve_between_timestamps(method, smoother, thresholds):
    """Computes agreement between all timestamps within one model"""
    fcn = data_loading.MODELS[0]
    sfno = data_loading.MODELS[1]

    fcn_agreements = np.zeros(thresholds.shape[0])
    fcn_attributions = []
    sfno_agreements = np.zeros(thresholds.shape[0])
    sfno_attributions = []

    for idx, time in enumerate(data_loading.DATETIMES):
        fcn_attributions.append(
            data_loading.get_attribution(model=fcn, smoother=smoother, attribution_method=method, time_stamp=time))
        sfno_attributions.append(
            data_loading.get_attribution(model=sfno, smoother=smoother, attribution_method=method, time_stamp=time))

        if (idx + 1) % 3 == 0:
            fcn_agreements += compute_threshold_agreements(fcn_attributions, thresholds)
            fcn_attributions = []
            sfno_agreements += compute_threshold_agreements(sfno_attributions, thresholds)
            sfno_attributions = []

    return fcn_agreements / 3.0, sfno_agreements / 3.0


def compute_average_difference_between_models(method, smoother):
    """Computes difference at timestamp t between the two models. Then averages over all timestamps."""
    fcn = data_loading.MODELS[0]
    sfno = data_loading.MODELS[1]

    differences = []

    for time in data_loading.DATETIMES:
        fcn_attribution = data_loading.get_attribution(model=fcn, smoother=smoother, attribution_method=method,
                                                       time_stamp=time)
        sfno_attribution = data_loading.get_attribution(model=sfno, smoother=smoother, attribution_method=method,
                                                        time_stamp=time)
        differences.append(compute_average_difference([fcn_attribution, sfno_attribution]))

    return np.array(differences).mean()


def compute_average_difference_between_timestamps(method, smoother):
    """Computes average difference between all timestamps within one model"""
    fcn = data_loading.MODELS[0]
    sfno = data_loading.MODELS[1]

    fcn_difference = 0.0
    fcn_attributions = []
    sfno_difference = 0.0
    sfno_attributions = []

    for idx, time in enumerate(data_loading.DATETIMES):
        fcn_attributions.append(
            data_loading.get_attribution(model=fcn, smoother=smoother, attribution_method=method, time_stamp=time))
        sfno_attributions.append(
            data_loading.get_attribution(model=sfno, smoother=smoother, attribution_method=method, time_stamp=time))

        if (idx + 1) % 3 == 0:
            fcn_difference += compute_average_difference(fcn_attributions)
            fcn_attributions = []
            sfno_difference += compute_average_difference(sfno_attributions)
            sfno_attributions = []

    return fcn_difference / 3.0, sfno_difference / 3.0


def compute_mean_variance(method, smoother):
    fcn = data_loading.MODELS[0]
    sfno = data_loading.MODELS[1]

    variances = []
    for idx, time in enumerate(data_loading.DATETIMES):
        fcn_attribution = data_loading.get_attribution(model=fcn, smoother=smoother, attribution_method=method,
                                                       time_stamp=time)
        sfno_attribution = data_loading.get_attribution(model=sfno, smoother=smoother, attribution_method=method,
                                                        time_stamp=time)

        stacked = np.stack([fcn_attribution, sfno_attribution], axis=0)
        variance = np.var(stacked, axis=0)
        variances.append(np.mean(variance))

    return np.mean(variances)


def compute_mean_attribution():
    fcn = data_loading.MODELS[0]
    sfno = data_loading.MODELS[1]

    fcn_avg = np.zeros((24, 720, 1440))
    sfno_avg = np.zeros((24, 720, 1440))

    n = 0
    for method in data_loading.METHODS:
        for smoother in data_loading.SMOOTHERS:
            for time in data_loading.DATETIMES:
                fcn_attribution = data_loading.get_attribution(model=fcn, smoother=smoother, attribution_method=method,
                                                               time_stamp=time)
                fcn_avg += fcn_attribution
                sfno_attribution = data_loading.get_attribution(model=sfno, smoother=smoother,
                                                                attribution_method=method, time_stamp=time)
                sfno_avg += sfno_attribution
                n += 1

    fcn_avg /= n
    sfno_avg /= n

    return fcn_avg, sfno_avg


def compute_per_method_mean_attribution(method, smoother):
    fcn = data_loading.MODELS[0]
    sfno = data_loading.MODELS[1]

    fcn_avg = np.zeros((24, 720, 1440))
    sfno_avg = np.zeros((24, 720, 1440))

    n = 0
    for time in data_loading.DATETIMES:
        fcn_attribution = data_loading.get_attribution(model=fcn, smoother=smoother, attribution_method=method,
                                                       time_stamp=time)
        fcn_avg += fcn_attribution
        sfno_attribution = data_loading.get_attribution(model=sfno, smoother=smoother, attribution_method=method,
                                                        time_stamp=time)
        sfno_avg += sfno_attribution
        n += 1

    fcn_avg /= n
    sfno_avg /= n

    return fcn_avg, sfno_avg


def compute_pairwise_mean_ssim_between_methods(smoother, model):
    n = len(data_loading.METHODS)
    M = np.eye(n, dtype=np.float64)

    for i in range(n):
        method_i = data_loading.METHODS[i]
        attribution_i = data_loading.load_per_method_average_attribution(method=method_i, smoother=smoother,
                                                                         model=model)
        for j in range(i + 1, n):
            method_j = data_loading.METHODS[j]
            attribution_j = data_loading.load_per_method_average_attribution(method=method_j, smoother=smoother,
                                                                             model=model)
            score = ssim(attribution_i, attribution_j, data_range=1.0, channel_axis=0, gaussian_weights=True)
            M[i, j] = M[j, i] = score

    return M


def compute_pairwise_mean_ssim_between_smoothers(method, model):
    n = len(data_loading.SMOOTHERS)
    M = np.eye(n, dtype=np.float64)

    for i in range(n):
        smoother_i = data_loading.SMOOTHERS[i]
        attribution_i = data_loading.load_per_method_average_attribution(method=method, smoother=smoother_i,
                                                                         model=model)
        for j in range(i + 1, n):
            smoother_j = data_loading.SMOOTHERS[j]
            attribution_j = data_loading.load_per_method_average_attribution(method=method, smoother=smoother_j,
                                                                             model=model)
            score = ssim(attribution_i, attribution_j, data_range=1.0, channel_axis=0, gaussian_weights=True)
            M[i, j] = M[j, i] = score

    return M


def compute_pairwise_mean_ssim_between_combos(model):
    methods = data_loading.METHODS
    smoothers = data_loading.SMOOTHERS
    combos = list(itertools.product(methods, smoothers))
    n = len(combos)
    print(n)
    M = np.eye(n, dtype=np.float64)

    for i in range(n):
        combo_i = combos[i]
        method, smoother = combo_i
        attribution_i = data_loading.load_per_method_average_attribution(method=method, smoother=smoother, model=model)
        for j in range(i + 1, n):
            combo_j = combos[j]
            method, smoother = combo_j
            attribution_j = data_loading.load_per_method_average_attribution(method=method, smoother=smoother,
                                                                             model=model)
            score = ssim(attribution_i, attribution_j, data_range=1.0, channel_axis=0, gaussian_weights=True)
            M[i, j] = M[j, i] = score

    return M


def run_full_analysis():
    thresholds = np.linspace(0.0, 1.0, 21)
    np.save(Folders.output / f"lin_thresholds.npy", thresholds)

    model_diff_rows = []
    fcn_diff_rows = []
    sfno_diff_rows = []
    mean_variance_rows = []
    for method in data_loading.METHODS:
        for smoother in data_loading.SMOOTHERS:
            fcn_method_avg, sfno_method_avg = compute_per_method_mean_attribution(method, smoother)
            np.save(Folders.output / f"{method}_{smoother}_fcn_average_attribution.npy", fcn_method_avg)
            np.save(Folders.output / f"{method}_{smoother}_sfno_average_attribution.npy", sfno_method_avg)

            mean_variance = compute_mean_variance(method, smoother)
            mean_variance_rows.append({"method": method, "smoother": smoother, "mean_variance": mean_variance})

    agreement_curves = compute_agreement_curve_between_models(method, smoother, thresholds)
    np.save(Folders.output / f"{method}_{smoother}_lin_threshold_separate_curves.npy", agreement_curves)

    fcn_timestamp_agreements, sfno_timestamp_agreements = compute_agreement_curve_between_timestamps(method, smoother,
                                                                                                     thresholds)
    np.save(Folders.output / f"{method}_{smoother}_fcn_lin_threshold_curve.npy", fcn_timestamp_agreements)
    np.save(Folders.output / f"{method}_{smoother}_sfno_lin_threshold_curve.npy", sfno_timestamp_agreements)

    model_difference = compute_average_difference_between_models(method, smoother)
    fcn_difference, sfno_difference = compute_average_difference_between_timestamps(method, smoother)

    model_diff_rows.append({"method": method, "smoother": smoother, "difference": model_difference})
    fcn_diff_rows.append({"method": method, "smoother": smoother, "difference": fcn_difference})
    sfno_diff_rows.append({"method": method, "smoother": smoother, "difference": sfno_difference})

    model_differences = pd.DataFrame(model_diff_rows, columns=["method", "smoother", "difference"])
    fcn_differences = pd.DataFrame(fcn_diff_rows, columns=["method", "smoother", "difference"])
    sfno_differences = pd.DataFrame(sfno_diff_rows, columns=["method", "smoother", "difference"])

    model_differences.to_csv(Folders.output / f"model_differences.csv", index=False)
    fcn_differences.to_csv(Folders.output / f"fcn_differences.csv", index=False)
    sfno_differences.to_csv(Folders.output / f"sfno_differences.csv", index=False)

    mean_variances = pd.DataFrame(mean_variance_rows, columns=["method", "smoother", "mean_variance"])
    mean_variances.to_csv(Folders.output / "mean_variances.csv", index=False)

    for model in data_loading.MODELS:
        similarities = compute_pairwise_mean_ssim_between_combos(model=model)
        np.save(Folders.output / "similarities" / f"{model}_similarities.npy", similarities)


if __name__ == '__main__':
    run_full_analysis()
