import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.ticker import FuncFormatter

import data_loading
from paths import Folders


def plot_threshold_agreements_curve(thresholds, threshold_agreements, label=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_title("Threshold Agreements")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Agreement")
        ax.grid(True)

    ax.set_yscale("log")
    ax.plot(thresholds, threshold_agreements, label=label)
    ax.legend()

    return ax


def plot_attribution(attribution, title=""):
    fig, ax = plt.subplots()
    ax.set_title(title)

    im = ax.imshow(attribution, origin="lower", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)

    return ax


def plot_distribution(arr, bins=10000, title="Value distribution"):
    values = arr.ravel()  # flatten
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins, density=True, alpha=0.7, edgecolor='black')
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True, ls="--", alpha=0.5)
    plt.show()


def format_title(string):
    string = string.replace("_", " ")
    string = string.title()
    return string


def format_file_name(string):
    string = string.replace(" ", "_")
    string = string.lower()
    return string


def plot_method_agreement_curve(method, model=None):
    thresholds = data_loading.load_thresholds()
    xticks = np.linspace(thresholds.min(), thresholds.max(), 11)
    plt.figure(figsize=(8, 4))
    if model is None:
        title = f"{format_title(method)} Threshold Agreement"
    else:
        title = f"{format_title(method)} Threshold Agreement in {model.upper()}"
    plt.title(title)
    plt.xlabel("Threshold")
    plt.ylabel("Agreement")
    plt.yscale("log")
    plt.grid(True, which="both", axis="x", linestyle="--", linewidth=0.5)

    plt.xticks(xticks, [f"{t:.1f}" for t in xticks])

    exponents = np.linspace(0.0, 9.0, 10)
    yticks = 10 ** (-exponents)
    plt.gca().set_yticks(yticks)

    def format_func(y, _):
        if y >= 1e-2:
            return f"{y:.2f}"
        else:
            exp = int(np.log10(y))
            return f"$10^{{{exp}}}$"

    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func))

    plt.xlim(-0.05, 1.05)
    plt.ylim(10 ** (-10), 10 ** 1)

    curves = []
    for smoother in data_loading.SMOOTHERS:
        curve = data_loading.load_curve(method, smoother=smoother, model=model)
        if model is not None:
            curve /= 10 / 3  # to fix a bug when generating the agreement
        curves.append(curve)
        plt.plot(thresholds, curve, label=format_title(smoother))

    curves = np.stack(curves, axis=0)
    curves = curves[:, 16:]

    means = np.mean(curves, axis=1)  # per smoother mean
    max_mean = np.max(means, axis=0)
    argmax_mean = np.argmax(means, axis=0)
    best_mean_smoother = data_loading.SMOOTHERS[argmax_mean]
    min_mean = np.min(means, axis=0)
    argmin_mean = np.argmin(means, axis=0)
    worst_mean_smoother = data_loading.SMOOTHERS[argmin_mean]

    medians = np.median(curves, axis=1)
    max_median = np.max(medians, axis=0)
    argmax_median = np.argmax(medians, axis=0)
    best_median_smoother = data_loading.SMOOTHERS[argmax_median]
    min_median = np.min(medians, axis=0)
    argmin_median = np.argmin(medians, axis=0)
    worst_median_smoother = data_loading.SMOOTHERS[argmin_median]

    average_curve = curves.mean(axis=0)
    overall_average = average_curve.mean()
    median_average = np.median(average_curve)

    plt.legend()
    plt.tight_layout()

    file = f"{format_file_name(title)}.pdf"
    plt.savefig(Folders.plots / file, format="pdf")

    stats = {"Method": format_title(method), "Overall Mean": overall_average, "Overall Median": median_average,
             "Maximum Mean": max_mean, "Mean Best Smoother": format_title(best_mean_smoother),
             "Minimum Mean": min_mean, "Mean Worst Smoother": format_title(worst_mean_smoother),
             "Maximum Median": max_median, "Median Best Smoother": format_title(best_median_smoother),
             "Minimum Median": min_median, "Median Worst Smoother": format_title(worst_median_smoother)}

    return stats


def plot_channel_importance():
    fcn = data_loading.MODELS[0]
    sfno = data_loading.MODELS[1]

    fcn_avg = data_loading.load_average_attribution(fcn)
    sfno_avg = data_loading.load_average_attribution(sfno)

    fcn_channel_importance = fcn_avg.mean(axis=(1, 2))
    sfno_channel_importance = sfno_avg.mean(axis=(1, 2))

    labels = data_loading.COMMON_VARIABLES

    fcn_idx = np.argsort(fcn_channel_importance)
    fcn_sorted = fcn_channel_importance[fcn_idx]
    fcn_labels = [labels[i] for i in fcn_idx]
    print("FCN")
    for lab, val in zip(fcn_labels, fcn_sorted):
        print(f"{lab}: {val:.4e}")

    sfno_idx = np.argsort(sfno_channel_importance)
    sfno_sorted = sfno_channel_importance[sfno_idx]
    sfno_labels = [labels[i] for i in sfno_idx]
    print("SFNO")
    for lab, val in zip(sfno_labels, sfno_sorted):
        print(f"{lab}: {val:.4e}")

    y = np.arange(len(labels))

    bar_height = 0.4

    plt.figure(figsize=(8, 8))
    plt.title("Average Variable Importance")

    plt.barh(y + bar_height / 2, fcn_channel_importance, height=bar_height, label="FCN")
    plt.barh(y - bar_height / 2, sfno_channel_importance, height=bar_height, label="SFNO")

    plt.yticks(y, labels=labels)
    plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    plt.gca().ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    plt.xlabel("Importance")
    plt.ylabel("Variable")
    plt.grid(True, which="both", axis="x", linestyle="--", linewidth=0.5)

    plt.legend()
    plt.tight_layout()
    plt.savefig(Folders.plots / "variable_importance.pdf", format="pdf")


def plot_average_channel_importance():
    fcn = data_loading.MODELS[0]
    sfno = data_loading.MODELS[1]

    fcn_avg = data_loading.load_average_attribution(fcn)
    sfno_avg = data_loading.load_average_attribution(sfno)

    fcn_channel_importance = fcn_avg.mean(axis=(1, 2))
    sfno_channel_importance = sfno_avg.mean(axis=(1, 2))
    average_channel_importance = (fcn_channel_importance + sfno_channel_importance) / 2

    idx = np.argsort(average_channel_importance)
    print(idx)
    # idx = idx[::-1]
    # print(idx)
    sorted_importance = average_channel_importance[idx]
    labels = [data_loading.COMMON_VARIABLES[i] for i in idx]

    y = np.arange(len(labels))

    bar_height = 0.4

    plt.figure(figsize=(8, 8))
    plt.title("Combined Average Variable Importance")

    plt.barh(y, sorted_importance, height=bar_height)

    plt.yticks(y, labels=labels)
    plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    plt.gca().ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    plt.xlabel("Importance")
    plt.ylabel("Variable")
    plt.grid(True, which="both", axis="x", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(Folders.plots / "average_variable_importance.pdf", format="pdf")


def make_similarity_plot(similarity_matrix, labels, title):
    n = len(similarity_matrix)
    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(similarity_matrix, vmin=0.990, vmax=1.0, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("SSIM")

    ax.set_title(title)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Method-Smoother Combination")
    ax.set_ylabel("Method-Smoother Combination")

    # Show gridlines between cells (nice for small N)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="w", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n):
        for j in range(n):
            if similarity_matrix[i, j] >= 0.99999:
                ax.text(j, i, f"{similarity_matrix[i, j]:.1f}",
                        ha="center", va="center",
                        fontsize=9, color="black")
            elif similarity_matrix[i, j] <= 0.997:
                num = f"{similarity_matrix[i, j]:.4f}"
                l = f"..{num[4:]}"
                ax.text(j, i, l,
                        ha="center", va="center",
                        fontsize=9, color="black")

    fig.tight_layout()
    plt.savefig(Folders.plots / f"{format_file_name(title)}.pdf", format="pdf")


def plot_similarities():
    method_similarities = []
    smoother_similarities = []
    for model in data_loading.MODELS:
        for smoother in data_loading.SMOOTHERS:
            method_similarities.append(data_loading.load_similarities(model=model, smoother=smoother))
        for method in data_loading.METHODS:
            smoother_similarities.append(data_loading.load_similarities(model=model, method=method))

    method_similarities = np.stack(method_similarities, axis=0)
    smoother_similarities = np.stack(smoother_similarities, axis=0)

    mean_method_similarities = np.mean(method_similarities, axis=0)
    mean_smoother_similarities = np.mean(smoother_similarities, axis=0)

    make_similarity_plot(mean_method_similarities, data_loading.METHODS, "Mean Method Similarity")
    make_similarity_plot(mean_smoother_similarities, data_loading.SMOOTHERS, "Mean Smoother Similarity")


def abbreviate(string: str):
    string = string.split("_")
    new_string = "".join([s[0] for s in string])
    return new_string.upper()
