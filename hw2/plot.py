import argparse
import itertools
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10

from config import DATA_ROOT, get_config, main_methods, output_dir, score_names, window_score
from data import MEAN, STD

EXAMPLE_COUNT = 10


def method_label(method):
    if method == "random":
        return "Random"
    if method == "grand_e0":
        return "GraNd, initialization"
    name, epoch = method.split("_e")
    return f"{'EL2N' if name == 'el2n' else 'GraNd'}, epoch {epoch}"


def load_results(directory, key_fields):
    groups = defaultdict(list)
    for path in sorted(directory.glob("*.json")):
        result = json.loads(path.read_text())
        groups[tuple(result[field] for field in key_fields)].append(result["test_accuracy"])
    return groups


def write_table(groups, header, path):
    lines = [",".join(header) + ",runs,mean_test_accuracy,std_test_accuracy"]
    for key, accs in sorted(groups.items()):
        lines.append(",".join(str(value) for value in key) + f",{len(accs)},{np.mean(accs):.4f},{np.std(accs):.4f}")
    path.write_text("\n".join(lines) + "\n")


def mean_std_percent(accs):
    return 100 * np.mean(accs), 100 * np.std(accs)


def draw_baselines(ax, groups, fraction):
    """Horizontal lines for training on the full dataset and on a random subset with the given fraction pruned."""
    full = groups.get(("full", 0.0))
    if full:
        ax.axhline(100 * np.mean(full), color="black", linestyle="--", label="Full dataset")
    random = groups.get(("random", fraction))
    if random:
        ax.axhline(100 * np.mean(random), color="gray", linestyle=":", label=f"Random, {round(100 * fraction)}% pruned")


def draw_curve(ax, xs, accs_list, label):
    stats = np.array([mean_std_percent(accs) for accs in accs_list])
    ax.plot(xs, stats[:, 0], marker="o", label=label)
    ax.fill_between(xs, stats[:, 0] - stats[:, 1], stats[:, 0] + stats[:, 1], alpha=0.2)


def finish_figure(fig, ax, xlabel, title, path):
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test accuracy, %")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_main(groups, cfg, path):
    """Test accuracy against the fraction of pruned training data for every main method."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    draw_baselines(ax, groups, fraction=None)
    for method in main_methods(cfg):
        fractions = [f for f in cfg["fractions"] if (method, f) in groups]
        if fractions:
            draw_curve(ax, 100 * np.array(fractions), [groups[(method, f)] for f in fractions], method_label(method))
    finish_figure(fig, ax, "Fraction of training data pruned, %", "Pruning the lowest-scoring examples", path)


def plot_early(groups, cfg, path):
    """Test accuracy at early_fraction pruned against the epoch at which the scores are computed."""
    fraction = cfg["early_fraction"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    draw_baselines(ax, groups, fraction)
    for prefix, label, epochs in (("el2n", "EL2N", cfg["score_epochs"]), ("grand", "GraNd", [0, *cfg["score_epochs"]])):
        epochs = [epoch for epoch in epochs if (f"{prefix}_e{epoch}", fraction) in groups]
        if epochs:
            draw_curve(ax, epochs, [groups[(f"{prefix}_e{epoch}", fraction)] for epoch in epochs], label)
    ax.set_xticks([0, *cfg["score_epochs"]])
    title = f"{round(100 * fraction)}% of training data pruned"
    finish_figure(fig, ax, "Epoch at which scores are computed", title, path)


def plot_window(window_groups, groups, cfg, path):
    """Test accuracy against the number of highest-scoring examples excluded from the training window."""
    keep_fraction = cfg["window_keep_fraction"]
    offsets = sorted(offset for (offset,) in window_groups)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    draw_baselines(ax, groups, round(1 - keep_fraction, 2))
    if offsets:
        draw_curve(ax, offsets, [window_groups[(offset,)] for offset in offsets], method_label(window_score(cfg)))
    positive_offsets = [offset for offset in offsets if offset > 0]
    if positive_offsets:
        ax.set_xscale("symlog", linthresh=min(positive_offsets))
        ax.set_xticks(offsets)
        ax.set_xticklabels([str(offset) for offset in offsets])
    title = f"{round(100 * keep_fraction)}% of training data kept after excluding the highest-scoring examples"
    finish_figure(fig, ax, "Number of highest-scoring examples excluded", title, path)


def plot_examples(cfg, out, dataset, path):
    """Top row: examples with the lowest EL2N. Bottom row: examples with the highest EL2N."""
    name = main_methods(cfg)[1]
    score = np.load(out / "scores" / f"{name}.npy")
    order = np.argsort(score, kind="stable")
    rows = [order[:EXAMPLE_COUNT], order[::-1][:EXAMPLE_COUNT]]
    fig, axes = plt.subplots(2, EXAMPLE_COUNT, figsize=(1.6 * EXAMPLE_COUNT, 4.2))
    for row, indices in enumerate(rows):
        for col, index in enumerate(indices):
            ax = axes[row, col]
            ax.imshow(dataset.data[index])
            ax.set_title(f"{dataset.classes[dataset.targets[index]]}\n{score[index]:.3f}", fontsize=8)
            ax.axis("off")
    fig.text(0.005, 0.73, f"Lowest {method_label(name)}", rotation=90, va="center", fontsize=9)
    fig.text(0.005, 0.27, f"Highest {method_label(name)}", rotation=90, va="center", fontsize=9)
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def spearman(a, b):
    rank_a = np.argsort(np.argsort(a))
    rank_b = np.argsort(np.argsort(b))
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def write_rank_correlations(cfg, out, dataset, path):
    """Spearman rank correlation between every pair of main scores and the L2 norm of the normalized input."""
    scores = {name: np.load(out / "scores" / f"{name}.npy") for name in main_methods(cfg)[1:]}
    x = dataset.data.astype(np.float32) / 255
    x = (x - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    scores["input_norm"] = np.linalg.norm(x.reshape(len(x), -1), axis=1)
    lines = [f"spearman({a}, {b}) = {spearman(scores[a], scores[b]):.3f}" for a, b in itertools.combinations(scores, 2)]
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main(cfg, out):
    groups = load_results(out / "results", ("method", "fraction"))
    write_table(groups, ("method", "fraction_pruned"), out / "results.csv")
    plot_main(groups, cfg, out / "accuracy_vs_pruned.png")
    plot_early(groups, cfg, out / "accuracy_vs_score_epoch.png")

    window_groups = load_results(out / "window_results", ("offset",))
    write_table(window_groups, ("offset",), out / "window_results.csv")
    plot_window(window_groups, groups, cfg, out / "accuracy_vs_window_offset.png")

    if all((out / "scores" / f"{name}.npy").exists() for name in score_names(cfg)):
        dataset = CIFAR10(DATA_ROOT, train=True, download=False)
        plot_examples(cfg, out, dataset, out / "el2n_examples.png")
        write_rank_correlations(cfg, out, dataset, out / "rank_correlations.txt")
    print(f"results are in {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(get_config(args.smoke), output_dir(args.smoke))
