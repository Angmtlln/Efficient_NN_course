import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import least_squares, nnls

import equations as eq

RESULTS = Path(__file__).parent / "results"


def load_measurements(path):
    df = pd.read_csv(path)
    df["oom"] = df["status"].eq("OOM")
    df["memory_bytes"] = pd.to_numeric(df["memory_bytes"], errors="coerce")
    df["is_validation"] = df["is_validation"].astype(str).str.lower().eq("true")
    return df


def fit_latency(S, B, T):
    def residuals(log_theta):
        return np.log(eq.latency(S, B, np.exp(log_theta))) - np.log(T)

    best = None
    for start in itertools.product((5e-6, 2e-5), (5e12, 20e12), (200e9, 500e9)):
        res = least_squares(residuals, np.log(start), diff_step=1e-3)
        if best is None or res.cost < best.cost:
            best = res
    tau, P, BW = np.exp(best.x)
    return dict(tau=tau, P=P, BW=BW)


def fit_energy(S, B, E, theta):
    A = np.stack([eq.latency(S, B, theta), eq.flops(S, B), eq.bytes_moved(S, B)], axis=1)
    scale = A.max(axis=0) 
    coef, _ = nnls(A / scale / E[:, None], np.ones_like(E))
    P_s, e_F, e_M = coef / scale
    return dict(P_s=P_s, e_F=e_F, e_M=e_M, **theta)


def rel_error(pred, meas):
    return (pred - meas) / meas


def summary(err):
    err = np.abs(np.asarray(err))
    return dict(mape=float(np.mean(err)), median_ape=float(np.median(err)), max_ape=float(np.max(err)))


TRAIN, VAL = "#2a78d6", "#eb6834"
REGIME_COLORS = {"launch": "#1baf7a", "memory": "#eb6834", "compute": "#2a78d6"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "legend.frameon": False, "font.size": 10, "lines.linewidth": 1.6,
})


def s_colors(sizes):
    cmap = plt.get_cmap("Blues")
    return {s: cmap(v) for s, v in zip(sizes, np.linspace(0.35, 0.95, len(sizes)))}


def per_size_plot(ax, ok, value_col, predict, sizes, ylabel, scale=1.0):
    colors = s_colors(sizes)
    b_grid = np.geomspace(1, 256, 200)
    for S in sizes:
        c = colors[S]
        ax.plot(b_grid, predict(S, b_grid) * scale, color=c, lw=1.6)
        part = ok[ok["S"] == S]
        base, val = part[~part["is_validation"]], part[part["is_validation"]]
        ax.plot(base["B"], base[value_col] * scale, "o", color=c, ms=5, mec="white", mew=0.8)
        ax.plot(val["B"], val[value_col] * scale, "o", mfc="white", mec=c, ms=5, mew=1.4)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size B")
    ax.set_ylabel(ylabel)
    handles = [Line2D([], [], color=colors[S], lw=1.6, label=f"S = {S}") for S in sizes]
    handles += [Line2D([], [], color=INK_MUTED, lw=1.6, label="predicted"),
                Line2D([], [], ls="", marker="o", color=INK_MUTED, label="measured"),
                Line2D([], [], ls="", marker="o", mfc="white", mec=INK_MUTED, mew=1.4,
                       label="measured (validation B)")]
    ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=2)


def parity(ax, ok, meas_col, pred_col, title, unit, scale=1.0):
    for split, color, label in ((False, TRAIN, "calibration (base grid)"), (True, VAL, "validation")):
        part = ok[ok["is_validation"] == split]
        ax.plot(part[meas_col] * scale, part[pred_col] * scale, "o", color=color, ms=4.5,
                alpha=0.85, mec="white", mew=0.5, label=label)
    lo = min(ok[meas_col].min(), ok[pred_col].min()) * scale
    hi = max(ok[meas_col].max(), ok[pred_col].max()) * scale
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, lw=1, ls="--", label="ideal: predicted = measured")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"measured {unit}")
    ax.set_ylabel(f"predicted {unit}")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")


def make_figures(fig_dir, ok, raw, theta, theta_e, env):
    fig_dir.mkdir(parents=True, exist_ok=True)
    ok = ok.copy()
    gpu = env.get("gpu", "GPU")
    env.setdefault("gpu_memory_bytes", float("nan"))
    sizes = sorted(s for s in ok["S"].unique() if s not in set(env.get("extra_S", [])))


    fig, ax = plt.subplots(figsize=(8, 5.5))
    per_size_plot(ax, ok, "latency_s", lambda S, B: eq.latency(S, B, theta), sizes,
                  "latency of one forward pass, ms", scale=1e3)
    ax.set_title(f"Latency vs batch size ({gpu}, FP32)")
    fig.savefig(fig_dir / "latency_vs_batch.png")
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(8, 5.5))
    per_size_plot(ax, ok, "memory_bytes", eq.memory, sizes, "peak allocated memory, MiB",
                  scale=1 / 2**20)
    ax.axhline(env["gpu_memory_bytes"] / 2**20, color=INK_MUTED, ls=":", lw=1)
    ax.text(1.1, env["gpu_memory_bytes"] / 2**20 * 0.8, "GPU memory", color=INK_MUTED, fontsize=8)
    oom = raw[raw["status"] == "OOM"]
    if len(oom):
        ax.plot(oom["B"], np.full(len(oom), env["gpu_memory_bytes"] / 2**20), "x", color=INK,
                ms=7, label="OOM (measured)")
    ax.set_title("Peak memory vs batch size (max_memory_allocated)")
    fig.savefig(fig_dir / "memory_vs_batch.png")
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(8, 5.5))
    per_size_plot(ax, ok, "energy_j", lambda S, B: eq.energy(S, B, theta_e), sizes,
                  "energy of one forward pass, mJ", scale=1e3)
    ax.set_title("Energy vs batch size (whole GPU, NVML)")
    fig.savefig(fig_dir / "energy_vs_batch.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    per_size_plot(ax, ok, "flops_counted", eq.flops, sizes, "GFLOPs of one forward pass", scale=1e-9)
    ax.set_title("FLOPs: formula (lines) vs torch FlopCounterMode (points)")
    fig.savefig(fig_dir / "flops_vs_batch.png")
    plt.close(fig)


    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    parity(axes[0, 0], ok, "flops_counted", "flops_pred", "FLOPs", "GFLOP", 1e-9)
    parity(axes[0, 1], ok, "memory_bytes", "memory_pred", "Memory", "MiB", 1 / 2**20)
    parity(axes[1, 0], ok, "latency_s", "latency_pred", "Latency", "ms", 1e3)
    parity(axes[1, 1], ok, "energy_j", "energy_pred", "Energy", "mJ", 1e3)
    fig.suptitle("Predicted vs measured, all (S, B) points", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "parity.png")
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(8, 5.5))
    ok["throughput"] = ok["flops_counted"] / ok["latency_s"]
    per_size_plot(ax, ok, "throughput",
                  lambda S, B: eq.flops(S, B) / eq.latency(S, B, theta), sizes,
                  "achieved throughput FLOPs / latency, TFLOP/s", scale=1e-12)
    ax.axhline(theta["P"] / 1e12, color=INK_MUTED, ls=":", lw=1)
    ax.text(1.1, theta["P"] / 1e12 * 1.08, f"fitted P = {theta['P'] / 1e12:.1f} TFLOP/s",
            color=INK_MUTED, fontsize=8)
    ax.legend_.set_loc("lower right")
    ax.set_title("Achieved FP32 throughput vs batch size")
    fig.savefig(fig_dir / "throughput_vs_batch.png")
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(8, 5.5))
    ai = eq.flops(ok["S"], ok["B"]) / eq.bytes_moved(ok["S"], ok["B"])
    for name, color in REGIME_COLORS.items():
        m = ok["regime_pred"] == name
        if m.any():
            ax.plot(ai[m], ok.loc[m, "throughput"] / 1e12, "o", color=color, ms=5, mec="white",
                    mew=0.5, label=f"measured, predicted regime: {name}-bound")
    ax.plot(ai, ok["flops_pred"] / ok["latency_pred"] / 1e12, "o", mfc="none", mec=INK_MUTED,
            ms=7, mew=0.8, label="predicted by Latency(S, B, θ)")
    ai_grid = np.geomspace(ai.min() * 0.8, ai.max() * 1.2, 200)
    ax.plot(ai_grid, np.minimum(theta["P"], ai_grid * theta["BW"]) / 1e12, color=INK, lw=1.6,
            label="roofline min(P, AI·BW) with fitted P, BW")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity FLOPs / Bytes, FLOP/byte")
    ax.set_ylabel("achieved throughput, TFLOP/s")
    ax.set_title("Roofline: whole forward pass")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(fig_dir / "roofline.png")
    plt.close(fig)

    all_S, all_B = sorted(ok["S"].unique()), sorted(ok["B"].unique())
    names = list(REGIME_COLORS)
    grid = np.full((len(all_S), len(all_B)), np.nan)
    fig, ax = plt.subplots(figsize=(11, 7))
    for _, r in ok.iterrows():
        grid[all_S.index(r["S"]), all_B.index(r["B"])] = names.index(r["regime_pred"])
    cmap = ListedColormap([matplotlib.colors.to_rgba(c, 0.35) for c in REGIME_COLORS.values()])
    ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto", origin="lower")
    for _, r in ok.iterrows():
        i, j = all_S.index(r["S"]), all_B.index(r["B"])
        ax.text(j, i, f"{r['latency_rel_err'] * 100:+.0f}%", ha="center", va="center", fontsize=8,
                color=INK, fontweight="bold" if abs(r["latency_rel_err"]) > 0.2 else "normal")
    for _, r in raw[raw["status"] == "OOM"].iterrows():
        ax.text(all_B.index(r["B"]), all_S.index(r["S"]), "OOM", ha="center", va="center", fontsize=8)
    extra_S, extra_B = set(env.get("extra_S", [])), set(env.get("extra_B", []))
    ax.set_xticks(range(len(all_B)), [f"{b}*" if b in extra_B else str(b) for b in all_B])
    ax.set_yticks(range(len(all_S)), [f"{s}*" if s in extra_S else str(s) for s in all_S])
    ax.set_xticks(np.arange(len(all_B)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(all_S)) - 0.5, minor=True)
    ax.grid(True, which="minor", color="white", lw=1.5)
    ax.grid(False, which="major")
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("batch size B   (* = randomly sampled validation value)")
    ax.set_ylabel("image size S")
    ax.set_title("Predicted dominant regime (colour) and latency error (predicted - measured) / measured")
    ax.legend(handles=[Patch(color=matplotlib.colors.to_rgba(c, 0.35), label=f"{n}-bound")
                       for n, c in REGIME_COLORS.items()],
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(fig_dir / "regime_map.png")
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=RESULTS / "measurements.csv")
    args = parser.parse_args()
    out_dir = args.csv.parent

    df = load_measurements(args.csv)
    ok = df[~df["oom"]].copy()
    train, val = ok[~ok["is_validation"]], ok[ok["is_validation"]]
    print(f"rows: {len(df)}  ok: {len(ok)}  OOM: {int(df['oom'].sum())}  "
          f"train: {len(train)}  validation: {len(val)}")

    theta = fit_latency(train["S"].values, train["B"].values, train["latency_s"].values)
    theta_energy = fit_energy(train["S"].values, train["B"].values, train["energy_j"].values, theta)

    S, B = ok["S"].values, ok["B"].values
    ok["flops_pred"] = eq.flops(S, B)
    ok["memory_pred"] = eq.memory(S, B)
    ok["latency_pred"] = eq.latency(S, B, theta)
    ok["energy_pred"] = eq.energy(S, B, theta_energy)
    shares = eq.regime(S, B, theta)
    ok["share_launch"], ok["share_memory"], ok["share_compute"] = shares
    ok["regime_pred"] = np.array(["launch", "memory", "compute"])[shares.argmax(axis=0)]
    for name, col in [("flops", "flops_counted"), ("memory", "memory_bytes"),
                      ("latency", "latency_s"), ("energy", "energy_j")]:
        ok[f"{name}_rel_err"] = rel_error(ok[f"{name}_pred"], ok[col])

    errors = {}
    for name in ("flops", "memory", "latency", "energy"):
        errors[name] = {split: summary(part[f"{name}_rel_err"])
                        for split, part in (("train", ok[~ok["is_validation"]]),
                                            ("validation", ok[ok["is_validation"]]))}

    env_path, theta_path = out_dir / "env.json", out_dir / "theta.json"
    if env_path.exists():
        env = json.loads(env_path.read_text())
    elif theta_path.exists():
        env = json.loads(theta_path.read_text()).get("env", {})
    else:
        env = {}

    gpu_mem = env.get("gpu_memory_bytes")
    oom = None
    if gpu_mem is not None:
        predicted_oom = eq.memory(df["S"].values, df["B"].values) > gpu_mem
        oom = dict(gpu_memory_bytes=gpu_mem,
                   measured=int(df["oom"].sum()), predicted=int(predicted_oom.sum()),
                   agree=int((predicted_oom == df["oom"].values).sum()), total=len(df))

    result = dict(
        latency=theta,
        energy=theta_energy,
        units=dict(tau="s", P="FLOP/s", BW="bytes/s", P_s="W", e_F="J/FLOP", e_M="J/byte"),
        errors=errors,
        oom=oom,
        n_train=len(train),
        n_validation=len(val),
        env=env,
    )
    (out_dir / "theta.json").write_text(json.dumps(result, indent=2))
    make_figures(out_dir / "figures", ok, df, theta, theta_energy, env)

    print(f"\nlatency theta: tau = {theta['tau'] * 1e6:.2f} us, P = {theta['P'] / 1e12:.2f} TFLOP/s, "
          f"BW = {theta['BW'] / 1e9:.1f} GB/s")
    print(f"energy theta:  P_s = {theta_energy['P_s']:.1f} W, e_F = {theta_energy['e_F'] * 1e12:.2f} pJ/FLOP, "
          f"e_M = {theta_energy['e_M'] * 1e12:.2f} pJ/byte")
    print("\nrelative error |pred - meas| / meas:")
    for name, parts in errors.items():
        print(f"  {name:8s} " + "  ".join(
            f"{split}: MAPE {s['mape'] * 100:6.1f}%  max {s['max_ape'] * 100:6.1f}%"
            for split, s in parts.items()))
    if oom:
        print(f"\nOOM measured: {oom['measured']}, predicted: {oom['predicted']}, "
              f"agree on {oom['agree']}/{oom['total']} configs")
    print(f"\nsaved {out_dir / 'theta.json'} and {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
