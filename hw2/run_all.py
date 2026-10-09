import argparse
import subprocess
import sys
import time

import numpy as np

import plot
from config import (
    DATA_ROOT,
    ROOT,
    early_methods,
    get_config,
    main_methods,
    output_dir,
    run_name,
    score_names,
    window_run_name,
    window_score,
)
from data import download

POLL_SECONDS = 5


def score_done(cfg, out, seed):
    raw_dir = out / "scores" / "raw"
    return all((raw_dir / f"{name}_seed{seed}.npy").exists() for name in score_names(cfg))


def run_jobs(jobs, workers, log_dir):
    """Run train.py jobs as separate processes, at most `workers` at a time. Return names of failed jobs."""
    pending = list(jobs)
    running = []
    failed = []
    finished = 0
    while pending or running:
        while pending and len(running) < workers:
            name, argv = pending.pop(0)
            log = open(log_dir / f"{name}.log", "w")
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "train.py"), *argv],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
            )
            running.append((name, process, log))
        time.sleep(POLL_SECONDS)
        still_running = []
        for name, process, log in running:
            exit_code = process.poll()
            if exit_code is None:
                still_running.append((name, process, log))
                continue
            log.close()
            finished += 1
            status = "done" if exit_code == 0 else f"failed with exit code {exit_code}"
            print(f"[{time.strftime('%H:%M:%S')}] {finished}/{len(jobs)} {name} {status}", flush=True)
            if exit_code != 0:
                failed.append(name)
        running = still_running
    return failed


def aggregate_scores(cfg, out):
    for name in score_names(cfg):
        arrays = [np.load(out / "scores" / "raw" / f"{name}_seed{seed}.npy") for seed in cfg["score_seeds"]]
        np.save(out / "scores" / f"{name}.npy", np.mean(arrays, axis=0))


def training_jobs(cfg, out, smoke_flag):
    """Return (name, argv) for every training run without a saved result.

    Jobs are ordered by experiment: main experiment first, then scores from different epochs, then the sliding window.
    Within one experiment, jobs with larger training sets start first.
    """
    jobs = {}

    def add_fit_job(stage, method, fraction, seed):
        name = run_name(method, fraction, seed)
        if name in jobs or (out / "results" / f"{name}.json").exists():
            return
        argv = ["fit", "--method", method, "--fraction", str(fraction), "--seed", str(seed), *smoke_flag]
        jobs[name] = (stage, -(1 - fraction), name, argv)

    for seed in cfg["fit_seeds"]:
        add_fit_job(0, "full", 0.0, seed)
        for fraction in cfg["fractions"]:
            for method in main_methods(cfg):
                add_fit_job(0, method, fraction, seed)
        for method in ["random", *early_methods(cfg)]:
            add_fit_job(1, method, cfg["early_fraction"], seed)

    score = window_score(cfg)
    for seed in cfg["fit_seeds"]:
        for offset in cfg["window_offsets"]:
            name = window_run_name(score, offset, seed)
            if (out / "window_results" / f"{name}.json").exists():
                continue
            argv = ["window", "--score", score, "--offset", str(offset), "--seed", str(seed), *smoke_flag]
            jobs[name] = (2, -cfg["window_keep_fraction"], name, argv)

    return [(name, argv) for _, _, name, argv in sorted(jobs.values())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = get_config(args.smoke)
    out = output_dir(args.smoke)
    log_dir = out / "logs"
    for path in (out / "scores" / "raw", out / "results", out / "window_results", log_dir):
        path.mkdir(parents=True, exist_ok=True)
    download(DATA_ROOT)
    smoke_flag = ["--smoke"] if args.smoke else []

    score_jobs = [
        (f"score_seed{seed}", ["score", "--seed", str(seed), *smoke_flag])
        for seed in cfg["score_seeds"]
        if not score_done(cfg, out, seed)
    ]
    print(f"score jobs to run: {len(score_jobs)}", flush=True)
    failed = run_jobs(score_jobs, args.workers, log_dir)
    if failed:
        print(f"score jobs failed: {', '.join(failed)}. Logs are in {log_dir}", flush=True)
        sys.exit(1)
    aggregate_scores(cfg, out)

    jobs = training_jobs(cfg, out, smoke_flag)
    print(f"training jobs to run: {len(jobs)}", flush=True)
    failed = run_jobs(jobs, args.workers, log_dir)

    plot.main(cfg, out)
    if failed:
        print(f"training jobs failed: {', '.join(failed)}. Logs are in {log_dir}. Run the same command again to retry them.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
