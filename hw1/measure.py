import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import pynvml
import torch
from torch.utils.flop_counter import FlopCounterMode

from models import build_model

BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
RESULTS = Path(__file__).parent / "results"
FIELDS = ["S", "B", "latency_s", "memory_bytes", "energy_j", "is_validation", "status",
          "flops_counted", "latency_p10_s", "latency_p90_s", "avg_power_w",
          "n_latency_runs", "n_energy_runs"]


def sample_grid(seed: int):
    rng = np.random.default_rng(seed)
    s_pool = [s for s in range(32, 513, 16) if s not in BASE_S]
    b_pool = [b for b in range(1, 257) if b & (b - 1) != 0]
    extra_S = sorted(int(s) for s in rng.choice(s_pool, size=4, replace=False))
    extra_B = sorted(int(b) for b in rng.choice(b_pool, size=3, replace=False))
    return extra_S, extra_B


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def measure_config(model, S, B, nvml_handle, latency_budget, energy_budget):
    x = torch.randn(B, 3, S, S, device="cuda")

    for _ in range(3):
        model(x)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    model(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    times = []
    start = time.perf_counter()
    while len(times) < 20 or (time.perf_counter() - start < latency_budget and len(times) < 1000):
        t0 = sync_time()
        model(x)
        times.append(sync_time() - t0)

    read_energy = lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(nvml_handle)
    e = read_energy()
    while (e0 := read_energy()) == e:
        pass
    t0 = time.perf_counter()
    n_en, e_tail = 0, None
    while True:
        model(x)
        torch.cuda.synchronize()
        n_en += 1
        if time.perf_counter() - t0 >= energy_budget and n_en >= 5:
            e1 = read_energy()
            if e_tail is None:
                e_tail = e1
            elif e1 != e_tail: 
                break
    t1 = time.perf_counter()
    energy_j = (e1 - e0) / 1000.0 / n_en

    with FlopCounterMode(display=False) as counter:
        model(x)
    flops_counted = counter.get_total_flops()

    return dict(
        latency_s=float(np.median(times)),
        latency_p10_s=float(np.percentile(times, 10)),
        latency_p90_s=float(np.percentile(times, 90)),
        memory_bytes=int(peak),
        energy_j=energy_j,
        avg_power_w=(e1 - e0) / 1000.0 / (t1 - t0),
        flops_counted=int(flops_counted),
        n_latency_runs=len(times),
        n_energy_runs=n_en,
    )


def environment_info(handle):
    props = torch.cuda.get_device_properties(0)
    return dict(
        gpu=torch.cuda.get_device_name(0),
        gpu_memory_bytes=props.total_memory,
        driver=pynvml.nvmlSystemGetDriverVersion(),
        power_limit_w=pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        python=platform.python_version(),
        os=platform.platform(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="few configurations, short runs")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--latency-budget", type=float, default=1.0, help="seconds per config")
    parser.add_argument("--energy-budget", type=float, default=2.0, help="seconds per config")
    parser.add_argument("--out", type=Path, default=RESULTS / "measurements.csv")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)

    model = build_model().cuda().eval()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    extra_S, extra_B = sample_grid(args.seed)
    if args.quick:
        sizes, batches = [32, 224, extra_S[0]], [1, 16, extra_B[0]]
        args.latency_budget, args.energy_budget = 0.3, 0.5
    else:
        sizes, batches = sorted(BASE_S + extra_S), sorted(BASE_B + extra_B)
    print(f"image sizes: {sizes}\nbatch sizes: {batches}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    info = environment_info(handle)
    info.update(seed=args.seed, extra_S=extra_S, extra_B=extra_B)
    (args.out.parent / "env.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))

    with torch.inference_mode():
        x = torch.randn(32, 3, 224, 224, device="cuda")
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < (1 if args.quick else 10):
            model(x)
        torch.cuda.synchronize()
        del x

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for S in sizes:
            for B in batches:
                row = dict(S=S, B=B, is_validation=(S in extra_S) or (B in extra_B))
                try:
                    row.update(measure_config(model, S, B, handle,
                                              args.latency_budget, args.energy_budget))
                    row["status"] = "ok"
                    print(f"S={S:3d} B={B:3d}  latency={row['latency_s'] * 1e3:9.3f} ms  "
                          f"memory={row['memory_bytes'] / 2**20:9.1f} MiB  "
                          f"energy={row['energy_j'] * 1e3:9.3f} mJ  "
                          f"power={row['avg_power_w']:5.1f} W", flush=True)
                except torch.cuda.OutOfMemoryError:
                    row.update(status="OOM", memory_bytes="OOM")
                    print(f"S={S:3d} B={B:3d}  OOM", flush=True)
                torch.cuda.empty_cache()
                writer.writerow(row)
                f.flush()

    pynvml.nvmlShutdown()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
