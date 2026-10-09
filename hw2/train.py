import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import DATA_ROOT, NUM_CLASSES, TRAIN_SIZE, get_config, output_dir, run_name, window_run_name
from data import augment, load_split, prepare
from model import ResNet18
from scores import calibrate_bn, el2n, grand

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def save_array(path, array):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
    os.replace(tmp, path)


def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def make_model():
    return ResNet18(NUM_CLASSES).to(DEVICE).to(memory_format=torch.channels_last)


def fit_model(model, x, y, cfg, epochs, seed, on_epoch_end=None):
    """Train with SGD on (x, y) for the given number of epochs using the learning rate schedule from cfg.

    on_epoch_end is called with the number of completed epochs after every epoch.
    """
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        nesterov=cfg["nesterov"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg["lr_milestones"], cfg["lr_gamma"])
    generator = torch.Generator(device=x.device).manual_seed(seed)
    batch_size = cfg["batch_size"]
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        order = torch.randperm(len(x), device=x.device, generator=generator)
        for start in range(0, len(x), batch_size):
            index = order[start:start + batch_size]
            inputs = prepare(augment(x[index], generator))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(inputs), y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"epoch {epoch + 1}/{epochs} loss {loss.item():.4f} seconds {time.time() - start_time:.1f}", flush=True)
        if on_epoch_end is not None:
            on_epoch_end(epoch + 1)


@torch.no_grad()
def accuracy(model, x, y, batch_size=1024):
    model.eval()
    correct = 0
    for start in range(0, len(x), batch_size):
        logits = model(prepare(x[start:start + batch_size]))
        correct += (logits.argmax(dim=1) == y[start:start + batch_size]).sum().item()
    return correct / len(x)


def run_score(cfg, out, seed):
    """Compute GraNd at initialization, then train and compute EL2N and GraNd after every epoch in score_epochs."""
    raw_dir = out / "scores" / "raw"
    x, y = load_split(DATA_ROOT, train=True, device=DEVICE)
    torch.manual_seed(seed)
    model = make_model()

    init_model = copy.deepcopy(model)
    calibrate_bn(init_model, x)
    save_array(raw_dir / f"grand_e0_seed{seed}.npy", grand(init_model, x, y, cfg["grad_chunk_size"]))
    del init_model

    def save_scores(epoch):
        if epoch not in cfg["score_epochs"]:
            return
        save_array(raw_dir / f"el2n_e{epoch}_seed{seed}.npy", el2n(model, x, y, NUM_CLASSES))
        save_array(raw_dir / f"grand_e{epoch}_seed{seed}.npy", grand(model, x, y, cfg["grad_chunk_size"]))

    fit_model(model, x, y, cfg, max(cfg["score_epochs"]), seed, on_epoch_end=save_scores)


def train_and_evaluate(cfg, keep, seed):
    """Train a model from a new initialization on the training examples with indices keep and return test accuracy."""
    x, y = load_split(DATA_ROOT, train=True, device=DEVICE)
    x_test, y_test = load_split(DATA_ROOT, train=False, device=DEVICE)
    keep = keep.to(DEVICE)
    torch.manual_seed(seed)
    model = make_model()
    start_time = time.time()
    fit_model(model, x[keep], y[keep], cfg, cfg["epochs"], seed)
    return {
        "train_size": len(keep),
        "test_accuracy": accuracy(model, x_test, y_test),
        "train_seconds": round(time.time() - start_time, 1),
    }


def run_fit(cfg, out, method, fraction, seed):
    """Prune the given fraction of examples with the lowest score (or at random) and train on the rest."""
    keep_count = round(TRAIN_SIZE * (1 - fraction))
    if method == "full":
        keep = torch.arange(TRAIN_SIZE)
    elif method == "random":
        keep = torch.randperm(TRAIN_SIZE, generator=torch.Generator().manual_seed(seed))[:keep_count]
    else:
        score = np.load(out / "scores" / f"{method}.npy")
        keep = torch.from_numpy(np.argsort(score, kind="stable")[TRAIN_SIZE - keep_count:])
    result = {"method": method, "fraction": fraction, "seed": seed, **train_and_evaluate(cfg, keep, seed)}
    save_json(out / "results" / f"{run_name(method, fraction, seed)}.json", result)
    print(json.dumps(result), flush=True)


def run_window(cfg, out, score_name, offset, seed):
    """Exclude the `offset` highest-scoring examples and train on the next window_keep_fraction of the data."""
    score = np.load(out / "scores" / f"{score_name}.npy")
    keep_count = round(TRAIN_SIZE * cfg["window_keep_fraction"])
    descending = np.argsort(-score, kind="stable")
    keep = torch.from_numpy(descending[offset:offset + keep_count].copy())
    result = {
        "score": score_name,
        "offset": offset,
        "keep_fraction": cfg["window_keep_fraction"],
        "seed": seed,
        **train_and_evaluate(cfg, keep, seed),
    }
    save_json(out / "window_results" / f"{window_run_name(score_name, offset, seed)}.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--seed", type=int, required=True)
    score_parser.add_argument("--smoke", action="store_true")
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--method", required=True)
    fit_parser.add_argument("--fraction", type=float, required=True)
    fit_parser.add_argument("--seed", type=int, required=True)
    fit_parser.add_argument("--smoke", action="store_true")
    window_parser = commands.add_parser("window")
    window_parser.add_argument("--score", required=True)
    window_parser.add_argument("--offset", type=int, required=True)
    window_parser.add_argument("--seed", type=int, required=True)
    window_parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = get_config(args.smoke)
    out = output_dir(args.smoke)
    if args.command == "score":
        run_score(cfg, out, args.seed)
    elif args.command == "fit":
        run_fit(cfg, out, args.method, args.fraction, args.seed)
    else:
        run_window(cfg, out, args.score, args.offset, args.seed)


if __name__ == "__main__":
    main()
