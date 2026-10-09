from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
RUNS_ROOT = ROOT / "runs"
NUM_CLASSES = 10
TRAIN_SIZE = 50000

FULL = {
    "epochs": 200,
    "batch_size": 128,
    "lr": 0.1,
    "momentum": 0.9,
    "nesterov": True,
    "weight_decay": 5e-4,
    "lr_milestones": [60, 120, 160],
    "lr_gamma": 0.2,
    "score_seeds": list(range(10)),
    "score_epochs": [1, 2, 5, 10, 20],
    "grad_chunk_size": 128,
    "fit_seeds": [100, 101, 102],
    "main_score_epoch": 20,
    "fractions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "early_fraction": 0.5,
    "window_score_epoch": 10,
    "window_keep_fraction": 0.4,
    "window_offsets": [0, 250, 500, 1000, 2000, 4000, 8000],
}

SMOKE = {
    **FULL,
    "epochs": 2,
    "lr_milestones": [1],
    "score_seeds": [0, 1],
    "score_epochs": [1],
    "grad_chunk_size": 32,
    "fit_seeds": [100],
    "main_score_epoch": 1,
    "fractions": [0.5],
    "window_score_epoch": 1,
    "window_offsets": [0, 500],
}


def get_config(smoke):
    return SMOKE if smoke else FULL


def output_dir(smoke):
    return RUNS_ROOT / ("smoke" if smoke else "full")


def score_names(cfg):
    """All scores saved by a score run: GraNd at initialization, EL2N and GraNd after every epoch in score_epochs."""
    epochs = cfg["score_epochs"]
    return ["grand_e0", *[f"el2n_e{epoch}" for epoch in epochs], *[f"grand_e{epoch}" for epoch in epochs]]


def main_methods(cfg):
    """Pruning methods compared in the main experiment over all fractions."""
    epoch = cfg["main_score_epoch"]
    return ["random", f"el2n_e{epoch}", "grand_e0", f"grand_e{epoch}"]


def early_methods(cfg):
    """Pruning methods compared in the experiment with scores from different epochs at early_fraction."""
    epochs = cfg["score_epochs"]
    return [*[f"el2n_e{epoch}" for epoch in epochs], "grand_e0", *[f"grand_e{epoch}" for epoch in epochs]]


def window_score(cfg):
    return f"el2n_e{cfg['window_score_epoch']}"


def run_name(method, fraction, seed):
    return f"{method}_f{fraction:.2f}_seed{seed}"


def window_run_name(score, offset, seed):
    return f"window_{score}_o{offset}_seed{seed}"
