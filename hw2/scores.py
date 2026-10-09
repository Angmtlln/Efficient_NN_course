import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from data import prepare

EVAL_BATCH_SIZE = 1024


@torch.no_grad()
def calibrate_bn(model, x):
    """Set BatchNorm running statistics to the cumulative average of batch statistics over x."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None
    model.train()
    for start in range(0, len(x), EVAL_BATCH_SIZE):
        model(prepare(x[start:start + EVAL_BATCH_SIZE]))
    model.eval()


@torch.no_grad()
def el2n(model, x, y, num_classes):
    """L2 norm of softmax(logits) minus one_hot(label) for every example, model in eval mode."""
    model.eval()
    norms = []
    for start in range(0, len(x), EVAL_BATCH_SIZE):
        logits = model(prepare(x[start:start + EVAL_BATCH_SIZE])).float()
        one_hot = F.one_hot(y[start:start + EVAL_BATCH_SIZE], num_classes).float()
        norms.append((logits.softmax(dim=1) - one_hot).norm(dim=1))
    return torch.cat(norms).cpu().numpy()


def grand(model, x, y, chunk_size):
    """L2 norm of the per-example cross-entropy gradient over all parameters, model in eval mode."""
    model.eval()
    params = {name: p.detach() for name, p in model.named_parameters()}
    buffers = {name: b.detach() for name, b in model.named_buffers()}

    def example_loss(params, buffers, xi, yi):
        logits = functional_call(model, (params, buffers), (xi.unsqueeze(0),))
        return F.cross_entropy(logits, yi.unsqueeze(0))

    example_grads = vmap(grad(example_loss), in_dims=(None, None, 0, 0))
    norms = []
    for start in range(0, len(x), chunk_size):
        grads = example_grads(params, buffers, prepare(x[start:start + chunk_size]), y[start:start + chunk_size])
        squared = sum(g.flatten(start_dim=1).pow(2).sum(dim=1) for g in grads.values())
        norms.append(squared.sqrt())
    return torch.cat(norms).cpu().numpy()
