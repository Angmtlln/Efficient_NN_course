import numpy as np

BYTES_PER_ELEM = 4  

N_PARAMS = 4704 + 51200 + 73728 + 32768 + 589824 + 131072 + 131328 + 25700 
WEIGHT_BYTES = BYTES_PER_ELEM * N_PARAMS 

KERNELS = [
    ("conv1",    2352,   0,        44,    0,     4 * 4704),
    ("relu1",    0,      0,        64,    0,     0),
    ("maxpool",  0,      0,        56,    0,     0),   
    ("conv2",    6400,   0,        24,    0,     4 * 51200),
    ("relu2",    0,      0,        32,    0,     0),
    ("conv3",    2304,   0,        24,    0,     4 * 73728),
    ("relu3",    0,      0,        16,    0,     0),
    ("conv4",    1024,   0,        24,    0,     4 * 32768),
    ("relu4",    0,      0,        32,    0,     0),
    ("conv5",    4608,   0,        20,    0,     4 * 589824),
    ("relu5",    0,      0,        8,     0,     0),
    ("conv6",    1024,   0,        12,    0,     4 * 131072),
    ("relu6",    0,      0,        16,    0,     0),
    ("gap",      0,      0,        8,     2048,  0),
    ("fc1",      0,      262144,   0,     3072,  4 * 131328),
    ("relu_fc",  0,      0,        0,     2048,  0),
    ("fc2",      0,      51200,    0,     1424,  4 * 25700),
]
N_KERNELS = len(KERNELS) 

PEAK_BYTES_PER_BS2 = 4 * (3 + 8 + 2) + 8 * 2  


def _as_arrays(image_size, batch):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    return S, B


def _theta(theta, keys):
    if isinstance(theta, dict):
        return [float(theta[k]) for k in keys]
    return [float(v) for v in theta]


def kernel_costs(image_size, batch):
    S, B = _as_arrays(image_size, batch)
    BS2 = B * S ** 2
    F = np.stack([f_bs2 * BS2 + f_b * B for _, f_bs2, f_b, _, _, _ in KERNELS])
    M = np.stack([m_bs2 * BS2 + m_b * B + m_c + 0 * BS2 for _, _, _, m_bs2, m_b, m_c in KERNELS])
    return F, M


def flops(image_size, batch):
    S, B = _as_arrays(image_size, batch)
    return B * (17712.0 * S ** 2 + 313344.0)


def memory(image_size, batch):
    S, B = _as_arrays(image_size, batch)
    return WEIGHT_BYTES + PEAK_BYTES_PER_BS2 * B * S ** 2


def bytes_moved(image_size, batch):
    S, B = _as_arrays(image_size, batch)
    return 380.0 * B * S ** 2 + 8592.0 * B + WEIGHT_BYTES


def latency(image_size, batch, theta):
    tau, P, BW = _theta(theta, ("tau", "P", "BW"))
    F, M = kernel_costs(image_size, batch)
    return np.maximum(tau, np.maximum(F / P, M / BW)).sum(axis=0)


def energy(image_size, batch, theta_energy):
    P_s, e_F, e_M, tau, P, BW = _theta(theta_energy, ("P_s", "e_F", "e_M", "tau", "P", "BW"))
    T = latency(image_size, batch, (tau, P, BW))
    return P_s * T + e_F * flops(image_size, batch) + e_M * bytes_moved(image_size, batch)


def regime(image_size, batch, theta):
    tau, P, BW = _theta(theta, ("tau", "P", "BW"))
    F, M = kernel_costs(image_size, batch)
    t_launch = np.full_like(F, tau)
    t_compute, t_memory = F / P, M / BW
    t = np.maximum(t_launch, np.maximum(t_compute, t_memory))
    is_launch = t_launch >= np.maximum(t_compute, t_memory)
    is_compute = ~is_launch & (t_compute >= t_memory)
    is_memory = ~is_launch & ~is_compute
    total = t.sum(axis=0)
    return np.stack([(t * m).sum(axis=0) / total for m in (is_launch, is_memory, is_compute)])


if __name__ == "__main__":
    for S, B in [(32, 1), (224, 1), (224, 64), (512, 256)]:
        print(f"S={S:3d} B={B:3d}  FLOPs={flops(S, B):.4g}  Memory={memory(S, B):.4g} B  "
              f"Bytes={bytes_moved(S, B):.4g} B")
    F, M = kernel_costs(224, 1)
    assert np.isclose(F.sum(), flops(224, 1)) and np.isclose(M.sum(), bytes_moved(224, 1))
    S = np.array([32, 64, 128])[:, None]
    B = np.array([1, 16, 256])[None, :]
    print(latency(S, B, dict(tau=10e-6, P=20e12, BW=400e9)).shape)
