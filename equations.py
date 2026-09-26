import numpy as np


def flops(image_size, batch):
    image_size = np.asarray(image_size, dtype=np.float64)
    batch = np.asarray(batch, dtype=np.float64)
    return batch * (17712 * image_size**2 + 313344)


def memory(image_size, batch):
    image_size = np.asarray(image_size, dtype=np.float64)
    batch = np.asarray(batch, dtype=np.float64)
    return 4161296 + 68 * batch * image_size**2


def bytes_moved(image_size, batch):
    image_size = np.asarray(image_size, dtype=np.float64)
    batch = np.asarray(batch, dtype=np.float64)
    return 364 * batch * image_size**2 + 8592 * batch + 4161296


def latency(image_size, batch, theta):
    compute_time = flops(image_size, batch) / theta["compute_rate"]
    memory_time = bytes_moved(image_size, batch) / theta["bandwidth"]
    return theta["t0"] + np.maximum(compute_time, memory_time)


def energy(image_size, batch, theta_energy):
    return (
        theta_energy["p0"] * latency(image_size, batch, theta_energy)
        + theta_energy["joules_per_flop"] * flops(image_size, batch)
        + theta_energy["joules_per_byte"] * bytes_moved(image_size, batch)
    )
