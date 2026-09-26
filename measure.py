
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import equations as eq
from models import SmallCNN

SEED = 20260926
BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
WARMUP = 10
REPEATS = 31
ENERGY_SECONDS = 1.0
ENERGY_WINDOWS = 3


def save_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def configure():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError('Нужна CUDA-версия PyTorch и доступная NVIDIA GPU.')
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.set_device(0)


def make_grid():
    rng = np.random.default_rng(SEED)
    extra_s = sorted(map(int, rng.choice([s for s in range(32, 513, 16) if s not in BASE_S], 4, replace=False)))
    extra_b = sorted(map(int, rng.choice([b for b in range(1, 257) if b not in BASE_B], 3, replace=False)))
    sizes, batches = sorted(BASE_S + extra_s), sorted(BASE_B + extra_b)
    grid = [(s, b, s not in BASE_S or b not in BASE_B) for s in sizes for b in batches]
    rng.shuffle(grid)  # Порядок перемешан, чтобы нагрев не совпадал с ростом S и B.
    assert len(grid) == len(set((s, b) for s, b, _ in grid)) == 132
    assert sum(not v for _, _, v in grid) == 63
    return grid, {'seed': SEED, 'base_S': BASE_S, 'base_B': BASE_B,
                  'random_S': extra_s, 'random_B': extra_b, 'S': sizes, 'B': batches,
                  'order': [list(row) for row in grid]}


@torch.inference_mode()
def verify():
    model = SmallCNN().eval()
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert len(convs) == 6
    assert [(m.in_channels, m.out_channels, m.kernel_size[0], m.stride[0]) for m in convs] == [
        (3, 32, 7, 2), (32, 64, 5, 1), (64, 128, 3, 2),
        (128, 256, 1, 1), (256, 256, 3, 2), (256, 512, 1, 1)]
    assert all(m.bias is None and m.padding == (m.kernel_size[0] // 2,) * 2 for m in convs)
    assert all(isinstance(model.features[i+1], nn.ReLU) for i, m in enumerate(model.features) if isinstance(m, nn.Conv2d))
    assert all(m.inplace for m in model.modules() if isinstance(m, nn.ReLU))
    assert isinstance(model.features[2], nn.MaxPool2d)
    assert (model.features[2].kernel_size, model.features[2].stride, model.features[2].padding) == (3, 2, 1)
    assert isinstance(model.pool, nn.AdaptiveAvgPool2d) and model.pool.output_size == 1
    assert [(m.in_features, m.out_features) for m in model.modules() if isinstance(m, nn.Linear)] == [(512, 256), (256, 100)]
    assert isinstance(model.classifier[2], nn.ReLU)
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == 1040324
    assert all(p.dtype == torch.float32 for p in model.parameters())
    shapes = []
    operations = []
    def hook(name):
        def record(layer, inputs, output):
            shapes.append({'layer': name, 'type': type(layer).__name__, 'shape': list(output.shape)})
            if isinstance(layer, nn.Conv2d):
                operations.append(2 * output.numel() * layer.in_channels * layer.kernel_size[0] ** 2)
            if isinstance(layer, nn.Linear):
                operations.append(2 * output.numel() * layer.in_features)
        return record
    handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if not list(m.children())]
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    for h in handles:
        h.remove()
    assert y.shape == (2, 100)
    expected = [[2,32,16,16],[2,32,16,16],[2,32,8,8],[2,64,8,8],[2,64,8,8],
                [2,128,4,4],[2,128,4,4],[2,256,4,4],[2,256,4,4],
                [2,256,2,2],[2,256,2,2],[2,512,2,2],[2,512,2,2],
                [2,512,1,1],[2,512],[2,256],[2,256],[2,100]]
    assert [v['shape'] for v in shapes] == expected
    assert sum(operations) == eq.flops(32, 2)
    # Независимая сумма логических чтений/записей при B=2, S=32.
    elements = x.numel()
    previous = x.numel()
    reads_writes = 0
    for row in shapes:
        current = int(np.prod(row['shape']))
        if row['type'] != 'Flatten':
            reads_writes += previous + current
        previous = current
    assert 4 * (reads_writes + parameters) == eq.bytes_moved(32, 2)
    # MaxPool временно создаёт int64-индексы размера своего выхода.
    pool_elements = 2 * 32 * 8 * 8
    memory_from_tensors = 4 * parameters + 4 * (elements + 2*32*16*16 + pool_elements) + 8 * pool_elements
    assert eq.memory(32, 2) == memory_from_tensors
    assert eq.memory(512, 256) == 4567564048
    theta = dict(t0=.001, compute_rate=1e12, bandwidth=1e11,
                 p0=10., joules_per_flop=1e-10, joules_per_byte=1e-9)
    s = np.array([32, 128, 512], dtype=np.int32)[:, None]
    b = np.array([1, 8, 256], dtype=np.int32)[None, :]
    for fn in [eq.flops, eq.memory, eq.bytes_moved, eq.latency, eq.energy]:
        args = (theta,) if fn in (eq.latency, eq.energy) else ()
        values = fn(s, b, *args)
        assert values.shape == (3, 3) and np.all(np.isfinite(values))
        for i in range(3):
            for j in range(3):
                assert np.isclose(values[i,j], fn(int(s[i,0]), int(b[0,j]), *args))
    assert eq.flops(512, 256) == 256 * (17712 * 512**2 + 313344)
    return {'parameters': parameters, 'parameter_bytes': 4*parameters, 'shapes_B2_S32': shapes,
            'flops_from_layers': sum(operations), 'broadcasting_including_int32': 'PASS',
            'bytes_from_layer_reads_writes': 4*(reads_writes+parameters),
            'memory_formula': '4161296 + 68 * B * S**2',
            'memory_from_tensors_B2_S32': memory_from_tensors,
            'maxpool_indices_bytes_B2_S32': 8 * pool_elements,
            'memory_B256_S512_bytes': int(eq.memory(512, 256))}


class GPUEnergy:
    """NVML всей выбранной GPU: счётчик mJ; резервный метод - интеграл мощности."""
    def __init__(self):
        self.mode, self.reason, self.nv, self.handle = 'unavailable', '', None, None
        try:
            import pynvml
            self.nv = pynvml
            pynvml.nvmlInit()
            # UUID связывает CUDA и NVML даже при CUDA_VISIBLE_DEVICES.
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(str(torch.cuda.get_device_properties(0).uuid))
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
                self.mode = 'nvml_energy_counter'
            except pynvml.NVMLError as exc:
                self.reason = f'Счётчик недоступен: {exc}'
                pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.mode = 'nvml_power_integration'
        except Exception as exc:
            self.reason = f'{type(exc).__name__}: {exc}'

    def close(self):
        if self.nv is not None:
            self.nv.nvmlShutdown()

    @torch.inference_mode()
    def measure(self, model, x, latency):
        if self.mode == 'unavailable':
            return None, {'method': self.mode, 'reason': self.reason}
        nv, handle = self.nv, self.handle
        # Каждый блок синхронизируется; это энергия серии, а не сумма изолированных таймеров.
        block = max(1, min(256, int(.05 / latency)))
        def run_for(seconds, sample_power=False):
            count, samples = 0, []
            torch.cuda.synchronize()
            start = time.perf_counter()
            if sample_power:
                samples.append((start, nv.nvmlDeviceGetPowerUsage(handle)/1000))
            while time.perf_counter() - start < seconds:
                for _ in range(block):
                    output = model(x)
                    del output
                torch.cuda.synchronize()
                count += block
                if sample_power:
                    samples.append((time.perf_counter(), nv.nvmlDeviceGetPowerUsage(handle)/1000))
            return count, time.perf_counter()-start, samples
        try:
            run_for(1.2 if self.mode == 'nvml_power_integration' else .3)
            windows = []
            for _ in range(ENERGY_WINDOWS):
                if self.mode == 'nvml_energy_counter':
                    before = nv.nvmlDeviceGetTotalEnergyConsumption(handle)
                    count, duration, _ = run_for(ENERGY_SECONDS)
                    after = nv.nvmlDeviceGetTotalEnergyConsumption(handle)
                    joules = (after-before)/1000
                    if joules <= 0:
                        raise ValueError('Счётчик энергии не увеличился.')
                    record = dict(forwards=count, seconds=duration, start_mJ=before, end_mJ=after,
                                  total_joules=joules, joules_per_forward=joules/count)
                else:
                    count, duration, samples = run_for(2.0, sample_power=True)
                    samples = np.asarray(samples)
                    joules = float(np.trapezoid(samples[:,1], samples[:,0]))
                    record = dict(forwards=count, seconds=duration, total_joules=joules,
                                  joules_per_forward=joules/count, power_samples=samples.tolist())
                windows.append(record)
            return float(np.median([v['joules_per_forward'] for v in windows])), {'method': self.mode, 'windows': windows, 'note': self.reason}
        except (nv.NVMLError, ValueError) as exc:
            return None, {'method': 'unavailable', 'reason': f'{type(exc).__name__}: {exc}'}


@torch.inference_mode()
def run_experiment(results='results'):
    configure()
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    checks = verify()
    save_json(results/'checks.json', checks)
    grid, grid_info = make_grid()
    hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['models.py', 'equations.py', 'measure.py']}
    csv_path, meta_path = results/'measurements.csv', results/'metadata.json'
    previous = []
    if csv_path.exists():
        previous = list(csv.DictReader(csv_path.open(encoding='utf-8', newline='')))
        old = json.loads(meta_path.read_text(encoding='utf-8'))
        if old['source_sha256'] != hashes or old['grid'] != grid_info:
            raise RuntimeError('Код или сетка изменились. Используйте новую папку --results для нового запуска.')
    sensor = GPUEnergy()
    prop = torch.cuda.get_device_properties(0)
    metadata = {'started_utc': datetime.now(timezone.utc).isoformat(), 'gpu': prop.name,
                'gpu_uuid': str(prop.uuid), 'gpu_total_memory_bytes': prop.total_memory,
                'python': platform.python_version(), 'platform': platform.platform(),
                'packages': {p: importlib.metadata.version(p) for p in ['torch','numpy','scipy','matplotlib','pandas','nvidia-ml-py','nbformat','nbclient','ipykernel']},
                'cuda_build': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                'flags': {'cudnn.benchmark': False, 'cudnn.allow_tf32': False, 'cuda.matmul.allow_tf32': False},
                'dtype': 'float32', 'warmup': WARMUP, 'latency_repeats': REPEATS,
                'energy_seconds_per_window': ENERGY_SECONDS, 'energy_windows': ENERGY_WINDOWS,
                'energy_method': sensor.mode, 'energy_note': sensor.reason, 'grid': grid_info,
                'source_sha256': hashes, 'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True, encoding='utf-8', errors='replace')}
    if previous:
        metadata = old
        metadata.setdefault('resumed_utc', []).append(datetime.now(timezone.utc).isoformat())
    save_json(meta_path, metadata)
    model = SmallCNN().cuda().eval()
    done = {(int(r['S']), int(r['B'])) for r in previous}
    fields = ['S','B','latency','memory','energy','is_validation','status']
    try:
        with csv_path.open('a', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not previous:
                writer.writeheader()
                stream.flush()
            for index, (s, b, validation) in enumerate(grid, 1):
                if (s,b) in done:
                    continue
                x = output = None
                torch.cuda.empty_cache()
                detail = {'S': s, 'B': b, 'order': index, 'free_before_bytes': torch.cuda.mem_get_info()[0]}
                row = dict(S=s, B=b, latency=None, memory=None, energy=None, is_validation=validation, status='OK')
                try:
                    # Отдельный seed точки делает продолженный запуск воспроизводимым.
                    torch.cuda.manual_seed(SEED + s*1000 + b)
                    x = torch.randn(b,3,s,s,device='cuda',dtype=torch.float32)
                    for _ in range(WARMUP):
                        output = model(x)
                        del output
                    output = None
                    timings = []
                    for _ in range(REPEATS):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        output = model(x)
                        torch.cuda.synchronize()
                        timings.append(time.perf_counter()-start)
                        del output
                    output = None
                    row['latency'] = float(np.median(timings))
                    # Выход прошлого forward удалён, вход и параметры уже на GPU.
                    torch.cuda.synchronize()
                    detail['baseline_allocated_bytes'] = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    output = model(x)
                    torch.cuda.synchronize()
                    row['memory'] = torch.cuda.max_memory_allocated()
                    assert tuple(output.shape) == (b, 100)
                    del output
                    output = None
                    row['energy'], energy_detail = sensor.measure(model, x, row['latency'])
                    detail.update(latency_seconds=timings, latency_q25=float(np.quantile(timings,.25)),
                                  latency_q75=float(np.quantile(timings,.75)), energy=energy_detail)
                except torch.cuda.OutOfMemoryError as exc:
                    row.update(status='OOM', latency=None, memory=None, energy=None)
                    detail['error'] = str(exc)
                finally:
                    x = output = None
                    gc.collect()
                    torch.cuda.empty_cache()
                # CSV — контрольная точка; подробности сохраняются перед ней.
                with (results/'details.jsonl').open('a', encoding='utf-8') as details:
                    details.write(json.dumps(detail, ensure_ascii=False, allow_nan=False)+'\n')
                    details.flush()
                writer.writerow(row)
                stream.flush()
                os.fsync(stream.fileno())
                print(f"[{index:3}/132] S={s:3}, B={b:3}: {row['status']}; t={row['latency']}; E={row['energy']}", flush=True)
    finally:
        sensor.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save_json(meta_path, metadata)
    return pd.read_csv(csv_path)


@torch.inference_mode()
def profile_flops(results='results'):
    configure()
    model = SmallCNN().cuda().eval()
    rows = []
    for s,b in [(32,1),(128,4),(512,1)]:
        x = torch.randn(b,3,s,s,device='cuda')
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
                output = model(x)
                torch.cuda.synchronize()
            estimate = sum(event.flops for event in prof.key_averages())
            rows.append(dict(S=s,B=b,analytical_flops=float(eq.flops(s,b)),profiler_flops=estimate,
                             status='OK' if estimate else 'UNSUPPORTED'))
            del output
        except (RuntimeError, NotImplementedError) as exc:
            rows.append(dict(S=s,B=b,analytical_flops=float(eq.flops(s,b)),profiler_flops=None,status=str(exc)))
        del x
    del model
    torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(Path(results)/'profiler_flops.csv', index=False)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results')
    args = parser.parse_args()
    run_experiment(args.results)
    profile_flops(args.results)
