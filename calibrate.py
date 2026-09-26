import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

import equations as eq


def metrics(real, predicted):
    real, predicted = np.asarray(real, float), np.asarray(predicted, float)
    valid = np.isfinite(real) & np.isfinite(predicted) & (real > 0)
    real, predicted = real[valid], predicted[valid]
    if not len(real):
        return {'n': 0}
    relative = np.abs(predicted-real)/real
    return dict(n=len(real), mape_percent=float(100*relative.mean()),
                median_ape_percent=float(100*np.median(relative)),
                p90_ape_percent=float(100*np.quantile(relative,.9)),
                max_ape_percent=float(100*relative.max()),
                rmse=float(np.sqrt(np.mean((real-predicted)**2))),
                signed_mean_percent=float(100*np.mean((predicted-real)/real)))


def fit_latency(frame):
    # Масштабирование: ms, GFLOP, MB. Минимизируем относительную ошибку.
    f = eq.flops(frame.S.to_numpy(), frame.B.to_numpy())/1e9
    d = eq.bytes_moved(frame.S.to_numpy(), frame.B.to_numpy())/1e6
    y = frame.latency.to_numpy()*1e3
    def residual(q):
        return (q[0] + np.maximum(q[1]*f, q[2]*d)-y)/y
    fits = []
    for ratio in [.01, .1, .3, 1., 3., 10., 100.]:
        start = [max(.01, np.min(y)*.5), np.median(y/f)*.5, np.median(y/d)*ratio]
        result = least_squares(residual, start, bounds=([0.,1e-9,1e-9],[np.inf]*3),
                               max_nfev=2500, xtol=1e-11, ftol=1e-11, gtol=1e-11)
        if result.success:
            fits.append(result)
    if not fits:
        raise RuntimeError('Оптимизация latency не сошлась.')
    best = min(fits, key=lambda r: np.sum(r.fun**2))
    q = best.x
    theta = dict(t0=float(q[0]/1e3), compute_rate=float(1e12/q[1]), bandwidth=float(1e9/q[2]))
    compute = q[1]*f >= q[2]*d
    norms = np.linalg.norm(best.jac, axis=0)
    normalized = best.jac/np.where(norms > 0, norms, 1)
    singular = np.linalg.svd(normalized, compute_uv=False)
    rank = int(np.linalg.matrix_rank(normalized))
    near = [r.x.tolist() for r in fits if np.sum(r.fun**2) <= np.sum(best.fun**2)*1.01 + 1e-12]
    warnings = ['R и W — эффективные коэффициенты этой модели, а не паспортные характеристики GPU.']
    if rank < 3 or min(int(compute.sum()), int((~compute).sum())) < 5:
        warnings.append('Все три параметра нельзя надёжно разделить: одна ветвь max почти не активна либо Якобиан вырожден.')
    condition = float(singular[0]/singular[-1]) if singular[-1] > 1e-14 else None
    if condition is None or condition > 100:
        warnings.append('Параметры плохо обусловлены; высокая точность их записи не означает точность определения.')
    identifiability = {'t0': 'effective fit coefficient', 'compute_rate': 'effective fit coefficient',
                       'bandwidth': 'effective fit coefficient'}
    if not compute.any():
        identifiability['compute_rate'] = {
            'status': 'not_identifiable',
            'conditional_lower_bound_FLOP_per_s': float(np.max(f*1e9 * theta['bandwidth']/(d*1e6))),
            'note': 'Число compute_rate в theta — представитель плоской области loss. Любое R выше границы даёт тот же прогноз на calibration при этих t0/W.'}
    if compute.all():
        identifiability['bandwidth'] = {
            'status': 'not_identifiable',
            'conditional_lower_bound_byte_per_s': float(np.max(d*1e6 * theta['compute_rate']/(f*1e9))),
            'note': 'Число bandwidth в theta — представитель плоской области loss. Любое W выше границы даёт тот же прогноз на calibration при этих t0/R.'}
    return theta, dict(objective='mean squared relative error', calibration_n=len(frame),
                       parameter_identifiability=identifiability,
                       calibration_compute_branch=int(compute.sum()), calibration_memory_branch=int((~compute).sum()),
                       normalized_jacobian_rank=rank, normalized_jacobian_condition=condition,
                       normalized_jacobian_singular_values=singular.tolist(),
                       near_best_starts_scaled_parameters=near, warnings=warnings)


def fit_energy(frame, latency_theta):
    frame = frame[frame.energy.notna() & (frame.energy > 0)]
    if len(frame) < 3:
        return None, {'calibration_n':len(frame), 'warnings':['Недостаточно измерений энергии для калибровки.']}
    s, b = frame.S.to_numpy(), frame.B.to_numpy()
    x = np.column_stack([eq.latency(s,b,latency_theta), eq.flops(s,b), eq.bytes_moved(s,b)])
    y = frame.energy.to_numpy()
    scales = np.linalg.norm(x,axis=0)
    normalized = x/scales
    coefficients, _ = nnls(normalized/y[:,None], np.ones(len(y)))
    parameters = coefficients/scales
    singular = np.linalg.svd(normalized,compute_uv=False)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 1e-14 else None
    # Bootstrap только calibration, при уже выбранной latency-модели.
    rng = np.random.default_rng(20260926)
    bootstrap = []
    for _ in range(100):
        idx = rng.integers(0,len(y),len(y))
        estimate, _ = nnls(normalized[idx]/y[idx,None], np.ones(len(idx)))
        bootstrap.append(estimate/scales)
    bootstrap = np.asarray(bootstrap)
    names = ['p0','joules_per_flop','joules_per_byte']
    ranges = {name:np.quantile(bootstrap[:,i],[.05,.95]).tolist() for i,name in enumerate(names)}
    warnings = ['Bootstrap условный: latency-параметры фиксированы; систематическая ошибка NVML и модели не учтена.',
                'P0 не измеряется на простое отдельно; это коэффициент регрессии, а не доказанная idle power.']
    if condition is None or condition > 100:
        warnings.append('Признаки энергии сильно коррелируют: отдельные P0, eF, eD нельзя интерпретировать как надёжные физические константы.')
    if np.any(parameters == 0):
        warnings.append('Нулевой коэффициент выбран ограничением NNLS; это не доказательство нулевого физического расхода.')
    theta = dict(latency_theta, **dict(zip(names,map(float,parameters))))
    return theta, dict(calibration_n=len(frame), normalized_design_condition=condition,
                       normalized_design_rank=int(np.linalg.matrix_rank(normalized)),
                       bootstrap_5_95_percentiles=ranges, warnings=warnings)


def make_figures(data, theta, root, only=None):
    folder = root/'figures'
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10, 'axes.grid':True, 'grid.alpha':.25})
    targets = [('latency',1e3,'Время одного forward, мс'),
               ('memory',1/2**20,'Абсолютный пик памяти, МиБ')]
    if theta['energy'] is not None and data.energy.notna().any():
        targets.append(('energy',1.,'Энергия одного forward, Дж'))
    if only is not None:
        targets = [target for target in targets if target[0] in only]
    for key, scale, label in targets:
        fig, axes = plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
        valid = data[(data.status == 'OK') & data[key].notna()]
        for v,color,marker,name in [(False,'#2865a3','o','Калибровка'),(True,'#d46a24','^','Проверка')]:
            part = valid[valid.is_validation == v]
            axes[0].scatter(part[key]*scale,part[key+'_pred']*scale,s=24,c=color,marker=marker,label=name,alpha=.8)
        lo = min(valid[key].min(),valid[key+'_pred'].min())*scale*.8
        hi = max(valid[key].max(),valid[key+'_pred'].max())*scale*1.2
        axes[0].plot([lo,hi],[lo,hi],'k--',lw=1,label='Идеальное совпадение')
        axes[0].set(xscale='log',yscale='log',xlabel='Измерение: '+label,ylabel='Прогноз: '+label)
        axes[0].legend(fontsize=8)
        for s,color in zip([32,128,512],['#2865a3','#38a06a','#d46a24']):
            part = valid[valid.S==s].sort_values('B')
            axes[1].plot(part.B,part[key+'_pred']*scale,color=color,label=f'Формула, S={s}')
            for v,marker in [(False,'o'),(True,'^')]:
                selected = part[part.is_validation == v]
                axes[1].scatter(selected.B,selected[key]*scale,color=color,marker=marker,s=25)
        axes[1].set(xscale='log',yscale='log',xlabel='Размер батча B, изображений',ylabel=label,
                    title='Точки — измерения; линии — формула')
        axes[1].legend(fontsize=8)
        fig.suptitle({'latency':'Время: настенные часы с синхронизацией CUDA',
                      'memory':'Память: параметры и тензоры против пика PyTorch',
                      'energy':'Энергия всей GPU: серия проходов'}[key])
        fig.savefig(folder/(key+'.png'),dpi=160)
        plt.close(fig)
    # Сетка целиком: поверхность модели по B для каждого S и все замеры.
    for key,scale,label in targets:
        sizes = sorted(data.S.unique())
        fig,axes = plt.subplots(3,4,figsize=(14,9),layout='constrained')
        for ax,s in zip(axes.flat,sizes):
            part = data[data.S==s].sort_values('B')
            ax.plot(part.B,part[key+'_pred']*scale,'k-',lw=1,label='Формула')
            for v,color,marker,name in [(False,'#2865a3','o','Калибровка'),(True,'#d46a24','^','Проверка')]:
                points = part[(part.status=='OK') & (part.is_validation==v) & part[key].notna()]
                ax.scatter(points.B,points[key]*scale,s=18,c=color,marker=marker,label=name)
            oom = part[part.status=='OOM']
            if len(oom):
                ax.scatter(oom.B,oom[key+'_pred']*scale,c='red',marker='x',label='OOM: оценка')
            ax.set(title=f'S = {s} пикс.',xscale='log',yscale='log',xlabel='B, шт.',
                   ylabel={'latency':'Время, мс','memory':'Память, МиБ','energy':'Энергия, Дж'}[key])
        axes.flat[-1].axis('off')
        handles,labels = axes.flat[0].get_legend_handles_labels()
        axes.flat[-1].legend(handles,labels,loc='center')
        fig.suptitle(label+' — все 132 конфигурации')
        fig.savefig(folder/(key+'_grid.png'),dpi=140)
        plt.close(fig)
    prof_path = root/'profiler_flops.csv'
    if prof_path.exists() and (only is None or 'flops' in only):
        prof = pd.read_csv(prof_path)
        prof = prof[prof.status=='OK']
        if len(prof):
            fig,ax = plt.subplots(figsize=(7,4),layout='constrained')
            positions = np.arange(len(prof))
            ax.plot(positions,prof.analytical_flops/1e9,'o-',label='Аналитическая формула')
            ax.scatter(positions,prof.profiler_flops/1e9,marker='x',s=90,label='Оценка PyTorch profiler')
            ax.set(xticks=positions,xticklabels=[f'S={r.S}, B={r.B}' for r in prof.itertuples()],
                   ylabel='GFLOP на forward',xlabel='Конфигурация',title='FLOPs: две оценки числа операций')
            ax.legend()
            fig.savefig(folder/'flops.png',dpi=160)
            plt.close(fig)


def calibrate(results='results'):
    root = Path(results)
    data = pd.read_csv(root/'measurements.csv')
    assert list(data.columns) == ['S','B','latency','memory','energy','is_validation','status']
    assert not data.duplicated(['S','B']).any()
    if data.is_validation.dtype != bool:
        data['is_validation'] = data.is_validation.astype(str).str.lower().map({'true':True,'false':False})
    assert data.is_validation.notna().all()
    meta = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    expected = (~data.S.isin(meta['grid']['base_S'])) | (~data.B.isin(meta['grid']['base_B']))
    assert (data.is_validation == expected).all()
    train = data[(~data.is_validation) & (data.status=='OK') & data.latency.notna()].copy()
    if len(train)<3:
        raise RuntimeError('Недостаточно calibration-точек.')
    latency_theta, latency_info = fit_latency(train)
    energy_theta, energy_info = fit_energy(train,latency_theta)
    theta = {'latency':latency_theta, 'energy':energy_theta,
             'units':{'t0':'s','compute_rate':'FLOP/s','bandwidth':'byte/s','p0':'W',
                      'joules_per_flop':'J/FLOP','joules_per_byte':'J/byte'},
             'latency_diagnostics':latency_info,'energy_diagnostics':energy_info,
             'calibration_points':train[['S','B']].to_numpy().tolist()}
    s,b = data.S.to_numpy(),data.B.to_numpy()
    data['flops_pred'] = eq.flops(s,b)
    data['bytes_pred'] = eq.bytes_moved(s,b)
    data['memory_pred'] = eq.memory(s,b)
    data['latency_pred'] = eq.latency(s,b,latency_theta)
    data['energy_pred'] = np.nan if energy_theta is None else eq.energy(s,b,energy_theta)
    summary = {'counts':{'total':len(data),'OK':int((data.status=='OK').sum()),'OOM':int((data.status=='OOM').sum()),
                          'calibration':int((~data.is_validation).sum()),'validation':int(data.is_validation.sum()),
                          'energy_measured':int(data.energy.notna().sum())}}
    for name,flag in [('calibration',False),('validation',True)]:
        part = data[(data.status=='OK') & (data.is_validation==flag)]
        summary[name] = {key:metrics(part[key],part[key+'_pred']) for key in ['latency','memory','energy']}
    ok = data[data.status=='OK']
    summary['memory'] = {'max_measured_bytes':int(ok.memory.max()),'max_predicted_bytes':float(data.memory_pred.max()),
                         'min_measured_to_predicted':float((ok.memory/ok.memory_pred).min()),
                         'max_measured_to_predicted':float((ok.memory/ok.memory_pred).max()),
                         'predicted_above_total_memory':int((data.memory_pred>meta['gpu_total_memory_bytes']).sum())}
    ftime = data.flops_pred/latency_theta['compute_rate']
    dtime = data.bytes_pred/latency_theta['bandwidth']
    summary['model_regimes'] = {'overhead_larger_than_both':int((latency_theta['t0']>np.maximum(ftime,dtime)).sum()),
                                'compute_branch':int((ftime>=dtime).sum()),'memory_branch':int((dtime>ftime).sum()),
                                'arithmetic_intensity_min_FLOP_per_byte':float((data.flops_pred/data.bytes_pred).min()),
                                'arithmetic_intensity_max_FLOP_per_byte':float((data.flops_pred/data.bytes_pred).max())}
    # Оценки не смешиваются с measurements.csv.
    data.to_csv(root/'predictions.csv',index=False)
    for name,obj in [('theta.json',theta),('summary.json',summary)]:
        (root/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    make_figures(data,theta,root)
    return summary


def refresh_memory(results='results'):
    """Пересчитать только память по сохранённому CSV, без GPU и калибровки theta."""
    root = Path(results)
    measured = pd.read_csv(root/'measurements.csv', float_precision='round_trip')
    prediction_path = root/'predictions.csv'
    with prediction_path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    pairs = [(int(row['S']), int(row['B'])) for row in rows]
    assert pairs == list(measured[['S','B']].itertuples(index=False, name=None))
    assert len(pairs) == len(set(pairs))
    for row, point in zip(rows, measured.itertuples()):
        assert row['status'] == point.status
        assert point.status != 'OK' or float(row['memory']) == point.memory
        row['memory_pred'] = str(float(eq.memory(point.S, point.B)))
    # Остальные поля сохраняются как строки, без округления или пересчёта.
    temporary = prediction_path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(prediction_path)
    data = measured.copy()
    data['memory_pred'] = eq.memory(data.S.to_numpy(), data.B.to_numpy())
    summary_path = root/'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    for name, flag in [('calibration',False), ('validation',True)]:
        part = data[(data.status=='OK') & (data.is_validation==flag)]
        summary[name]['memory'] = metrics(part.memory, part.memory_pred)
    ok = data[data.status=='OK']
    meta = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    summary['memory'] = {
        'max_measured_bytes': int(ok.memory.max()),
        'max_predicted_bytes': float(data.memory_pred.max()),
        'min_measured_to_predicted': float((ok.memory/ok.memory_pred).min()),
        'max_measured_to_predicted': float((ok.memory/ok.memory_pred).max()),
        'predicted_above_total_memory': int((data.memory_pred>meta['gpu_total_memory_bytes']).sum()),
    }
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    theta = json.loads((root/'theta.json').read_text(encoding='utf-8'))
    make_figures(data,theta,root,only={'memory'})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',default='results')
    parser.add_argument('--memory-only', action='store_true', help='Обновить только память по сохранённым измерениям, без калибровки')
    args = parser.parse_args()
    action = refresh_memory if args.memory_only else calibrate
    print(json.dumps(action(args.results),ensure_ascii=False,indent=2))
