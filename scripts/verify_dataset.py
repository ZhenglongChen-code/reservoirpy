import numpy as np
import json

data = np.load('dataset_test/data.npz')
print('=== Data Shapes ===')
for key in data.files:
    print(f'  {key}: {data[key].shape}, dtype={data[key].dtype}')

print()
print('=== Normalization Check ===')
print(f'  log_perm_norm: mean={data["log_perm_norm"].mean():.4f}, std={data["log_perm_norm"].std():.4f}')
print(f'  pressure_norm: mean={data["pressure_norm"].mean():.4f}, std={data["pressure_norm"].std():.4f}')
print(f'  pressure_norm range: [{data["pressure_norm"].min():.4f}, {data["pressure_norm"].max():.4f}]')

print()
with open('dataset_test/configs.json') as f:
    cfg = json.load(f)
print('=== Sample Diversity ===')
for i, c in enumerate(cfg['configs']):
    w = c['wells']
    n_inj = sum(1 for ww in w if ww['is_injector'])
    n_prod = len(w) - n_inj
    print(f'  Sample {i}: range={c["major_range"]:.0f}m, az={c["azimuth"]:.0f}deg, '
          f'mu={c["viscosity_mPas"]:.2f}mPa.s, phi={c["porosity"]:.2f}, '
          f'wells={n_inj}inj+{n_prod}prod, '
          f'logK_mean={c["mean_log_perm"]:.2f}')

print()
with open('dataset_test/metadata.json') as f:
    meta = json.load(f)
print('=== Metadata ===')
print(json.dumps(meta, indent=2))
