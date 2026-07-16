import json

with open('.openclaw/tmp/user-files/materials-futures-sprint.ipynb', 'r') as f:
    nb = json.load(f)

# Update metadata for Colab
nb['metadata']['accelerator'] = 'GPU'
nb['metadata']['gpu'] = {'driver_version': '535.161.08', 'count': 1}
if 'tpu' in nb['metadata']:
    del nb['metadata']['tpu']

# Fix Cell 1: Colab-compatible setup
colab_setup = """# Colab + GPU setup
!pip install -q mp-api jarvis-tools einops flax optax scikit-learn xgboost

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["font.size"] = 12
from functools import partial
import warnings, time, json, os, pickle, zipfile, hashlib
warnings.filterwarnings("ignore")

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Num devices: {jax.device_count()}")
print(f"Backend: {jax.default_backend()}")

SEED = 42
np.random.seed(SEED)

EXP_VERSION = "v1"
EXP_ROOT = f"experiments/{EXP_VERSION}"

dirs = [
    f"{EXP_ROOT}/configs", f"{EXP_ROOT}/checkpoints", f"{EXP_ROOT}/predictions",
    f"{EXP_ROOT}/logs", f"{EXP_ROOT}/metrics", f"{EXP_ROOT}/figures",
    f"{EXP_ROOT}/figure_data", f"{EXP_ROOT}/tables", f"{EXP_ROOT}/metadata",
    f"{EXP_ROOT}/paper",
    "materials_futures_sprint/figures", "materials_futures_sprint/results",
    "materials_futures_sprint/data", "materials_futures_sprint/checkpoints",
    "materials_futures_sprint/paper",
]
for d in dirs:
    os.makedirs(d, exist_ok=True)

START_TIME = time.time()
EXP_ID_PREFIX = time.strftime("exp_%Y%m%d_%H%M%S")
print(f"Experiment: {EXP_VERSION} | ID prefix: {EXP_ID_PREFIX}")

def make_exp_id(seed, model):
    return f"{EXP_ID_PREFIX}_{model}_seed{seed}"

def file_hash(path, algo='sha256'):
    h = hashlib.new(algo)
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    except:
        return None
"""

nb['cells'][1]['source'] = colab_setup.split('\n')

# Fix Cell 2: Colab-compatible API key input
nb['cells'][2]['source'] = [
    '# API Key - enter when prompted\n',
    'import getpass\n',
    'MP_API_KEY = getpass.getpass("Enter your Materials Project API key: ")\n',
    'print("API key loaded")\n',
]

# Reduce batch sizes for T4 GPU (16GB VRAM)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        new_src = []
        for line in cell['source']:
            if 'batch_size = 256' in line:
                line = line.replace('batch_size = 256', 'batch_size = 128')
                print(f'  Cell {i}: batch_size 256 -> 128')
            new_src.append(line)
        cell['source'] = new_src

with open('.openclaw/tmp/user-files/materials-futures-colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f'\nColab notebook: {len(nb["cells"])} cells')
print(f'Batch size: 128 (T4-safe)')
