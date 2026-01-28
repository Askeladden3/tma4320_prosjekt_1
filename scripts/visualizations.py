import os
import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots
import pandas as pd
import json
import yaml

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_nn,
)

def save_arch(folder_name):
    full_dir = r'output\\cached_output' + '\\' + folder_name
    try:
        os.makedirs(full_dir)
    except:
        pass
    cfg = load_config("config.yaml")
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)
    nn_params, loss_dict = train_nn(sensor_data, cfg)

    nn_df = pd.DataFrame()
    nn_dict = {}
    for idx, (w,b) in enumerate(nn_params):
        nn_dict[f'w_{idx}'] = pd.Series(w.flatten().tolist())
        nn_dict[f'b_{idx}'] = pd.Series(b.flatten().tolist())

    nn_df = pd.DataFrame(nn_dict)
    nn_df.to_csv(full_dir + '\\nn_params.csv')

    with open("config.yaml") as f:
        data = yaml.safe_load(f)
    with open(full_dir + '\\modelconfig.yaml', 'w') as fil:
        yaml.safe_dump(data, fil)

def load_arch(folder_name):
    full_dir = r'output\\cached_output' + '\\' + folder_name
    nn_params = []
    with open(full_dir + '\\modelconfig.yaml', 'r') as fil:
        config_data = yaml.safe_load(fil)
    
    layer_sizes =  config_data["training"]["layer_sizes"]
    nn_df = pd.read_csv(full_dir + '\\nn_params.csv')
    n_layers = nn_df.shape[1] // 2
    for idx in range(n_layers):
        w_shape = (layer_sizes[idx], layer_sizes[idx+1])
        w, b = nn_df[f'w_{idx}'].to_numpy(), nn_df[f'b_{idx}'].to_numpy()
        w_mask = ~np.isnan(w)
        b_mask = ~np.isnan(b)
        nn_params.append((w[w_mask].reshape(w_shape), b[b_mask]))

    return nn_params, config_data



#Eksempelkjøring save_arch
#save_arch('standard')

#Eksempelkjøring load_arch
#nn_params, config = load_arch('standard')

    