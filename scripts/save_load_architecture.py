import os
import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots
import pandas as pd
import json
import yaml
import argparse

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_nn,
    train_pinn
)

def save_arch(folder_name, model_type):
    '''Trener modellen med config.yaml og lagrer vekter samt config i cached_output under folder_name'''
    full_dir = r'output\\cached_output' + '\\' + folder_name
    try:
        os.makedirs(full_dir)
    except:
        pass
    cfg = load_config("config.yaml")
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)
    if model_type == 'nn':
        nn_params= train_nn(sensor_data, cfg)

    elif model_type == 'pinn':
        pinn_params = train_pinn(sensor_data, cfg)
    else:
        print('Invalid model type')
        return None

    nn_dict = {}
    model_params = nn_params if model_type == 'nn' else pinn_params['nn']
    for idx, (w,b) in enumerate(model_params):
        nn_dict[f'w_{idx}'] = pd.Series(w.flatten().tolist())
        nn_dict[f'b_{idx}'] = pd.Series(b.flatten().tolist())

    nn_df = pd.DataFrame(nn_dict)
    nn_df.to_csv(full_dir + f'\\{model_type}_params.csv')

    fysikk_param = {}
    for key, val in pinn_params.items():
        if key != 'nn':
            param = np.exp(val)
            fysikk_param[key.split("_")[-1]] = param

    fys_df = pd.DataFrame(fysikk_param)
    fys_df.to_csv(full_dir + f'\\{model_type}_fysikk_params.csv')


    with open("config.yaml") as f:
        data = yaml.safe_load(f)
    with open(full_dir + '\\modelconfig.yaml', 'w') as fil:
        yaml.safe_dump(data, fil)

def load_arch(folder_name, model_type):
    '''Henter ut parametre og config fra folder_name i cached_output-mappen'''
    full_dir = r'output\\cached_output' + '\\' + folder_name
    nn_params = []
    config_data = load_config(full_dir + '\\modelconfig.yaml')
    
    layer_sizes =  config_data.layer_sizes
    nn_df = pd.read_csv(full_dir + f'\\{model_type}_params.csv')
    n_layers = nn_df.shape[1] // 2
    for idx in range(n_layers):
        w_shape = (layer_sizes[idx], layer_sizes[idx+1])
        w, b = nn_df[f'w_{idx}'].to_numpy(), nn_df[f'b_{idx}'].to_numpy()
        w_mask = ~np.isnan(w)
        b_mask = ~np.isnan(b)
        nn_params.append((w[w_mask].reshape(w_shape), b[b_mask]))

    return nn_params, config_data



def main():
    parser = argparse.ArgumentParser(description='instructions')
    parser.add_argument("-fname", action = "store", dest="filename", default="general_output" )
    args = parser.parse_args()

  
    save_arch(args.filename, "pinn")

if __name__ == "__main__":
    main()