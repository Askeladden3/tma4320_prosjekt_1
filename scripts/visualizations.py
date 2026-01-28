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
    '''Trener modellen med config.yaml og lagrer vekter samt config i cached_output under folder_name'''
    full_dir = r'output\\cached_output' + '\\' + folder_name
    try:
        os.makedirs(full_dir)
    except:
        pass
    cfg = load_config("config.yaml")
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)
    nn_params, loss_dict = train_nn(sensor_data, cfg)

    loss_df = pd.DataFrame(loss_dict)

    nn_dict = {}
    for idx, (w,b) in enumerate(nn_params):
        nn_dict[f'w_{idx}'] = pd.Series(w.flatten().tolist())
        nn_dict[f'b_{idx}'] = pd.Series(b.flatten().tolist())

    nn_df = pd.DataFrame(nn_dict)
    nn_df.to_csv(full_dir + '\\nn_params.csv')
    loss_df.to_csv(full_dir + '\\losses.csv')

    with open("config.yaml") as f:
        data = yaml.safe_load(f)
    with open(full_dir + '\\modelconfig.yaml', 'w') as fil:
        yaml.safe_dump(data, fil)

def load_arch(folder_name):
    '''Henter ut parametre og config fra folder_name i cached_output-mappen'''
    full_dir = r'output\\cached_output' + '\\' + folder_name
    nn_params = []
    config_data = load_config(full_dir + '\\modelconfig.yaml')
    
    layer_sizes =  config_data.layer_sizes
    nn_df = pd.read_csv(full_dir + '\\nn_params.csv')
    n_layers = nn_df.shape[1] // 2
    for idx in range(n_layers):
        w_shape = (layer_sizes[idx], layer_sizes[idx+1])
        w, b = nn_df[f'w_{idx}'].to_numpy(), nn_df[f'b_{idx}'].to_numpy()
        w_mask = ~np.isnan(w)
        b_mask = ~np.isnan(b)
        nn_params.append((w[w_mask].reshape(w_shape), b[b_mask]))

    return nn_params, config_data

def difference_plot(folder_name):
    full_dir = r'output\\cached_output' + '\\' + folder_name

    nn_params, cfg = load_arch(folder_name)
    
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)
    T_pred = predict_grid(nn_params, x, y, t, cfg)

    T_diff = T_fdm - T_pred

    create_animation(
        x, y, t, T_diff, title="Differansetemp", save_path=full_dir + "\\difftemp.gif"
    )

    plt.imshow(T_diff[0, :, :])
    plt.show()

def loss_plot(folder_name):
    'Plotter total loss over epokene'
    full_dir = r'output\\cached_output' + '\\' + folder_name

    nn_params, cfg = load_arch(folder_name)
    loss_df = pd.read_csv(full_dir + '\\losses.csv')

    epoker = np.arange(0, cfg.num_epochs)
    plt.title('Total loss over epoker')
    plt.ylabel('total loss')
    plt.xlabel('epoke')
    plt.plot(epoker, loss_df['total'])
    plt.show()
   




#Eksempelkjøring save_arch
#save_arch('standard')

#Eksempelkjøring load_arch
#nn_params, config = load_arch('standard')

#difference_plot('standard')

#loss_plot('standard')

    