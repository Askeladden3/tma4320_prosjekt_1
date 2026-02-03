import os
import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots
import pandas as pd
import json
import yaml
import jax
import jax.numpy as jnp
from jax import jit
import argparse


from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_nn
)

from save_load_architecture import save_arch, load_arch

def difference_plot(folder_name, model_type):
    full_dir = r'output\\cached_output' + '\\' + folder_name

    nn_params, cfg = load_arch(folder_name, model_type)
    
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)
    T_pred = predict_grid(nn_params, x, y, t, cfg)

    T_diff = T_fdm - T_pred


    plt.imshow(T_diff[0, :, :])
    plt.show()

def create_animations(folder_name, model_type, makegif):
    full_dir = r'output\\cached_output' + '\\' + folder_name

    nn_params, cfg = load_arch(folder_name, model_type)
    
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)
    T_pred = predict_grid(nn_params, x, y, t, cfg)

    T_diff = T_fdm - T_pred

    if makegif:
        plot_snapshots(x,y,t,T_pred, "Predikert temperatur over tid", save_path=full_dir + f"\\{model_type}_predtempsnapshot.png")

    create_animation(
        x, y, t, T_diff, title="Differansetemp", save_path=full_dir + f"\\{model_type}_difftemp.gif"
    )
    create_animation(
        x, y, t, T_pred, title="Predikert temperatur over tid", save_path=full_dir + f"\\{model_type}_temppred.gif"
    )

def loss_plot(folder_name, model_type):
    'Plotter total loss over epokene'
    full_dir = r'output\\cached_output' + '\\' + folder_name

    nn_params, cfg = load_arch(folder_name, model_type)
    loss_df = pd.read_csv(full_dir + '\\losses.csv')

    epoker = np.arange(0, cfg.num_epochs)
    plt.title('Total loss over epoker')
    plt.ylabel('total loss')
    plt.xlabel('epoke')
    plt.plot(epoker, loss_df['total'])
    plt.show()
   




#Eksempelkjøring save_arch
#save_arch('standard', 'pinn')

#Eksempelkjøring load_arch
#nn_params, config = load_arch('standard')

def main():
    parser = argparse.ArgumentParser(description='instructions')
    parser.add_argument("-fname", action = "store", dest="filename", default="general_output" )
    parser.add_argument("-g", action = "store_true", dest="makegif")
    args = parser.parse_args()
    create_animations(args.filename, 'pinn', args.makegif)



if __name__ == "__main__":
    main()
    