"""Script for training and plotting the PINN model."""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_pinn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 5.4: Start
    #######################################################################
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)

    pinn_params, loss_dict = train_pinn(sensor_data, cfg)

    T_pred = predict_grid(pinn_params['nn'], x, y, t, cfg)

    print("\nFerdig lærte fysiske parametre:\n")
    for key, val in pinn_params.items():
        if key != 'nn':
            param = jnp.exp(val)
            print(f'{key.split('_')[-1]}: {param: .4f}')

    print("\nGenerating Physics-informed Neural Net temp visualizations...")
    plot_snapshots(
        x,
        y,
        t,
        T_pred,
        save_path="output/pinn/pinn_snapshots.png",
    )
    create_animation(
        x, y, t, T_pred, title="PINN", save_path="output/pinn/pinn_animation.gif"
    )

    #X-akse i plottene
    epoch_list = np.arange(0, cfg.num_epochs)

    plt.plot(epoch_list, loss_dict['total'])
    

    plt.show()
    #######################################################################
    # Oppgave 5.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
