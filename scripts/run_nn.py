"""Script for training and plotting the NN model."""

import os

import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_nn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 4.4: Start
    #######################################################################
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)

    nn_params, loss_dict = train_nn(sensor_data, cfg)

    T_pred = predict_grid(nn_params, x, y, t, cfg)

    print("\nGenerating Neural Net temp visualizations...")
    plot_snapshots(
        x,
        y,
        t,
        T_pred,
        save_path="output/nn/nn_snapshots.png",
    )
    create_animation(
        x, y, t, T_pred, title="NN", save_path="output/nn/nn_animation.gif"
    )

    #X-akse i plottene
    epoch_list = np.arange(0, cfg.num_epochs)

    plt.plot(epoch_list, loss_dict['total'])
    

    plt.show()


    #######################################################################
    # Oppgave 4.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
