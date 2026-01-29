"""Training routines for NN and PINN models."""

import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm

from .config import Config
from .loss import bc_loss, data_loss, ic_loss, physics_loss
from .model import init_nn_params, init_pinn_params
from .optim import adam_step, init_adam
from .sampling import sample_bc, sample_ic, sample_interior


def train_nn(
    sensor_data: jnp.ndarray, cfg: Config
) -> tuple[list[tuple[jnp.ndarray, jnp.ndarray]], dict]:
    """Train a standard neural network on sensor data only.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        params: Trained network parameters
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    nn_params = init_nn_params(cfg)
    adam_state = init_adam(nn_params)

    losses = {"total": [], "data": [], "ic": []}  # Fill with loss histories

    #######################################################################
    # Oppgave 4.3: Start
    #######################################################################

    def total_loss(nn_params, sensor_data, ic_points):
        dl = data_loss(nn_params, sensor_data, cfg)
        icl = ic_loss(nn_params,ic_points,cfg)
        return cfg.lambda_data*dl + cfg.lambda_ic*icl, (dl, icl)

  

    for _ in tqdm(range(cfg.num_epochs), desc="Training NN"):
        ic_epoch, _ = sample_ic(key, cfg)
        (loss_tot, loss_parts), grads = (jax.value_and_grad(total_loss, argnums=0, has_aux=True)(nn_params, sensor_data, ic_epoch))
        losses["total"].append(loss_tot)
        losses["data"].append(loss_parts[0])
        losses["ic"].append(loss_parts[1])
        nn_params, adam_state = adam_step(nn_params, grads, adam_state, lr=cfg.learning_rate)
    

        ...

    #######################################################################
    # Oppgave 4.3: Slutt
    #######################################################################

    return nn_params, {k: jnp.array(v) for k, v in losses.items()}


def train_pinn(sensor_data: jnp.ndarray, cfg: Config) -> tuple[dict, dict]:
    """Train a physics-informed neural network.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        pinn_params: Trained parameters (nn weights + alpha)
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    pinn_params = init_pinn_params(cfg)
    opt_state = init_adam(pinn_params)

    losses = {"total": [], "data": [], "physics": [], "ic": [], "bc": []}

    #######################################################################
    # Oppgave 5.3: Start
    #######################################################################


    def total_loss(pinn_params, sensor_data, ic_points, interior_points, bc_points):
        dl = data_loss(pinn_params['nn'], sensor_data, cfg)
        ic = ic_loss(pinn_params['nn'],ic_points,cfg)
        ph = physics_loss(pinn_params, interior_points, cfg)
        bc = bc_loss(pinn_params, bc_points, cfg)
        tot_loss =  cfg.lambda_data*dl + cfg.lambda_ic*ic + cfg.lambda_bc*bc + cfg.lambda_physics * ph
        return tot_loss, (tot_loss, dl, ph, ic, bc)
  
    for _ in tqdm(range(cfg.num_epochs), desc="Training PINN"):
        ic_epoch, _ = sample_ic(key, cfg)
        interior_epoch, _ = sample_interior(key, cfg)
        bc_epoch, _ = sample_bc(key, cfg)
        (loss_tot, loss_parts), grads = (jax.value_and_grad(total_loss, argnums=0, has_aux=True)(pinn_params, sensor_data, ic_epoch, interior_epoch, bc_epoch))

        keys = losses.keys()
        for dict_key, value in zip(keys, loss_parts):
            losses[dict_key].append(value)
        
        pinn_params, opt_state = adam_step(pinn_params, grads, opt_state, lr=cfg.learning_rate)

    #######################################################################
    # Oppgave 5.3: Slutt
    #######################################################################

    return pinn_params, {k: jnp.array(v) for k, v in losses.items()}
