"""This code is modified from PureJaxRL:

  https://github.com/luchris429/purejaxrl

Please refer to their work if you use this example in your research."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import NamedTuple, Literal
from pgx.bridge_bidding import BridgeBidding, download_dds_results
import time
import os
import json
from pprint import pprint


import pickle
from omegaconf import OmegaConf
from pydantic import BaseModel
import wandb

from src.models import make_forward_pass
from src.evaluation import make_simple_duplicate_evaluate
from src.roll_out import make_roll_out
from src.gae import make_calc_gae
from src.update import make_update_step


print(jax.default_backend())
print(jax.local_devices())


class PPOConfig(BaseModel):
    seed: int = 0  # Seed for random number generation.
    lr: float = 0.000001  # Learning rate for Adam optimizer.
    num_envs: int = 8192  # Number of parallel environments for actor rollout.
    num_steps: int = 32  # Number of steps per actor rollout.
    total_timesteps: int = 2_621_440_000  # Total number of steps by the end of training.
    update_epochs: int = 10  # Number of epochs for PPO update.
    minibatch_size: int = 1024  # Minibatch size.
    num_minibatches: int = 128  # Number of minibatches per epoch.
    num_updates: int = 10000  # Number of parameter updates until training concludes.
    # dataset config
    dds_results_dir: str = "dds_results"  # Path to the directory where dds_results are located.
    # eval config
    num_eval_envs: int = 100_000  # Number of parallel environments for evaluation.
    eval_opp_activation: str = "relu"  # Activation function of the opponent during evaluation.
    eval_opp_model_type: Literal["DeepMind", "FAIR"] = "DeepMind"  # Model type of the opponent during evaluation.
    eval_opp_model_path: str = "bridge_models/model-sl.pkl"  # Path to the baseline model prepared for evaluation.
    num_eval_step: int = 100  # Interval for evaluation.
    # log config
    save_model: bool = True  # Whether to save the trained model.
    save_model_interval: int = 100  # Interval for saving the trained model.
    log_path: str = "rl_log"  # Path to the directory where training settings and trained models are saved.
    exp_name: str = "exp_0000"  # Name of the experiment.
    save_model_path: str = "rl_params"  # Path to the directory where the trained model is saved.
    # actor config
    initial_model_path: str | None = None  # Path to the initial model for the neural network.
    actor_activation: str = "relu"  # Activation function of the model being trained.
    actor_model_type: Literal["DeepMind", "FAIR"] = "DeepMind"  # Model type being trained.
    # opposite config
    use_fsp: bool = False  # Whether to use fictitious self-play.
    num_latest_models: int = 1_000_000_000  # Number of latest models to use for fictitious self-play.
    # GAE config
    gamma: float = 1  # Discount factor gamma.
    gae_lambda: float = 0.95  # GAE lambda for advantage estimation.
    # loss config
    clip_eps: float = 0.2  # Clipping epsilon for PPO.
    ent_coef: float = 0.001  # Entropy coefficient for exploration.
    vf_coef: float = 0.5  # Coefficient for value loss.
    # PPO code optimization
    max_grad_norm: float = 0.5  # Maximum norm for gradients.
    reward_scale: float = 7600  # Hyperparameter for normalizing rewards.


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    legal_action_mask: jnp.ndarray


def train(config, rng, optimizer):
    config.num_updates = (
        config.total_timesteps // config.num_steps // config.num_envs
    )
    config.num_minibatches = (
        config.num_envs * config.num_steps // config.minibatch_size
    )
    if not os.path.isdir("dds_results"):
        download_dds_results()
    env = BridgeBidding("dds_results/dds_results_10M.npy")

    actor_forward_pass = make_forward_pass(
        activation=config.actor_activation,
        model_type=config.actor_model_type,
    )
    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1,) + env.observation_shape)
    params = actor_forward_pass.init(_rng, init_x)  # params  # DONE
    opt_state = optimizer.init(params=params)  # DONE

    if config.initial_model_path is not None:
        params = pickle.load(open(config.initial_model_path, "rb"))
        print(f"load initial params for actor: {config.initial_model_path}")

    # MAKE EVAL
    rng, eval_rng = jax.random.split(rng)
    eval_env = BridgeBidding("dds_results/dds_results_500K.npy")
    simple_duplicate_evaluate = make_simple_duplicate_evaluate(
        eval_env=eval_env,
        team1_activation=config.actor_activation,
        team1_model_type=config.actor_model_type,
        team2_activation=config.actor_activation,
        team2_model_type=config.actor_model_type,
        num_eval_envs=config.num_eval_envs,
    )
    jit_simple_duplicate_evaluate = jax.jit(simple_duplicate_evaluate)

    # INIT UPDATE FUNCTION

    opp_forward_pass = make_forward_pass(
        activation=config.actor_activation,
        model_type=config.actor_model_type,
    )

    init = jax.jit(jax.vmap(env.init))
    roll_out = jax.jit(make_roll_out(config, env, actor_forward_pass, opp_forward_pass))
    calc_gae = jax.jit(make_calc_gae(config, actor_forward_pass))
    update_step = jax.jit(
        make_update_step(config, actor_forward_pass, optimizer=optimizer)
    )

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config.num_envs)
    env_state = init(reset_rng)

    steps = 0
    terminated_count = 0
    rng, _rng = jax.random.split(rng)
    runner_state = (
        params,
        opt_state,
        env_state,
        env_state.observation,
        terminated_count,
        _rng,
    )  # DONE

    save_model_path = os.path.join(config.log_path, config.exp_name, config.save_model_path)
    os.makedirs(save_model_path, exist_ok=True)
    with open(config.eval_opp_model_path, "rb") as f:
        eval_opp_params = pickle.load(f)
    print("start training")
    for i in range(config.num_updates):
        print(f"--------------iteration {i}---------------")
        # save model
        if i % config.save_model_interval == 0:
            if config.save_model:
                with open(os.path.join(save_model_path, f"params-{i:08}.pkl"), "wb") as writer:
                    pickle.dump(runner_state[0], writer)

        # eval
        if i % config.num_eval_step == 0:
            time_du_sta = time.time()
            log_info, _, _ = jit_simple_duplicate_evaluate(runner_state[0], eval_opp_params, eval_rng)
            eval_log = {"eval/IMP_reward": log_info[0].item(), "eval/IMP_SE": log_info[1].item()}
            time_du_end = time.time()
            print(f"duplicate eval time: {time_du_end-time_du_sta}")

        if config.use_fsp:
            params_list = list(sorted([path for path in os.listdir(save_model_path) if "params" in path]))[-config.num_latest_models:]
            params_path = np.random.choice(params_list)
            print(f"opposite params: {params_path}")
            with open(os.path.join(save_model_path, params_path), "rb") as f:
                opp_params = pickle.load(f)
        else:
            print("opposite params: latest")
            opp_params = runner_state[0]

        time1 = time.time()
        runner_state, traj_batch = roll_out(
            runner_state=runner_state, opp_params=opp_params
        )
        time2 = time.time()
        advantages, targets = calc_gae(runner_state=runner_state, traj_batch=traj_batch)
        time3 = time.time()
        runner_state, loss_info = update_step(
            runner_state=runner_state,
            traj_batch=traj_batch,
            advantages=advantages,
            targets=targets,
        )
        time4 = time.time()

        print(f"rollout time: {time2 - time1}")
        print(f"calc gae time: {time3 - time2}")
        print(f"update time: {time4 - time3}")
        steps += config.num_envs * config.num_steps

        total_loss, (
            value_loss,
            loss_actor,
            entropy,
            approx_kl,
            clipflacs,
            illegal_action_prob_sum,
        ) = loss_info

        # make log
        log = {
            "train/total_loss": float(total_loss.item()),
            "train/value_loss": float(value_loss.item()),
            "train/loss_actor": float(loss_actor.item()),
            "train/illegal_action_prob_sum": float(illegal_action_prob_sum.item()),
            "train/policy_entropy": float(entropy.item()),
            "train/clipflacs": float(clipflacs.item()),
            "train/approx_kl": float(approx_kl.item()),
            "train/ix0": jnp.sum(traj_batch.step_ix.flatten() % 4 == 0).sum().item(),
            "train/ix1": jnp.sum(traj_batch.step_ix.flatten() % 4 == 1).sum().item(),
            "train/ix2": jnp.sum(traj_batch.step_ix.flatten() % 4 == 2).sum().item(),
            "train/ix3": jnp.sum(traj_batch.step_ix.flatten() % 4 == 3).sum().item(),
            "board_num": int(runner_state[4]),
            "steps": steps,
        }
        print(log)
        if i % config.num_eval_step == 0:
            log = {**log, **eval_log}
        wandb.log(log)
    
    if config.save_model:
        with open(os.path.join(save_model_path, f"params-{i + 1:08}.pkl"), "wb") as writer:
            pickle.dump(runner_state[0], writer)

    return runner_state


if __name__ == "__main__":
    config = PPOConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
    print(config)
    wandb.init(
        project="ppo-bridge",
        name=config.exp_name,
        config=config.model_dump(),
        save_code=True,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.lr, eps=1e-5),
    )

    rng = jax.random.PRNGKey(config.seed)
    sta = time.time()
    out = train(config, rng, optimizer)
    end = time.time()
    print("training: time", end - sta)
