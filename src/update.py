import jax
import jax.numpy as jnp
import distrax
import numpy as np
import optax
from src.utils import masked_policy


def make_warmup_step(config, actor_forward_pass, optimizer):

    def update_step(runner_state, traj_batch, advantages, targets):
        (
            params,
            opt_state,
            env_state,
            last_obs,
            terminated_count,
            rng,
        ) = runner_state

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(tup, batch_info):
                params, opt_state = tup
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    # RERUN NETWORK
                    logits, value = actor_forward_pass.apply(
                        params, traj_batch.obs.astype(jnp.float32)
                    )  # DONE
                    mask = traj_batch.legal_action_mask
                    pi = masked_policy(mask, logits)
                    entropy = pi.entropy().mean()

                    loss_actor = optax.losses.softmax_cross_entropy(
                        logits, 
                        mask.astype(jnp.float32) / mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
                    )
                    loss_actor = loss_actor.mean()

                    pi = distrax.Categorical(logits=logits)
                    illegal_action_probabilities = pi.probs * ~mask
                    illegal_action_prob_sum = illegal_action_probabilities.sum(axis=-1)

                    total_loss = loss_actor

                    return total_loss, (
                        jnp.float32(0), # value_loss.mean()
                        loss_actor.mean(),
                        entropy.mean(),
                        jnp.float32(0),  # approx_kl.mean(),
                        jnp.float32(0),  # clipflacs.mean(),
                        illegal_action_prob_sum.mean(),
                    )

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                loss_info, grads = grad_fn(
                    params, traj_batch, advantages, targets
                )  # DONE
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)  # DONE
                loss_info = jax.tree_map(jnp.mean, loss_info)
                return (
                    params,
                    opt_state,
                ), loss_info  # DONE

            (
                params,
                opt_state,
                traj_batch,
                advantages,
                targets,
                rng,
            ) = update_state  # DONE
            rng, _rng = jax.random.split(rng)
            batch_size = config.minibatch_size * config.num_minibatches
            assert (
                batch_size == config.num_steps * config.num_envs
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config.num_minibatches, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            (params, opt_state), loss_info = jax.lax.scan(
                _update_minbatch, (params, opt_state), minibatches
            )  # DONE
            update_state = (
                params,
                opt_state,
                traj_batch,
                advantages,
                targets,
                rng,
            )  # DONE
            loss_info = jax.tree_map(jnp.mean, loss_info)
            return update_state, loss_info

        update_state = (
            params,
            opt_state,
            traj_batch,
            advantages,
            targets,
            rng,
        )  # DONE
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.update_epochs
        )
        # print(loss_info)
        params, opt_state, _, _, _, rng = update_state  # DONE

        runner_state = (
            params,
            opt_state,
            env_state,
            last_obs,
            terminated_count,
            rng,
        )  # DONE
        loss_info = jax.tree_map(jnp.mean, loss_info)
        return runner_state, loss_info

    return update_step


def make_update_step(config, actor_forward_pass, optimizer):
    def value_loss_fn(value, traj_batch, targets):
        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
            -config.clip_eps, config.clip_eps
        )
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (
            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        )
        return value_loss

    def update_step(runner_state, traj_batch, advantages, targets):
        (
            params,
            opt_state,
            env_state,
            last_obs,
            terminated_count,
            rng,
        ) = runner_state

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(tup, batch_info):
                params, opt_state = tup
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    # RERUN NETWORK
                    logits, value = actor_forward_pass.apply(
                        params, traj_batch.obs.astype(jnp.float32)
                    )  # DONE
                    mask = traj_batch.legal_action_mask
                    pi = masked_policy(mask, logits)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_loss = value_loss_fn(
                        value=value, traj_batch=traj_batch, targets=targets
                    )
                    """
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["clip_eps"], config["clip_eps"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )
                    """

                    # CALCULATE ACTOR LOSS
                    logratio = log_prob - traj_batch.log_prob
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)

                    # gae標準化
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config.clip_eps,
                            1.0 + config.clip_eps,
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()

                    entropy = pi.entropy().mean()

                    pi = distrax.Categorical(logits=logits)
                    illegal_action_probabilities = pi.probs * ~mask
                    illegal_action_prob_sum = illegal_action_probabilities.sum(axis=-1)

                    total_loss = (
                        loss_actor
                        + config.vf_coef * value_loss
                        - config.ent_coef * entropy
                    )
                    """
                    total_loss = -config["ent_coef"] * entropy
                    """
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipflacs = jnp.float32(
                        jnp.abs((ratio - 1.0)) > config.clip_eps
                    ).mean()

                    return total_loss, (
                        value_loss.mean(),
                        loss_actor.mean(),
                        entropy.mean(),
                        approx_kl.mean(),
                        clipflacs.mean(),
                        illegal_action_prob_sum.mean(),
                    )

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                loss_info, grads = grad_fn(
                    params, traj_batch, advantages, targets
                )  # DONE
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)  # DONE
                loss_info = jax.tree_map(jnp.mean, loss_info)
                return (
                    params,
                    opt_state,
                ), loss_info  # DONE

            (
                params,
                opt_state,
                traj_batch,
                advantages,
                targets,
                rng,
            ) = update_state  # DONE
            rng, _rng = jax.random.split(rng)
            batch_size = config.minibatch_size * config.num_minibatches
            assert (
                batch_size == config.num_steps * config.num_envs
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config.num_minibatches, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            (params, opt_state), loss_info = jax.lax.scan(
                _update_minbatch, (params, opt_state), minibatches
            )  # DONE
            update_state = (
                params,
                opt_state,
                traj_batch,
                advantages,
                targets,
                rng,
            )  # DONE
            loss_info = jax.tree_map(jnp.mean, loss_info)
            return update_state, loss_info

        update_state = (
            params,
            opt_state,
            traj_batch,
            advantages,
            targets,
            rng,
        )  # DONE
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.update_epochs
        )
        # print(loss_info)
        params, opt_state, _, _, _, rng = update_state  # DONE

        runner_state = (
            params,
            opt_state,
            env_state,
            last_obs,
            terminated_count,
            rng,
        )  # DONE
        loss_info = jax.tree_map(jnp.mean, loss_info)
        return runner_state, loss_info

    return update_step
