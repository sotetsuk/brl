import jax
import jax.numpy as jnp
import distrax
from pgx.experimental.wrappers import auto_reset

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


def teammate_player(target_player):
    return jnp.int32([1, 0, 3, 2])[target_player]


def make_skip_fn(
    init_fn, step_fn, forward_pass, actor_params, opp_params, target_player=0
):
    def skip_fn(state, rng):
        state, rewards, _ = jax.lax.while_loop(
            lambda x: x[0].current_player != target_player,
            body_fn,
            (state, state.rewards, rng),
        )
        return state.replace(rewards=rewards)  # todo: fix

    def body_fn(x):
        state, rewards, rng = x
        batch_size = rewards.shape[0]

        params = jax.lax.cond(
            state.current_player == teammate_player(target_player),
            lambda: actor_params,
            lambda: opp_params,
        )

        logits, _ = forward_pass.apply(
            params,
            state.observation.astype(jnp.float32),
        )
        logits = logits + jnp.finfo(jnp.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        rng, _rng = jax.random.split(rng)
        action = pi.sample(seed=_rng)
 
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, batch_size)
        state = jax.vmap(auto_reset(step_fn, init_fn))(state, action, rngs)

        return state, rewards + state.rewards, rng
        
    return skip_fn


def single_play_step_two_policy_commpetitive(
    init_fn, step_fn, actor_forward_pass, actor_params, opp_forward_pass, opp_params
):
    skip_fn = make_skip_fn(
        init_fn,
        step_fn,
        actor_forward_pass,
        actor_params,
        opp_params,
        target_player=0,
    )

    def wrapped_step_fn(state, action, rng):
        batch_size = action.shape[0]

        rng1, rng2 = jax.random.split(rng)
        rngs = jax.random.split(rng1, batch_size)
        state = jax.vmap(auto_reset(step_fn, init_fn))(state, action, rngs)

        state = skip_fn(state, rng2)

        return state

    return wrapped_step_fn


def normal_step(step_fn):
    def wrapped_step_fn(state, action, rng):
        batch_size = action.shape[0]
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, batch_size)
        state = jax.vmap(step_fn)(state, action, rngs)
        return state

    return wrapped_step_fn
