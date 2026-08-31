import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from . import train_value


def test_value_tokenizer_accepts_common_prompt_transform_arguments():
    class FakeTokenizer:
        def encode(self, _text, *, add_bos, add_eos):
            assert add_bos is True
            assert add_eos is False
            return [1, 2]

    tokenizer = train_value.GemmaValueTokenizer(max_len=4)
    tokenizer._tokenizer = FakeTokenizer()  # noqa: SLF001

    tokens, mask = tokenizer.tokenize(
        "insert the plug",
        state=None,
        adv_ind="positive",
        adv_ind_dropout=True,
    )

    assert tokens.tolist() == [1, 2, 0, 0]
    assert mask.tolist() == [True, True, False, False]


def test_train_state_is_registered_as_jax_pytree():
    model = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    params = nnx.state(model)
    optimizer = optax.adam(1e-3)
    state = train_value.TrainState(
        step=0,
        params=params,
        model_def=nnx.graphdef(model),
        opt_state=optimizer.init(params),
        ema_params=jax.tree.map(lambda value: value, params),
    )

    leaves = jax.tree.leaves(state)
    mesh = jax.make_mesh((1, 1), ("batch", "fsdp"))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    placed = jax.tree.map(
        lambda value: jax.device_put(value, replicated),
        state,
        is_leaf=lambda value: hasattr(value, "shape"),
    )

    assert all(not isinstance(leaf, train_value.TrainState) for leaf in leaves)
    assert isinstance(placed, train_value.TrainState)
    assert isinstance(placed.params, nnx.State)
    assert isinstance(placed.model_def, nnx.GraphDef)


def test_checkpoint_round_trip_restores_optimizer_state(tmp_path):
    def make_state(step: int):
        model = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
        params = nnx.state(model)
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3))
        opt_state = optimizer.init(params)
        grads = jax.tree.map(jnp.ones_like, params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        state = train_value.TrainState(
            step=step,
            params=params,
            model_def=nnx.graphdef(model),
            opt_state=opt_state,
            ema_params=jax.tree.map(lambda value: value, params),
        )
        return state, optimizer

    saved_state, _ = make_state(step=7)
    train_value.save_checkpoint(saved_state, tmp_path)

    fresh_state, optimizer = make_state(step=0)
    restored_state = train_value.load_checkpoint(tmp_path / "step_00000007", fresh_state)

    assert train_value._host_step(restored_state.step) == 7  # noqa: SLF001
    for restored_tree, expected_tree in (
        (restored_state.params, saved_state.params),
        (restored_state.ema_params, saved_state.ema_params),
        (restored_state.opt_state, saved_state.opt_state),
    ):
        assert jax.tree.structure(restored_tree) == jax.tree.structure(expected_tree)
        for restored, expected in zip(
            jax.tree.leaves(restored_tree),
            jax.tree.leaves(expected_tree),
            strict=True,
        ):
            assert jnp.array_equal(restored, expected)

    grads = jax.tree.map(jnp.ones_like, restored_state.params)
    optimizer.update(grads, restored_state.opt_state, restored_state.params)
