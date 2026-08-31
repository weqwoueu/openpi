import flax.nnx as nnx
import jax
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
