import itertools

import pytest
import torch

from fp import config
from fp.models.ours import Probe, ProbeConfig, all_configs

D, B, T = 8, 3, 7  # tiny widths for most tests: 8 columns per modality


def tiny(cfg: ProbeConfig) -> ProbeConfig:
    return ProbeConfig(**{**cfg.__dict__, "d_modality": D, "hidden": 16, "proj": 8, "n_modes": 5, "clip_dim": 6})


TINY = [tiny(c) for c in all_configs()]
IDS = [c.name for c in TINY]


def make(cfg, seed=0) -> Probe:
    torch.manual_seed(seed)
    return Probe(cfg).eval()


def batch(cfg, seed=1, steps=T):
    return torch.randn(B, steps, cfg.d_in, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------- the suite of architectures
def test_the_suite_has_sixteen_distinct_architectures():
    cfgs = all_configs()
    assert len(cfgs) == 16 and len({c.name for c in cfgs}) == 16
    assert {c.name for c in cfgs} >= {"lstm_plain", "lstm_modes_clip", "mlp_clip", "lstm_mha_modes", "mlp_mha_modes_clip"}


# ---------------------------------------------------------------- shapes and heads
@pytest.mark.parametrize("cfg", TINY, ids=IDS)
def test_outputs_are_exactly_the_configured_heads_with_the_right_shapes(cfg):
    out = make(cfg)(batch(cfg))
    assert set(out) == {"score"} | ({"mode"} if cfg.modes else set()) | ({"sem"} if cfg.clip else set())
    assert out["score"].shape == (B, T)
    if cfg.modes:
        assert out["mode"].shape == (B, T, cfg.n_modes)
    if cfg.clip:
        assert out["sem"].shape == (B, T, cfg.clip_dim)


@pytest.mark.parametrize("cfg", TINY, ids=IDS)
def test_every_parameter_receives_a_gradient(cfg):
    model = make(cfg).train()
    out = model(batch(cfg))
    sum(v.sum() for v in out.values()).backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


# ---------------------------------------------------------------- causality and padding
@pytest.mark.parametrize("cfg", TINY, ids=IDS)
def test_an_output_never_depends_on_later_steps(cfg):
    model, x = make(cfg), batch(cfg)
    changed = x.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:])  # new content from step 4 on
    a, b = model(x), model(changed)
    for key in a:
        assert torch.allclose(a[key][:, :4], b[key][:, :4], atol=1e-6), key
    assert not torch.allclose(a["score"][:, 4:], b["score"][:, 4:])  # and the later outputs do change, so the test can fail


@pytest.mark.parametrize("cfg", TINY, ids=IDS)
def test_zero_padding_after_the_real_steps_does_not_change_the_real_outputs(cfg):
    model, x = make(cfg), batch(cfg)
    padded = torch.cat([x, torch.zeros(B, 5, cfg.d_in)], dim=1)
    a, b = model(x), model(padded)
    for key in a:
        assert torch.allclose(a[key], b[key][:, :T], atol=1e-6), key


# ---------------------------------------------------------------- the attention switch
def test_attention_changes_the_width_the_trunk_reads():
    plain, att = tiny(ProbeConfig(trunk="lstm")), tiny(ProbeConfig(trunk="lstm", mha=True))
    assert make(plain).trunk.input_size == 3 * D and make(att).trunk.input_size == 3 * att.proj
    assert make(plain).attention is None and make(att).attention is not None


def test_attention_mixes_steps_within_each_modality_only():
    """Each modality has its own projection and attention block, and reads only its own columns: changing the wrist columns changes the
    wrist output of the block and leaves the base output untouched."""
    cfg = tiny(ProbeConfig(mha=True))
    att = make(cfg).attention
    x = batch(cfg)
    other = x.clone()
    other[..., D : 2 * D] = torch.randn_like(other[..., D : 2 * D])  # change the wrist columns only
    a, b = att(x), att(other)
    assert torch.allclose(a[..., : cfg.proj], b[..., : cfg.proj])  # base output unchanged
    assert not torch.allclose(a[..., cfg.proj : 2 * cfg.proj], b[..., cfg.proj : 2 * cfg.proj])  # wrist output changed


# ---------------------------------------------------------------- inputs, config, reproducibility
@pytest.mark.parametrize("mha", [False, True])
def test_a_wrong_number_of_feature_columns_is_an_error(mha):
    cfg = tiny(ProbeConfig(mha=mha))
    with pytest.raises(ValueError, match="expected"):
        make(cfg)(torch.zeros(B, T, cfg.d_in + 1))


def test_bad_configs_are_errors():
    with pytest.raises(ValueError, match="unknown trunk"):
        ProbeConfig(trunk="gru")
    with pytest.raises(ValueError, match="divisible"):
        ProbeConfig(mha=True, proj=130)


def test_the_same_seed_gives_the_same_model():
    cfg = tiny(ProbeConfig(trunk="lstm", mha=True, modes=True, clip=True))
    x = batch(cfg)
    assert torch.equal(make(cfg, seed=3)(x)["score"], make(cfg, seed=3)(x)["score"])
    assert not torch.equal(make(cfg, seed=3)(x)["score"], make(cfg, seed=4)(x)["score"])


def test_names_are_built_from_the_switches():
    assert ProbeConfig().name == "lstm_plain"
    assert ProbeConfig(trunk="mlp", clip=True).name == "mlp_clip"
    assert ProbeConfig(mha=True, modes=True, clip=True).name == "lstm_mha_modes_clip"


# ---------------------------------------------------------------- real sizes
@pytest.mark.parametrize("mha", [False, True])
def test_the_headline_sizes_run_on_6144_columns(mha):
    cfg = ProbeConfig(mha=mha, modes=True, clip=True)
    assert cfg.d_in == config.FEATURE_DIM == 6144
    out = make(cfg)(torch.randn(2, 5, 6144))
    assert out["score"].shape == (2, 5) and out["mode"].shape == (2, 5, 10) and out["sem"].shape == (2, 5, 64)
