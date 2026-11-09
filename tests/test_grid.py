from fp.grid import VARIANTS, Unit, split_priority, units

NAMES = {
    "liberox_groot": ["signature_seed1", "task_seed0", "skill_b", "task_seed1", "signature_seed0", "skill_a", "task_seed2", "signature_seed2"],
    "libero10_groot": ["paper_seed2", "paper_seed0", "paper_seed1"],
    "libero10_pi05": ["paper_seed0", "paper_seed1", "paper_seed2"],
}


def test_a_unit_holds_the_16_architectures_of_its_label_variant_with_distinct_names():
    methods = Unit("liberox_groot", "task_seed0", "onset_pre").methods
    assert len(methods) == len(set(methods)) == 16 and all(m.endswith("__onset_pre") for m in methods)
    assert "lstm_plain__onset_pre" in methods and "mlp_mha_modes_clip__onset_pre" in methods


def test_seed_zero_of_every_split_kind_comes_before_the_other_seeds():
    order = [u.split for u in units(NAMES) if u.dataset == "liberox_groot" and u.variant == "onset_pre"]
    assert order == ["task_seed0", "signature_seed0", "skill_a", "skill_b", "task_seed1", "signature_seed1", "task_seed2", "signature_seed2"]


def test_the_pre_onset_variant_goes_first_on_every_split_and_the_datasets_keep_their_order():
    run = units(NAMES)
    first_pass = run[: len(run) // len(VARIANTS)]
    assert {u.variant for u in first_pass} == {"onset_pre"} and run[0] == Unit("liberox_groot", "task_seed0", "onset_pre")
    assert [u.dataset for u in first_pass if split_priority(u.split)[0] == 0] == ["liberox_groot"] * 4 + ["libero10_groot", "libero10_pi05"]  # task, signature and 2 skills at seed 0
    assert [v for v in dict.fromkeys(u.variant for u in run)] == list(VARIANTS)


def test_every_probe_appears_exactly_once_on_every_split():
    run = units(NAMES)
    runs = [(u.dataset, u.split, m) for u in run for m in u.methods]
    assert len(run) == 4 * 14 and len(runs) == len(set(runs)) == 4 * 14 * 16


def test_split_priority_reads_seed_and_kind_from_the_name():
    assert split_priority("task_seed2") == (2, 0, "task_seed2") and split_priority("skill_stack_objects") == (0, 2, "skill_stack_objects")
    assert split_priority("signature_seed1")[:2] == (1, 1) and split_priority("paper_seed0")[:2] == (0, 0)
