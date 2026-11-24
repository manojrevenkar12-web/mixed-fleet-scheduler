from __future__ import annotations

from mfs.world.world import Site, build_site


def test_same_seed_same_site(tiny: Site) -> None:
    again = build_site(5, "tiny")
    assert again.profiles() == tiny.profiles()
    assert (again.demand == tiny.demand).all()
    assert again.step_s("llama-8b", "A100", 1, 16, 3) == tiny.step_s("llama-8b", "A100", 1, 16, 3)


def test_other_seed_other_site(tiny: Site) -> None:
    other = build_site(6, "tiny")
    assert other.step_s("llama-8b", "A100", 1, 16) != tiny.step_s("llama-8b", "A100", 1, 16)


def test_bigger_batches_take_longer_but_serve_more(tiny: Site) -> None:
    t1, t32 = tiny.step_s("llama-8b", "H100", 1, 1), tiny.step_s("llama-8b", "H100", 1, 32)
    assert t32 > t1
    assert 32 / t32 > 8 / t1  # batching pays while decode is memory-bound


def test_tensor_parallel_cuts_the_step_of_a_large_model(tiny: Site) -> None:
    assert tiny.step_s("llama-70b", "H100", 4, 8) < tiny.step_s("llama-70b", "H100", 2, 8)


def test_a_driver_change_slows_the_gpu_from_its_hour(tiny: Site) -> None:
    before = tiny.step_s("llama-8b", "L40S", 1, 16, 11)
    after = tiny.step_s("llama-8b", "L40S", 1, 16, 12)
    assert after > before * 1.1
    assert tiny.step_s("llama-8b", "A100", 1, 16, 12) == tiny.step_s("llama-8b", "A100", 1, 16, 11)


def test_a_model_that_does_not_fit_is_refused(tiny: Site) -> None:
    assert not tiny.fits("llama-70b", "L40S", 1, 1)  # 140 GB of weights on a 48 GB card
    assert tiny.fits("llama-8b", "L40S", 1, 8)


def test_held_back_pairs_have_no_profiles(tiny: Site) -> None:
    pairs = {(p.model, p.gpu) for p in tiny.profiles()}
    assert ("llama-70b", "L40S") not in pairs
    assert ("llama-8b", "H100") in pairs


def test_a_measurement_is_noisy_but_near_the_truth(tiny: Site) -> None:
    p = tiny.profiles()[0]
    truth = tiny.step_s(p.model, p.gpu, p.tp, p.batch) * 1000
    assert abs(p.step_ms / truth - 1) < 0.25
