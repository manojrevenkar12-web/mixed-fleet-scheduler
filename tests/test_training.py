from __future__ import annotations

from typing import Any

from mfs.domain import TrainingJob, TrainingReport
from mfs.platform.build import build_platform
from mfs.platform.estimators import Uniform
from mfs.platform.scheduler import Scheduler
from mfs.world.world import Site


def _demand(site: Site, h: int, scale: float = 1.0) -> dict[str, float]:
    return {m: float(site.demand[h, j]) * scale for j, m in enumerate(site.models)}


def test_world_training_jobs_are_the_same_for_every_design(tiny: Site) -> None:
    assert tiny.training_jobs() == tiny.training_jobs()
    assert len(tiny.training_jobs()) == 6
    assert all(p.gpus == 8 for p in tiny.training_profiles())


def test_a_slower_link_and_fewer_flops_train_slower(tiny: Site) -> None:
    assert tiny.train_rate("llama-8b", "H100", 8) > tiny.train_rate("llama-8b", "A100", 8) > 0
    assert not tiny.train_fits("llama-13b", "V100", 8)  # 26 GB of sharded state plus activations on a 32 GB card


def test_training_fills_free_gpus_and_never_takes_inference_gpus(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    sched, _ = build_platform(tiny.services, tiny.fleet, tiny.profiles(), fast_cfg, seed=tiny.seed, training_profiles=tiny.training_profiles())
    sched.submit(TrainingJob(1, "llama-8b", 8, 5e8, 0))
    p = sched.plan(0, _demand(tiny, 0), _demand(tiny, 0))
    assert [a.job for a in p.training] == [1]
    used, train = p.gpus_used(), p.training_gpus()
    for g in tiny.fleet:
        assert used.get(g, 0) + train.get(g, 0) <= tiny.fleet[g]


def test_inference_preempts_training_when_it_needs_the_gpus(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    sched, _ = build_platform(tiny.services, tiny.fleet, tiny.profiles(), fast_cfg, seed=tiny.seed, training_profiles=tiny.training_profiles())
    for i in range(8):
        sched.submit(TrainingJob(i, "llama-8b", 8, 5e10, 0))
    quiet = sched.plan(0, _demand(tiny, 0, 0.1), _demand(tiny, 0, 0.1))
    assert len(quiet.training) >= 2
    busy = sched.plan(1, _demand(tiny, 0, 6.0), _demand(tiny, 0, 6.0))
    assert busy.preempted
    assert sched.preemptions == len(busy.preempted)


def test_a_job_that_ran_out_of_memory_is_not_placed_there_again(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    sched = Scheduler(Uniform(fast_cfg, tiny.services), fast_cfg, tiny.services, {"V100": 16}, admission=False, feedback=False)
    sched.submit(TrainingJob(1, "llama-13b", 8, 1e9, 0))
    quiet = {m: 0.0 for m in tiny.models}
    p = sched.plan(0, quiet, quiet)
    assert p.training and p.training[0].gpu == "V100"  # an A100-minded scheduler thinks it fits
    sched.observe_training(0, [TrainingReport(1, "V100", 8, 0.0, crashed=True)])
    assert not sched.plan(1, quiet, quiet).training


def test_measured_throughput_corrects_the_estimate(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    sched, _ = build_platform(tiny.services, tiny.fleet, tiny.profiles(), fast_cfg, seed=tiny.seed)
    sched.submit(TrainingJob(1, "llama-8b", 8, 1e12, 0))
    quiet = {m: 0.0 for m in tiny.models}
    a = sched.plan(0, quiet, quiet).training[0]
    for h in range(6):
        sched.observe_training(h, [TrainingReport(1, a.gpu, 8, a.tokens_per_s * 0.7)])
    assert abs(sched.est.train_rate("llama-8b", a.gpu, 8) / a.tokens_per_s - 0.7) < 0.02
