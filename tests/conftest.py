from __future__ import annotations

from typing import Any

import pytest

from mfs import config
from mfs.world.world import Site, build_site


@pytest.fixture(scope="session")
def tiny() -> Site:
    return build_site(5, "tiny")


@pytest.fixture(scope="session")
def fast_cfg() -> dict[str, Any]:
    """The platform's settings with a shorter training run, so the suite stays fast on a laptop CPU."""
    return config.merged(config.platform(), {"perf_model": {"epochs": 150, "members": 3}})
