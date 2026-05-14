"""pytest config — registers `gpu` marker and lets users skip via -m 'not gpu'."""

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_gpu = pytest.mark.skip(reason="ROCm GPU not available (run with --rocm or set ROCMOLKIT_HAS_GPU=1)")
    has_gpu = config.getoption("--rocm") or _env_truthy("ROCMOLKIT_HAS_GPU")
    if has_gpu:
        return
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--rocm",
        action="store_true",
        default=False,
        help="enable tests that require an AMD GPU + ROCm runtime",
    )


def _env_truthy(name: str) -> bool:
    import os

    val = os.environ.get(name, "")
    return val.lower() in {"1", "true", "yes", "on"}
