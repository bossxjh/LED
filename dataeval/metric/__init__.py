__all__ = [
    "compute_leanability_from_npzdata",
    "compute_task_diversity_entropy",
    "compute_baselines_from_npzdata",
]


def __getattr__(name):
    if name == "compute_leanability_from_npzdata":
        from .leanability import compute_leanability_from_npzdata

        return compute_leanability_from_npzdata
    if name == "compute_task_diversity_entropy":
        from .diversity import compute_task_diversity_entropy

        return compute_task_diversity_entropy
    if name == "compute_baselines_from_npzdata":
        from .baselines import compute_baselines_from_npzdata

        return compute_baselines_from_npzdata
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
