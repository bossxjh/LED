__all__ = ["extract_features", "extract_features_with_metadata"]


def __getattr__(name):
    if name in __all__:
        from .api import extract_features, extract_features_with_metadata

        return {
            "extract_features": extract_features,
            "extract_features_with_metadata": extract_features_with_metadata,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
