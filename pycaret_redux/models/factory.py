"""Resolve model ID to estimator instance."""

from __future__ import annotations

from typing import Any

from sklearn.base import clone

from pycaret_redux.models.registry import ModelRegistry


def create_estimator(
    id_or_estimator: str | Any,
    registry: ModelRegistry,
    **kwargs,
) -> Any:
    """Create a fresh estimator instance.

    Parameters
    ----------
    id_or_estimator : str or estimator
        Model ID string (e.g. "lr") or a pre-built estimator instance.
    registry : ModelRegistry
        The model registry to look up IDs.
    **kwargs
        Additional arguments passed to the constructor (override defaults).

    Returns
    -------
    estimator
        A fresh (unfitted) estimator instance.
    """
    if isinstance(id_or_estimator, str):
        return registry.create_instance(id_or_estimator, **kwargs)
    # Clone the provided estimator to get a fresh copy
    return clone(id_or_estimator).set_params(**kwargs) if kwargs else clone(id_or_estimator)
