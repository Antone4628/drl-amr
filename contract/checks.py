"""Backend-independent checks on code written against the v2 contract.

training_only_params  which of a function's parameters could receive a
                      training-only type. Used to assert that the
                      observation builder shared with deployment accepts
                      none.
"""
from __future__ import annotations

import inspect
import typing
from collections.abc import Callable

from contract.reports import AdaptReport, AuxFields, IntervalReport

# Types that exist in training and not at deployment. AuxFields is the
# auxiliary channel; IntervalReport contains it; AdaptReport is a training
# diagnostic the deployed loop never builds.
TRAINING_ONLY = (AdaptReport, AuxFields, IntervalReport)


def _mentions_training_only(annotation: object) -> bool:
    """True if an annotation is, or contains at any depth, a training-only
    type or typing.Any."""
    if annotation is typing.Any:
        return True
    if isinstance(annotation, (list, tuple)):
        return any(_mentions_training_only(a) for a in annotation)
    if any(annotation is t for t in TRAINING_ONLY):
        return True
    return any(_mentions_training_only(a) for a in typing.get_args(annotation))


def training_only_params(fn: Callable) -> list[str]:
    """Names of fn's parameters that could receive a training-only type.

    A parameter is reported when its annotation is or contains one of
    TRAINING_ONLY (AuxFields | None, dict[str, AuxFields], ...), when it is
    annotated Any, and when it has no annotation at all: an unannotated
    parameter accepts everything.

    Args:
        fn: The function to inspect, e.g. the actor observation builder.

    Returns:
        The offending parameter names, in signature order; empty when the
        signature is clean.
    """
    hints = typing.get_type_hints(fn)
    offending = []
    for name in inspect.signature(fn).parameters:
        if name not in hints or _mentions_training_only(hints[name]):
            offending.append(name)
    return offending