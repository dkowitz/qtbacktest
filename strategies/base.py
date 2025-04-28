"""Base class for back-testing strategy plug-ins."""

from __future__ import annotations

from typing import Dict, Any


class Strategy:
    """Abstract strategy skeleton.

    Sub-classes **must** implement :meth:`next` and may override
    :meth:`prepare`.
    """

    # Concrete strategies must provide a param_schema mapping; see examples.
    param_schema: Dict[str, Any] = {}

    def __init__(self, **params):
        self.params = params

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def prepare(self, df):  # noqa: D401 – simple stub
        """Pre-compute indicators on *df* before the walk-forward loop."""

    # ------------------------------------------------------------------
    # Required hooks
    # ------------------------------------------------------------------

    def next(self, i, row, position):  # pragma: no cover – abstract
        """Called bar-by-bar during back-test – must be implemented."""

        raise NotImplementedError
