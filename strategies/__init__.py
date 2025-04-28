"""Strategy plug-in package.

Any ``.py`` file placed in this directory that defines subclasses of
:class:`Strategy` will automatically be discovered by the GUI at start-up
and become selectable in the *Strategies* dock.
"""

from __future__ import annotations

# The base class lives in ``base`` to avoid import cycles.
from .base import Strategy  # noqa: F401 – re-export for convenience
