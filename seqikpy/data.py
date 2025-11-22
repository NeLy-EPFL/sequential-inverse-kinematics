"""Deprecated data module."""

import warnings
from .body_config import neuromechfly_body_config as _default_nmf_config

# Issue deprecation warning upon import
message = (
    f"Constants in `seqikpy.data` (i.e. `INITIAL_ANGLES`, `BOUNDS`, `NMF_SIZE`, "
    "`PTS2ALIGN`, `SKELETON`, and `NMF_TEMPLATE`) are deprecated and will be removed "
    "in a future version. Use the `seqikpy.body_config.BodyConfig` dataclass instead. "
    "The new way to access these constants is to import the default configuration "
    "`seqikpy.body_config.neuromechfly_body_config` and access its attributes. "
    "For example, instead of `seqikpy.data.NMF_TEMPLATE`, use "
    "`seqikpy.body_config.neuromechfly_body_config.template`."
)
warnings.warn(message, DeprecationWarning, stacklevel=2)

# Define deprecated constants from the new body_config module
INITIAL_ANGLES = _default_nmf_config.initial_angles_rad
BOUNDS = _default_nmf_config.dof_bounds_rad
NMF_SIZE = _default_nmf_config.segment_sizes
PTS2ALIGN = _default_nmf_config.points_to_align
SKELETON = _default_nmf_config.skeleton
NMF_TEMPLATE = _default_nmf_config.template
