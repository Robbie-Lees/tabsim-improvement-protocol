"""Enhanced tab-sim simulations and score-compressed SBI for RFI-affected calibration.

Built on tab-sim (https://github.com/chrisfinlay/tab-sim), which it imports
but never modifies. See ``docs/00_overview.md`` for the map from the protocol
(tabsim_sbi_protocol_v2) to the code.
"""

from tabsim_sbi import _jax  # noqa: F401  (enables JAX x64 before anything else)
from tabsim_sbi.gains import GainParameterisation
from tabsim_sbi.geometry import ObservationGeometry, from_tabsim_zarr, synthetic_geometry
from tabsim_sbi.simulator import SimConfig, Simulator

__version__ = "0.1.0"

__all__ = [
    "GainParameterisation",
    "ObservationGeometry",
    "SimConfig",
    "Simulator",
    "from_tabsim_zarr",
    "synthetic_geometry",
]
