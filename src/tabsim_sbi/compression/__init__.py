"""Score compression, matrix-free nuisance hardening and the fiducial fit (protocol Section 3)."""

from tabsim_sbi.compression.fiducial import FitResult, MapFitter, chi2_with, map_fit
from tabsim_sbi.compression.hardening import Hardener
from tabsim_sbi.compression.model import CompressionModel
from tabsim_sbi.compression.score import ScoreCompressor, WhitenedModel

__all__ = ["CompressionModel", "FitResult", "Hardener", "MapFitter", "ScoreCompressor", "WhitenedModel",
           "chi2_with", "map_fit"]
