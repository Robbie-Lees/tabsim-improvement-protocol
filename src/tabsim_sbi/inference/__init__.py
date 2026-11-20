"""Density estimation and simulation proposals (protocol Sections 3.4-3.5)."""

from tabsim_sbi.inference.mdn import MDN, train_mdn
from tabsim_sbi.inference.proposal import GaussianProposal, sir_correct, standard_normal_logpdf

__all__ = ["MDN", "GaussianProposal", "sir_correct", "standard_normal_logpdf", "train_mdn"]
