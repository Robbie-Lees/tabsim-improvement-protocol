"""Simulator enhancements E1-E8 of the protocol (E7 deferred, see docs).

Each module is a set of pure JAX functions that draw one effect's latent
variables and apply it; :mod:`tabsim_sbi.simulator` composes them.

E1 :mod:`.emitters`     unmodelled emitters and TLE errors
E2 :mod:`.rfi_signal`   power process, per-antenna and complex beams
E3 :mod:`.noise`        signal-dependent self-noise
E4 :mod:`.sky`          unmodelled faint / off-axis / extended sky
E5 :mod:`.gain_errors`  gain residual beyond the model, non-closing errors
E6 :mod:`.receiver`     receiver compression
E8 :mod:`.heavy_tail`   Student-t catch-all
"""
