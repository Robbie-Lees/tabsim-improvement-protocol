"""Validation: V1 calibration (:mod:`.calibration`), V3 realism (:mod:`.realism`).

V0 (Gaussian limit) is the simulator/forward parity tests plus running the
protocol with every enhancement off; V2 (compression loss) compares the
Gaussian score, the Student-t score and a learned compressor on a reduced
array; V4 (posterior predictive checks) reuses the simulator with posterior
draws and the statistics in :mod:`.realism`.
"""
