import numpy as np

import scipy
import scipy.constants
import scipy.linalg as la
import scipy.sparse.linalg as sla

from types import SimpleNamespace

##################################################################
# Pauli matrices
pauli = SimpleNamespace(
s0 = np.array([[1.0, 0.0], [0.0, 1.0]]),
sx = np.array([[0.0, 1.0], [1.0, 0.0]]),
sy = np.array([[0.0, -1j], [1j, 0.0]]),
sz = np.array([[1.0, 0.0], [0.0, -1.0]]),
)

# Kron products
t0s0 = np.kron(s0, s0),
t0sx = np.kron(s0, sx),
t0sy = np.kron(s0, sy),
t0sz = np.kron(s0, sz),

txs0 = np.kron(sx, s0),
txsx = np.kron(sx, sx),
txsy = np.kron(sx, sy),
txsz = np.kron(sx, sz),

tys0 = np.kron(sy, s0),
tysx = np.kron(sy, sx),
tysy = np.kron(sy, sy),
tysz = np.kron(sy, sz),

tzs0 = np.kron(sz, s0),
tzsx = np.kron(sz, sx),
tzsy = np.kron(sz, sy),
tzsz = np.kron(sz, sz),

pauli.sp = (pauli.sx + 1j * pauli.sy) / 2
pauli.sm = (pauli.sx - 1j * pauli.sy) / 2


###################################################################
# Units
# Length : nm
# Energy : meV
# Current : nA
# Temperature : K

constants = SimpleNamespace(
    hbar=scipy.constants.hbar,
    m_e=scipy.constants.m_e,
    eV=scipy.constants.eV,
    e=scipy.constants.e,
    meV=scipy.constants.eV * 1e-3,
    k_B=scipy.constants.k / (scipy.constants.eV * 1e-3),
    mu_B=scipy.constants.physical_constants["Bohr magneton"][0]
    / (scipy.constants.eV * 1e-3),
    current_unit=scipy.constants.k * scipy.constants.e / scipy.constants.hbar * 1e9,
)