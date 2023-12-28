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
t0s0 = np.kron(pauli.s0, pauli.s0),
t0sx = np.kron(pauli.s0, pauli.sx),
t0sy = np.kron(pauli.s0, pauli.sy),
t0sz = np.kron(pauli.s0, pauli.sz),

txs0 = np.kron(pauli.sx, pauli.s0),
txsx = np.kron(pauli.sx, pauli.sx),
txsy = np.kron(pauli.sx, pauli.sy),
txsz = np.kron(pauli.sx, pauli.sz),

tys0 = np.kron(pauli.sy, pauli.s0),
tysx = np.kron(pauli.sy, pauli.sx),
tysy = np.kron(pauli.sy, pauli.sy),
tysz = np.kron(pauli.sy, pauli.sz),

tzs0 = np.kron(pauli.sz, pauli.s0),
tzsx = np.kron(pauli.sz, pauli.sx),
tzsy = np.kron(pauli.sz, pauli.sy),
tzsz = np.kron(pauli.sz, pauli.sz),

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