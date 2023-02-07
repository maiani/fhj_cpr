import kwant
import numpy as np
import sympy as sym
from kwant.continuum import discretize, sympify
from physics_utils import pauli


def fnwjj_hamiltonian_sym(dims):
    """
    Generate a symbolic Hamiltonian of the effective model for a ferromagnetic hybird junction.

        Parameters:
            dims (int) : Number of dimension (1 or 2)

    """

    hamiltonian = sympify(
        """ 
        k_c * (k_x**2 + k_y**2) * kron(sigma_z, sigma_0)
        + V(x, y) * kron(sigma_z, sigma_0) 
        + (+ alpha_z(x, y) * k_y) * kron(sigma_z, sigma_x)
        + (- alpha_z(x, y) * k_x) * kron(sigma_z, sigma_y)
        + beta * (k_x * kron(sigma_z, sigma_x) - k_y * kron(sigma_z, sigma_y))
        + h_x(x, y) * kron(sigma_0, sigma_x)
        + h_y(x, y) * kron(sigma_0, sigma_y)
        + h_z(x, y) * kron(sigma_0, sigma_z) 
        + Delta(x, y) * ( cos(theta(x,y)) * kron(sigma_x, sigma_0) - sin(theta(x,y)) * kron(sigma_y, sigma_0) )
    """
    )
    
    if dims == 1:
        hamiltonian = hamiltonian.subs(sym.Symbol("k_y", commutative=False), 0)
        hamiltonian = hamiltonian.subs(sym.Symbol("y", commutative=False), 0)
    elif dims == 2:
        pass
    else:
        raise ValueError("Only dimensions 1 and 2 are implemented.")

    return sym.simplify(hamiltonian)


def make_2D_system(L_x, L_y, a_x, a_y, finalized=True):
    """
    Create a Kwant system of a finite width ferromagnetic hybrid junction.
    """
    lat = kwant.lattice.Monatomic(
        [[a_x, 0], [0, a_y]], offset=None, name="wire", norbs=4
    )
    hamiltonian = fnwjj_hamiltonian_sym(dims=2)
    template = discretize(hamiltonian, grid=lat)

    def shape(site):
        (x, y) = site.pos
        return (abs(x) <= L_x / 2) and (abs(y) <= L_y / 2)

    syst = kwant.Builder()
    syst.fill(template, shape, (0.0, 0.0))


    if finalized:
        syst = syst.finalized()

    return syst, lat


def get_default_pf_params_tripartite():

    """
    Default potential parameters.
    """

    D0 = 0.250  # Gap in meV

    pf_params = dict(
        L_junction=400,
        mu_C=2 * D0,
        mu_L=4.0,
        mu_R=4.0,
        L_barr_L=20,
        L_barr_R=20,
        V_barr_L=-5,  # 4 * D0,
        V_barr_R=-5,  # 4 * D0,
        Delta_0=D0,
        theta_LR=np.pi,
        alpha_z_L=2.0,
        alpha_z_C=2.0,
        alpha_z_R=2.0,
        h_x_L=0.0 * D0,
        h_x_C=0.0 * D0,
        h_x_R=0.0 * D0,
        h_y_L=0.0 * D0,
        h_y_C=0.0 * D0,
        h_y_R=0.0 * D0,
    )

    return pf_params


def generate_pf_tripartite(
    L_junction,  # L_C + L_barr_R + L_barr_L
    mu_C,
    mu_L,
    mu_R,
    L_barr_L,
    L_barr_R,
    V_barr_L,
    V_barr_R,
    Delta_0,
    theta_LR,
    alpha_z_L,
    alpha_z_C,
    alpha_z_R,
    h_x_L,
    h_x_C,
    h_x_R,
    h_y_L,
    h_y_C,
    h_y_R,
):

    """
    Generate the potential functions V(x, y), Delta(x, y), theta(x, y), h_x(x, y), h_y(x, y), h_z(x, y)
    """

    def sig(x, l_smth=1e-3):
        return 0.5 * (1 + np.tanh(0.5 * x / l_smth))

    def f_R(x):
        return sig(x - L_junction / 2)

    def f_L(x):
        return 1 - sig(x + L_junction / 2)

    def f_C(x):
        return sig(x + L_junction / 2 - L_barr_L) - sig(x - L_junction / 2 + L_barr_R)

    def f_BR(x):
        return sig(x - L_junction / 2 + L_barr_R) - sig(x - L_junction / 2)

    def f_BL(x):
        return sig(x + L_junction / 2) - sig(x + L_junction / 2 - L_barr_L)

    def V(x, y):
        return (
            -mu_C * f_C(x)
            - mu_L * f_L(x)
            - mu_R * f_R(x)
            + V_barr_R * f_BR(x)
            + V_barr_L * f_BL(x)
        )

    def Delta(x, y):
        return +Delta_0 * f_L(x) + Delta_0 * f_R(x)

    def theta(x, y):
        return 0 * f_L(x) + theta_LR * f_R(x)

    def alpha_x(x, y):
        return +0 * (f_C(x) + f_BL(x) + f_BR(x)) + 0 * f_L(x) + 0 * f_R(x)

    def alpha_y(x, y):
        return +0 * (f_C(x) + f_BL(x) + f_BR(x)) + 0 * f_L(x) + 0 * f_R(x)

    def alpha_z(x, y):
        return (
            +alpha_z_C * (f_C(x) + f_BL(x) + f_BR(x))
            + alpha_z_L * f_L(x)
            + alpha_z_R * f_R(x)
        )

    def h_x(x, y):
        return +h_x_C * (f_C(x) + f_BL(x) + f_BR(x)) + h_x_L * f_L(x) + h_x_R * f_R(x)

    def h_y(x, y):
        return +h_y_C * (f_C(x) + f_BL(x) + f_BR(x)) + h_y_L * f_L(x) + h_y_R * f_R(x)

    def h_z(x, y):
        return +0 * (f_C(x) + f_BL(x) + f_BR(x)) + 0 * f_L(x) + 0 * f_R(x)

    return V, Delta, theta, alpha_x, alpha_y, alpha_z, h_x, h_y, h_z


def get_default_pf_params_domains():

    """
    Default potential parameters.
    """

    D0 = 0.200  # Gap in meV

    pf_params = dict(
        L_junction=800,
        mu_C=2 * D0,
        mu_L=4.0,
        mu_R=4.0,
        L_barr_L=20,
        L_barr_R=20,
        V_barr_L=-4,  # 4 * D0,
        V_barr_R=-4,  # 4 * D0,
        Delta_0=D0,
        theta_LR=np.pi,
        h=0.9 * D0,
        period=80,
    )

    return pf_params


def generate_pf_domains(
    L_junction,
    mu_C,
    mu_L,
    mu_R,
    L_barr_L,
    L_barr_R,
    V_barr_L,
    V_barr_R,
    Delta_0,
    theta_LR,
    h,
    period,
):

    """
    Generate the potential functions.
    """

    def sig(x, l_smth=1e-3):
        return 0.5 * (1 + np.tanh(0.5 * x / l_smth))

    def f_R(x):
        return sig(x - L_junction / 2)

    def f_L(x):
        return 1 - sig(x + L_junction / 2)

    def f_C(x):
        return sig(x + L_junction / 2 - L_barr_L) - sig(x - L_junction / 2 + L_barr_R)

    def f_BR(x):
        return sig(x - L_junction / 2 + L_barr_R) - sig(x - L_junction / 2)

    def f_BL(x):
        return sig(x + L_junction / 2) - sig(x + L_junction / 2 - L_barr_L)

    def V(x, y):
        return (
            -mu_C * f_C(x)
            - mu_L * f_L(x)
            - mu_R * f_R(x)
            + V_barr_R * f_BR(x)
            + V_barr_L * f_BL(x)
        )

    def Delta(x, y):
        return +Delta_0 * f_L(x) + Delta_0 * f_R(x)

    def theta(x, y):
        return 0 * f_L(x) + theta_LR * f_R(x)

    def h_x(x, y):
        return h * (
            signal.square(x * (2 * np.pi) / period, duty=0.25)
            - signal.square((x + period / 2) * (2 * np.pi) / period, duty=0.25)
        )

    def h_y(x, y):
        return h * (
            signal.square((x + period / 4) * (2 * np.pi) / period, duty=0.25)
            - signal.square((x + 3 * period / 4) * (2 * np.pi) / period, duty=0.25)
        )

    def h_z(x, y):
        return 0

    return V, Delta, theta, alpha_x, alpha_y, alpha_z, h_x, h_y, h_z


def get_default_params() -> dict:
    """
    Generate the default prameters dict.
    """

    ############### POTENTIAL LANDSCAPE #######################

    params = {
        # Material parameters
        "k_c": 38.0998212 / 0.026,
        # Dresselhaus spin-orbit coupling
        "beta": 0,
        ##########
        "alpha_c": 0,
        # Electrostatic potential landscape
        "V": lambda *args: 0,
        # Superconductive pairing potential
        "Delta": lambda *args: 0,
        "theta": lambda *args: 0,
        # SOC field
        "alpha_x": lambda *args: 0,
        "alpha_y": lambda *args: 0,
        "alpha_z": lambda *args: 0,
        # Zeeman field
        "h_x": lambda *args: 0,
        "h_y": lambda *args: 0,
        "h_z": lambda *args: 0,
        # Functions
        "cos": np.cos,
        "sin": np.sin,
    }
    return params
