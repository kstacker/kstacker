"""Numerically stable, vectorized Keplerian sky-plane geometry.

The orbital state is (a, lambda_ref, stellar_mass, h, k, p, q).
Distances are in AU and times in years. Coordinates are North and East.
"""
from __future__ import annotations
import numpy as np

def wrap_2pi(x: np.ndarray | float) -> np.ndarray | float:
    """
    Wrap angles to the half-open interval [0, 2π).

    Used to keep mean longitude, mean anomaly, and argument of periapsis in
    canonical range after arithmetic operations.
    """
    return np.mod(x + 2.0 * np.pi, 2.0 * np.pi)


def mean_motion(a: np.ndarray, m0: np.ndarray) -> np.ndarray:
    """
    Keplerian mean motion: n = 2π √(m0 / a³)  [rad / year].

    With a in AU and m0 in solar masses, and using G·M_sun = 4π² AU³/yr²,
    this gives the mean motion in radians per year.
    """
    return 2.0 * np.pi * np.sqrt(m0 / (a ** 3))


def lambda_to_M0(la0: np.ndarray, h: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Convert mean longitude at the reference epoch to a gauge-chosen mean anomaly.

    This helper is retained only for human-readable derived classical outputs.
    The MCMC likelihood itself never uses this conversion.

    For e>0, varpi = Ω+ω = atan2(h,k) and M0 = λ0-varpi.
    At e=0, varpi is undefined; for reporting we choose varpi=0, so M0=λ0.
    """
    la0 = np.asarray(la0, dtype=float)
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    e = np.hypot(h, k)
    varpi = np.where(e > 0.0, np.arctan2(h, k), 0.0)
    return wrap_2pi(la0 - varpi)


def M0_to_t0(
    M0: np.ndarray, a: np.ndarray, m0: np.ndarray, t_ref: float
) -> np.ndarray:
    """Convert a gauge-chosen M0 to the corresponding reported t0."""
    n = mean_motion(a, m0)
    return t_ref - (M0 / n)


def solve_equinoctial_kepler(
    lambda_value: np.ndarray,
    h: np.ndarray,
    k: np.ndarray,
    *,
    tolerance: float = 1e-13,
    max_iterations: int = 20,
) -> np.ndarray:
    """
    Solve the equinoctial Kepler equation

        λ = F + h cos(F) - k sin(F)

    where F = E + Ω + ω is the eccentric longitude.  The solver is fully
    vectorized over walkers and epochs.  At e=0, h=k=0 and F=λ exactly.
    """
    lam = np.asarray(lambda_value, dtype=float)
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    F = np.array(lam, dtype=float, copy=True)
    for _ in range(int(max_iterations)):
        sF = np.sin(F)
        cF = np.cos(F)
        residual = F + h * cF - k * sF - lam
        derivative = 1.0 - h * sF - k * cF
        step = residual / derivative
        F -= step
        if np.all(np.abs(step) < float(tolerance)):
            break
    return F


def equinoctial_xy_from_hk(
    a: np.ndarray,
    lambda_value: np.ndarray,
    h: np.ndarray,
    k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Direct orbital-plane coordinates from (a, λ, h, k), without reconstructing
    Ω, ω, M, E, or t0.

    With e²=h²+k², η=sqrt(1-e²), β=1/(1+η):

        X/a = (1-βh²)cosF + βhk sinF - k
        Y/a = βhk cosF + (1-βk²)sinF - h
    """
    a = np.asarray(a, dtype=float)
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    e2 = h*h + k*k
    eta = np.sqrt(np.maximum(0.0, 1.0 - e2))
    beta = 1.0 / (1.0 + eta)
    F = solve_equinoctial_kepler(lambda_value, h, k)
    cF = np.cos(F)
    sF = np.sin(F)
    X = a * ((1.0 - beta*h*h) * cF + beta*h*k * sF - k)
    Y = a * (beta*h*k * cF + (1.0 - beta*k*k) * sF - h)
    return X, Y


def project_equinoctial_sin_i2(
    X: np.ndarray,
    Y: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Direct sky projection for the bounded inclination variables

        p = sin(i/2) sin(Ω)
        q = sin(i/2) cos(Ω).

    In the project's (North, East) convention:

        North = (1-2p²) X + 2pq Y
        East  = -2pq X - (1-2q²) Y

    No atan2, angular reconstruction, division, or square root is required.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    north = (1.0 - 2.0*p*p) * X + 2.0*p*q * Y
    east = -2.0*p*q * X - (1.0 - 2.0*q*q) * Y
    return north, east


def _equinoctial_sky_tracks(
    a: np.ndarray,
    la0: np.ndarray,
    m0: np.ndarray,
    h: np.ndarray,
    k: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    ts: np.ndarray,
    *,
    t_ref: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized EqOE propagation to North/East sky coordinates."""
    a = np.asarray(a, dtype=float)
    la0 = np.asarray(la0, dtype=float)
    m0 = np.asarray(m0, dtype=float)
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    ts = np.asarray(ts, dtype=float)

    n = mean_motion(a, m0)
    lam = la0[..., None] + n[..., None] * (ts - float(t_ref))
    X, Y = equinoctial_xy_from_hk(
        a[..., None], lam, h[..., None], k[..., None]
    )
    north, east = project_equinoctial_sin_i2(
        X, Y, p[..., None], q[..., None]
    )
    return north, east


def compute_projection_matrices_from_hkpq(
    k: np.ndarray, h: np.ndarray, p: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """
    Compatibility helper returning the direct EqOE sky projection matrix for
    the bounded inclination variables p=sin(i/2)sinΩ, q=sin(i/2)cosΩ.

    The matrix acts on the equinoctial (X,Y) basis, not the 
    periapsis-aligned classical (x_orb,y_orb) basis.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    r00 = 1.0 - 2.0*p*p
    r01 = 2.0*p*q
    r10 = -2.0*p*q
    r11 = -(1.0 - 2.0*q*q)
    rot = np.stack([
        np.stack([r00, r01], axis=-1),
        np.stack([r10, r11], axis=-1),
    ], axis=-2)
    return rot.astype(np.float32, copy=False)


