"""
physics.py - the maths behind the solar system simulation.

Everything here is plain NumPy (plus physical constants from SciPy), so it
can be tested without opening a window.

Improvements over the classic "Planet class" tutorial version:
  * Every body pulls on every other body (true N-body), not just Sun <-> planet.
  * Forces are computed for all bodies at once with NumPy arrays (no nested loops).
  * Velocity Verlet integration instead of Euler, so orbits don't slowly
    spiral outwards and total energy stays (almost) constant.
"""

import numpy as np
from scipy.constants import G, astronomical_unit as AU

DAY = 24 * 3600  # seconds in a day

# ---------------------------------------------------------------------------
# Body data: name, mass (kg), distance from Sun (AU), orbital speed (m/s),
#            draw radius (pixels), colour (RGB)
# Planets start on the +x axis moving in the +y direction (anticlockwise).
# ---------------------------------------------------------------------------
BODIES = [
    ("Sun",     1.98892e30, 0.000,     0.0, 16, (255, 210,  60)),
    ("Mercury", 3.30e23,    0.387, 47_400,  4, (170, 170, 170)),
    ("Venus",   4.8685e24,  0.723, 35_020,  7, (230, 200, 140)),
    ("Earth",   5.9742e24,  1.000, 29_783,  8, (100, 149, 237)),
    ("Mars",    6.39e23,    1.524, 24_077,  6, (188,  39,  50)),
    ("Jupiter", 1.898e27,   5.203, 13_070, 12, (210, 170, 120)),
    ("Saturn",  5.683e26,   9.537,  9_680, 11, (230, 210, 150)),
    ("Uranus",  8.681e25,  19.191,  6_800,  9, (150, 220, 230)),
    ("Neptune", 1.024e26,  30.070,  5_430,  9, ( 70, 110, 230)),
]

# A tiny "softening" length (metres) stops the force blowing up to infinity
# if two bodies ever pass through each other.
SOFTENING = 1e7


def initial_state():
    """Return (positions, velocities, masses) as NumPy arrays.

    positions / velocities have shape (N, 2) in metres and m/s.
    """
    masses = np.array([b[1] for b in BODIES], dtype=float)
    pos = np.array([[b[2] * AU, 0.0] for b in BODIES])
    vel = np.array([[0.0, b[3]] for b in BODIES])

    # Give the Sun a small opposite velocity so the total momentum is zero.
    # Without this the whole system slowly drifts off the screen.
    vel[0] = -(masses[1:, None] * vel[1:]).sum(axis=0) / masses[0]
    return pos, vel, masses


def accelerations(pos, masses):
    """Gravitational acceleration on every body from every other body.

    Uses broadcasting: diff[i, j] is the vector from body i to body j.
        a_i = sum_j  G * m_j * (r_j - r_i) / |r_j - r_i|^3
    """
    diff = pos[None, :, :] - pos[:, None, :]          # (N, N, 2)
    dist_sq = (diff ** 2).sum(axis=-1) + SOFTENING ** 2
    inv_dist3 = dist_sq ** -1.5
    np.fill_diagonal(inv_dist3, 0.0)                  # no self-attraction
    return G * (diff * (masses[None, :] * inv_dist3)[:, :, None]).sum(axis=1)


def step(pos, vel, acc, masses, dt):
    """Advance the system by one time step using velocity Verlet.

    Verlet is "symplectic": it conserves energy far better than the simple
    Euler method (vel += acc*dt; pos += vel*dt) used in most tutorials.
    Returns the new (pos, vel, acc).
    """
    vel_half = vel + 0.5 * acc * dt
    pos = pos + vel_half * dt
    acc = accelerations(pos, masses)
    vel = vel_half + 0.5 * acc * dt
    return pos, vel, acc


def total_energy(pos, vel, masses):
    """Kinetic + potential energy (J). Should stay nearly constant."""
    kinetic = 0.5 * (masses * (vel ** 2).sum(axis=1)).sum()
    diff = pos[None, :, :] - pos[:, None, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1) + SOFTENING ** 2)
    i, j = np.triu_indices(len(masses), k=1)          # each pair once
    potential = -(G * masses[i] * masses[j] / dist[i, j]).sum()
    return kinetic + potential
