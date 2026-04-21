"""
Decentralized CLF-CBF-QP Control for Two Vehicles on a Straight Road
====================================================================

Updated version
---------------
This version is built for the straight-road head-to-head scenario and includes:

- inter-vehicle circle-pair CBF constraints
- road-boundary CBF constraints
- randomized initial conditions over multiple rollouts
- average metrics over 10 rollouts
- plots and animation shown only for the last rollout
- acceleration effort and steering-rate effort added back

Randomized initial conditions
-----------------------------
For each rollout:
- vehicle i:
    y0   ~ Uniform[-1.0, 1.0]
    psi0 ~ Uniform[-20 deg, +20 deg]
- vehicle j:
    y0   ~ Uniform[-1.0, 1.0]
    psi0 ~ Uniform[pi-20 deg, pi+20 deg]

Metrics
-------
Metrics
-------
For each vehicle, averaged over 10 rollouts:
- time to goal:
    time needed to travel 30 m horizontally along the intended direction
    (NaN if not reached within the simulation horizon)
- average absolute lateral deviation:
    average over time of |y|
- acceleration effort
- steering-rate effort
- inter-vehicle collision rate
- boundary collision rate
- minimum inter-vehicle clearance
- minimum boundary clearance
- number of deadlocks:
    counted when both vehicles fail to reach their 30 m goal
    within the simulation horizon

Notes
-----
- The metrics table shows averages over all rollouts.
- The plots and video correspond only to the last rollout.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


# =========================
# Utility functions
# =========================

def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rot2d(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]])


def unit_heading(psi: float) -> np.ndarray:
    return np.array([math.cos(psi), math.sin(psi)])


def unit_lateral(psi: float) -> np.ndarray:
    return np.array([-math.sin(psi), math.cos(psi)])


def deg2rad(deg: float) -> float:
    return deg * np.pi / 180.0


def adaptive_lambda_from_clearance(
    clearance: float,
    clearance_gain: float,
    lambda_lower_bound: float,
    lambda_upper_bound: float,
) -> float:
    """
    Distance-based adaptive lambda:
        lambda_cbf = clip(clearance_gain * clearance,
                          lambda_lower_bound,
                          lambda_upper_bound)
    """
    lam = clearance_gain * clearance
    """
    if clearance_gain == 0.2/4:
        if clearance >= 3.0:
            lam = lambda_upper_bound
        else:
            lam = lambda_lower_bound
    else:
        if clearance >= 12.0:
            lam = lambda_upper_bound
        else:
            lam = lambda_lower_bound
    """
    return float(np.clip(lam, lambda_lower_bound, lambda_upper_bound))


# =========================
# Data classes
# =========================

@dataclass
class VehicleParams:
    length: float = 4.0
    width: float = 2.0
    wheelbase: float = 2.5
    max_speed: float = 10.0
    min_speed: float = -2.0
    max_accel: float = 5.0
    min_accel: float = -5.0
    max_steer: float = 1.0
    min_steer: float = -1.0
    max_steer_rate: float = 1.0
    min_steer_rate: float = -1.0


@dataclass
class CLFParams:
    w_v: float = 10.0
    w_delta: float = 2.0
    desired_speed: float = 6.0
    nominal_speed_gain: float = 1.0
    nominal_heading_gain: float = 1.0
    nominal_steer_rate_gain: float = 1.0
    clf_rate: float = 10.0
    clf_slack_weight: float = 1.0


@dataclass
class CBFParams:
    lambda_cbf: float = 0.2
    cbf_slack_weight: float = 1e8
    eps_dist: float = 1e-6


@dataclass
class QPWeights:
    u_weight: np.ndarray = field(default_factory=lambda: np.diag([100.0, 1.0]))


@dataclass
class RouteSpec:
    name: str
    road_half_width: float
    lane_axis: str  # "horizontal" or "vertical"


@dataclass
class SimParams:
    dt: float = 0.1
    T: float = 10.0

    @property
    def steps(self) -> int:
        return int(round(self.T / self.dt))

    @property
    def Dt(self) -> float:
        return 2.0 * self.dt


@dataclass
class RolloutConfig:
    n_rollouts: int = 10
    y_min: float = -1.0
    y_max: float = 1.0
    psi_i_min_deg: float = -20.0
    psi_i_max_deg: float = 20.0
    psi_j_offset_deg: float = 20.0
    seed: int = 7
    goal_distance_x: float = 30.0


# =========================
# Geometry for circle approximation
# =========================

def circle_approximation(length: float, width: float, n_circles: int = 3) -> Tuple[np.ndarray, float]:
    if n_circles != 3:
        raise ValueError("This implementation uses exactly 3 circles.")
    offsets = np.array([-length / 3.0, 0.0, length / 3.0], dtype=float)
    radius = math.sqrt((length / 6.0) ** 2 + (width / 2.0) ** 2)
    return offsets, radius


# =========================
# Vehicle dynamics
# =========================

def bicycle_dynamics(state: np.ndarray, control: np.ndarray, vp: VehicleParams) -> np.ndarray:
    px, py, psi, v, delta = state
    a, delta_rate = control
    L = vp.wheelbase

    return np.array([
        v * math.cos(psi),
        v * math.sin(psi),
        v * math.tan(delta) / L,
        a,
        delta_rate,
    ], dtype=float)


def step_vehicle(state: np.ndarray, control: np.ndarray, vp: VehicleParams, dt: float) -> np.ndarray:
    xnext = state + dt * bicycle_dynamics(state, control, vp)
    xnext[2] = wrap_to_pi(xnext[2])
    xnext[3] = np.clip(xnext[3], vp.min_speed, vp.max_speed)
    xnext[4] = np.clip(xnext[4], vp.min_steer, vp.max_steer)
    return xnext


# =========================
# Circle center kinematics
# =========================

def circle_center_and_kinematics_affine(
    state: np.ndarray,
    s_offset: float,
    vp: VehicleParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    c_ddot = c_ddot_0 + B @ u
    with u = [a, delta_rate]
    """
    px, py, psi, v, delta = state
    L = vp.wheelbase

    e = unit_heading(psi)
    ep = unit_lateral(psi)

    psi_dot = v * math.tan(delta) / L

    c = np.array([px, py], dtype=float) + s_offset * e
    c_dot = v * e + s_offset * psi_dot * ep

    c_ddot_0 = (-s_offset * psi_dot**2) * e + (v * psi_dot) * ep

    B_a = e + s_offset * (math.tan(delta) / L) * ep
    B_sr = s_offset * (v / (L * math.cos(delta)**2)) * ep

    B = np.column_stack((B_a, B_sr))
    return c, c_dot, c_ddot_0, B


# =========================
# Goal-based geometry
# =========================

def front_circle_offset(vp: VehicleParams) -> float:
    return vp.length / 3.0


def desired_heading_to_goal(
    state: np.ndarray,
    goal_point: np.ndarray,
    vp: VehicleParams,
    s_track: float = None,
) -> float:
    if s_track is None:
        s_track = front_circle_offset(vp)

    c_track, _, _, _ = circle_center_and_kinematics_affine(state, s_track, vp)
    dx = goal_point[0] - c_track[0]
    dy = goal_point[1] - c_track[1]
    return math.atan2(dy, dx)


def desired_steer_from_heading_error(
    state: np.ndarray,
    goal_point: np.ndarray,
    vp: VehicleParams,
    clfp: CLFParams,
    s_track: float = None,
) -> Tuple[float, float]:
    psi = state[2]
    psi_des = desired_heading_to_goal(state, goal_point, vp, s_track=s_track)
    e_psi = wrap_to_pi(psi - psi_des)

    delta_des = np.clip(-clfp.nominal_heading_gain * e_psi, vp.min_steer, vp.max_steer)
    return float(delta_des), float(e_psi)


# =========================
# Nominal controller
# =========================

def nominal_control(
    state: np.ndarray,
    clfp: CLFParams,
    vp: VehicleParams,
    goal_point: np.ndarray,
) -> np.ndarray:
    _, _, _, v, delta = state
    v_err = clfp.desired_speed - v

    a_nom = clfp.nominal_speed_gain * v_err
    a_nom = np.clip(a_nom, vp.min_accel, vp.max_accel)

    delta_des, _ = desired_steer_from_heading_error(
        state=state,
        goal_point=goal_point,
        vp=vp,
        clfp=clfp,
    )

    delta_rate_nom = clfp.nominal_steer_rate_gain * (delta_des - delta)
    delta_rate_nom = np.clip(delta_rate_nom, vp.min_steer_rate, vp.max_steer_rate)

    return np.array([a_nom, delta_rate_nom], dtype=float)


# =========================
# CLF terms
# =========================

def clf_terms(
    state: np.ndarray,
    vp: VehicleParams,
    clfp: CLFParams,
    goal_point: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    _, _, _, v, delta = state
    v_ref = clfp.desired_speed

    delta_des, epsi = desired_steer_from_heading_error(
        state=state,
        goal_point=goal_point,
        vp=vp,
        clfp=clfp,
    )

    V = (
        + clfp.w_v * ((v - v_ref)**2)
        + clfp.w_delta * ((delta - delta_des)**2)
    )

    f = bicycle_dynamics(state, np.array([0.0, 0.0]), vp)

    gradV = np.array([
        0.0,
        0.0,
        0.0,
        2.0 * clfp.w_v * (v - v_ref),
        2.0 * clfp.w_delta * (delta - delta_des),
    ], dtype=float)

    g1 = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=float)
    g2 = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)

    Vdot_const = float(gradV @ f)
    Vdot_u = np.array([gradV @ g1, gradV @ g2], dtype=float)

    return V, Vdot_const, Vdot_u


# =========================
# CBF builders
# =========================

def pairwise_circle_cbf_affine(
    ego_state: np.ndarray,
    other_state: np.ndarray,
    ego_s: float,
    other_s: float,
    ego_vp: VehicleParams,
    other_vp: VehicleParams,
    circle_radius: float,
    cbfp: CBFParams,
) -> Tuple[float, float, np.ndarray]:
    """
    Build:
        h_ddot = hddot_const + hddot_u @ u_ego
    where
        h = ||d|| - (r1 + r2)
    """
    c1, c1_dot, c1_ddot_0, B1 = circle_center_and_kinematics_affine(ego_state, ego_s, ego_vp)
    c2, c2_dot, c2_ddot_0, _ = circle_center_and_kinematics_affine(other_state, other_s, other_vp)

    c2_ddot_known = c2_ddot_0

    d = c1 - c2
    d_dot = c1_dot - c2_dot

    q = max(np.linalg.norm(d), cbfp.eps_dist)
    n = d / q
    R = 2.0 * circle_radius

    h = q - R
    h_dot = float(n @ d_dot)

    tangential_term = (np.dot(d_dot, d_dot) - (np.dot(n, d_dot))**2) / q

    hddot_const = float(n @ (c1_ddot_0 - c2_ddot_known) + tangential_term)
    hddot_u = np.array([n @ B1[:, 0], n @ B1[:, 1]], dtype=float)

    return h, h_dot, np.array([hddot_const, *hddot_u], dtype=float)


def road_boundary_cbf_affine(
    ego_state: np.ndarray,
    ego_s: float,
    ego_vp: VehicleParams,
    route: RouteSpec,
    boundary_side: str,
) -> Tuple[float, float, np.ndarray]:
    """
    Build one road-boundary CBF for one circle.
    Returns
    -------
    h, h_dot, [hddot_const, coeff_a, coeff_delta_rate]
    """
    c, c_dot, c_ddot_0, B = circle_center_and_kinematics_affine(ego_state, ego_s, ego_vp)

    half_w = route.road_half_width

    if route.lane_axis == "horizontal":
        if boundary_side == "left":
            h = half_w - c[1]
            h_dot = -c_dot[1]
            hddot_const = -c_ddot_0[1]
            hddot_u = -B[1, :]
        elif boundary_side == "right":
            h = c[1] + half_w
            h_dot = c_dot[1]
            hddot_const = c_ddot_0[1]
            hddot_u = B[1, :]
        else:
            raise ValueError("boundary_side must be 'left' or 'right'.")
    else:
        raise ValueError("This straight-road script uses only horizontal roads.")

    return float(h), float(h_dot), np.array([hddot_const, hddot_u[0], hddot_u[1]], dtype=float)


# =========================
# Decentralized QP for one vehicle
# =========================

def solve_vehicle_qp(
    ego_state: np.ndarray,
    other_state: np.ndarray,
    ego_route: RouteSpec,
    ego_vp: VehicleParams,
    other_vp: VehicleParams,
    clfp: CLFParams,
    cbfp: CBFParams,
    qpw: QPWeights,
    sim: SimParams,
    circle_offsets: np.ndarray,
    circle_radius: float,
    goal_point: np.ndarray,
) -> Tuple[np.ndarray, Dict]:

    u_nom = nominal_control(ego_state, clfp, ego_vp, goal_point)

    n_cbf = 9 + 3 + 3   # inter-vehicle + left boundary + right boundary

    u = cp.Variable(2)
    s_cbf = cp.Variable(n_cbf, nonneg=True)
    s_clf = cp.Variable(1, nonneg=True)

    constraints = []

    # input bounds
    constraints += [
        u[0] >= ego_vp.min_accel,
        u[0] <= ego_vp.max_accel,
        u[1] >= ego_vp.min_steer_rate,
        u[1] <= ego_vp.max_steer_rate,
    ]

    dt = sim.dt

    # one-step consistency bounds
    constraints += [
        ego_state[4] + dt * u[1] >= ego_vp.min_steer,
        ego_state[4] + dt * u[1] <= ego_vp.max_steer,
        ego_state[3] + dt * u[0] >= ego_vp.min_speed,
        ego_state[3] + dt * u[0] <= ego_vp.max_speed,
    ]

    # CLF
    V, Vdot_const, Vdot_u = clf_terms(ego_state, ego_vp, clfp, goal_point)
    constraints += [
        Vdot_const + Vdot_u @ u <= -clfp.clf_rate * V + s_clf[0]
    ]

    Dt = sim.Dt
    lambda_cbf = cbfp.lambda_cbf
    cbf_index = 0

    # inter-vehicle CBFs
    for s_i in circle_offsets:
        for s_j in circle_offsets:
            h, h_dot, hddot_aff = pairwise_circle_cbf_affine(
                ego_state=ego_state,
                other_state=other_state,
                ego_s=s_i,
                other_s=s_j,
                ego_vp=ego_vp,
                other_vp=other_vp,
                circle_radius=circle_radius,
                cbfp=cbfp,
            )
            hddot_const = hddot_aff[0]
            hddot_u = hddot_aff[1:]

            constraints += [
                0.5 * Dt**2 * (0.5 * hddot_const + hddot_u @ u)
                + 0.5 * Dt * h_dot
                + 0.5 * lambda_cbf * h
                + s_cbf[cbf_index] >= 0.0
            ]
            cbf_index += 1

    # left boundary CBFs
    for s_i in circle_offsets:
        h, h_dot, hddot_aff = road_boundary_cbf_affine(
            ego_state=ego_state,
            ego_s=s_i,
            ego_vp=ego_vp,
            route=ego_route,
            boundary_side="left",
        )
        hddot_const = hddot_aff[0]
        hddot_u = hddot_aff[1:]
        h_eff = h - circle_radius

        constraints += [
            0.5 * Dt**2 * (hddot_const + hddot_u @ u)
            + Dt * h_dot
            + lambda_cbf * h_eff
            + s_cbf[cbf_index] >= 0.0
        ]
        cbf_index += 1

    # right boundary CBFs
    for s_i in circle_offsets:
        h, h_dot, hddot_aff = road_boundary_cbf_affine(
            ego_state=ego_state,
            ego_s=s_i,
            ego_vp=ego_vp,
            route=ego_route,
            boundary_side="right",
        )
        hddot_const = hddot_aff[0]
        hddot_u = hddot_aff[1:]
        h_eff = h - circle_radius

        constraints += [
            0.5 * Dt**2 * (hddot_const + hddot_u @ u)
            + Dt * h_dot
            + lambda_cbf * h_eff
            + s_cbf[cbf_index] >= 0.0
        ]
        cbf_index += 1

    assert cbf_index == n_cbf

    objective = cp.Minimize(
        0.5 * cp.quad_form(u - u_nom, qpw.u_weight)
        + clfp.clf_slack_weight * cp.sum_squares(s_clf)
        + cbfp.cbf_slack_weight * cp.sum_squares(s_cbf)
    )

    prob = cp.Problem(objective, constraints)

    qp_info = {
        "status": None,
        "u_nom": u_nom.copy(),
        "clf_V": V,
        "cbf_slack_inter_vehicle": np.full(9, np.nan, dtype=float),
    }

    try:
        prob.solve(
            solver=cp.OSQP,
            warm_start=True,
            verbose=False,
            eps_abs=1e-5,
            eps_rel=1e-5,
            max_iter=20000,
        )
    except Exception:
        prob.solve(warm_start=True, verbose=False)

    qp_info["status"] = prob.status
    if s_cbf.value is not None:
        qp_info["cbf_slack_inter_vehicle"] = np.array(s_cbf.value[:9], dtype=float)

    if u.value is None or prob.status not in ["optimal", "optimal_inaccurate"]:
        u_sol = np.array([
            np.clip(u_nom[0], ego_vp.min_accel, ego_vp.max_accel),
            np.clip(u_nom[1], ego_vp.min_steer_rate, ego_vp.max_steer_rate),
        ], dtype=float)
        qp_info["fallback"] = True
    else:
        u_sol = np.array(u.value).reshape(2)
        qp_info["fallback"] = False

    return u_sol, qp_info


# =========================
# Safety / metric helpers
# =========================

def get_vehicle_circle_centers(state: np.ndarray, vp: VehicleParams, circle_offsets: np.ndarray) -> List[np.ndarray]:
    centers = []
    for s in circle_offsets:
        c, _, _, _ = circle_center_and_kinematics_affine(state, s, vp)
        centers.append(c)
    return centers


def min_intervehicle_clearance(
    x_i: np.ndarray,
    x_j: np.ndarray,
    vp_i: VehicleParams,
    vp_j: VehicleParams,
    circle_offsets: np.ndarray,
    circle_radius: float,
) -> float:
    centers_i = get_vehicle_circle_centers(x_i, vp_i, circle_offsets)
    centers_j = get_vehicle_circle_centers(x_j, vp_j, circle_offsets)

    min_clear = np.inf
    for ci in centers_i:
        for cj in centers_j:
            d = np.linalg.norm(ci - cj) - 2.0 * circle_radius
            min_clear = min(min_clear, d)
    return float(min_clear)


def min_boundary_clearance(
    x: np.ndarray,
    route: RouteSpec,
    vp: VehicleParams,
    circle_offsets: np.ndarray,
    circle_radius: float,
) -> float:
    centers = get_vehicle_circle_centers(x, vp, circle_offsets)
    half_w = route.road_half_width

    min_clear = np.inf
    for c in centers:
        clearance_top = half_w - c[1] - circle_radius
        clearance_bottom = c[1] + half_w - circle_radius
        min_clear = min(min_clear, clearance_top, clearance_bottom)
    return float(min_clear)


def compute_time_to_goal_x(hist_x: np.ndarray, t: np.ndarray, direction: str, distance_goal: float = 30.0) -> float:
    x0 = hist_x[0]
    if direction == "positive":
        progress = hist_x - x0
        idx = np.where(progress >= distance_goal)[0]
    elif direction == "negative":
        progress = x0 - hist_x
        idx = np.where(progress >= distance_goal)[0]
    else:
        raise ValueError("direction must be 'positive' or 'negative'.")

    if len(idx) == 0:
        return np.nan
    return float(t[idx[0]])


# =========================
# Plot helpers
# =========================

def vehicle_corners(state: np.ndarray, vp: VehicleParams) -> np.ndarray:
    px, py, psi, _, _ = state
    L = vp.length
    W = vp.width

    corners_local = np.array([
        [ L / 2.0,  W / 2.0],
        [ L / 2.0, -W / 2.0],
        [-L / 2.0, -W / 2.0],
        [-L / 2.0,  W / 2.0],
    ], dtype=float)

    R = rot2d(psi)
    corners_world = (R @ corners_local.T).T + np.array([px, py], dtype=float)
    return corners_world


# =========================
# One rollout simulation
# =========================

def run_single_rollout(
    init_state_i: np.ndarray,
    init_state_j: np.ndarray,
    cbfp_schedule: str = "constant_normal",
):
    sim = SimParams(dt=0.1, T=10.0)

    vp_i = VehicleParams()
    vp_j = VehicleParams()

    clfp_i = CLFParams()
    clfp_j = CLFParams()

    # --------------------------------------
    # Baseline method parameters
    # --------------------------------------
    lambda_constant_normal = 0.2
    lambda_constant_conservative = 0.15

    # --------------------------------------
    # Adaptive method parameters
    # Easy to modify
    # --------------------------------------
    clearance_gain_i = 0.2/4
    lambda_lower_bound_i = 0.1
    lambda_upper_bound_i = 0.4

    clearance_gain_j = 0.05/4
    lambda_lower_bound_j = 0.1
    lambda_upper_bound_j = 0.4

    qpw_i = QPWeights(u_weight=np.diag([10.0, 1.0]))
    qpw_j = QPWeights(u_weight=np.diag([10.0, 1.0]))

    circle_offsets, circle_radius = circle_approximation(vp_i.length, vp_i.width, n_circles=3)

    route_i = RouteSpec(name="vehicle_i", road_half_width=3.1, lane_axis="horizontal")
    route_j = RouteSpec(name="vehicle_j", road_half_width=3.1, lane_axis="horizontal")

    x_i = init_state_i.copy()
    x_j = init_state_j.copy()

    current_clearance = min_intervehicle_clearance(
        x_i, x_j, vp_i, vp_j, circle_offsets, circle_radius
    )

    # initialize cbf parameters
    if cbfp_schedule == "constant_normal":
        cbfp_i = CBFParams(lambda_cbf=lambda_constant_normal, cbf_slack_weight=1e8)
        cbfp_j = CBFParams(lambda_cbf=lambda_constant_normal, cbf_slack_weight=1e8)

    elif cbfp_schedule == "constant_conservative":
        cbfp_i = CBFParams(lambda_cbf=lambda_constant_normal, cbf_slack_weight=1e8)
        cbfp_j = CBFParams(lambda_cbf=lambda_constant_conservative, cbf_slack_weight=1e8)

    elif cbfp_schedule == "adaptive":
        lambda_i0 = adaptive_lambda_from_clearance(
            clearance=current_clearance,
            clearance_gain=clearance_gain_i,
            lambda_lower_bound=lambda_lower_bound_i,
            lambda_upper_bound=lambda_upper_bound_i,
        )
        lambda_j0 = adaptive_lambda_from_clearance(
            clearance=current_clearance,
            clearance_gain=clearance_gain_j,
            lambda_lower_bound=lambda_lower_bound_j,
            lambda_upper_bound=lambda_upper_bound_j,
        )
        cbfp_i = CBFParams(lambda_cbf=lambda_i0, cbf_slack_weight=1e8)
        cbfp_j = CBFParams(lambda_cbf=lambda_j0, cbf_slack_weight=1e8)

    else:
        raise ValueError("Unknown cbfp_schedule.")

    hist = {
        "x_i": [x_i.copy()],
        "x_j": [x_j.copy()],
        "u_i": [],
        "u_j": [],
        "qp_i": [],
        "qp_j": [],
        "epsi_i": [],
        "epsi_j": [],
        "goal_i": [],
        "goal_j": [],
        "lambda_i": [cbfp_i.lambda_cbf],
        "lambda_j": [cbfp_j.lambda_cbf],
        "clearance_iv": [current_clearance],
        "t": [0.0],
    }

    for k in range(sim.steps):
        current_clearance = min_intervehicle_clearance(
            x_i, x_j, vp_i, vp_j, circle_offsets, circle_radius
        )

        bi_clear = min_boundary_clearance(x_i, route_i, vp_i, circle_offsets, circle_radius)
        bj_clear = min_boundary_clearance(x_j, route_j, vp_j, circle_offsets, circle_radius)

        if cbfp_schedule == "adaptive":
            cbfp_i.lambda_cbf = adaptive_lambda_from_clearance(
                clearance=current_clearance,
                clearance_gain=clearance_gain_i,
                lambda_lower_bound=lambda_lower_bound_i,
                lambda_upper_bound=lambda_upper_bound_i,
            )
            cbfp_j.lambda_cbf = adaptive_lambda_from_clearance(
                clearance=current_clearance,
                clearance_gain=clearance_gain_j,
                lambda_lower_bound=lambda_lower_bound_j,
                lambda_upper_bound=lambda_upper_bound_j,
            )

        lookahead_i = 8.0
        lookahead_j = 8.0

        goal_i = np.array([x_i[0] + lookahead_i, 0.0], dtype=float)
        goal_j = np.array([x_j[0] - lookahead_j, 0.0], dtype=float)

        u_i, qp_i = solve_vehicle_qp(
            ego_state=x_i.copy(),
            other_state=x_j.copy(),
            ego_route=route_i,
            ego_vp=vp_i,
            other_vp=vp_j,
            clfp=clfp_i,
            cbfp=cbfp_i,
            qpw=qpw_i,
            sim=sim,
            circle_offsets=circle_offsets,
            circle_radius=circle_radius,
            goal_point=goal_i,
        )

        u_j, qp_j = solve_vehicle_qp(
            ego_state=x_j.copy(),
            other_state=x_i.copy(),
            ego_route=route_j,
            ego_vp=vp_j,
            other_vp=vp_i,
            clfp=clfp_j,
            cbfp=cbfp_j,
            qpw=qpw_j,
            sim=sim,
            circle_offsets=circle_offsets,
            circle_radius=circle_radius,
            goal_point=goal_j,
        )

        _, epsi_i = desired_steer_from_heading_error(
            state=x_i,
            goal_point=goal_i,
            vp=vp_i,
            clfp=clfp_i,
        )

        _, epsi_j = desired_steer_from_heading_error(
            state=x_j,
            goal_point=goal_j,
            vp=vp_j,
            clfp=clfp_j,
        )

        msg = (
            f"t = {k * sim.dt:.1f} s | "
            f"clearance = {current_clearance:.3f} m | "
            f"bi_clear = {bi_clear:.3f} m | "
            f"bj_clear = {bj_clear:.3f} m | "

            f"lambda_i = {cbfp_i.lambda_cbf:.3f} | "
            f"lambda_j = {cbfp_j.lambda_cbf:.3f} |"
            #f"acc_i = {u_i[0]:.3f} | "
            #f"acc_j = {u_j[0]:.3f}"
            #f"v_i = {x_i[3]:.3f} | "
            #f"v_j = {x_j[3]:.3f}"
            #f"dis_i = {x_i[0]:.3f} | "
            #f"dis_j = {x_j[0]:.3f}"
        )
        print(msg.ljust(95), flush=True)

        if qp_i.get("status") in ["infeasible", "infeasible_inaccurate"]:
            print("  WARNING: vehicle i QP infeasible")
        if qp_j.get("status") in ["infeasible", "infeasible_inaccurate"]:
            print("  WARNING: vehicle j QP infeasible")

        x_i = step_vehicle(x_i, u_i, vp_i, sim.dt)
        x_j = step_vehicle(x_j, u_j, vp_j, sim.dt)

        next_clearance = min_intervehicle_clearance(
            x_i, x_j, vp_i, vp_j, circle_offsets, circle_radius
        )

        hist["x_i"].append(x_i.copy())
        hist["x_j"].append(x_j.copy())
        hist["u_i"].append(u_i.copy())
        hist["u_j"].append(u_j.copy())
        hist["qp_i"].append(qp_i)
        hist["qp_j"].append(qp_j)
        hist["epsi_i"].append(epsi_i)
        hist["epsi_j"].append(epsi_j)
        hist["goal_i"].append(goal_i.copy())
        hist["goal_j"].append(goal_j.copy())
        hist["lambda_i"].append(cbfp_i.lambda_cbf)
        hist["lambda_j"].append(cbfp_j.lambda_cbf)
        hist["clearance_iv"].append(next_clearance)
        hist["t"].append((k + 1) * sim.dt)

    print()

    for key in ["x_i", "x_j", "u_i", "u_j", "epsi_i", "epsi_j", "goal_i", "goal_j", "t", "lambda_i", "lambda_j", "clearance_iv"]:
        hist[key] = np.array(hist[key])

    metrics = compute_rollout_metrics(
        hist=hist,
        vp_i=vp_i,
        vp_j=vp_j,
        route_i=route_i,
        route_j=route_j,
        circle_offsets=circle_offsets,
        circle_radius=circle_radius,
        goal_distance_x=30.0,
    )

    return hist, metrics, sim, vp_i, vp_j, route_i, route_j, circle_offsets, circle_radius


# =========================
# Rollout metrics
# =========================

def compute_rollout_metrics(
    hist,
    vp_i,
    vp_j,
    route_i,
    route_j,
    circle_offsets,
    circle_radius,
    goal_distance_x: float = 30.0,
):
    x_i = hist["x_i"]
    x_j = hist["x_j"]
    u_i = hist["u_i"]
    u_j = hist["u_j"]
    t = hist["t"]

    dt = t[1] - t[0]

    acc_eff_i = float(np.sum(np.abs(u_i[:, 0])) * dt)
    acc_eff_j = float(np.sum(np.abs(u_j[:, 0])) * dt)

    steer_eff_i = float(np.sum(np.abs(u_i[:, 1])) * dt)
    steer_eff_j = float(np.sum(np.abs(u_j[:, 1])) * dt)

    min_iv_clear = np.inf
    min_bound_clear_i = np.inf
    min_bound_clear_j = np.inf

    intervehicle_collision_i = False
    intervehicle_collision_j = False
    boundary_collision_i = False
    boundary_collision_j = False

    for k in range(len(t)):
        xi = x_i[k]
        xj = x_j[k]

        iv_clear = min_intervehicle_clearance(xi, xj, vp_i, vp_j, circle_offsets, circle_radius)
        bi_clear = min_boundary_clearance(xi, route_i, vp_i, circle_offsets, circle_radius)
        bj_clear = min_boundary_clearance(xj, route_j, vp_j, circle_offsets, circle_radius)

        min_iv_clear = min(min_iv_clear, iv_clear)
        min_bound_clear_i = min(min_bound_clear_i, bi_clear)
        min_bound_clear_j = min(min_bound_clear_j, bj_clear)

        if iv_clear < 0.0:
            intervehicle_collision_i = True
            intervehicle_collision_j = True
        if bi_clear < 0.0:
            boundary_collision_i = True
        if bj_clear < 0.0:
            boundary_collision_j = True

    time_to_goal_i = compute_time_to_goal_x(
        x_i[:, 0], t, direction="positive", distance_goal=goal_distance_x
    )
    time_to_goal_j = compute_time_to_goal_x(
        x_j[:, 0], t, direction="negative", distance_goal=goal_distance_x
    )

    avg_abs_y_i = float(np.mean(np.abs(x_i[:, 1])))
    avg_abs_y_j = float(np.mean(np.abs(x_j[:, 1])))

    # Deadlock:
    # both vehicles fail to reach their goal within the simulation horizon
    deadlock = int(np.isnan(time_to_goal_i) and np.isnan(time_to_goal_j))

    return {
        "i": {
            "time_to_goal": time_to_goal_i,
            "avg_abs_y": avg_abs_y_i,
            "acc_effort": acc_eff_i,
            "steer_rate_effort": steer_eff_i,
            "intervehicle_collision": int(intervehicle_collision_i),
            "boundary_collision": int(boundary_collision_i),
            "min_intervehicle_clearance": float(min_iv_clear),
            "min_boundary_clearance": float(min_bound_clear_i),
        },
        "j": {
            "time_to_goal": time_to_goal_j,
            "avg_abs_y": avg_abs_y_j,
            "acc_effort": acc_eff_j,
            "steer_rate_effort": steer_eff_j,
            "intervehicle_collision": int(intervehicle_collision_j),
            "boundary_collision": int(boundary_collision_j),
            "min_intervehicle_clearance": float(min_iv_clear),
            "min_boundary_clearance": float(min_bound_clear_j),
        },
        "deadlock": deadlock,
    }


# =========================
# Monte Carlo simulation
# =========================

def build_random_initial_states(cfg: RolloutConfig):
    rng = np.random.default_rng(cfg.seed)

    states = []
    psi_i_min = deg2rad(cfg.psi_i_min_deg)
    psi_i_max = deg2rad(cfg.psi_i_max_deg)
    psi_j_span = deg2rad(cfg.psi_j_offset_deg)

    for _ in range(cfg.n_rollouts):
        y_i = rng.uniform(cfg.y_min, cfg.y_max)
        y_j = rng.uniform(cfg.y_min, cfg.y_max)

        psi_i = rng.uniform(psi_i_min, psi_i_max)
        psi_j = rng.uniform(np.pi - psi_j_span, np.pi + psi_j_span)

        x_i0 = np.array([-15.0, y_i, psi_i, 6.0, 0.0], dtype=float)
        x_j0 = np.array([ 15.0, y_j, psi_j, 6.0, 0.0], dtype=float)

        states.append((x_i0, x_j0))

    return states


def run_monte_carlo(cbfp_schedule: str = "constant_normal", n_rollouts: int = 10):
    rollout_cfg = RolloutConfig(n_rollouts=n_rollouts)
    initial_states = build_random_initial_states(rollout_cfg)

    all_metrics = []
    last_hist = None
    last_bundle = None

    for r_idx, (x_i0, x_j0) in enumerate(initial_states, start=1):
        print(f"\n=== Rollout {r_idx}/{n_rollouts} ===")
        print(f"Initial vehicle i: y={x_i0[1]:.3f}, psi={x_i0[2]:.3f} rad")
        print(f"Initial vehicle j: y={x_j0[1]:.3f}, psi={x_j0[2]:.3f} rad")

        hist, metrics, sim, vp_i, vp_j, route_i, route_j, circle_offsets, circle_radius = run_single_rollout(
            init_state_i=x_i0,
            init_state_j=x_j0,
            cbfp_schedule=cbfp_schedule,
        )

        all_metrics.append(metrics)
        last_hist = hist
        last_bundle = (sim, vp_i, vp_j, route_i, route_j, circle_offsets, circle_radius)

    avg_metrics = average_rollout_metrics(all_metrics)

    return avg_metrics, last_hist, last_bundle


def average_rollout_metrics(all_metrics: List[Dict]) -> Dict:
    avg = {"i": {}, "j": {}}

    for veh in ["i", "j"]:
        avg[veh]["time_to_goal"] = float(np.nanmean([m[veh]["time_to_goal"] for m in all_metrics]))
        avg[veh]["avg_abs_y"] = float(np.mean([m[veh]["avg_abs_y"] for m in all_metrics]))
        avg[veh]["acc_effort"] = float(np.mean([m[veh]["acc_effort"] for m in all_metrics]))
        avg[veh]["steer_rate_effort"] = float(np.mean([m[veh]["steer_rate_effort"] for m in all_metrics]))
        avg[veh]["intervehicle_collision"] = float(np.mean([m[veh]["intervehicle_collision"] for m in all_metrics]))
        avg[veh]["boundary_collision"] = float(np.mean([m[veh]["boundary_collision"] for m in all_metrics]))
        avg[veh]["min_intervehicle_clearance"] = float(np.mean([m[veh]["min_intervehicle_clearance"] for m in all_metrics]))
        avg[veh]["min_boundary_clearance"] = float(np.mean([m[veh]["min_boundary_clearance"] for m in all_metrics]))
        # Global deadlock statistics
        avg["deadlock_count"] = int(np.sum([m["deadlock"] for m in all_metrics]))
        avg["deadlock_rate"] = float(np.mean([m["deadlock"] for m in all_metrics]))

    return avg


# =========================
# Metrics table (average over rollouts)
# =========================

def show_average_metrics_table(avg_metrics, n_rollouts: int):
    def fmt_nan(x):
        return "NaN" if np.isnan(x) else f"{x:.3f}"

    row_labels = [
        "Time to Goal [s]",
        "Avg |y|",
        "Acceleration Effort",
        "Steering-Rate Effort",
        "Inter-Vehicle Collision Rate",
        "Boundary Collision Rate",
        "Min Inter-Vehicle Clearance [m]",
        "Min Boundary Clearance [m]",
        "Deadlocks [count]",
        "Deadlock Rate",
    ]

    deadlock_count_str = f"{avg_metrics['deadlock_count']}/{n_rollouts}"
    deadlock_rate_str = f"{avg_metrics['deadlock_rate']:.3f}"

    table_data = [
        [fmt_nan(avg_metrics["i"]["time_to_goal"]), fmt_nan(avg_metrics["j"]["time_to_goal"])],
        [f"{avg_metrics['i']['avg_abs_y']:.3f}", f"{avg_metrics['j']['avg_abs_y']:.3f}"],
        [f"{avg_metrics['i']['acc_effort']:.3f}", f"{avg_metrics['j']['acc_effort']:.3f}"],
        [f"{avg_metrics['i']['steer_rate_effort']:.3f}", f"{avg_metrics['j']['steer_rate_effort']:.3f}"],
        [f"{avg_metrics['i']['intervehicle_collision']:.3f}", f"{avg_metrics['j']['intervehicle_collision']:.3f}"],
        [f"{avg_metrics['i']['boundary_collision']:.3f}", f"{avg_metrics['j']['boundary_collision']:.3f}"],
        [f"{avg_metrics['i']['min_intervehicle_clearance']:.3f}", f"{avg_metrics['j']['min_intervehicle_clearance']:.3f}"],
        [f"{avg_metrics['i']['min_boundary_clearance']:.3f}", f"{avg_metrics['j']['min_boundary_clearance']:.3f}"],
        [deadlock_count_str, deadlock_count_str],
        [deadlock_rate_str, deadlock_rate_str],
    ]

    fig, ax = plt.subplots(figsize=(10, 6.0))
    ax.axis("off")
    ax.set_title(f"Average Metrics Summary over {n_rollouts} Rollouts", fontsize=13, pad=12)

    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=["Vehicle i", "Vehicle j"],
        cellLoc="center",
        rowLoc="center",
        loc="center",
        bbox=[0.05, 0.02, 0.90, 0.92],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(0.085)
        else:
            cell.set_height(0.088)

        if col == -1:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    plt.show()

    return avg_metrics


# =========================
# Static plots (last rollout only)
# =========================

def plot_results(hist, route_i, filename: str = "two_vehicle_road_results.png"):
    t_state = hist["t"]
    t_u = t_state[:-1]

    x_i = hist["x_i"]
    x_j = hist["x_j"]
    u_i = hist["u_i"]
    u_j = hist["u_j"]

    fig, axs = plt.subplots(5, 2, figsize=(12, 16), constrained_layout=True)

    half_w = route_i.road_half_width
    axs[0, 0].axhspan(-half_w, half_w, color="lightgray", alpha=0.4)
    axs[0, 0].axhline(half_w, color="k", linewidth=1.2)
    axs[0, 0].axhline(-half_w, color="k", linewidth=1.2)
    axs[0, 0].plot(x_i[:, 0], x_i[:, 1], label="vehicle i")
    axs[0, 0].plot(x_j[:, 0], x_j[:, 1], label="vehicle j")
    axs[0, 0].scatter(x_i[0, 0], x_i[0, 1], marker="o", s=60, label="i start")
    axs[0, 0].scatter(x_j[0, 0], x_j[0, 1], marker="o", s=60, label="j start")
    axs[0, 0].scatter(x_i[-1, 0], x_i[-1, 1], marker="x", s=70, label="i end")
    axs[0, 0].scatter(x_j[-1, 0], x_j[-1, 1], marker="x", s=70, label="j end")
    axs[0, 0].axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.6)
    axs[0, 0].set_title("Trajectories (Last Rollout)")
    axs[0, 0].set_xlabel("x [m]")
    axs[0, 0].set_ylabel("y [m]")
    axs[0, 0].axis("equal")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(t_state, x_i[:, 3], label="vehicle i")
    axs[0, 1].plot(t_state, x_j[:, 3], label="vehicle j")
    axs[0, 1].set_title("Speed")
    axs[0, 1].set_xlabel("t [s]")
    axs[0, 1].set_ylabel("v [m/s]")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 0].plot(t_state, x_i[:, 2], label="vehicle i")
    axs[1, 0].plot(t_state, x_j[:, 2], label="vehicle j")
    axs[1, 0].set_title("Heading")
    axs[1, 0].set_xlabel("t [s]")
    axs[1, 0].set_ylabel("psi [rad]")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].plot(t_state, x_i[:, 4], label="vehicle i")
    axs[1, 1].plot(t_state, x_j[:, 4], label="vehicle j")
    axs[1, 1].set_title("Steering Angle")
    axs[1, 1].set_xlabel("t [s]")
    axs[1, 1].set_ylabel("delta [rad]")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    axs[2, 0].plot(t_u, u_i[:, 0], label="vehicle i")
    axs[2, 0].plot(t_u, u_j[:, 0], label="vehicle j")
    axs[2, 0].set_title("Acceleration Input")
    axs[2, 0].set_xlabel("t [s]")
    axs[2, 0].set_ylabel("a [m/s^2]")
    axs[2, 0].grid(True)
    axs[2, 0].legend()

    axs[2, 1].plot(t_u, u_i[:, 1], label="vehicle i")
    axs[2, 1].plot(t_u, u_j[:, 1], label="vehicle j")
    axs[2, 1].set_title("Steering Rate Input")
    axs[2, 1].set_xlabel("t [s]")
    axs[2, 1].set_ylabel("delta_rate [rad/s]")
    axs[2, 1].grid(True)
    axs[2, 1].legend()

    axs[3, 0].plot(t_state, hist["lambda_i"], label="lambda_i")
    axs[3, 0].plot(t_state, hist["lambda_j"], label="lambda_j")
    axs[3, 0].set_title("CBF Lambda")
    axs[3, 0].set_xlabel("t [s]")
    axs[3, 0].set_ylabel("lambda_cbf")
    axs[3, 0].grid(True)
    axs[3, 0].legend()

    axs[3, 1].plot(t_state, hist["clearance_iv"], label="inter-vehicle clearance")
    axs[3, 1].set_title("Inter-Vehicle Clearance")
    axs[3, 1].set_xlabel("t [s]")
    axs[3, 1].set_ylabel("clearance [m]")
    axs[3, 1].grid(True)
    axs[3, 1].legend()

    axs[4, 0].plot(t_state, x_i[:, 1], label="vehicle i y")
    axs[4, 0].plot(t_state, x_j[:, 1], label="vehicle j y")
    axs[4, 0].axhline(half_w, color="k", linewidth=1.0)
    axs[4, 0].axhline(-half_w, color="k", linewidth=1.0)
    axs[4, 0].axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.6)
    axs[4, 0].set_title("Lateral Position")
    axs[4, 0].set_xlabel("t [s]")
    axs[4, 0].set_ylabel("y [m]")
    axs[4, 0].grid(True)
    axs[4, 0].legend()

    axs[4, 1].axis("off")

    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {filename}")
    plt.show()


# =========================
# Animation and video export (last rollout only)
# =========================

def animate_simulation(
    hist,
    vp_i,
    vp_j,
    route_i,
    circle_offsets,
    circle_radius,
    filename: str = "two_vehicle_road.mp4",
):
    import os
    from matplotlib import animation
    from matplotlib.patches import Polygon, Circle as PatchCircle

    x_i = hist["x_i"]
    x_j = hist["x_j"]
    goal_i_hist = hist["goal_i"]
    goal_j_hist = hist["goal_j"]
    t = hist["t"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(-30, 30)
    ax.set_ylim(-8, 8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Decentralized CLF-CBF-QP on a Straight Road (Last Rollout)")
    ax.grid(True)

    half_w = route_i.road_half_width
    ax.fill_between([-30, 30], -half_w, half_w, color="lightgray", alpha=0.5)
    ax.plot([-30, 30], [ half_w,  half_w], "k-", linewidth=1.5)
    ax.plot([-30, 30], [-half_w, -half_w], "k-", linewidth=1.5)
    ax.axhline(0.0, linestyle="--", color="gray", alpha=0.6, linewidth=1.0)

    traj_i_line, = ax.plot([], [], linewidth=1.5, color="tab:blue", label="vehicle i")
    traj_j_line, = ax.plot([], [], linewidth=1.5, color="tab:orange", label="vehicle j")

    poly_i = Polygon(vehicle_corners(x_i[0], vp_i), closed=True, fill=False, edgecolor="tab:blue", linewidth=2.0)
    poly_j = Polygon(vehicle_corners(x_j[0], vp_j), closed=True, fill=False, edgecolor="tab:orange", linewidth=2.0)
    ax.add_patch(poly_i)
    ax.add_patch(poly_j)

    circle_patches_i = []
    circle_patches_j = []
    for _ in circle_offsets:
        ci = PatchCircle((0.0, 0.0), circle_radius, fill=False, edgecolor="tab:blue", linestyle="--", alpha=0.8)
        cj = PatchCircle((0.0, 0.0), circle_radius, fill=False, edgecolor="tab:orange", linestyle="--", alpha=0.8)
        ax.add_patch(ci)
        ax.add_patch(cj)
        circle_patches_i.append(ci)
        circle_patches_j.append(cj)

    heading_i_line, = ax.plot([], [], color="tab:blue", linewidth=2.0)
    heading_j_line, = ax.plot([], [], color="tab:orange", linewidth=2.0)

    goal_i_scatter = ax.scatter([], [], marker="x", s=80, color="tab:blue", label="goal i")
    goal_j_scatter = ax.scatter([], [], marker="x", s=80, color="tab:orange", label="goal j")

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    ax.legend(loc="upper right")

    def set_heading_line(line_obj, state, length=1.8):
        px, py, psi, _, _ = state
        ex = length * np.cos(psi)
        ey = length * np.sin(psi)
        line_obj.set_data([px, px + ex], [py, py + ey])

    def init():
        traj_i_line.set_data([], [])
        traj_j_line.set_data([], [])

        poly_i.set_xy(vehicle_corners(x_i[0], vp_i))
        poly_j.set_xy(vehicle_corners(x_j[0], vp_j))

        for k, s in enumerate(circle_offsets):
            ci, _, _, _ = circle_center_and_kinematics_affine(x_i[0], s, vp_i)
            cj, _, _, _ = circle_center_and_kinematics_affine(x_j[0], s, vp_j)
            circle_patches_i[k].center = (ci[0], ci[1])
            circle_patches_j[k].center = (cj[0], cj[1])

        set_heading_line(heading_i_line, x_i[0])
        set_heading_line(heading_j_line, x_j[0])

        goal_i_scatter.set_offsets(goal_i_hist[0])
        goal_j_scatter.set_offsets(goal_j_hist[0])

        time_text.set_text(f"t = {t[0]:.1f} s")

        return [
            traj_i_line, traj_j_line, poly_i, poly_j,
            heading_i_line, heading_j_line, goal_i_scatter, goal_j_scatter,
            time_text, *circle_patches_i, *circle_patches_j
        ]

    def update(frame):
        if frame == 0:
            gi = goal_i_hist[0]
            gj = goal_j_hist[0]
        else:
            gi = goal_i_hist[frame - 1]
            gj = goal_j_hist[frame - 1]

        traj_i_line.set_data(x_i[:frame + 1, 0], x_i[:frame + 1, 1])
        traj_j_line.set_data(x_j[:frame + 1, 0], x_j[:frame + 1, 1])

        poly_i.set_xy(vehicle_corners(x_i[frame], vp_i))
        poly_j.set_xy(vehicle_corners(x_j[frame], vp_j))

        for k, s in enumerate(circle_offsets):
            ci, _, _, _ = circle_center_and_kinematics_affine(x_i[frame], s, vp_i)
            cj, _, _, _ = circle_center_and_kinematics_affine(x_j[frame], s, vp_j)
            circle_patches_i[k].center = (ci[0], ci[1])
            circle_patches_j[k].center = (cj[0], cj[1])

        set_heading_line(heading_i_line, x_i[frame])
        set_heading_line(heading_j_line, x_j[frame])

        goal_i_scatter.set_offsets(gi)
        goal_j_scatter.set_offsets(gj)

        time_text.set_text(f"t = {t[frame]:.1f} s")

        return [
            traj_i_line, traj_j_line, poly_i, poly_j,
            heading_i_line, heading_j_line, goal_i_scatter, goal_j_scatter,
            time_text, *circle_patches_i, *circle_patches_j
        ]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        interval=100,
        blit=False,
        repeat=False,
    )

    try:
        writer = animation.FFMpegWriter(fps=10, bitrate=1800)
        ani.save(filename, writer=writer)
        print(f"Saved animation to: {filename}")
    except Exception as e:
        print(f"MP4 export failed: {e}")
        gif_name = os.path.splitext(filename)[0] + ".gif"
        try:
            ani.save(gif_name, writer="pillow", fps=10)
            print(f"Saved animation to: {gif_name}")
        except Exception as e2:
            print(f"GIF export also failed: {e2}")

    plt.show()


# =========================
# Main
# =========================

if __name__ == "__main__":
    print("Select lambda_cbf baseline method:")
    print("  1. Constant, normal value (0.2) for both vehicles")
    print("  2. Constant, normal value (0.2) for vehicle i and conservative value (0.15) for vehicle j")
    print("  3. Distance-adaptive values for both vehicles")
    choice = input("Enter 1, 2, or 3 [default: 1]: ").strip()

    schedule_map = {
        "1": "constant_normal",
        "2": "constant_conservative",
        "3": "adaptive",
    }
    cbfp_schedule = schedule_map.get(choice, "constant_normal")
    choice_tag = choice if choice in schedule_map else "1"

    if cbfp_schedule == "constant_normal":
        print("Both vehicles will use constant lambda_cbf = 0.2")
    elif cbfp_schedule == "constant_conservative":
        print("Vehicle i will use constant lambda_cbf = 0.2, and vehicle j will use constant lambda_cbf = 0.15")
    else:
        print("Both vehicles will use distance-adaptive lambda_cbf values")
        print("Rule: lambda_cbf = clip(clearance_gain * min_intervehicle_clearance, lambda_min, lambda_max)")

    n_rollouts = 2

    figure_filename = f"two_vehicle_road_results_choice{choice_tag}.png"
    video_filename = f"two_vehicle_road_choice{choice_tag}.mp4"

    avg_metrics, last_hist, last_bundle = run_monte_carlo(
        cbfp_schedule=cbfp_schedule,
        n_rollouts=n_rollouts,
    )

    show_average_metrics_table(avg_metrics, n_rollouts=n_rollouts)

    sim, vp_i, vp_j, route_i, route_j, circle_offsets, circle_radius = last_bundle

    plot_results(last_hist, route_i, filename=figure_filename)

    animate_simulation(
        last_hist,
        vp_i,
        vp_j,
        route_i,
        circle_offsets,
        circle_radius,
        filename=video_filename,
    )