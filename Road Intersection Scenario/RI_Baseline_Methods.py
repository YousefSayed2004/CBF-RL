"""
Rounded road-intersection CLF-CBF-QP scenario.

This intermediate script is intentionally standalone so the intersection
geometry and tuning parameters can be adjusted without touching the H2H or
lane-merging scenarios.
"""

from dataclasses import dataclass, field
import argparse
import math
import os
from typing import Dict, List, Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np


# =========================
# Utility functions
# =========================

def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def deg2rad(deg: float) -> float:
    return deg * np.pi / 180.0


def circle_approximation(length: float, width: float, n_circles: int = 3) -> Tuple[np.ndarray, float]:
    if n_circles != 3:
        raise ValueError("This implementation uses exactly 3 circles.")
    offsets = np.array([-length / 3.0, 0.0, length / 3.0], dtype=float)
    radius = math.sqrt((length / 6.0) ** 2 + (width / 2.0) ** 2)
    return offsets, radius


def left_normal(direction: np.ndarray) -> np.ndarray:
    return np.array([-direction[1], direction[0]], dtype=float)


def right_normal(direction: np.ndarray) -> np.ndarray:
    return np.array([direction[1], -direction[0]], dtype=float)


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
    w_v: float = 1.0
    w_delta: float = 15.0
    desired_speed: float = 5.0
    nominal_speed_gain: float = 1.0
    nominal_heading_gain: float = 1.0
    nominal_steer_rate_gain: float = 5.0
    clf_rate: float = 10.0
    clf_slack_weight: float = 1.0


@dataclass
class CBFParams:
    lambda_cbf: float = 0.25
    cbf_slack_weight: float = 1e8
    eps_dist: float = 1e-6


@dataclass
class QPWeights:
    u_weight: np.ndarray = field(default_factory=lambda: np.diag([1.0, 15.0]))
    eta_weight: float = 1e5
    eta_min: float = 0.0
    eta_max: float = 1.0
    # Set this to True if you want the aTTCBF baseline to use CBF slack again.
    attcbf_use_cbf_slack: bool = True


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
class IntersectionGeometry:
    road_width: float = 6.0
    corner_radius: float = 2.0
    lookahead_dist: float = 8.0
    start_distance: float = 15.0
    switch_offset: float = 2.0

    @property
    def road_half_width(self) -> float:
        return 0.5 * self.road_width

    @property
    def lane_offset(self) -> float:
        return 0.25 * self.road_width

    @property
    def corner_offset(self) -> float:
        return self.road_half_width + self.corner_radius


@dataclass
class RouteSpec:
    start: int
    exit: int
    geom: IntersectionGeometry

    @property
    def start_point(self) -> np.ndarray:
        lane = self.geom.lane_offset
        dist = self.geom.start_distance
        direction = self.start_dir
        return -dist * direction + lane * right_normal(direction)

    @property
    def start_dir(self) -> np.ndarray:
        dirs = {
            1: np.array([0.0, 1.0], dtype=float),
            2: np.array([-1.0, 0.0], dtype=float),
            3: np.array([0.0, -1.0], dtype=float),
            4: np.array([1.0, 0.0], dtype=float),
        }
        return dirs[self.start]

    @property
    def entry_point(self) -> np.ndarray:
        lane = self.geom.lane_offset
        c = self.geom.corner_offset
        direction = self.start_dir
        return -c * direction + lane * right_normal(direction)

    @property
    def switch_entry_point(self) -> np.ndarray:
        lane = self.geom.lane_offset
        c = self.geom.corner_offset + self.geom.switch_offset
        direction = self.start_dir
        return -c * direction + lane * right_normal(direction)

    @property
    def exit_point(self) -> np.ndarray:
        lane = self.geom.lane_offset
        c = self.geom.corner_offset
        direction = self.exit_dir
        return c * direction + lane * right_normal(direction)

    @property
    def exit_dir(self) -> np.ndarray:
        dirs = {
            1: np.array([0.0, -1.0], dtype=float),
            2: np.array([1.0, 0.0], dtype=float),
            3: np.array([0.0, 1.0], dtype=float),
            4: np.array([-1.0, 0.0], dtype=float),
        }
        return dirs[self.exit]

    @property
    def initial_heading(self) -> float:
        d = self.start_dir
        return math.atan2(d[1], d[0])


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


def circle_center_and_kinematics_affine(
    state: np.ndarray,
    s_offset: float,
    vp: VehicleParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    px, py, psi, v, delta = state
    L = vp.wheelbase

    e = np.array([math.cos(psi), math.sin(psi)], dtype=float)
    ep = np.array([-math.sin(psi), math.cos(psi)], dtype=float)

    c = np.array([px, py], dtype=float) + s_offset * e

    psi_dot = v * math.tan(delta) / L
    c_dot = v * e + s_offset * psi_dot * ep

    c_ddot_0 = (-s_offset * psi_dot**2) * e + (v * psi_dot) * ep

    B_a = e + s_offset * (math.tan(delta) / L) * ep
    B_sr = s_offset * (v / (L * math.cos(delta)**2)) * ep

    B = np.column_stack((B_a, B_sr))
    return c, c_dot, c_ddot_0, B


# =========================
# Goal and route helpers
# =========================

def front_circle_offset(vp: VehicleParams) -> float:
    return vp.length / 3.0


def route_goal_from_position(position: np.ndarray, route: RouteSpec) -> Tuple[np.ndarray, str]:
    """Single source of truth for all lookahead switching."""
    geom = route.geom
    entry = route.switch_entry_point
    start_dir = route.start_dir

    position = np.asarray(position, dtype=float)
    inbound_s = float((position - entry) @ start_dir)
    inbound_goal = entry + (inbound_s + geom.lookahead_dist) * start_dir
    if inbound_s < 0.0:
        return inbound_goal, "inbound_lookahead"

    exit_line_point = route.exit_point
    exit_dir = route.exit_dir
    exit_s = float((position - exit_line_point) @ exit_dir)
    return exit_line_point + (exit_s + geom.lookahead_dist) * exit_dir, "outbound_lookahead"


def first_time_to_exit(x_hist: np.ndarray, t: np.ndarray, route: RouteSpec) -> float:
    progress = (x_hist[:, :2] - route.exit_point) @ route.exit_dir
    idx = np.where(progress >= 0.0)[0]
    if len(idx) == 0:
        return float("nan")
    return float(t[idx[0]])


def line_centerline_deviation(point: np.ndarray, line_point: np.ndarray, direction: np.ndarray) -> float:
    return float(abs((point - line_point) @ left_normal(direction)))


def turn_arc_center_and_radius(route: RouteSpec) -> Tuple[np.ndarray, float]:
    entry = route.entry_point
    exit_point = route.exit_point
    exit_dir = route.exit_dir
    center = entry + float((exit_point - entry) @ exit_dir) * exit_dir
    radius = float(np.linalg.norm(entry - center))
    return center, radius


def centerline_deviation(point: np.ndarray, route: RouteSpec) -> float:
    point = np.asarray(point, dtype=float)
    entry = route.entry_point
    exit_point = route.exit_point
    start_dir = route.start_dir
    exit_dir = route.exit_dir

    if float(start_dir @ exit_dir) > 0.5:
        return line_centerline_deviation(point, entry, start_dir)

    inbound_s = float((point - entry) @ start_dir)
    if inbound_s < 0.0:
        return line_centerline_deviation(point, entry, start_dir)

    exit_s = float((point - exit_point) @ exit_dir)
    if exit_s > 0.0:
        return line_centerline_deviation(point, exit_point, exit_dir)

    center, radius = turn_arc_center_and_radius(route)
    return float(abs(np.linalg.norm(point - center) - radius))


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


def clf_terms(
    state: np.ndarray,
    vp: VehicleParams,
    clfp: CLFParams,
    goal_point: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    _, _, _, v, delta = state
    v_ref = clfp.desired_speed

    delta_des, _ = desired_steer_from_heading_error(
        state=state,
        goal_point=goal_point,
        vp=vp,
        clfp=clfp,
    )

    V = clfp.w_v * ((v - v_ref) ** 2) + clfp.w_delta * ((delta - delta_des) ** 2)

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
# Boundary geometry
# =========================

def intersection_sector(point: np.ndarray) -> str:
    x, y = float(point[0]), float(point[1])
    if y < -abs(x):
        return "bottom"
    if x >= abs(y):
        return "right"
    if y >= abs(x):
        return "top"
    return "left"


def in_corner_transition_box(point: np.ndarray, geom: IntersectionGeometry) -> bool:
    c = geom.corner_offset
    return bool(abs(point[0]) <= c and abs(point[1]) <= c)


def corner_centers(geom: IntersectionGeometry) -> Dict[str, np.ndarray]:
    c = geom.corner_offset
    return {
        "sw": np.array([-c, -c], dtype=float),
        "se": np.array([c, -c], dtype=float),
        "ne": np.array([c, c], dtype=float),
        "nw": np.array([-c, c], dtype=float),
    }


def active_boundary_specs(point: np.ndarray, geom: IntersectionGeometry) -> List[Tuple[str, str]]:
    sector = intersection_sector(point)
    use_arc = in_corner_transition_box(point, geom)

    if sector == "bottom":
        return [("arc", "sw"), ("arc", "se")] if use_arc else [("line", "x_min"), ("line", "x_max")]
    if sector == "right":
        return [("arc", "se"), ("arc", "ne")] if use_arc else [("line", "y_min"), ("line", "y_max")]
    if sector == "top":
        return [("arc", "nw"), ("arc", "ne")] if use_arc else [("line", "x_min"), ("line", "x_max")]
    return [("arc", "sw"), ("arc", "nw")] if use_arc else [("line", "y_min"), ("line", "y_max")]


def line_boundary_terms(
    spec: str,
    c: np.ndarray,
    c_dot: np.ndarray,
    c_ddot_0: np.ndarray,
    B: np.ndarray,
    geom: IntersectionGeometry,
    circle_radius: float,
) -> Tuple[float, float, np.ndarray]:
    h = geom.road_half_width

    if spec == "x_min":
        value = c[0] + h - circle_radius
        dot = c_dot[0]
        ddot_const = c_ddot_0[0]
        ddot_u = B[0, :]
    elif spec == "x_max":
        value = h - c[0] - circle_radius
        dot = -c_dot[0]
        ddot_const = -c_ddot_0[0]
        ddot_u = -B[0, :]
    elif spec == "y_min":
        value = c[1] + h - circle_radius
        dot = c_dot[1]
        ddot_const = c_ddot_0[1]
        ddot_u = B[1, :]
    elif spec == "y_max":
        value = h - c[1] - circle_radius
        dot = -c_dot[1]
        ddot_const = -c_ddot_0[1]
        ddot_u = -B[1, :]
    else:
        raise ValueError(f"Unknown line boundary spec: {spec}")

    return float(value), float(dot), np.array([ddot_const, ddot_u[0], ddot_u[1]], dtype=float)


def arc_boundary_terms(
    spec: str,
    c: np.ndarray,
    c_dot: np.ndarray,
    c_ddot_0: np.ndarray,
    B: np.ndarray,
    geom: IntersectionGeometry,
    circle_radius: float,
    eps_dist: float,
) -> Tuple[float, float, np.ndarray]:
    center = corner_centers(geom)[spec]
    d = c - center
    q = max(float(np.linalg.norm(d)), eps_dist)
    n = d / q

    value = q - geom.corner_radius - circle_radius
    dot = float(n @ c_dot)
    tangential_term = (float(c_dot @ c_dot) - dot**2) / q
    ddot_const = float(n @ c_ddot_0 + tangential_term)
    ddot_u = np.array([n @ B[:, 0], n @ B[:, 1]], dtype=float)

    return float(value), float(dot), np.array([ddot_const, ddot_u[0], ddot_u[1]], dtype=float)


def intersection_boundary_cbf_affines(
    ego_state: np.ndarray,
    ego_s: float,
    ego_vp: VehicleParams,
    geom: IntersectionGeometry,
    circle_radius: float,
    cbfp: CBFParams,
) -> List[Tuple[float, float, np.ndarray, str]]:
    c, c_dot, c_ddot_0, B = circle_center_and_kinematics_affine(ego_state, ego_s, ego_vp)
    terms = []

    for kind, spec in active_boundary_specs(c, geom):
        if kind == "line":
            h, h_dot, hddot_aff = line_boundary_terms(spec, c, c_dot, c_ddot_0, B, geom, circle_radius)
        else:
            h, h_dot, hddot_aff = arc_boundary_terms(
                spec, c, c_dot, c_ddot_0, B, geom, circle_radius, cbfp.eps_dist
            )
        terms.append((h, h_dot, hddot_aff, f"{kind}:{spec}"))

    return terms


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
    c1, c1_dot, c1_ddot_0, B1 = circle_center_and_kinematics_affine(ego_state, ego_s, ego_vp)
    c2, c2_dot, c2_ddot_0, _ = circle_center_and_kinematics_affine(other_state, other_s, other_vp)

    d = c1 - c2
    d_dot = c1_dot - c2_dot

    q = max(float(np.linalg.norm(d)), cbfp.eps_dist)
    n = d / q
    R = 2.0 * circle_radius

    h = q - R
    h_dot = float(n @ d_dot)
    tangential_term = (float(d_dot @ d_dot) - h_dot**2) / q

    hddot_const = float(n @ (c1_ddot_0 - c2_ddot_0) + tangential_term)
    hddot_u = np.array([n @ B1[:, 0], n @ B1[:, 1]], dtype=float)

    return h, h_dot, np.array([hddot_const, *hddot_u], dtype=float)


# =========================
# Decentralized QP
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
    use_attcbf: bool = False,
) -> Tuple[np.ndarray, Dict]:
    u_nom = nominal_control(ego_state, clfp, ego_vp, goal_point)

    n_pairwise = len(circle_offsets) * len(circle_offsets)
    n_boundary = 2 * len(circle_offsets)
    n_cbf = n_pairwise + n_boundary

    u = cp.Variable(2)
    s_cbf = cp.Variable(n_cbf, nonneg=True)
    s_clf = cp.Variable(1, nonneg=True)
    eta = cp.Variable() if use_attcbf else None
    use_cbf_slack = (not use_attcbf) or qpw.attcbf_use_cbf_slack

    constraints = [
        u[0] >= ego_vp.min_accel,
        u[0] <= ego_vp.max_accel,
        u[1] >= ego_vp.min_steer_rate,
        u[1] <= ego_vp.max_steer_rate,
    ]

    dt = sim.dt
    constraints += [
        ego_state[4] + dt * u[1] >= ego_vp.min_steer,
        ego_state[4] + dt * u[1] <= ego_vp.max_steer,
        ego_state[3] + dt * u[0] >= ego_vp.min_speed,
        ego_state[3] + dt * u[0] <= ego_vp.max_speed,
    ]
    if use_attcbf:
        constraints += [
            eta >= qpw.eta_min,
            eta <= qpw.eta_max,
        ]

    V, Vdot_const, Vdot_u = clf_terms(ego_state, ego_vp, clfp, goal_point)
    constraints += [
        Vdot_const + Vdot_u @ u <= -clfp.clf_rate * V + s_clf[0]
    ]

    Dt = sim.Dt
    lambda_cbf = cbfp.lambda_cbf
    cbf_gain = eta if use_attcbf else lambda_cbf
    cbf_index = 0
    boundary_specs = []

    def cbf_slack_term(index: int):
        return s_cbf[index] if use_cbf_slack else 0.0

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

    for s_i in circle_offsets:
        for h, h_dot, hddot_aff, spec in intersection_boundary_cbf_affines(
            ego_state=ego_state,
            ego_s=s_i,
            ego_vp=ego_vp,
            geom=ego_route.geom,
            circle_radius=circle_radius,
            cbfp=cbfp,
        ):
            hddot_const = hddot_aff[0]
            hddot_u = hddot_aff[1:]
            constraints += [
                0.5 * Dt**2 * (hddot_const + hddot_u @ u)
                + Dt * h_dot
                + lambda_cbf * h
                + s_cbf[cbf_index] >= 0.0
            ]
            boundary_specs.append(spec)
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
        "solver": None,
        "solver_attempts": [],
        "u_nom": u_nom.copy(),
        "clf_V": V,
        "boundary_specs": boundary_specs,
    }

    optimal_statuses = {"optimal", "optimal_inaccurate"}
    solver_attempts = []
    solver_plan = [
        ("OSQP", {
            "solver": cp.OSQP,
            "warm_start": True,
            "verbose": False,
            "eps_abs": 1e-5,
            "eps_rel": 1e-5,
            "max_iter": 20000,
        }),
        ("CLARABEL", {
            "solver": "CLARABEL",
            "warm_start": True,
            "verbose": False,
        }),
        ("SCS", {
            "solver": cp.SCS,
            "warm_start": True,
            "verbose": False,
        }),
    ]

    for solver_name, solve_kwargs in solver_plan:
        try:
            prob.solve(**solve_kwargs)
            solver_attempts.append({"solver": solver_name, "status": prob.status})
        except cp.SolverError as exc:
            solver_attempts.append({
                "solver": solver_name,
                "status": f"exception:{type(exc).__name__}",
                "exception": str(exc),
            })
            continue
        except Exception as exc:
            solver_attempts.append({
                "solver": solver_name,
                "status": f"exception:{type(exc).__name__}",
                "exception": str(exc),
            })
            continue

        if prob.status in optimal_statuses:
            qp_info["solver"] = solver_name
            break

    qp_info["status"] = prob.status
    qp_info["solver_attempts"] = solver_attempts
    if qp_info["status"] is None and solver_attempts:
        qp_info["status"] = solver_attempts[-1]["status"]
        if "exception" in solver_attempts[-1]:
            qp_info["exception"] = solver_attempts[-1]["exception"]

    if u.value is None or prob.status not in optimal_statuses:
        u_sol = np.array([
            np.clip(u_nom[0], ego_vp.min_accel, ego_vp.max_accel),
            np.clip(u_nom[1], ego_vp.min_steer_rate, ego_vp.max_steer_rate),
        ], dtype=float)
        qp_info["fallback"] = True
    else:
        u_sol = np.array(u.value).reshape(2)
        qp_info["fallback"] = False

    return u_sol, qp_info


def solve_vehicle_qp_against_many(
    ego_state: np.ndarray,
    other_states: List[np.ndarray],
    ego_route: RouteSpec,
    ego_vp: VehicleParams,
    other_vps: List[VehicleParams],
    clfp: CLFParams,
    cbfp: CBFParams,
    qpw: QPWeights,
    sim: SimParams,
    circle_offsets: np.ndarray,
    circle_radius: float,
    goal_point: np.ndarray,
    use_attcbf: bool = False,
) -> Tuple[np.ndarray, Dict]:
    u_nom = nominal_control(ego_state, clfp, ego_vp, goal_point)

    n_pairwise = len(other_states) * len(circle_offsets) * len(circle_offsets)
    n_boundary = 2 * len(circle_offsets)
    n_cbf = n_pairwise + n_boundary

    u = cp.Variable(2)
    s_cbf = cp.Variable(n_cbf, nonneg=True)
    s_clf = cp.Variable(1, nonneg=True)
    eta = cp.Variable() if use_attcbf else None
    use_cbf_slack = (not use_attcbf) or qpw.attcbf_use_cbf_slack

    constraints = [
        u[0] >= ego_vp.min_accel,
        u[0] <= ego_vp.max_accel,
        u[1] >= ego_vp.min_steer_rate,
        u[1] <= ego_vp.max_steer_rate,
    ]

    dt = sim.dt
    constraints += [
        ego_state[4] + dt * u[1] >= ego_vp.min_steer,
        ego_state[4] + dt * u[1] <= ego_vp.max_steer,
        ego_state[3] + dt * u[0] >= ego_vp.min_speed,
        ego_state[3] + dt * u[0] <= ego_vp.max_speed,
    ]
    if use_attcbf:
        constraints += [
            eta >= qpw.eta_min,
            eta <= qpw.eta_max,
        ]

    V, Vdot_const, Vdot_u = clf_terms(ego_state, ego_vp, clfp, goal_point)
    constraints += [
        Vdot_const + Vdot_u @ u <= -clfp.clf_rate * V + s_clf[0]
    ]

    Dt = sim.Dt
    lambda_cbf = cbfp.lambda_cbf
    cbf_gain = eta if use_attcbf else lambda_cbf
    cbf_index = 0
    boundary_specs = []

    def cbf_slack_term(index: int):
        return s_cbf[index] if use_cbf_slack else 0.0

    for other_state, other_vp in zip(other_states, other_vps):
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
                    + 0.5 * cbf_gain * h
                    + cbf_slack_term(cbf_index) >= 0.0
                ]
                cbf_index += 1

    for s_i in circle_offsets:
        for h, h_dot, hddot_aff, spec in intersection_boundary_cbf_affines(
            ego_state=ego_state,
            ego_s=s_i,
            ego_vp=ego_vp,
            geom=ego_route.geom,
            circle_radius=circle_radius,
            cbfp=cbfp,
        ):
            hddot_const = hddot_aff[0]
            hddot_u = hddot_aff[1:]
            constraints += [
                0.5 * Dt**2 * (hddot_const + hddot_u @ u)
                + Dt * h_dot
                + cbf_gain * h
                + cbf_slack_term(cbf_index) >= 0.0
            ]
            boundary_specs.append(spec)
            cbf_index += 1

    assert cbf_index == n_cbf

    objective_terms = (
        0.5 * cp.quad_form(u - u_nom, qpw.u_weight)
        + clfp.clf_slack_weight * cp.sum_squares(s_clf)
    )
    if use_cbf_slack:
        objective_terms += cbfp.cbf_slack_weight * cp.sum_squares(s_cbf)
    if use_attcbf:
        objective_terms += qpw.eta_weight * cp.square(eta)

    objective = cp.Minimize(objective_terms)

    prob = cp.Problem(objective, constraints)
    qp_info = {
        "status": None,
        "solver": None,
        "solver_attempts": [],
        "u_nom": u_nom.copy(),
        "clf_V": V,
        "boundary_specs": boundary_specs,
        "eta": None,
    }

    optimal_statuses = {"optimal", "optimal_inaccurate"}
    solver_attempts = []
    solver_plan = [
        ("OSQP", {
            "solver": cp.OSQP,
            "warm_start": True,
            "verbose": False,
            "eps_abs": 1e-5,
            "eps_rel": 1e-5,
            "max_iter": 20000,
        }),
        ("CLARABEL", {
            "solver": "CLARABEL",
            "warm_start": True,
            "verbose": False,
        }),
        ("SCS", {
            "solver": cp.SCS,
            "warm_start": True,
            "verbose": False,
        }),
    ]

    for solver_name, solve_kwargs in solver_plan:
        try:
            prob.solve(**solve_kwargs)
            solver_attempts.append({"solver": solver_name, "status": prob.status})
        except cp.SolverError as exc:
            solver_attempts.append({
                "solver": solver_name,
                "status": f"exception:{type(exc).__name__}",
                "exception": str(exc),
            })
            continue
        except Exception as exc:
            solver_attempts.append({
                "solver": solver_name,
                "status": f"exception:{type(exc).__name__}",
                "exception": str(exc),
            })
            continue

        if prob.status in optimal_statuses:
            qp_info["solver"] = solver_name
            break

    qp_info["status"] = prob.status
    qp_info["solver_attempts"] = solver_attempts
    if qp_info["status"] is None and solver_attempts:
        qp_info["status"] = solver_attempts[-1]["status"]
        if "exception" in solver_attempts[-1]:
            qp_info["exception"] = solver_attempts[-1]["exception"]

    if u.value is None or prob.status not in optimal_statuses:
        u_sol = np.array([
            np.clip(u_nom[0], ego_vp.min_accel, ego_vp.max_accel),
            np.clip(u_nom[1], ego_vp.min_steer_rate, ego_vp.max_steer_rate),
        ], dtype=float)
        qp_info["fallback"] = True
    else:
        u_sol = np.array(u.value).reshape(2)
        qp_info["fallback"] = False
        if use_attcbf and eta.value is not None:
            qp_info["eta"] = float(np.clip(float(eta.value), qpw.eta_min, qpw.eta_max))

    return u_sol, qp_info


# =========================
# Safety and metric helpers
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
    cbfp = CBFParams()
    min_clear = np.inf
    for s in circle_offsets:
        for h, _, _, _ in intersection_boundary_cbf_affines(
            ego_state=x,
            ego_s=s,
            ego_vp=vp,
            geom=route.geom,
            circle_radius=circle_radius,
            cbfp=cbfp,
        ):
            min_clear = min(min_clear, h)
    return float(min_clear)


def vehicle_corners(state: np.ndarray, vp: VehicleParams) -> np.ndarray:
    px, py, psi, _, _ = state
    L = vp.length
    W = vp.width
    local = np.array([
        [L / 2, W / 2],
        [L / 2, -W / 2],
        [-L / 2, -W / 2],
        [-L / 2, W / 2],
    ])
    c = math.cos(psi)
    s = math.sin(psi)
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([px, py])


# =========================
# Rollout
# =========================

def initial_state_from_route(route: RouteSpec, speed: float) -> np.ndarray:
    p = route.start_point
    return np.array([p[0], p[1], route.initial_heading, speed, 0.0], dtype=float)


def run_single_rollout(
    route_i: RouteSpec,
    route_j: RouteSpec,
    vp_i: VehicleParams,
    vp_j: VehicleParams,
    clfp_i: CLFParams,
    clfp_j: CLFParams,
    cbfp_i: CBFParams,
    cbfp_j: CBFParams,
    qpw_i: QPWeights,
    qpw_j: QPWeights,
    sim: SimParams,
    speed_i: float,
    speed_j: float,
) -> Tuple[Dict, Dict, Tuple]:
    circle_offsets, circle_radius = circle_approximation(vp_i.length, vp_i.width, n_circles=3)

    x_i = initial_state_from_route(route_i, speed_i)
    x_j = initial_state_from_route(route_j, speed_j)

    current_clearance = min_intervehicle_clearance(x_i, x_j, vp_i, vp_j, circle_offsets, circle_radius)
    bi_clear = min_boundary_clearance(x_i, route_i, vp_i, circle_offsets, circle_radius)
    bj_clear = min_boundary_clearance(x_j, route_j, vp_j, circle_offsets, circle_radius)

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
        "goal_mode_i": [],
        "goal_mode_j": [],
        "lambda_i": [cbfp_i.lambda_cbf],
        "lambda_j": [cbfp_j.lambda_cbf],
        "clearance_iv": [current_clearance],
        "boundary_i": [bi_clear],
        "boundary_j": [bj_clear],
        "events": [],
        "t": [0.0],
    }

    def print_step_details(t_now, clearance, bound_i, bound_j, mode_i="", mode_j=""):
        msg = (
            f"t = {t_now:.1f} s | "
            f"clearance = {clearance:.3f} m | "
            f"bi_clear = {bound_i:.3f} m | "
            f"bj_clear = {bound_j:.3f} m | "
            f"lambda_i = {cbfp_i.lambda_cbf:.3f} | "
            f"lambda_j = {cbfp_j.lambda_cbf:.3f} | "
            #f"mode_i = {mode_i or '-'} | "
            #f"mode_j = {mode_j or '-'}"
        )
        print(msg, flush=True)

    for k in range(sim.steps):
        t_now = k * sim.dt

        current_clearance = min_intervehicle_clearance(x_i, x_j, vp_i, vp_j, circle_offsets, circle_radius)
        bi_clear = min_boundary_clearance(x_i, route_i, vp_i, circle_offsets, circle_radius)
        bj_clear = min_boundary_clearance(x_j, route_j, vp_j, circle_offsets, circle_radius)

        if current_clearance < -1e-3:
            hist["events"].append("collision")
            print_step_details(t_now, current_clearance, bi_clear, bj_clear)
            print(f"  terminating at t = {t_now:.1f} s: inter-vehicle collision")
            break
        if bi_clear < -1e-3 or bj_clear < -1e-3:
            hist["events"].append("boundary_collision")
            print_step_details(t_now, current_clearance, bi_clear, bj_clear)
            print(f"  terminating at t = {t_now:.1f} s: boundary collision")
            break

        goal_i, mode_i = route_goal_from_position(x_i[:2], route_i)
        goal_j, mode_j = route_goal_from_position(x_j[:2], route_j)

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

        infeasible_i = qp_i.get("status") not in ["optimal", "optimal_inaccurate"]
        infeasible_j = qp_j.get("status") not in ["optimal", "optimal_inaccurate"]

        if infeasible_i or infeasible_j:
            print_step_details(t_now, current_clearance, bi_clear, bj_clear, mode_i, mode_j)
            print(
                f"  terminating at t = {t_now:.1f} s: infeasible QP "
                f"(status_i={qp_i.get('status')}, status_j={qp_j.get('status')})"
            )
            hist["x_i"].append(x_i.copy())
            hist["x_j"].append(x_j.copy())
            hist["u_i"].append(u_i.copy())
            hist["u_j"].append(u_j.copy())
            hist["qp_i"].append(qp_i)
            hist["qp_j"].append(qp_j)
            hist["goal_i"].append(goal_i.copy())
            hist["goal_j"].append(goal_j.copy())
            hist["goal_mode_i"].append(mode_i)
            hist["goal_mode_j"].append(mode_j)
            hist["lambda_i"].append(cbfp_i.lambda_cbf)
            hist["lambda_j"].append(cbfp_j.lambda_cbf)
            hist["clearance_iv"].append(current_clearance)
            hist["boundary_i"].append(bi_clear)
            hist["boundary_j"].append(bj_clear)
            hist["events"].append("infeasible_qp")
            hist["t"].append((k + 1) * sim.dt)
            break

        _, epsi_i = desired_steer_from_heading_error(x_i, goal_i, vp_i, clfp_i)
        _, epsi_j = desired_steer_from_heading_error(x_j, goal_j, vp_j, clfp_j)

        print_step_details(t_now, current_clearance, bi_clear, bj_clear, mode_i, mode_j)

        x_i = step_vehicle(x_i, u_i, vp_i, sim.dt)
        x_j = step_vehicle(x_j, u_j, vp_j, sim.dt)

        next_clearance = min_intervehicle_clearance(x_i, x_j, vp_i, vp_j, circle_offsets, circle_radius)
        next_bi_clear = min_boundary_clearance(x_i, route_i, vp_i, circle_offsets, circle_radius)
        next_bj_clear = min_boundary_clearance(x_j, route_j, vp_j, circle_offsets, circle_radius)

        if next_clearance < -1e-3:
            event = "collision"
        elif next_bi_clear < -1e-3 or next_bj_clear < -1e-3:
            event = "boundary_collision"
        else:
            event = "running"

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
        hist["goal_mode_i"].append(mode_i)
        hist["goal_mode_j"].append(mode_j)
        hist["lambda_i"].append(cbfp_i.lambda_cbf)
        hist["lambda_j"].append(cbfp_j.lambda_cbf)
        hist["clearance_iv"].append(next_clearance)
        hist["boundary_i"].append(next_bi_clear)
        hist["boundary_j"].append(next_bj_clear)
        hist["events"].append(event)
        hist["t"].append((k + 1) * sim.dt)

        if event != "running":
            print_step_details((k + 1) * sim.dt, next_clearance, next_bi_clear, next_bj_clear, mode_i, mode_j)
            print(f"  terminating at t = {(k + 1) * sim.dt:.1f} s: {event.replace('_', ' ')}")
            break

    print()

    for key in [
        "x_i", "x_j", "u_i", "u_j", "epsi_i", "epsi_j", "goal_i", "goal_j",
        "t", "lambda_i", "lambda_j", "clearance_iv", "boundary_i", "boundary_j",
    ]:
        hist[key] = np.array(hist[key])
    for key in ["u_i", "u_j"]:
        if hist[key].size == 0:
            hist[key] = hist[key].reshape(0, 2)
    for key in ["goal_i", "goal_j"]:
        if hist[key].size == 0:
            hist[key] = hist[key].reshape(0, 2)

    metrics = compute_rollout_metrics(hist, route_i, route_j, vp_i, vp_j, circle_offsets, circle_radius)
    bundle = (sim, vp_i, vp_j, route_i, route_j, circle_offsets, circle_radius)
    return hist, metrics, bundle


def min_multi_intervehicle_clearance(
    states: Dict[str, np.ndarray],
    vps: Dict[str, VehicleParams],
    circle_offsets: np.ndarray,
    circle_radius: float,
) -> float:
    names = list(states.keys())
    min_clear = np.inf
    for a_idx, name_a in enumerate(names):
        for name_b in names[a_idx + 1:]:
            clear = min_intervehicle_clearance(
                states[name_a],
                states[name_b],
                vps[name_a],
                vps[name_b],
                circle_offsets,
                circle_radius,
            )
            min_clear = min(min_clear, clear)
    return float(min_clear)


def run_multi_vehicle_rollout(
    vehicle_data: Dict[str, Dict],
    sim: SimParams,
    lambda_update_fn=None,
    qp_solve_fn=None,
    use_attcbf: bool = False,
) -> Tuple[Dict, Dict, Tuple]:
    names = list(vehicle_data.keys())
    first_vp = vehicle_data[names[0]]["vp"]
    circle_offsets, circle_radius = circle_approximation(first_vp.length, first_vp.width, n_circles=3)

    states = {
        name: initial_state_from_route(data["route"], data["speed"])
        for name, data in vehicle_data.items()
    }

    inter_clear = min_multi_intervehicle_clearance(
        states,
        {name: data["vp"] for name, data in vehicle_data.items()},
        circle_offsets,
        circle_radius,
    )
    boundary_clear = {
        name: min_boundary_clearance(
            states[name],
            data["route"],
            data["vp"],
            circle_offsets,
            circle_radius,
        )
        for name, data in vehicle_data.items()
    }

    hist = {
        "names": names,
        "x": {name: [states[name].copy()] for name in names},
        "u": {name: [] for name in names},
        "qp": {name: [] for name in names},
        "epsi": {name: [] for name in names},
        "goal": {name: [] for name in names},
        "goal_mode": {name: [] for name in names},
        "lambda": {name: [vehicle_data[name]["cbfp"].lambda_cbf] for name in names},
        "cbf_gain_label": "eta" if use_attcbf else "lambda",
        "clearance_iv": [inter_clear],
        "boundary": {name: [boundary_clear[name]] for name in names},
        "events": [],
        "t": [0.0],
    }

    def print_step_details(t_now, clearance, boundary):
        lambda_text = " | ".join(
            f"{hist['cbf_gain_label']}_{name} = {vehicle_data[name]['cbfp'].lambda_cbf:.3f}"
            for name in names
        )
        min_boundary = min(boundary.values()) if boundary else float("nan")
        msg = (
            f"t = {t_now:.1f} s | "
            f"clearance = {clearance:.3f} m | "
            f"min_boundary = {min_boundary:.3f} m | "
            f"{lambda_text} |"
        )
        print(msg, flush=True)

    optimal_statuses = {"optimal", "optimal_inaccurate"}

    for k in range(sim.steps):
        t_now = k * sim.dt
        vps = {name: data["vp"] for name, data in vehicle_data.items()}
        inter_clear = min_multi_intervehicle_clearance(states, vps, circle_offsets, circle_radius)
        boundary_clear = {
            name: min_boundary_clearance(
                states[name],
                vehicle_data[name]["route"],
                vehicle_data[name]["vp"],
                circle_offsets,
                circle_radius,
            )
            for name in names
        }

        if inter_clear < -1e-3:
            hist["events"].append("collision")
            print_step_details(t_now, inter_clear, boundary_clear)
            print(f"  terminating at t = {t_now:.1f} s: inter-vehicle collision")
            break
        if any(clear < -1e-3 for clear in boundary_clear.values()):
            hist["events"].append("boundary_collision")
            print_step_details(t_now, inter_clear, boundary_clear)
            print(f"  terminating at t = {t_now:.1f} s: boundary collision")
            break

        if lambda_update_fn is not None:
            lambda_update_fn(inter_clear, vehicle_data, states, circle_offsets, circle_radius)

        goals = {}
        modes = {}
        controls = {}
        qp_infos = {}
        epsi_values = {}

        for name in names:
            other_names = [other for other in names if other != name]
            other_states = [states[other].copy() for other in other_names]
            other_vps = [vehicle_data[other]["vp"] for other in other_names]
            goals[name], modes[name] = route_goal_from_position(
                states[name][:2],
                vehicle_data[name]["route"],
            )
            solve_fn = qp_solve_fn or solve_vehicle_qp_against_many
            solve_kwargs = dict(
                ego_state=states[name].copy(),
                other_states=other_states,
                ego_route=vehicle_data[name]["route"],
                ego_vp=vehicle_data[name]["vp"],
                other_vps=other_vps,
                clfp=vehicle_data[name]["clfp"],
                cbfp=vehicle_data[name]["cbfp"],
                qpw=vehicle_data[name]["qpw"],
                sim=sim,
                circle_offsets=circle_offsets,
                circle_radius=circle_radius,
                goal_point=goals[name],
                use_attcbf=use_attcbf,
            )
            if qp_solve_fn is not None:
                solve_kwargs["name"] = name
                solve_kwargs["other_names"] = other_names
                solve_kwargs.pop("use_attcbf", None)
            controls[name], qp_infos[name] = solve_fn(**solve_kwargs)
            if use_attcbf and qp_infos[name].get("eta") is not None:
                vehicle_data[name]["cbfp"].lambda_cbf = qp_infos[name]["eta"]

        infeasible_names = [
            name for name in names
            if qp_infos[name].get("status") not in optimal_statuses
        ]
        if infeasible_names:
            print_step_details(k * sim.dt, inter_clear, boundary_clear)
            statuses = ", ".join(f"{name}={qp_infos[name].get('status')}" for name in infeasible_names)
            print(f"  terminating at t = {k * sim.dt:.1f} s: infeasible QP ({statuses})")
            for name in names:
                hist["x"][name].append(states[name].copy())
                hist["u"][name].append(controls[name].copy())
                hist["qp"][name].append(qp_infos[name])
                hist["goal"][name].append(goals[name].copy())
                hist["goal_mode"][name].append(modes[name])
                hist["lambda"][name].append(vehicle_data[name]["cbfp"].lambda_cbf)
                hist["boundary"][name].append(boundary_clear[name])
            hist["clearance_iv"].append(inter_clear)
            hist["events"].append("infeasible_qp")
            hist["t"].append((k + 1) * sim.dt)
            break

        for name in names:
            _, epsi_values[name] = desired_steer_from_heading_error(
                states[name],
                goals[name],
                vehicle_data[name]["vp"],
                vehicle_data[name]["clfp"],
            )

        print_step_details(k * sim.dt, inter_clear, boundary_clear)

        next_states = {
            name: step_vehicle(states[name], controls[name], vehicle_data[name]["vp"], sim.dt)
            for name in names
        }
        states = next_states

        vps = {name: data["vp"] for name, data in vehicle_data.items()}
        next_clearance = min_multi_intervehicle_clearance(states, vps, circle_offsets, circle_radius)
        next_boundary = {
            name: min_boundary_clearance(
                states[name],
                vehicle_data[name]["route"],
                vehicle_data[name]["vp"],
                circle_offsets,
                circle_radius,
            )
            for name in names
        }

        if next_clearance < -1e-3:
            event = "collision"
        elif any(clear < -1e-3 for clear in next_boundary.values()):
            event = "boundary_collision"
        else:
            event = "running"

        for name in names:
            hist["x"][name].append(states[name].copy())
            hist["u"][name].append(controls[name].copy())
            hist["qp"][name].append(qp_infos[name])
            hist["epsi"][name].append(epsi_values[name])
            hist["goal"][name].append(goals[name].copy())
            hist["goal_mode"][name].append(modes[name])
            hist["lambda"][name].append(vehicle_data[name]["cbfp"].lambda_cbf)
            hist["boundary"][name].append(next_boundary[name])
        hist["clearance_iv"].append(next_clearance)
        hist["events"].append(event)
        hist["t"].append((k + 1) * sim.dt)

        if event != "running":
            print_step_details((k + 1) * sim.dt, next_clearance, next_boundary)
            print(f"  terminating at t = {(k + 1) * sim.dt:.1f} s: {event.replace('_', ' ')}")
            break

    print()

    hist["t"] = np.array(hist["t"])
    hist["clearance_iv"] = np.array(hist["clearance_iv"])
    for name in names:
        for key in ["x", "u", "epsi", "goal", "lambda", "boundary"]:
            hist[key][name] = np.array(hist[key][name])
        if hist["u"][name].size == 0:
            hist["u"][name] = hist["u"][name].reshape(0, 2)
        if hist["goal"][name].size == 0:
            hist["goal"][name] = hist["goal"][name].reshape(0, 2)

    metrics = compute_multi_rollout_metrics(hist, vehicle_data, circle_offsets, circle_radius)
    bundle = (sim, vehicle_data, circle_offsets, circle_radius)
    return hist, metrics, bundle


# =========================
# Metrics
# =========================

def compute_rollout_metrics(
    hist: Dict,
    route_i: RouteSpec,
    route_j: RouteSpec,
    vp_i: VehicleParams,
    vp_j: VehicleParams,
    circle_offsets: np.ndarray,
    circle_radius: float,
) -> Dict:
    x_i = hist["x_i"]
    x_j = hist["x_j"]
    u_i = hist["u_i"]
    u_j = hist["u_j"]
    t = hist["t"]
    dt = t[1] - t[0] if len(t) > 1 else 0.1

    acc_eff_i = float(np.sum(np.abs(u_i[:, 0])) * dt) if len(u_i) else 0.0
    acc_eff_j = float(np.sum(np.abs(u_j[:, 0])) * dt) if len(u_j) else 0.0
    steer_eff_i = float(np.sum(np.abs(u_i[:, 1])) * dt) if len(u_i) else 0.0
    steer_eff_j = float(np.sum(np.abs(u_j[:, 1])) * dt) if len(u_j) else 0.0

    min_iv_clear = np.inf
    min_bound_clear_i = np.inf
    min_bound_clear_j = np.inf
    intervehicle_collision = False
    boundary_collision_i = False
    boundary_collision_j = False

    for k in range(len(t)):
        iv_clear = min_intervehicle_clearance(x_i[k], x_j[k], vp_i, vp_j, circle_offsets, circle_radius)
        bi_clear = min_boundary_clearance(x_i[k], route_i, vp_i, circle_offsets, circle_radius)
        bj_clear = min_boundary_clearance(x_j[k], route_j, vp_j, circle_offsets, circle_radius)

        min_iv_clear = min(min_iv_clear, iv_clear)
        min_bound_clear_i = min(min_bound_clear_i, bi_clear)
        min_bound_clear_j = min(min_bound_clear_j, bj_clear)

        if iv_clear < 0.0:
            intervehicle_collision = True
        if bi_clear < -1e-3:
            boundary_collision_i = True
        if bj_clear < -1e-3:
            boundary_collision_j = True

    time_to_goal_i = first_time_to_exit(x_i, t, route_i)
    time_to_goal_j = first_time_to_exit(x_j, t, route_j)
    infeasible_qp = int(any(ev == "infeasible_qp" for ev in hist.get("events", [])))
    deadlock = int(np.isnan(time_to_goal_i) or np.isnan(time_to_goal_j))

    avg_centerline_deviation_i = float(np.mean([centerline_deviation(state[:2], route_i) for state in x_i]))
    avg_centerline_deviation_j = float(np.mean([centerline_deviation(state[:2], route_j) for state in x_j]))

    invalid_motion = (
        infeasible_qp
        or intervehicle_collision
        or boundary_collision_i
        or boundary_collision_j
    )

    return {
        "i": {
            "time_to_goal": time_to_goal_i,
            "avg_centerline_deviation": avg_centerline_deviation_i,
            "acc_effort": acc_eff_i,
            "steer_rate_effort": steer_eff_i,
            "intervehicle_collision": int(intervehicle_collision),
            "boundary_collision": int(boundary_collision_i),
            "min_intervehicle_clearance": float(min_iv_clear),
            "min_boundary_clearance": float(min_bound_clear_i),
        },
        "j": {
            "time_to_goal": time_to_goal_j,
            "avg_centerline_deviation": avg_centerline_deviation_j,
            "acc_effort": acc_eff_j,
            "steer_rate_effort": steer_eff_j,
            "intervehicle_collision": int(intervehicle_collision),
            "boundary_collision": int(boundary_collision_j),
            "min_intervehicle_clearance": float(min_iv_clear),
            "min_boundary_clearance": float(min_bound_clear_j),
        },
        "deadlock": deadlock,
        "infeasible_qp": infeasible_qp,
        "valid_motion_metrics": int(not invalid_motion),
    }


def compute_multi_rollout_metrics(
    hist: Dict,
    vehicle_data: Dict[str, Dict],
    circle_offsets: np.ndarray,
    circle_radius: float,
) -> Dict:
    names = hist["names"]
    t = hist["t"]
    dt = t[1] - t[0] if len(t) > 1 else 0.1
    metrics = {}
    intervehicle_collision = bool(np.any(hist["clearance_iv"] < 0.0))
    infeasible_qp = int(any(ev == "infeasible_qp" for ev in hist.get("events", [])))

    for name in names:
        x_hist = hist["x"][name]
        u_hist = hist["u"][name]
        route = vehicle_data[name]["route"]
        vp = vehicle_data[name]["vp"]
        boundary_values = np.array([
            min_boundary_clearance(state, route, vp, circle_offsets, circle_radius)
            for state in x_hist
        ])
        time_to_goal = first_time_to_exit(x_hist, t, route)
        metrics[name] = {
            "time_to_goal": time_to_goal,
            "avg_centerline_deviation": float(np.mean([
                centerline_deviation(state[:2], route)
                for state in x_hist
            ])),
            "acc_effort": float(np.sum(np.abs(u_hist[:, 0])) * dt) if len(u_hist) else 0.0,
            "steer_rate_effort": float(np.sum(np.abs(u_hist[:, 1])) * dt) if len(u_hist) else 0.0,
            "intervehicle_collision": int(intervehicle_collision),
            "boundary_collision": int(np.any(boundary_values < -1e-3)),
            "min_intervehicle_clearance": float(np.min(hist["clearance_iv"])),
            "min_boundary_clearance": float(np.min(boundary_values)),
        }

    deadlock = int(any(np.isnan(metrics[name]["time_to_goal"]) for name in names))
    boundary_collision = any(metrics[name]["boundary_collision"] for name in names)
    metrics["deadlock"] = deadlock
    metrics["infeasible_qp"] = infeasible_qp
    metrics["valid_motion_metrics"] = int(
        not infeasible_qp and not intervehicle_collision and not boundary_collision
    )
    return metrics


def show_multi_metrics_summary(metrics: Dict, names: List[str]) -> None:
    print("Multi-vehicle metrics summary")
    print(f"  deadlock = {metrics['deadlock']}")
    print(f"  infeasible_qp = {metrics['infeasible_qp']}")
    print(f"  valid_motion_metrics = {metrics['valid_motion_metrics']}")
    for name in names:
        m = metrics[name]
        time_text = "NaN" if np.isnan(m["time_to_goal"]) else f"{m['time_to_goal']:.3f}"
        print(
            f"  vehicle {name}: "
            f"time_to_goal = {time_text} s | "
            f"avg_dev = {m['avg_centerline_deviation']:.3f} m | "
            f"min_iv_clear = {m['min_intervehicle_clearance']:.3f} m | "
            f"min_boundary = {m['min_boundary_clearance']:.3f} m"
        )


def average_rollout_metrics(all_metrics: List[Dict]) -> Dict:
    if not all_metrics:
        raise ValueError("all_metrics must contain at least one rollout.")
    vehicles = [
        key for key, value in all_metrics[0].items()
        if isinstance(value, dict) and "time_to_goal" in value
    ]
    n_rollouts = len(all_metrics)
    avg = {veh: {} for veh in vehicles}
    valid_motion_metrics = [m for m in all_metrics if m.get("valid_motion_metrics", 1)]
    completion_metrics = [m for m in valid_motion_metrics if not m.get("deadlock", 0)]

    def mean_valid(veh: str, key: str, nanmean: bool = False) -> float:
        if not valid_motion_metrics:
            return float("nan")
        values = [m[veh][key] for m in valid_motion_metrics]
        if nanmean:
            finite_values = [v for v in values if not np.isnan(v)]
            if not finite_values:
                return float("nan")
            return float(np.mean(finite_values))
        return float(np.mean(values))

    def rollout_completion_time(m: Dict) -> float:
        times = np.array([m[veh]["time_to_goal"] for veh in vehicles], dtype=float)
        if np.any(np.isnan(times)):
            return float("nan")
        return float(np.max(times))

    for veh in vehicles:
        avg[veh]["time_to_goal"] = mean_valid(veh, "time_to_goal", nanmean=True)
        avg[veh]["avg_centerline_deviation"] = mean_valid(veh, "avg_centerline_deviation")
        avg[veh]["acc_effort"] = mean_valid(veh, "acc_effort")
        avg[veh]["steer_rate_effort"] = mean_valid(veh, "steer_rate_effort")
        avg[veh]["intervehicle_collision"] = float(np.mean([m[veh]["intervehicle_collision"] for m in all_metrics]))
        avg[veh]["boundary_collision"] = float(np.mean([m[veh]["boundary_collision"] for m in all_metrics]))
        avg[veh]["min_intervehicle_clearance"] = mean_valid(veh, "min_intervehicle_clearance")
        avg[veh]["min_boundary_clearance"] = mean_valid(veh, "min_boundary_clearance")

    deadlock_count = int(np.sum([m["deadlock"] for m in valid_motion_metrics]))
    avg["deadlock_count"] = deadlock_count
    avg["deadlock_rate"] = 100.0 * deadlock_count / n_rollouts if n_rollouts else float("nan")
    avg["infeasible_qp_count"] = int(np.sum([m["infeasible_qp"] for m in all_metrics]))
    avg["infeasible_qp_rate"] = float(np.mean([m["infeasible_qp"] for m in all_metrics]))
    avg["valid_motion_rollouts"] = len(valid_motion_metrics)
    avg["valid_motion_rollout_rate"] = (
        100.0 * avg["valid_motion_rollouts"] / n_rollouts if n_rollouts else float("nan")
    )
    avg["intervehicle_collision_count"] = int(np.sum([
        int(any(m[veh]["intervehicle_collision"] for veh in vehicles))
        for m in all_metrics
    ]))
    avg["boundary_collision_count"] = int(np.sum([
        int(any(m[veh]["boundary_collision"] for veh in vehicles))
        for m in all_metrics
    ]))
    avg["collision_count"] = int(np.sum([
        int(any(m[veh]["intervehicle_collision"] or m[veh]["boundary_collision"] for veh in vehicles))
        for m in all_metrics
    ]))
    avg["collision_rate"] = (
        100.0 * avg["collision_count"] / n_rollouts
        if n_rollouts
        else float("nan")
    )

    def mean_completion_time() -> float:
        values = [rollout_completion_time(m) for m in completion_metrics]
        finite_values = [v for v in values if not np.isnan(v)]
        if not finite_values:
            return float("nan")
        return float(np.mean(finite_values))

    avg["system"] = {
        "valid_rollouts": avg["valid_motion_rollouts"],
        "valid_rollout_rate": avg["valid_motion_rollout_rate"],
        "deadlock_count": avg["deadlock_count"],
        "deadlock_rate": avg["deadlock_rate"],
        "collision_count": avg["collision_count"],
        "collision_rate": avg["collision_rate"],
        "completion_time": mean_completion_time(),
    }
    episode_returns = [
        m["episode_return"]
        for m in all_metrics
        if "episode_return" in m and not np.isnan(m["episode_return"])
    ]
    avg["avg_episode_return"] = float(np.mean(episode_returns)) if episode_returns else float("nan")
    return avg


def show_average_metrics_table(avg_metrics: Dict, n_rollouts: int) -> Dict:
    def fmt_nan(x):
        return "NaN" if np.isnan(x) else f"{x:.3f}"

    def fmt_percent(x):
        return "NaN" if np.isnan(x) else f"{x:.1f}%"

    system = avg_metrics["system"]
    row_labels = [
        "Task Completion Time [s]",
        "Collision Rate",
        "Deadlock Rate",
    ]
    table_data = [
        [fmt_nan(system["completion_time"])],
        [f"{system['collision_count']}/{n_rollouts} ({fmt_percent(system['collision_rate'])})"],
        [f"{system['deadlock_count']}/{n_rollouts} ({fmt_percent(system['deadlock_rate'])})"],
    ]

    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.axis("off")
    ax.set_title(f"Road Intersection Metrics Summary over {n_rollouts} Rollouts", fontsize=13, pad=10)
    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=["Process"],
        cellLoc="center",
        rowLoc="center",
        loc="center",
        bbox=[0.15, 0.05, 0.80, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(0.20)
        else:
            cell.set_height(0.20)
        if col == -1:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    plt.show()
    return avg_metrics


# =========================
# Plot helpers
# =========================

def draw_intersection(ax, geom: IntersectionGeometry, xlim=(-18, 18), ylim=(-18, 18)) -> None:
    from matplotlib.patches import Arc, Rectangle, Wedge

    h = geom.road_half_width
    r = geom.corner_radius
    c = geom.corner_offset
    xmin, xmax = xlim
    ymin, ymax = ylim

    road_color = "lightgray"
    ax.add_patch(Rectangle((xmin, -h), xmax - xmin, 2.0 * h, facecolor=road_color, alpha=0.45, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((-h, ymin), 2.0 * h, ymax - ymin, facecolor=road_color, alpha=0.45, edgecolor="none", zorder=0))
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            ax.add_patch(Rectangle(
                (sx * h if sx > 0 else -c, sy * h if sy > 0 else -c),
                r,
                r,
                facecolor=road_color,
                alpha=0.45,
                edgecolor="none",
                zorder=0,
            ))

    for name, center in corner_centers(geom).items():
        if name == "ne":
            theta1, theta2 = 180, 270
        elif name == "se":
            theta1, theta2 = 90, 180
        elif name == "sw":
            theta1, theta2 = 0, 90
        else:
            theta1, theta2 = 270, 360
        ax.add_patch(Wedge(center, r, theta1, theta2, facecolor="white", edgecolor="none", zorder=0.5))
        ax.add_patch(Arc(center, 2.0 * r, 2.0 * r, theta1=theta1, theta2=theta2, color="k", linewidth=1.5, zorder=2))

    ax.plot([-h, -h], [ymin, -c], "k-", linewidth=1.5)
    ax.plot([h, h], [ymin, -c], "k-", linewidth=1.5)
    ax.plot([-h, -h], [c, ymax], "k-", linewidth=1.5)
    ax.plot([h, h], [c, ymax], "k-", linewidth=1.5)
    ax.plot([xmin, -c], [-h, -h], "k-", linewidth=1.5)
    ax.plot([xmin, -c], [h, h], "k-", linewidth=1.5)
    ax.plot([c, xmax], [-h, -h], "k-", linewidth=1.5)
    ax.plot([c, xmax], [h, h], "k-", linewidth=1.5)

    ax.plot([0.0, 0.0], [ymin, -c], linestyle="--", color="gray", alpha=0.6, linewidth=1.0)
    ax.plot([0.0, 0.0], [c, ymax], linestyle="--", color="gray", alpha=0.6, linewidth=1.0)
    ax.plot([xmin, -c], [0.0, 0.0], linestyle="--", color="gray", alpha=0.6, linewidth=1.0)
    ax.plot([c, xmax], [0.0, 0.0], linestyle="--", color="gray", alpha=0.6, linewidth=1.0)


def plot_results(
    hist: Dict,
    route_i: RouteSpec,
    route_j: RouteSpec,
    filename: str = "RI_results.png",
) -> None:
    t_state = hist["t"]
    t_u = t_state[:-1]
    x_i = hist["x_i"]
    x_j = hist["x_j"]
    u_i = hist["u_i"]
    u_j = hist["u_j"]

    fig, axs = plt.subplots(5, 2, figsize=(12, 16), constrained_layout=True)

    draw_intersection(axs[0, 0], route_i.geom, xlim=(-16, 16), ylim=(-16, 16))
    axs[0, 0].plot(x_i[:, 0], x_i[:, 1], label="vehicle i")
    axs[0, 0].plot(x_j[:, 0], x_j[:, 1], label="vehicle j")
    axs[0, 0].scatter(x_i[0, 0], x_i[0, 1], marker="o", s=60, label="i start")
    axs[0, 0].scatter(x_j[0, 0], x_j[0, 1], marker="o", s=60, label="j start")
    axs[0, 0].scatter(x_i[-1, 0], x_i[-1, 1], marker="x", s=70, label="i end")
    axs[0, 0].scatter(x_j[-1, 0], x_j[-1, 1], marker="x", s=70, label="j end")
    axs[0, 0].set_title("Road Intersection Trajectories")
    axs[0, 0].set_xlabel("x [m]")
    axs[0, 0].set_ylabel("y [m]")
    axs[0, 0].axis("equal")
    axs[0, 0].set_xlim(-16, 16)
    axs[0, 0].set_ylim(-16, 16)
    axs[0, 0].grid(True)
    axs[0, 0].legend(fontsize=8)

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

    if len(u_i):
        axs[2, 0].plot(t_u, u_i[:, 0], label="vehicle i")
        axs[2, 0].plot(t_u, u_j[:, 0], label="vehicle j")
    axs[2, 0].set_title("Acceleration Input")
    axs[2, 0].set_xlabel("t [s]")
    axs[2, 0].set_ylabel("a [m/s^2]")
    axs[2, 0].grid(True)
    axs[2, 0].legend()

    if len(u_i):
        axs[2, 1].plot(t_u, u_i[:, 1], label="vehicle i")
        axs[2, 1].plot(t_u, u_j[:, 1], label="vehicle j")
    axs[2, 1].set_title("Steering Rate Input")
    axs[2, 1].set_xlabel("t [s]")
    axs[2, 1].set_ylabel("delta_rate [rad/s]")
    axs[2, 1].grid(True)
    axs[2, 1].legend()

    axs[3, 0].plot(t_state, hist["lambda_i"], label="vehicle i")
    axs[3, 0].plot(t_state, hist["lambda_j"], label="vehicle j")
    axs[3, 0].set_title("CBF Lambda")
    axs[3, 0].set_xlabel("t [s]")
    axs[3, 0].set_ylabel("lambda_cbf")
    axs[3, 0].grid(True)
    axs[3, 0].legend()

    axs[3, 1].plot(t_state, hist["clearance_iv"], label="inter-vehicle")
    axs[3, 1].plot(t_state, hist["boundary_i"], label="boundary i")
    axs[3, 1].plot(t_state, hist["boundary_j"], label="boundary j")
    axs[3, 1].set_title("Clearances")
    axs[3, 1].set_xlabel("t [s]")
    axs[3, 1].set_ylabel("clearance [m]")
    axs[3, 1].grid(True)
    axs[3, 1].legend()

    axs[4, 0].plot(t_state, [centerline_deviation(s[:2], route_i) for s in x_i], label="vehicle i")
    axs[4, 0].plot(t_state, [centerline_deviation(s[:2], route_j) for s in x_j], label="vehicle j")
    axs[4, 0].set_title("Centerline Deviation")
    axs[4, 0].set_xlabel("t [s]")
    axs[4, 0].set_ylabel("error [m]")
    axs[4, 0].grid(True)
    axs[4, 0].legend()

    axs[4, 1].axis("off")

    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {filename}")
    plt.show()


def plot_multi_vehicle_results(
    hist: Dict,
    vehicle_data: Dict[str, Dict],
    filename: str = "RI_results.png",
) -> None:
    names = hist["names"]
    colors = {
        "i": "tab:blue",
        "j": "tab:orange",
        "k": "tab:green",
        "l": "tab:red",
    }
    t_state = hist["t"]
    t_u = t_state[:-1]
    geom = vehicle_data[names[0]]["route"].geom

    fig, axs = plt.subplots(5, 2, figsize=(12, 16), constrained_layout=True)

    draw_intersection(axs[0, 0], geom, xlim=(-16, 16), ylim=(-16, 16))
    for name in names:
        x_hist = hist["x"][name]
        color = colors.get(name, None)
        axs[0, 0].plot(x_hist[:, 0], x_hist[:, 1], label=f"vehicle {name}", color=color)
        axs[0, 0].scatter(x_hist[0, 0], x_hist[0, 1], marker="o", s=50, color=color)
        axs[0, 0].scatter(x_hist[-1, 0], x_hist[-1, 1], marker="x", s=65, color=color)
    axs[0, 0].set_title("Road Intersection Trajectories")
    axs[0, 0].set_xlabel("x [m]")
    axs[0, 0].set_ylabel("y [m]")
    axs[0, 0].axis("equal")
    axs[0, 0].set_xlim(-16, 16)
    axs[0, 0].set_ylim(-16, 16)
    axs[0, 0].grid(True)
    axs[0, 0].legend(fontsize=8)

    for name in names:
        x_hist = hist["x"][name]
        color = colors.get(name, None)
        axs[0, 1].plot(t_state, x_hist[:, 3], label=f"vehicle {name}", color=color)
        axs[1, 0].plot(t_state, x_hist[:, 2], label=f"vehicle {name}", color=color)
        axs[1, 1].plot(t_state, x_hist[:, 4], label=f"vehicle {name}", color=color)
    axs[0, 1].set_title("Speed")
    axs[0, 1].set_xlabel("t [s]")
    axs[0, 1].set_ylabel("v [m/s]")
    axs[1, 0].set_title("Heading")
    axs[1, 0].set_xlabel("t [s]")
    axs[1, 0].set_ylabel("psi [rad]")
    axs[1, 1].set_title("Steering Angle")
    axs[1, 1].set_xlabel("t [s]")
    axs[1, 1].set_ylabel("delta [rad]")

    for ax in [axs[0, 1], axs[1, 0], axs[1, 1]]:
        ax.grid(True)
        ax.legend(fontsize=8)

    for name in names:
        u_hist = hist["u"][name]
        color = colors.get(name, None)
        if len(u_hist):
            axs[2, 0].plot(t_u[:len(u_hist)], u_hist[:, 0], label=f"vehicle {name}", color=color)
            axs[2, 1].plot(t_u[:len(u_hist)], u_hist[:, 1], label=f"vehicle {name}", color=color)
    axs[2, 0].set_title("Acceleration Input")
    axs[2, 0].set_xlabel("t [s]")
    axs[2, 0].set_ylabel("a [m/s^2]")
    axs[2, 1].set_title("Steering Rate Input")
    axs[2, 1].set_xlabel("t [s]")
    axs[2, 1].set_ylabel("delta_rate [rad/s]")

    for ax in [axs[2, 0], axs[2, 1]]:
        ax.grid(True)
        ax.legend(fontsize=8)

    for name in names:
        color = colors.get(name, None)
        axs[3, 0].plot(t_state, hist["lambda"][name], label=f"vehicle {name}", color=color)
        axs[3, 1].plot(t_state, hist["boundary"][name], label=f"boundary {name}", color=color)
    axs[3, 0].set_title("CBF Lambda")
    axs[3, 0].set_xlabel("t [s]")
    axs[3, 0].set_ylabel("lambda_cbf")
    axs[3, 1].plot(t_state, hist["clearance_iv"], label="min inter-vehicle", color="k", linewidth=1.8)
    axs[3, 1].set_title("Clearances")
    axs[3, 1].set_xlabel("t [s]")
    axs[3, 1].set_ylabel("clearance [m]")

    for ax in [axs[3, 0], axs[3, 1]]:
        ax.grid(True)
        ax.legend(fontsize=8)

    for name in names:
        color = colors.get(name, None)
        route = vehicle_data[name]["route"]
        axs[4, 0].plot(
            t_state,
            [centerline_deviation(state[:2], route) for state in hist["x"][name]],
            label=f"vehicle {name}",
            color=color,
        )
    axs[4, 0].set_title("Centerline Deviation")
    axs[4, 0].set_xlabel("t [s]")
    axs[4, 0].set_ylabel("error [m]")
    axs[4, 0].grid(True)
    axs[4, 0].legend(fontsize=8)
    axs[4, 1].axis("off")

    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {filename}")
    plt.show()


def animate_simulation(
    hist: Dict,
    vp_i: VehicleParams,
    vp_j: VehicleParams,
    route_i: RouteSpec,
    route_j: RouteSpec,
    circle_offsets: np.ndarray,
    circle_radius: float,
    filename: str = "RI_simulation.mp4",
) -> None:
    from matplotlib import animation
    from matplotlib.patches import Circle as PatchCircle
    from matplotlib.patches import Polygon

    x_i = hist["x_i"]
    x_j = hist["x_j"]
    goal_i_hist = hist["goal_i"]
    goal_j_hist = hist["goal_j"]
    t = hist["t"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(-18, 18)
    ax.set_ylim(-18, 18)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Rounded Road Intersection CLF-CBF-QP")
    ax.grid(True)
    draw_intersection(ax, route_i.geom)

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

    def goal_for_frame(goal_hist, frame):
        if len(goal_hist) == 0:
            return np.array([np.nan, np.nan])
        if frame == 0:
            return goal_hist[0]
        return goal_hist[min(frame - 1, len(goal_hist) - 1)]

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
        goal_i_scatter.set_offsets(goal_for_frame(goal_i_hist, 0))
        goal_j_scatter.set_offsets(goal_for_frame(goal_j_hist, 0))
        time_text.set_text(f"t = {t[0]:.1f} s")
        return [
            traj_i_line, traj_j_line, poly_i, poly_j,
            heading_i_line, heading_j_line, goal_i_scatter, goal_j_scatter,
            time_text, *circle_patches_i, *circle_patches_j,
        ]

    def update(frame):
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
        goal_i_scatter.set_offsets(goal_for_frame(goal_i_hist, frame))
        goal_j_scatter.set_offsets(goal_for_frame(goal_j_hist, frame))
        time_text.set_text(f"t = {t[frame]:.1f} s")
        return [
            traj_i_line, traj_j_line, poly_i, poly_j,
            heading_i_line, heading_j_line, goal_i_scatter, goal_j_scatter,
            time_text, *circle_patches_i, *circle_patches_j,
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
    except Exception as exc:
        print(f"MP4 export failed: {exc}")
        gif_name = os.path.splitext(filename)[0] + ".gif"
        try:
            ani.save(gif_name, writer="pillow", fps=10)
            print(f"Saved animation to: {gif_name}")
        except Exception as exc2:
            print(f"GIF export also failed: {exc2}")

    plt.show()


def animate_multi_vehicle_simulation(
    hist: Dict,
    vehicle_data: Dict[str, Dict],
    circle_offsets: np.ndarray,
    circle_radius: float,
    filename: str = "RI_simulation.mp4",
) -> None:
    from matplotlib import animation
    from matplotlib.patches import Circle as PatchCircle
    from matplotlib.patches import Polygon

    names = hist["names"]
    colors = {
        "i": "tab:blue",
        "j": "tab:orange",
        "k": "tab:green",
        "l": "tab:red",
    }
    t = hist["t"]
    geom = vehicle_data[names[0]]["route"].geom

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(-18, 18)
    ax.set_ylim(-18, 18)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Rounded Road Intersection CLF-CBF-QP")
    ax.grid(True)
    draw_intersection(ax, geom)

    traj_lines = {}
    polys = {}
    circle_patches = {}
    heading_lines = {}
    goal_scatters = {}

    for name in names:
        color = colors.get(name, None)
        x_hist = hist["x"][name]
        vp = vehicle_data[name]["vp"]
        traj_lines[name], = ax.plot([], [], linewidth=1.5, color=color, label=f"vehicle {name}")
        polys[name] = Polygon(vehicle_corners(x_hist[0], vp), closed=True, fill=False, edgecolor=color, linewidth=2.0)
        ax.add_patch(polys[name])
        circle_patches[name] = []
        for _ in circle_offsets:
            patch = PatchCircle((0.0, 0.0), circle_radius, fill=False, edgecolor=color, linestyle="--", alpha=0.8)
            ax.add_patch(patch)
            circle_patches[name].append(patch)
        heading_lines[name], = ax.plot([], [], color=color, linewidth=2.0)
        goal_scatters[name] = ax.scatter([], [], marker="x", s=80, color=color, label=f"goal {name}")

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    lambda_text = ax.text(
        0.02,
        0.92,
        "",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    ax.legend(loc="upper right")

    def set_heading_line(line_obj, state, length=1.8):
        px, py, psi, _, _ = state
        ex = length * np.cos(psi)
        ey = length * np.sin(psi)
        line_obj.set_data([px, px + ex], [py, py + ey])

    def goal_for_frame(goal_hist, frame):
        if len(goal_hist) == 0:
            return np.array([np.nan, np.nan])
        if frame == 0:
            return goal_hist[0]
        return goal_hist[min(frame - 1, len(goal_hist) - 1)]

    def lambda_text_for_frame(frame):
        lines = []
        for name in names:
            lam_hist = hist["lambda"][name]
            idx = min(frame, len(lam_hist) - 1)
            lines.append(f"lambda_{name} = {lam_hist[idx]:.3f}")
        return "\n".join(lines)

    def init():
        artists = [time_text, lambda_text]
        for name in names:
            x_hist = hist["x"][name]
            vp = vehicle_data[name]["vp"]
            traj_lines[name].set_data([], [])
            polys[name].set_xy(vehicle_corners(x_hist[0], vp))
            for idx, s in enumerate(circle_offsets):
                c, _, _, _ = circle_center_and_kinematics_affine(x_hist[0], s, vp)
                circle_patches[name][idx].center = (c[0], c[1])
            set_heading_line(heading_lines[name], x_hist[0])
            goal_scatters[name].set_offsets(goal_for_frame(hist["goal"][name], 0))
            artists.extend([traj_lines[name], polys[name], heading_lines[name], goal_scatters[name]])
            artists.extend(circle_patches[name])
        time_text.set_text(f"t = {t[0]:.1f} s")
        lambda_text.set_text(lambda_text_for_frame(0))
        return artists

    def update(frame):
        artists = [time_text, lambda_text]
        for name in names:
            x_hist = hist["x"][name]
            vp = vehicle_data[name]["vp"]
            traj_lines[name].set_data(x_hist[:frame + 1, 0], x_hist[:frame + 1, 1])
            polys[name].set_xy(vehicle_corners(x_hist[frame], vp))
            for idx, s in enumerate(circle_offsets):
                c, _, _, _ = circle_center_and_kinematics_affine(x_hist[frame], s, vp)
                circle_patches[name][idx].center = (c[0], c[1])
            set_heading_line(heading_lines[name], x_hist[frame])
            goal_scatters[name].set_offsets(goal_for_frame(hist["goal"][name], frame))
            artists.extend([traj_lines[name], polys[name], heading_lines[name], goal_scatters[name]])
            artists.extend(circle_patches[name])
        time_text.set_text(f"t = {t[frame]:.1f} s")
        lambda_text.set_text(lambda_text_for_frame(frame))
        return artists

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
    except Exception as exc:
        print(f"MP4 export failed: {exc}")
        gif_name = os.path.splitext(filename)[0] + ".gif"
        try:
            ani.save(gif_name, writer="pillow", fps=10)
            print(f"Saved animation to: {gif_name}")
        except Exception as exc2:
            print(f"GIF export also failed: {exc2}")

    plt.show()


# =========================
# Baseline methods runner
# =========================

BASELINE_VEHICLE_NAMES = ("i", "j", "k")
BASELINE_START_BY_NAME = {"i": 1, "j": 2, "k": 3}


@dataclass
class BaselineConfig:
    # Method 1: all vehicles use lambda_uniform.
    lambda_uniform: float = 0.232

    # Method 2: fixed heterogeneous lambda values.
    lambda_i: float = 0.194
    lambda_j: float = 0.4
    lambda_k: float = 0.4

    # Method 3: lambda_vehicle = clip(gain_vehicle * adaptive_clearance, lower, upper).
    gain_i: float = 0.2
    gain_j: float = 0.01
    gain_k: float = 0.6
    lambda_lower: float = 0.1
    lambda_upper: float = 0.4

    # Method 4: aTTCBF adaptive eta from the QP.
    eta_weight: float = 1e7


def adaptive_lambda_from_clearance(
    clearance: float,
    clearance_gain: float,
    lambda_lower_bound: float,
    lambda_upper_bound: float,
) -> float:
    return float(np.clip(clearance_gain * clearance, lambda_lower_bound, lambda_upper_bound))


def sample_baseline_routes(rng: np.random.Generator, geom: IntersectionGeometry) -> Dict[str, RouteSpec]:
    routes = {}
    for name in BASELINE_VEHICLE_NAMES:
        start = BASELINE_START_BY_NAME[name]
        possible_exits = [lane for lane in (1, 2, 3, 4) if lane != start]
        routes[name] = RouteSpec(start=start, exit=int(rng.choice(possible_exits)), geom=geom)
    return routes


def make_baseline_vehicle_data(
    routes: Dict[str, RouteSpec],
    cfg: BaselineConfig,
    initial_lambdas: Dict[str, float],
) -> Dict[str, Dict]:
    vehicle_data = {}
    for name in BASELINE_VEHICLE_NAMES:
        clfp = CLFParams()
        vehicle_data[name] = {
            "route": routes[name],
            "vp": VehicleParams(),
            "clfp": clfp,
            "cbfp": CBFParams(lambda_cbf=initial_lambdas[name]),
            "qpw": QPWeights(),
            "speed": clfp.desired_speed,
        }
    return vehicle_data


def run_baseline_monte_carlo(
    cfg: BaselineConfig,
    method: str = "2",
    n_rollouts: int = 10,
    seed: int = 7,
) -> Tuple[Dict, Dict, Tuple]:
    names = list(BASELINE_VEHICLE_NAMES)

    lambda_constant_same = cfg.lambda_uniform
    lambda_constant_i = cfg.lambda_i
    lambda_constant_j = cfg.lambda_j
    lambda_constant_k = cfg.lambda_k

    gain_i = cfg.gain_i
    gain_j = cfg.gain_j
    gain_k = cfg.gain_k
    eta_weight = cfg.eta_weight

    clearance_gains = {"i": gain_i, "j": gain_j, "k": gain_k}
    lambda_lower = {name: cfg.lambda_lower for name in names}
    lambda_upper = {name: cfg.lambda_upper for name in names}

    geom = IntersectionGeometry()
    sim = SimParams()
    rng = np.random.default_rng(seed)

    all_metrics = []
    last_hist = None
    last_bundle = None

    for rollout_idx in range(1, n_rollouts + 1):
        routes = sample_baseline_routes(rng, geom)
        if method == "1":
            initial_lambdas = {name: lambda_constant_same for name in names}
        elif method == "2":
            initial_lambdas = {
                "i": lambda_constant_i,
                "j": lambda_constant_j,
                "k": lambda_constant_k,
            }
        elif method == "3":
            initial_lambdas = {name: lambda_lower[name] for name in names}
        elif method == "4":
            initial_lambdas = {name: 0.0 for name in names}
        else:
            raise ValueError("Unknown baseline method.")

        vehicle_data = make_baseline_vehicle_data(routes, cfg, initial_lambdas)
        if method == "4":
            for name in names:
                vehicle_data[name]["qpw"].eta_weight = eta_weight

        print(f"\n=== Rollout {rollout_idx}/{n_rollouts} ===")
        for name in names:
            route = routes[name]
            print(
                f"vehicle {name}: start {route.start}, exit {route.exit}, "
                f"lambda_cbf = {vehicle_data[name]['cbfp'].lambda_cbf:.3f}"
            )

        def update_fn(_, data, states, offsets, radius):
            def pair_clear(a: str, b: str) -> float:
                return min_intervehicle_clearance(
                    states[a], states[b],
                    data[a]["vp"], data[b]["vp"],
                    offsets, radius,
                )

            clearance_ij = pair_clear("i", "j")
            clearance_ik = pair_clear("i", "k")
            clearance_jk = pair_clear("j", "k")

            def adaptive_clearance_for_vehicle(name: str) -> float:
                return clearance_ij if name in ("i", "j") else min(clearance_ik, clearance_jk)
                # return clearance_ik if name in ("i", "k") else min(clearance_ij, clearance_jk)
                # return clearance_jk if name in ("j", "k") else min(clearance_ij, clearance_ik)

            for name in names:
                data[name]["cbfp"].lambda_cbf = adaptive_lambda_from_clearance(
                    clearance=adaptive_clearance_for_vehicle(name),
                    clearance_gain=clearance_gains[name],
                    lambda_lower_bound=lambda_lower[name],
                    lambda_upper_bound=lambda_upper[name],
                )

        hist, metrics, bundle = run_multi_vehicle_rollout(
            vehicle_data=vehicle_data,
            sim=sim,
            lambda_update_fn=update_fn if method == "3" else None,
            use_attcbf=method == "4",
        )
        metrics["episode_return"] = float("nan")
        all_metrics.append(metrics)
        last_hist = hist
        last_bundle = bundle

    avg_metrics = average_rollout_metrics(all_metrics)
    return avg_metrics, last_hist, last_bundle


def main() -> None:
    cfg = BaselineConfig()
    parser = argparse.ArgumentParser(description="Three-vehicle RI baseline-method rollouts.")
    parser.add_argument("--method", choices=["1", "2", "3", "4"], default=None)
    parser.add_argument("--n_rollouts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip_animation", action="store_true")
    args = parser.parse_args()

    if args.method is None:
        print("Select lambda_cbf baseline method:")
        print("  1: Uniform fixed lambda")
        print("  2: Heterogeneous fixed lambdas")
        print("  3: Heuristic lambda = clip(gain * nearest_clearance, lower, upper)")
        print("  4: aTTCBF adaptive eta from QP")
        method = input("Enter 1, 2, 3, or 4 [default: 2]: ").strip() or "2"
        if method not in {"1", "2", "3", "4"}:
            method = "2"
    else:
        method = args.method

    print("\nRoad Intersection baseline settings")
    print(f"  method = {method}")
    print(f"  n_rollouts = {args.n_rollouts}, seed = {args.seed}")
    if method == "1":
        print(f"  uniform lambda_cbf = {cfg.lambda_uniform:.3f}")
    elif method == "2":
        print(
            "  heterogeneous lambdas: "
            f"i={cfg.lambda_i:.3f}, j={cfg.lambda_j:.3f}, k={cfg.lambda_k:.3f}"
        )
    elif method == "3":
        print(
            "  heuristic gains: "
            f"i={cfg.gain_i:.3f}, j={cfg.gain_j:.3f}, k={cfg.gain_k:.3f}; "
            f"lambda bounds = [{cfg.lambda_lower:.3f}, {cfg.lambda_upper:.3f}]"
        )
    else:
        print(f"  aTTCBF eta_weight = {cfg.eta_weight:.3g}")

    avg_metrics, last_hist, last_bundle = run_baseline_monte_carlo(
        cfg,
        method=method,
        n_rollouts=args.n_rollouts,
        seed=args.seed,
    )
    show_average_metrics_table(avg_metrics, n_rollouts=args.n_rollouts)

    _, vehicle_data, circle_offsets, circle_radius = last_bundle
    video_filename = "RI_baseline_simulation.mp4"

    if not args.skip_animation:
        animate_multi_vehicle_simulation(
            last_hist,
            vehicle_data,
            circle_offsets,
            circle_radius,
            filename=video_filename,
        )


if __name__ == "__main__":
    main()
