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
    return np.array([math.cos(psi), math.sin(psi)], dtype=float)


def unit_lateral(psi: float) -> np.ndarray:
    return np.array([-math.sin(psi), math.cos(psi)], dtype=float)


def deg2rad(deg: float) -> float:
    return deg * np.pi / 180.0


def adaptive_lambda_from_clearance(
    clearance: float,
    clearance_gain: float,
    lambda_lower_bound: float,
    lambda_upper_bound: float,
) -> float:
    lam = clearance_gain * clearance
    return float(np.clip(lam, lambda_lower_bound, lambda_upper_bound))


def lane_tangent(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)], dtype=float)


def lane_normal(theta: float) -> np.ndarray:
    return np.array([-math.sin(theta), math.cos(theta)], dtype=float)


def signed_line_coordinate(point: np.ndarray, pref: np.ndarray, vec: np.ndarray) -> float:
    return float((point - pref) @ vec)


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
    w_v: float = 0.5
    w_delta: float = 15.0
    desired_speed: float = 5.0
    nominal_speed_gain: float = 1.0
    nominal_heading_gain: float = 1.0
    nominal_steer_rate_gain: float = 5.0
    clf_rate: float = 1.0
    clf_slack_weight: float = 1.0


@dataclass
class CBFParams:
    lambda_cbf: float = 0.2
    cbf_slack_weight: float = 1e8
    eps_dist: float = 1e-6


@dataclass
class QPWeights:
    u_weight: np.ndarray = field(default_factory=lambda: np.diag([0.5, 50.0]))


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
    y_min: float = -0.25
    y_max: float = 0.25
    psi_min_deg: float = -10.0
    psi_max_deg: float = 10.0
    seed: int = 7
    s0: float = -15.0
    goal_x: float = 5.0


@dataclass
class MergeGeometry:
    lane_half_width: float = 2.0
    theta_upper_deg: float = -30.0
    theta_lower_deg: float = 30.0
    merge_point: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0], dtype=float))
    lookahead_dist: float = 5.0
    angled_lookahead_lateral_offset: float = -0.75
    entry_length: float = 30.0

    def __post_init__(self):
        self.theta_upper = deg2rad(self.theta_upper_deg)
        self.theta_lower = deg2rad(self.theta_lower_deg)
        self.theta_mid = 0.0

        self.t_upper = lane_tangent(self.theta_upper)
        self.n_upper = lane_normal(self.theta_upper)
        self.t_lower = lane_tangent(self.theta_lower)
        self.n_lower = lane_normal(self.theta_lower)
        self.t_mid = lane_tangent(self.theta_mid)
        self.n_mid = lane_normal(self.theta_mid)

        self.pref_upper = self.merge_point.copy()
        self.pref_lower = self.merge_point.copy()
        self.pref_mid = self.merge_point.copy()

        w = self.lane_half_width

        self.upper_start_center = self.pref_upper + (-self.entry_length) * self.t_upper
        self.lower_start_center = self.pref_lower + (-self.entry_length) * self.t_lower
        self.mid_start_center = self.pref_mid + (-self.entry_length) * self.t_mid

        # The reference paths meet at merge_point. The lane boundaries are the
        # straight middle corridor rotated about that same point, so they stay
        # parallel to each lane centerline instead of collapsing at the origin.
        self.upper_boundary_plus = self.pref_upper + w * self.n_upper
        self.upper_boundary_minus = self.pref_upper - w * self.n_upper
        self.lower_boundary_plus = self.pref_lower + w * self.n_lower
        self.lower_boundary_minus = self.pref_lower - w * self.n_lower
        self.mid_boundary_top = self.pref_mid + w * self.n_mid
        self.mid_boundary_bottom = self.pref_mid - w * self.n_mid

        # Opening corners from exact intersections of the rotated lane
        # boundaries with the straight middle-lane boundaries.
        m = math.tan(abs(self.theta_upper))
        b = w / math.cos(abs(self.theta_upper))

        self.x_open_left = (-w - b) / m
        self.x_open_right = (-w + b) / m

        self.left_top_corner = np.array([self.x_open_left, w], dtype=float)
        self.left_bottom_corner = np.array([self.x_open_left, -w], dtype=float)
        self.right_top_corner = np.array([self.x_open_right, w], dtype=float)
        self.right_bottom_corner = np.array([self.x_open_right, -w], dtype=float)

        self.upper_open = self.left_top_corner.copy()
        self.lower_open = self.left_bottom_corner.copy()

        self.x_switch = 0.5 * (self.x_open_left + self.x_open_right)


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
# Lane / route helpers
# =========================

def upper_has_entered_middle_corridor(point: np.ndarray, geom: MergeGeometry) -> bool:
    return bool(point[0] > geom.x_switch and point[1] < geom.lane_half_width)


def lower_has_entered_middle_corridor(point: np.ndarray, geom: MergeGeometry) -> bool:
    return bool(point[0] > geom.x_switch and point[1] > -geom.lane_half_width)


def has_crossed_boundary_corner(point: np.ndarray, corner: np.ndarray, tvec: np.ndarray) -> bool:
    return bool((point - corner) @ tvec >= 0.0)


def active_lane_frame_for_vehicle(vehicle_name: str, circle_center: np.ndarray, geom: MergeGeometry):
    if vehicle_name == "upper":
        if upper_has_entered_middle_corridor(circle_center, geom):
            return geom.pref_mid, geom.t_mid, geom.n_mid, "middle_horizontal"
        return geom.pref_upper, geom.t_upper, geom.n_upper, "upper_angled"
    elif vehicle_name == "lower":
        if lower_has_entered_middle_corridor(circle_center, geom):
            return geom.pref_mid, geom.t_mid, geom.n_mid, "middle_horizontal"
        return geom.pref_lower, geom.t_lower, geom.n_lower, "lower_angled"
    elif vehicle_name == "mid":
        return geom.pref_mid, geom.t_mid, geom.n_mid, "middle_horizontal"
    else:
        raise ValueError("Unknown vehicle_name")


def active_boundary_frame_for_vehicle(
    vehicle_name: str,
    circle_center: np.ndarray,
    geom: MergeGeometry,
    boundary_side: str,
):
    if vehicle_name == "mid":
        return geom.pref_mid, geom.t_mid, geom.n_mid, "middle_horizontal"

    if vehicle_name == "upper":
        if boundary_side == "left":
            if has_crossed_boundary_corner(circle_center, geom.right_top_corner, geom.t_upper):
                return geom.pref_mid, geom.t_mid, geom.n_mid, "middle_horizontal"
            return geom.pref_upper, geom.t_upper, geom.n_upper, "upper_angled"
        if circle_center[0] < geom.x_switch:
            return geom.pref_upper, geom.t_upper, geom.n_upper, "upper_angled"
        return geom.pref_mid, geom.t_mid, geom.n_mid, "middle_horizontal"

    if vehicle_name == "lower":
        if boundary_side == "right":
            if has_crossed_boundary_corner(circle_center, geom.right_bottom_corner, geom.t_lower):
                return geom.pref_mid, geom.t_mid, geom.n_mid, "middle_horizontal"
            return geom.pref_lower, geom.t_lower, geom.n_lower, "lower_angled"
        if circle_center[0] < geom.x_switch:
            return geom.pref_lower, geom.t_lower, geom.n_lower, "lower_angled"
        return geom.pref_mid, geom.t_mid, geom.n_mid, "middle_horizontal"

    raise ValueError("Unknown vehicle_name")


def lookahead_goal_from_track_point(
    track_point: np.ndarray,
    vehicle_name: str,
    geom: MergeGeometry,
) -> np.ndarray:
    pref, t, n, phase = active_lane_frame_for_vehicle(vehicle_name, track_point, geom)
    s = signed_line_coordinate(track_point, pref, t)
    lateral_offset = geom.angled_lookahead_lateral_offset if phase.endswith("_angled") else 0.0
    if phase == "lower_angled":
        lateral_offset *= -1.0
    return pref + (s + geom.lookahead_dist) * t + lateral_offset * n


def front_circle_offset(vp: VehicleParams) -> float:
    return vp.length / 3.0


def front_circle_center(state: np.ndarray, vp: VehicleParams) -> np.ndarray:
    c, _, _, _ = circle_center_and_kinematics_affine(state, front_circle_offset(vp), vp)
    return c


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

    q = max(np.linalg.norm(d), cbfp.eps_dist)
    n = d / q
    R = 2.0 * circle_radius

    h = q - R
    h_dot = float(n @ d_dot)

    tangential_term = (np.dot(d_dot, d_dot) - (np.dot(n, d_dot)) ** 2) / q

    hddot_const = float(n @ (c1_ddot_0 - c2_ddot_0) + tangential_term)
    hddot_u = np.array([n @ B1[:, 0], n @ B1[:, 1]], dtype=float)

    return h, h_dot, np.array([hddot_const, *hddot_u], dtype=float)


def active_lane_boundary_cbf_affine(
    ego_state: np.ndarray,
    ego_s: float,
    ego_vp: VehicleParams,
    geom: MergeGeometry,
    vehicle_name: str,
    boundary_side: str,
    circle_radius: float,
) -> Tuple[float, float, np.ndarray]:
    c, c_dot, c_ddot_0, B = circle_center_and_kinematics_affine(ego_state, ego_s, ego_vp)
    pref, _, nvec, _ = active_boundary_frame_for_vehicle(vehicle_name, c, geom, boundary_side)

    e = float((c - pref) @ nvec)
    e_dot = float(nvec @ c_dot)
    e_ddot_const = float(nvec @ c_ddot_0)
    e_ddot_u = np.array([nvec @ B[:, 0], nvec @ B[:, 1]], dtype=float)

    half_w = geom.lane_half_width

    if boundary_side == "left":
        h = half_w - circle_radius - e
        h_dot = -e_dot
        hddot_const = -e_ddot_const
        hddot_u = -e_ddot_u
    elif boundary_side == "right":
        h = half_w - circle_radius + e
        h_dot = e_dot
        hddot_const = e_ddot_const
        hddot_u = e_ddot_u
    else:
        raise ValueError("boundary_side must be 'left' or 'right'.")

    return float(h), float(h_dot), np.array([hddot_const, hddot_u[0], hddot_u[1]], dtype=float)


# =========================
# Decentralized QP for one vehicle
# =========================

def solve_vehicle_qp(
    ego_state: np.ndarray,
    other_states: List[np.ndarray],
    ego_name: str,
    ego_vp: VehicleParams,
    other_vps: List[VehicleParams],
    clfp: CLFParams,
    cbfp: CBFParams,
    qpw: QPWeights,
    sim: SimParams,
    geom: MergeGeometry,
    circle_offsets: np.ndarray,
    circle_radius: float,
    goal_point: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    u_nom = nominal_control(ego_state, clfp, ego_vp, goal_point)

    n_pairwise = len(other_states) * len(circle_offsets) * len(circle_offsets)
    n_boundary = 2 * len(circle_offsets)
    n_cbf = n_pairwise + n_boundary

    u = cp.Variable(2)
    s_cbf = cp.Variable(n_cbf, nonneg=True)
    s_clf = cp.Variable(1, nonneg=True)

    constraints = []
    constraints += [
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

    V, Vdot_const, Vdot_u = clf_terms(ego_state, ego_vp, clfp, goal_point)
    constraints += [Vdot_const + Vdot_u @ u <= -clfp.clf_rate * V + s_clf[0]]

    Dt = sim.Dt
    lambda_cbf = cbfp.lambda_cbf
    cbf_index = 0

    # Pairwise inter-vehicle constraints against every other vehicle.
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
                    + 0.5 * lambda_cbf * h
                    + s_cbf[cbf_index] >= 0.0
                ]
                cbf_index += 1

    for boundary_side in ["left", "right"]:
        for s_i in circle_offsets:
            h, h_dot, hddot_aff = active_lane_boundary_cbf_affine(
                ego_state=ego_state,
                ego_s=s_i,
                ego_vp=ego_vp,
                geom=geom,
                vehicle_name=ego_name,
                boundary_side=boundary_side,
                circle_radius=circle_radius,
            )
            hddot_const = hddot_aff[0]
            hddot_u = hddot_aff[1:]
            constraints += [
                0.5 * Dt**2 * (hddot_const + hddot_u @ u)
                + Dt * h_dot
                + lambda_cbf * h
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
        "fallback": False,
    }

    solver_exception = None

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
        try:
            prob.solve(warm_start=True, verbose=False)
        except Exception as exc:
            solver_exception = exc
            pass

    qp_info["status"] = prob.status
    if qp_info["status"] is None and solver_exception is not None:
        qp_info["status"] = f"exception:{type(solver_exception).__name__}"
        qp_info["exception"] = str(solver_exception)

    if u.value is None or prob.status not in ["optimal", "optimal_inaccurate"]:
        u_sol = np.array([
            np.clip(u_nom[0], ego_vp.min_accel, ego_vp.max_accel),
            np.clip(u_nom[1], ego_vp.min_steer_rate, ego_vp.max_steer_rate),
        ], dtype=float)
        qp_info["fallback"] = True
    else:
        u_sol = np.array(u.value).reshape(2)

    return u_sol, qp_info


# =========================
# Safety / metrics helpers
# =========================

def get_vehicle_circle_centers(state: np.ndarray, vp: VehicleParams, circle_offsets: np.ndarray) -> List[np.ndarray]:
    centers = []
    for s in circle_offsets:
        c, _, _, _ = circle_center_and_kinematics_affine(state, s, vp)
        centers.append(c)
    return centers


def min_pair_clearance(
    x_a: np.ndarray,
    x_b: np.ndarray,
    vp_a: VehicleParams,
    vp_b: VehicleParams,
    circle_offsets: np.ndarray,
    circle_radius: float,
) -> float:
    centers_a = get_vehicle_circle_centers(x_a, vp_a, circle_offsets)
    centers_b = get_vehicle_circle_centers(x_b, vp_b, circle_offsets)
    min_clear = np.inf
    for ca in centers_a:
        for cb in centers_b:
            d = np.linalg.norm(ca - cb) - 2.0 * circle_radius
            min_clear = min(min_clear, d)
    return float(min_clear)


def min_vehicle_clearance_to_others(
    ego_name: str,
    states: Dict[str, np.ndarray],
    vps: Dict[str, VehicleParams],
    circle_offsets: np.ndarray,
    circle_radius: float,
) -> float:
    names = [n for n in states if n != ego_name]
    vals = [min_pair_clearance(states[ego_name], states[n], vps[ego_name], vps[n], circle_offsets, circle_radius) for n in names]
    return float(min(vals))


def pairwise_vehicle_clearances(
    states: Dict[str, np.ndarray],
    vps: Dict[str, VehicleParams],
    circle_offsets: np.ndarray,
    circle_radius: float,
) -> Dict[str, float]:
    return {
        "upper_lower": min_pair_clearance(
            states["upper"], states["lower"], vps["upper"], vps["lower"], circle_offsets, circle_radius
        ),
        "mid_upper": min_pair_clearance(
            states["mid"], states["upper"], vps["mid"], vps["upper"], circle_offsets, circle_radius
        ),
        "mid_lower": min_pair_clearance(
            states["mid"], states["lower"], vps["mid"], vps["lower"], circle_offsets, circle_radius
        ),
    }


def vehicle_min_clearances_from_pairwise(pairwise_clearances: Dict[str, float]) -> Dict[str, float]:
    return {
        "upper": min(pairwise_clearances["upper_lower"], pairwise_clearances["mid_upper"]),
        "mid": min(pairwise_clearances["mid_upper"], pairwise_clearances["mid_lower"]),
        "lower": min(pairwise_clearances["upper_lower"], pairwise_clearances["mid_lower"]),
    }


def min_boundary_clearance(
    state: np.ndarray,
    vehicle_name: str,
    geom: MergeGeometry,
    vp: VehicleParams,
    circle_offsets: np.ndarray,
    circle_radius: float,
) -> float:
    centers = get_vehicle_circle_centers(state, vp, circle_offsets)
    min_clear = np.inf
    for c in centers:
        pref, _, nvec, _ = active_boundary_frame_for_vehicle(vehicle_name, c, geom, "left")
        e = float((c - pref) @ nvec)
        left_clear = geom.lane_half_width - circle_radius - e

        pref, _, nvec, _ = active_boundary_frame_for_vehicle(vehicle_name, c, geom, "right")
        e = float((c - pref) @ nvec)
        right_clear = geom.lane_half_width - circle_radius + e

        min_clear = min(min_clear, left_clear, right_clear)
    return float(min_clear)


def compute_time_to_goal_x(hist_x: np.ndarray, t: np.ndarray, x_goal: float = 5.0) -> float:
    idx = np.where(hist_x >= x_goal)[0]
    if len(idx) == 0:
        return np.nan
    return float(t[idx[0]])


def avg_abs_y_for_metrics(vehicle_name: str, x_hist: np.ndarray, geom: MergeGeometry) -> float:
    if vehicle_name == "upper":
        angled_hist = x_hist[x_hist[:, 0] <= geom.x_switch]
        if angled_hist.size == 0:
            return float("nan")
        lateral = (angled_hist[:, :2] - geom.pref_upper) @ geom.n_upper
        return float(np.mean(np.abs(lateral)))

    if vehicle_name == "lower":
        angled_hist = x_hist[x_hist[:, 0] <= geom.x_switch]
        if angled_hist.size == 0:
            return float("nan")
        lateral = (angled_hist[:, :2] - geom.pref_lower) @ geom.n_lower
        return float(np.mean(np.abs(lateral)))

    return float(np.mean(np.abs(x_hist[:, 1])))


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
    init_states: Dict[str, np.ndarray],
    cbfp_schedule: str = "constant_same",
    goal_x: float = 5.0,
):
    sim = SimParams(dt=0.1, T=10.0)
    geom = MergeGeometry()

    names = ["upper", "mid", "lower"]

    vps = {name: VehicleParams() for name in names}
    clfps = {name: CLFParams() for name in names}
    qpws = {name: QPWeights() for name in names}

    lambda_constant_aggresive = 0.4
    lambda_constant_normal = 0.15
    lambda_constant_conservative = 0.1

    clearance_gains = {"upper": 0.2 / 4, "mid": 0.4, "lower": 0.2 / 4}
    lambda_lower = {name: 0.1 for name in names}
    lambda_upper = {name: 0.4 for name in names}

    circle_offsets, circle_radius = circle_approximation(vps["upper"].length, vps["upper"].width, n_circles=3)

    states = {name: init_states[name].copy() for name in names}

    current_clearances = pairwise_vehicle_clearances(states, vps, circle_offsets, circle_radius)
    clearance_upper_lower = current_clearances["upper_lower"]
    clearance_mid_upper = current_clearances["mid_upper"]
    clearance_mid_lower = current_clearances["mid_lower"]

    cbfps = {}
    if cbfp_schedule == "constant_same":
        for name in names:
            cbfps[name] = CBFParams(lambda_cbf=0.2, cbf_slack_weight=5e7)
    elif cbfp_schedule == "constant_different":
        cbfps["upper"] = CBFParams(lambda_cbf=lambda_constant_normal, cbf_slack_weight=5e7)
        cbfps["mid"] = CBFParams(lambda_cbf=lambda_constant_aggresive, cbf_slack_weight=5e7)
        cbfps["lower"] = CBFParams(lambda_cbf=lambda_constant_conservative, cbf_slack_weight=5e7)
    elif cbfp_schedule == "adaptive":
        for name in names:
            cbfps[name] = CBFParams(
                lambda_cbf=adaptive_lambda_from_clearance(
                    clearance=clearance_mid_upper if (name == "upper" or name == "mid") else min(clearance_mid_lower, clearance_upper_lower),
                    clearance_gain=clearance_gains[name],
                    lambda_lower_bound=lambda_lower[name],
                    lambda_upper_bound=lambda_upper[name],
                ),
                cbf_slack_weight=1e7,
            )
    else:
        raise ValueError("Unknown cbfp_schedule.")

    hist = {"t": [0.0], "x_switch": geom.x_switch, "events": []}
    for name in names:
        hist[f"x_{name}"] = [states[name].copy()]
        hist[f"u_{name}"] = []
        hist[f"qp_{name}"] = []
        hist[f"goal_{name}"] = []
        hist[f"epsi_{name}"] = []
        hist[f"lambda_{name}"] = [cbfps[name].lambda_cbf]
    for pair_name, clearance in current_clearances.items():
        hist[f"clearance_{pair_name}"] = [clearance]

    def print_step_details(t_now, pairwise_clearances, boundary_clearances):
        msg = (
           # f"t = {t_now:.1f} s | x_switch = {geom.x_switch:.3f} | "
            #f"clr_ul = {pairwise_clearances['upper_lower']:.3f} | "
            #f"clr_mu = {pairwise_clearances['mid_upper']:.3f} | "
            #f"clr_ml = {pairwise_clearances['mid_lower']:.3f} | "
            f"bclr_u = {boundary_clearances['upper']:.3f} | "
            f"bclr_m = {boundary_clearances['mid']:.3f} | "
            f"bclr_l = {boundary_clearances['lower']:.3f} | "
            f"lam_u = {cbfps['upper'].lambda_cbf:.3f} | "
            f"lam_m = {cbfps['mid'].lambda_cbf:.3f} | "
            f"lam_l = {cbfps['lower'].lambda_cbf:.3f}"
        )
        print(msg, flush=True)

    for k in range(sim.steps):
        current_clearances = pairwise_vehicle_clearances(states, vps, circle_offsets, circle_radius)
        clearance_upper_lower = current_clearances["upper_lower"]
        clearance_mid_upper = current_clearances["mid_upper"]
        clearance_mid_lower = current_clearances["mid_lower"]
        boundary_clearances = {
            name: min_boundary_clearance(states[name], name, geom, vps[name], circle_offsets, circle_radius)
            for name in names
        }

        if min(current_clearances.values()) < -1e-3:
            hist["events"].append("collision")
            print_step_details(k * sim.dt, current_clearances, boundary_clearances)
            print(f"  terminating at t = {k * sim.dt:.1f} s: inter-vehicle collision")
            break
        if min(boundary_clearances.values()) < -1e-3:
            hist["events"].append("boundary_collision")
            print_step_details(k * sim.dt, current_clearances, boundary_clearances)
            print(f"  terminating at t = {k * sim.dt:.1f} s: boundary collision")
            break

        if cbfp_schedule == "adaptive":
            for name in names:
                if name == "upper" or name == "mid":
                    clearance = clearance_mid_upper   
                else:
                    clearance = min(clearance_mid_lower, clearance_upper_lower)
                cbfps[name].lambda_cbf = adaptive_lambda_from_clearance(
                    clearance,
                    clearance_gain=clearance_gains[name],
                    lambda_lower_bound=lambda_lower[name],
                    lambda_upper_bound=lambda_upper[name],
                )

        goals = {}
        controls = {}
        qp_infos = {}

        for name in names:
            fc = front_circle_center(states[name], vps[name])
            goals[name] = lookahead_goal_from_track_point(fc, name, geom)

            other_names = [n for n in names if n != name]
            try:
                controls[name], qp_infos[name] = solve_vehicle_qp(
                    ego_state=states[name].copy(),
                    other_states=[states[n].copy() for n in other_names],
                    ego_name=name,
                    ego_vp=vps[name],
                    other_vps=[vps[n] for n in other_names],
                    clfp=clfps[name],
                    cbfp=cbfps[name],
                    qpw=qpws[name],
                    sim=sim,
                    geom=geom,
                    circle_offsets=circle_offsets,
                    circle_radius=circle_radius,
                    goal_point=goals[name],
                )
            except Exception as exc:
                u_nom = nominal_control(states[name], clfps[name], vps[name], goals[name])
                controls[name] = np.array([
                    np.clip(u_nom[0], vps[name].min_accel, vps[name].max_accel),
                    np.clip(u_nom[1], vps[name].min_steer_rate, vps[name].max_steer_rate),
                ], dtype=float)
                qp_infos[name] = {
                    "status": f"exception:{type(exc).__name__}",
                    "u_nom": u_nom.copy(),
                    "clf_V": np.nan,
                    "fallback": True,
                    "exception": str(exc),
                }

        infeasible_names = [
            name for name in names
            if qp_infos[name].get("status") not in ["optimal", "optimal_inaccurate"]
        ]

        if infeasible_names:
            status_msg = ", ".join(
                f"{name}={qp_infos[name].get('status')}" for name in infeasible_names
            )
            print_step_details(k * sim.dt, current_clearances, boundary_clearances)
            print(f"  terminating rollout at t = {k * sim.dt:.1f} s: infeasible QP ({status_msg})")
            print("  moving to next rollout")

            for name in names:
                hist[f"x_{name}"].append(states[name].copy())
                hist[f"u_{name}"].append(controls[name].copy())
                hist[f"qp_{name}"].append(qp_infos[name])
                hist[f"goal_{name}"].append(goals[name].copy())
                hist[f"lambda_{name}"].append(cbfps[name].lambda_cbf)
            for pair_name, clearance in current_clearances.items():
                hist[f"clearance_{pair_name}"].append(clearance)
            hist["events"].append("infeasible_qp")
            hist["t"].append((k + 1) * sim.dt)
            break

        for name in names:
            _, epsi = desired_steer_from_heading_error(
                state=states[name],
                goal_point=goals[name],
                vp=vps[name],
                clfp=clfps[name],
            )
            hist[f"epsi_{name}"].append(epsi)

        print_step_details(k * sim.dt, current_clearances, boundary_clearances)

        for name in names:
            states[name] = step_vehicle(states[name], controls[name], vps[name], sim.dt)

        next_clearances = pairwise_vehicle_clearances(states, vps, circle_offsets, circle_radius)
        next_boundary_clearances = {
            name: min_boundary_clearance(states[name], name, geom, vps[name], circle_offsets, circle_radius)
            for name in names
        }

        if min(next_clearances.values()) < -1e-3:
            event = "collision"
        elif min(next_boundary_clearances.values()) < -1e-3:
            event = "boundary_collision"
        else:
            event = "running"

        for name in names:
            hist[f"x_{name}"].append(states[name].copy())
            hist[f"u_{name}"].append(controls[name].copy())
            hist[f"qp_{name}"].append(qp_infos[name])
            hist[f"goal_{name}"].append(goals[name].copy())
            hist[f"lambda_{name}"].append(cbfps[name].lambda_cbf)
        for pair_name, clearance in next_clearances.items():
            hist[f"clearance_{pair_name}"].append(clearance)
        hist["events"].append(event)

        hist["t"].append((k + 1) * sim.dt)

        if event != "running":
            print_step_details((k + 1) * sim.dt, next_clearances, next_boundary_clearances)
            print(f"  terminating at t = {(k + 1) * sim.dt:.1f} s: {event.replace('_', ' ')}")
            break

    for key in list(hist.keys()):
        if key.startswith("x_") or key.startswith("u_") or key.startswith("goal_") or key.startswith("epsi_") or key.startswith("lambda_") or key.startswith("clearance_") or key == "t":
            hist[key] = np.array(hist[key])
    for name in names:
        if hist[f"u_{name}"].size == 0:
            hist[f"u_{name}"] = hist[f"u_{name}"].reshape(0, 2)
        if hist[f"goal_{name}"].size == 0:
            hist[f"goal_{name}"] = hist[f"goal_{name}"].reshape(0, 2)

    metrics = compute_rollout_metrics(
        hist=hist,
        vps=vps,
        geom=geom,
        circle_offsets=circle_offsets,
        circle_radius=circle_radius,
        goal_x=goal_x,
    )

    return hist, metrics, sim, vps, geom, circle_offsets, circle_radius


# =========================
# Rollout metrics
# =========================

def compute_rollout_metrics(
    hist,
    vps,
    geom,
    circle_offsets,
    circle_radius,
    goal_x: float = 5.0,
):
    names = ["upper", "mid", "lower"]
    t = hist["t"]
    dt = t[1] - t[0] if len(t) > 1 else 0.0
    infeasible_qp = int(
        any(ev == "infeasible_qp" for ev in hist.get("events", []))
        or any(
            qp_info.get("status") not in ["optimal", "optimal_inaccurate"]
            for name in names
            for qp_info in hist.get(f"qp_{name}", [])
        )
    )

    states_now = {name: None for name in names}

    metrics = {name: {} for name in names}
    deadlock_flags = {}

    for name in names:
        x_hist = hist[f"x_{name}"]
        u_hist = hist[f"u_{name}"]

        metrics[name]["time_to_goal"] = compute_time_to_goal_x(x_hist[:, 0], t, x_goal=goal_x)
        metrics[name]["goal_reached"] = int(not np.isnan(metrics[name]["time_to_goal"]))
        metrics[name]["avg_abs_y"] = avg_abs_y_for_metrics(name, x_hist, geom)
        metrics[name]["acc_effort"] = float(np.sum(np.abs(u_hist[:, 0])) * dt)
        metrics[name]["steer_rate_effort"] = float(np.sum(np.abs(u_hist[:, 1])) * dt)
        metrics[name]["boundary_collision"] = 0
        metrics[name]["min_boundary_clearance"] = np.inf
        metrics[name]["intervehicle_collision"] = 0
        metrics[name]["min_intervehicle_clearance"] = np.inf
        deadlock_flags[name] = int(np.isnan(metrics[name]["time_to_goal"]))

    for k in range(len(t)):
        states_now = {name: hist[f"x_{name}"][k] for name in names}
        for name in names:
            b_clear = min_boundary_clearance(states_now[name], name, geom, vps[name], circle_offsets, circle_radius)
            metrics[name]["min_boundary_clearance"] = min(metrics[name]["min_boundary_clearance"], b_clear)
            if b_clear < 0.0:
                metrics[name]["boundary_collision"] = 1

        for i, name_i in enumerate(names):
            clears = []
            for j, name_j in enumerate(names):
                if i == j:
                    continue
                c = min_pair_clearance(states_now[name_i], states_now[name_j], vps[name_i], vps[name_j], circle_offsets, circle_radius)
                clears.append(c)
                if c < 0.0:
                    metrics[name_i]["intervehicle_collision"] = 1
            metrics[name_i]["min_intervehicle_clearance"] = min(metrics[name_i]["min_intervehicle_clearance"], min(clears))

    deadlock = int(all(deadlock_flags[name] == 1 for name in names))
    invalid_motion_metrics = infeasible_qp or any(
        metrics[name]["intervehicle_collision"] or metrics[name]["boundary_collision"]
        for name in names
    )

    return {
        "upper": metrics["upper"],
        "mid": metrics["mid"],
        "lower": metrics["lower"],
        "deadlock": deadlock,
        "infeasible_qp": infeasible_qp,
        "valid_motion_metrics": int(not invalid_motion_metrics),
    }


# =========================
# Monte Carlo simulation
# =========================

def sample_initial_point_on_lane(s0: float, e0: float, theta: float, pref: np.ndarray) -> np.ndarray:
    t = lane_tangent(theta)
    n = lane_normal(theta)
    return pref + s0 * t + e0 * n


def build_random_initial_states(cfg: RolloutConfig, geom: MergeGeometry):
    rng = np.random.default_rng(cfg.seed)
    states = []

    for _ in range(cfg.n_rollouts):
        e_upper = rng.uniform(cfg.y_min, cfg.y_max)
        e_mid = rng.uniform(cfg.y_min, cfg.y_max)
        e_lower = rng.uniform(cfg.y_min, cfg.y_max)

        dpsi_upper = deg2rad(rng.uniform(cfg.psi_min_deg, cfg.psi_max_deg))
        dpsi_mid = deg2rad(rng.uniform(cfg.psi_min_deg, cfg.psi_max_deg))
        dpsi_lower = deg2rad(rng.uniform(cfg.psi_min_deg, cfg.psi_max_deg))

        p_upper = sample_initial_point_on_lane(cfg.s0, e_upper, geom.theta_upper, geom.pref_upper)
        p_mid = sample_initial_point_on_lane(cfg.s0, e_mid, geom.theta_mid, geom.pref_mid)
        p_lower = sample_initial_point_on_lane(cfg.s0, e_lower, geom.theta_lower, geom.pref_lower)

        x_upper = np.array([p_upper[0], p_upper[1], geom.theta_upper + dpsi_upper, 5.0, 0.0], dtype=float)
        x_mid = np.array([p_mid[0], p_mid[1], geom.theta_mid + dpsi_mid, 5.0, 0.0], dtype=float)
        x_lower = np.array([p_lower[0], p_lower[1], geom.theta_lower + dpsi_lower, 5.0, 0.0], dtype=float)

        states.append({"upper": x_upper, "mid": x_mid, "lower": x_lower})

    return states


def run_monte_carlo(cbfp_schedule: str = "constant_normal", n_rollouts: int = 10):
    geom = MergeGeometry()
    rollout_cfg = RolloutConfig(n_rollouts=n_rollouts)
    initial_states = build_random_initial_states(rollout_cfg, geom)

    all_metrics = []
    last_hist = None
    last_bundle = None

    for r_idx, init_states in enumerate(initial_states, start=1):
        print(f"\n=== Rollout {r_idx}/{n_rollouts} ===")
        for name in ["upper", "mid", "lower"]:
            print(
                f"Initial {name}: x={init_states[name][0]:.3f}, y={init_states[name][1]:.3f}, "
                f"psi={init_states[name][2]:.3f} rad"
            )

        hist, metrics, sim, vps, geom, circle_offsets, circle_radius = run_single_rollout(
            init_states=init_states,
            cbfp_schedule=cbfp_schedule,
            goal_x=rollout_cfg.goal_x,
        )

        all_metrics.append(metrics)
        last_hist = hist
        last_bundle = (sim, vps, geom, circle_offsets, circle_radius)

    avg_metrics = average_rollout_metrics(all_metrics)
    return avg_metrics, last_hist, last_bundle


def average_rollout_metrics(all_metrics: List[Dict]) -> Dict:
    avg = {name: {} for name in ["upper", "mid", "lower"]}
    valid_motion_metrics = [m for m in all_metrics if m.get("valid_motion_metrics", 1)]

    def mean_valid(name: str, key: str, nanmean: bool = False) -> float:
        if not valid_motion_metrics:
            return float("nan")
        values = [m[name][key] for m in valid_motion_metrics]
        if nanmean:
            finite_values = [v for v in values if not np.isnan(v)]
            if not finite_values:
                return float("nan")
            return float(np.mean(finite_values))
        return float(np.mean(values))

    for name in ["upper", "mid", "lower"]:
        avg[name]["time_to_goal"] = mean_valid(name, "time_to_goal", nanmean=True)
        avg[name]["goal_reached_count"] = int(np.sum([m[name]["goal_reached"] for m in valid_motion_metrics]))
        avg[name]["avg_abs_y"] = mean_valid(name, "avg_abs_y")
        avg[name]["acc_effort"] = mean_valid(name, "acc_effort")
        avg[name]["steer_rate_effort"] = mean_valid(name, "steer_rate_effort")
        avg[name]["intervehicle_collision"] = float(np.mean([m[name]["intervehicle_collision"] for m in all_metrics]))
        avg[name]["boundary_collision"] = float(np.mean([m[name]["boundary_collision"] for m in all_metrics]))
        avg[name]["min_intervehicle_clearance"] = mean_valid(name, "min_intervehicle_clearance")
        avg[name]["min_boundary_clearance"] = mean_valid(name, "min_boundary_clearance")

    avg["deadlock_count"] = int(np.sum([m["deadlock"] for m in all_metrics]))
    avg["deadlock_rate"] = (
        float(np.mean([m["deadlock"] for m in valid_motion_metrics]))
        if valid_motion_metrics
        else float("nan")
    )
    avg["infeasible_qp_count"] = int(np.sum([m["infeasible_qp"] for m in all_metrics]))
    avg["infeasible_qp_rate"] = float(np.mean([m["infeasible_qp"] for m in all_metrics]))
    avg["valid_motion_rollouts"] = len(valid_motion_metrics)
    return avg


# =========================
# Metrics table (average over rollouts)
# =========================

def show_average_metrics_table(avg_metrics, n_rollouts: int):
    def fmt_nan(x):
        return "NaN" if np.isnan(x) else f"{x:.3f}"

    row_labels = [
        "Time to Goal [s]",
        "Goal Reached [count]",
        "Avg |y|",
        "Acceleration Effort",
        "Steering-Rate Effort",
        "Inter-Vehicle Collision Rate",
        "Boundary Collision Rate",
        "Min Inter-Vehicle Clearance [m]",
        "Min Boundary Clearance [m]",
        "Deadlock Rate",
        "Infeasible QP [count]",
        "Infeasible QP Rate",
        "Valid Motion Rollouts",
    ]

    deadlock_rate_str = fmt_nan(avg_metrics["deadlock_rate"])
    infeasible_count_str = f"{avg_metrics['infeasible_qp_count']}/{n_rollouts}"
    infeasible_rate_str = f"{avg_metrics['infeasible_qp_rate']:.3f}"
    valid_motion_rollouts_str = f"{avg_metrics['valid_motion_rollouts']}/{n_rollouts}"

    table_data = [
        [fmt_nan(avg_metrics["upper"]["time_to_goal"]), fmt_nan(avg_metrics["mid"]["time_to_goal"]), fmt_nan(avg_metrics["lower"]["time_to_goal"])],
        [f"{avg_metrics['upper']['goal_reached_count']}/{n_rollouts}", f"{avg_metrics['mid']['goal_reached_count']}/{n_rollouts}", f"{avg_metrics['lower']['goal_reached_count']}/{n_rollouts}"],
        [f"{avg_metrics['upper']['avg_abs_y']:.3f}", f"{avg_metrics['mid']['avg_abs_y']:.3f}", f"{avg_metrics['lower']['avg_abs_y']:.3f}"],
        [f"{avg_metrics['upper']['acc_effort']:.3f}", f"{avg_metrics['mid']['acc_effort']:.3f}", f"{avg_metrics['lower']['acc_effort']:.3f}"],
        [f"{avg_metrics['upper']['steer_rate_effort']:.3f}", f"{avg_metrics['mid']['steer_rate_effort']:.3f}", f"{avg_metrics['lower']['steer_rate_effort']:.3f}"],
        [f"{avg_metrics['upper']['intervehicle_collision']:.3f}", f"{avg_metrics['mid']['intervehicle_collision']:.3f}", f"{avg_metrics['lower']['intervehicle_collision']:.3f}"],
        [f"{avg_metrics['upper']['boundary_collision']:.3f}", f"{avg_metrics['mid']['boundary_collision']:.3f}", f"{avg_metrics['lower']['boundary_collision']:.3f}"],
        [f"{avg_metrics['upper']['min_intervehicle_clearance']:.3f}", f"{avg_metrics['mid']['min_intervehicle_clearance']:.3f}", f"{avg_metrics['lower']['min_intervehicle_clearance']:.3f}"],
        [f"{avg_metrics['upper']['min_boundary_clearance']:.3f}", f"{avg_metrics['mid']['min_boundary_clearance']:.3f}", f"{avg_metrics['lower']['min_boundary_clearance']:.3f}"],
        [deadlock_rate_str, deadlock_rate_str, deadlock_rate_str],
        [infeasible_count_str, infeasible_count_str, infeasible_count_str],
        [infeasible_rate_str, infeasible_rate_str, infeasible_rate_str],
        [valid_motion_rollouts_str, valid_motion_rollouts_str, valid_motion_rollouts_str],
    ]

    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.axis("off")
    ax.set_title(f"Average Metrics Summary over {n_rollouts} Rollouts", fontsize=13, pad=12)

    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=["Upper Vehicle", "Middle Vehicle", "Lower Vehicle"],
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
# Static plots
# =========================

def plot_results(hist, geom: MergeGeometry, filename: str = "three_vehicle_merge_results.png"):
    t_state = hist["t"]
    t_u = t_state[:-1]

    x_u = hist["x_upper"]
    x_m = hist["x_mid"]
    x_l = hist["x_lower"]
    u_u = hist["u_upper"]
    u_m = hist["u_mid"]
    u_l = hist["u_lower"]

    fig, axs = plt.subplots(5, 2, figsize=(13, 16), constrained_layout=True)

    axs[0, 0].plot(x_u[:, 0], x_u[:, 1], label="upper")
    axs[0, 0].plot(x_m[:, 0], x_m[:, 1], label="middle")
    axs[0, 0].plot(x_l[:, 0], x_l[:, 1], label="lower")
    axs[0, 0].axhline(geom.lane_half_width, color="k", linewidth=1.0)
    axs[0, 0].axhline(-geom.lane_half_width, color="k", linewidth=1.0)
    axs[0, 0].axvline(geom.x_switch, linestyle="--", color="gray", linewidth=1.0, alpha=0.7)
    axs[0, 0].set_title("Trajectories (Last Rollout)")
    axs[0, 0].set_xlabel("x [m]")
    axs[0, 0].set_ylabel("y [m]")
    axs[0, 0].grid(True)
    axs[0, 0].axis("equal")
    axs[0, 0].legend()

    axs[0, 1].plot(t_state, x_u[:, 3], label="upper")
    axs[0, 1].plot(t_state, x_m[:, 3], label="middle")
    axs[0, 1].plot(t_state, x_l[:, 3], label="lower")
    axs[0, 1].set_title("Speed")
    axs[0, 1].set_xlabel("t [s]")
    axs[0, 1].set_ylabel("v [m/s]")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 0].plot(t_state, x_u[:, 2], label="upper")
    axs[1, 0].plot(t_state, x_m[:, 2], label="middle")
    axs[1, 0].plot(t_state, x_l[:, 2], label="lower")
    axs[1, 0].set_title("Heading")
    axs[1, 0].set_xlabel("t [s]")
    axs[1, 0].set_ylabel("psi [rad]")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].plot(t_state, x_u[:, 4], label="upper")
    axs[1, 1].plot(t_state, x_m[:, 4], label="middle")
    axs[1, 1].plot(t_state, x_l[:, 4], label="lower")
    axs[1, 1].set_title("Steering Angle")
    axs[1, 1].set_xlabel("t [s]")
    axs[1, 1].set_ylabel("delta [rad]")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    axs[2, 0].plot(t_u, u_u[:, 0], label="upper")
    axs[2, 0].plot(t_u, u_m[:, 0], label="middle")
    axs[2, 0].plot(t_u, u_l[:, 0], label="lower")
    axs[2, 0].set_title("Acceleration Input")
    axs[2, 0].set_xlabel("t [s]")
    axs[2, 0].set_ylabel("a [m/s^2]")
    axs[2, 0].grid(True)
    axs[2, 0].legend()

    axs[2, 1].plot(t_u, u_u[:, 1], label="upper")
    axs[2, 1].plot(t_u, u_m[:, 1], label="middle")
    axs[2, 1].plot(t_u, u_l[:, 1], label="lower")
    axs[2, 1].set_title("Steering Rate Input")
    axs[2, 1].set_xlabel("t [s]")
    axs[2, 1].set_ylabel("delta_rate [rad/s]")
    axs[2, 1].grid(True)
    axs[2, 1].legend()

    axs[3, 0].plot(t_state, hist["lambda_upper"], label="lambda_upper")
    axs[3, 0].plot(t_state, hist["lambda_mid"], label="lambda_mid")
    axs[3, 0].plot(t_state, hist["lambda_lower"], label="lambda_lower")
    axs[3, 0].set_title("CBF Lambda")
    axs[3, 0].set_xlabel("t [s]")
    axs[3, 0].set_ylabel("lambda_cbf")
    axs[3, 0].grid(True)
    axs[3, 0].legend()

    axs[3, 1].plot(t_state, hist["clearance_upper_lower"], label="upper-lower")
    axs[3, 1].plot(t_state, hist["clearance_mid_upper"], label="mid-upper")
    axs[3, 1].plot(t_state, hist["clearance_mid_lower"], label="mid-lower")
    axs[3, 1].set_title("Pairwise Vehicle Clearances")
    axs[3, 1].set_xlabel("t [s]")
    axs[3, 1].set_ylabel("clearance [m]")
    axs[3, 1].grid(True)
    axs[3, 1].legend()

    axs[4, 0].plot(t_state, x_u[:, 1], label="upper y")
    axs[4, 0].plot(t_state, x_m[:, 1], label="middle y")
    axs[4, 0].plot(t_state, x_l[:, 1], label="lower y")
    axs[4, 0].axhline(geom.lane_half_width, color="k", linewidth=1.0)
    axs[4, 0].axhline(-geom.lane_half_width, color="k", linewidth=1.0)
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
    vps,
    geom: MergeGeometry,
    circle_offsets,
    circle_radius,
    filename: str = "three_vehicle_merge.mp4",
):
    import os
    from matplotlib import animation
    from matplotlib.patches import Polygon, Circle as PatchCircle

    names = ["upper", "mid", "lower"]
    colors = {"upper": "tab:blue", "mid": "tab:green", "lower": "tab:orange"}

    x_hist = {name: hist[f"x_{name}"] for name in names}
    goal_hist = {name: hist[f"goal_{name}"] for name in names}
    t = hist["t"]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")
    ax.set_xlim(-32, 32)
    ax.set_ylim(-20, 20)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Three-Vehicle Lane Merge: Decentralized CLF-CBF-QP (Last Rollout)")
    ax.grid(True)

    # -------------------------
    # Draw lane geometry
    # -------------------------

    # Middle lane boundaries: straight lane through the merge region.
    x_left = geom.mid_start_center[0]
    x_right = 30.0

    ax.plot([x_left, geom.x_open_left], [ geom.lane_half_width,  geom.lane_half_width],
            color="black", linewidth=1.6)
    ax.plot([geom.x_open_right, x_right], [ geom.lane_half_width,  geom.lane_half_width],
            color="black", linewidth=1.6)
    ax.plot([x_left, geom.x_open_left], [-geom.lane_half_width, -geom.lane_half_width],
            color="black", linewidth=1.6)
    ax.plot([geom.x_open_right, x_right], [-geom.lane_half_width, -geom.lane_half_width],
            color="black", linewidth=1.6)
    ax.fill_between([x_left, x_right],
                    -geom.lane_half_width, geom.lane_half_width,
                    color="lightgray", alpha=0.25)

    # Draw each angled boundary as a lane-parallel segment ending at its
    # assigned opening cross.
    def draw_boundary_to_corner(tvec, corner):
        start = corner - geom.entry_length * tvec
        ax.plot([start[0], corner[0]], [start[1], corner[1]],
                color="black", linewidth=1.6)

    # First cross pair.
    draw_boundary_to_corner(geom.t_lower, geom.left_bottom_corner)
    draw_boundary_to_corner(geom.t_upper, geom.left_top_corner)

    # Second/far cross pair.
    draw_boundary_to_corner(geom.t_lower, geom.right_bottom_corner)
    draw_boundary_to_corner(geom.t_upper, geom.right_top_corner)


    # -------------------------
    # Dynamic artists
    # -------------------------
    traj_lines = {}
    polys = {}
    circle_patches = {}
    heading_lines = {}
    goal_scatters = {}

    for name in names:
        traj_lines[name], = ax.plot([], [], linewidth=1.8, color=colors[name], label=name)

        polys[name] = Polygon(
            vehicle_corners(x_hist[name][0], vps[name]),
            closed=True,
            fill=False,
            edgecolor=colors[name],
            linewidth=2.0,
        )
        ax.add_patch(polys[name])

        circle_patches[name] = []
        for _ in circle_offsets:
            patch = PatchCircle(
                (0.0, 0.0),
                circle_radius,
                fill=False,
                edgecolor=colors[name],
                linestyle="--",
                alpha=0.8,
            )
            ax.add_patch(patch)
            circle_patches[name].append(patch)

        heading_lines[name], = ax.plot([], [], color=colors[name], linewidth=2.0)
        goal_scatters[name] = ax.scatter([], [], marker="x", s=70, color=colors[name])

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    ax.legend(loc="upper right")

    def set_heading_line(line_obj, state, length=1.8):
        px, py, psi, _, _ = state
        ex = length * np.cos(psi)
        ey = length * np.sin(psi)
        line_obj.set_data([px, px + ex], [py, py + ey])

    def init():
        artists = []
        for name in names:
            traj_lines[name].set_data([], [])
            polys[name].set_xy(vehicle_corners(x_hist[name][0], vps[name]))

            for k, s in enumerate(circle_offsets):
                c, _, _, _ = circle_center_and_kinematics_affine(x_hist[name][0], s, vps[name])
                circle_patches[name][k].center = (c[0], c[1])

            set_heading_line(heading_lines[name], x_hist[name][0])

            if len(goal_hist[name]) > 0:
                goal_scatters[name].set_offsets(goal_hist[name][0])

            artists.extend([
                traj_lines[name],
                polys[name],
                heading_lines[name],
                goal_scatters[name],
                *circle_patches[name],
            ])

        time_text.set_text(f"t = {t[0]:.1f} s")
        artists.append(time_text)
        return artists

    def update(frame):
        artists = []
        for name in names:
            traj_lines[name].set_data(
                x_hist[name][:frame + 1, 0],
                x_hist[name][:frame + 1, 1]
            )

            polys[name].set_xy(vehicle_corners(x_hist[name][frame], vps[name]))

            for k, s in enumerate(circle_offsets):
                c, _, _, _ = circle_center_and_kinematics_affine(x_hist[name][frame], s, vps[name])
                circle_patches[name][k].center = (c[0], c[1])

            set_heading_line(heading_lines[name], x_hist[name][frame])

            if len(goal_hist[name]) > 0:
                gi = goal_hist[name][0] if frame == 0 else goal_hist[name][frame - 1]
                goal_scatters[name].set_offsets(gi)

            artists.extend([
                traj_lines[name],
                polys[name],
                heading_lines[name],
                goal_scatters[name],
                *circle_patches[name],
            ])

        time_text.set_text(f"t = {t[frame]:.1f} s")
        artists.append(time_text)
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
    print("  1. Constant, normal value (0.2) for all vehicles")
    print("  2. Constant, normal value (0.2) for upper/mid and conservative value (0.15) for lower")
    print("  3. Distance-adaptive values for all vehicles")
    choice = input("Enter 1, 2, or 3 [default: 1]: ").strip()

    schedule_map = {
        "1": "constant_same",
        "2": "constant_different",
        "3": "adaptive",
    }
    cbfp_schedule = schedule_map.get(choice, "constant_same")
    choice_tag = choice if choice in schedule_map else "1"

    if cbfp_schedule == "constant_same":
        print("All vehicles will use constant lambda_cbf = 0.2")
    elif cbfp_schedule == "constant_different":
        print("Middle will use constant lambda_cbf = 0.4, upper will use constant lambda_cbf = 0.2, lower will use constant lambda_cbf = 0.1")
    else:
        print("All vehicles will use distance-adaptive lambda_cbf values")
        print("Rule: lambda_cbf = clip(clearance_gain * min_intervehicle_clearance, lambda_min, lambda_max)")

    n_rollouts = 20

    figure_filename = f"three_vehicle_merge_results_choice{choice_tag}.png"
    video_filename = f"three_vehicle_merge_choice{choice_tag}.mp4"

    avg_metrics, last_hist, last_bundle = run_monte_carlo(
        cbfp_schedule=cbfp_schedule,
        n_rollouts=n_rollouts,
    )

    show_average_metrics_table(avg_metrics, n_rollouts=n_rollouts)

    sim, vps, geom, circle_offsets, circle_radius = last_bundle
    plot_results(last_hist, geom, filename=figure_filename)

    animate_simulation(
        last_hist,
        vps,
        geom,
        circle_offsets,
        circle_radius,
        filename=video_filename,
    )
