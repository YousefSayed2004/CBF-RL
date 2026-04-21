import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np


# =========================
# Utility functions
# =========================

def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rot2d(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def unit_heading(psi: float) -> np.ndarray:
    return np.array([math.cos(psi), math.sin(psi)], dtype=float)


def unit_lateral(psi: float) -> np.ndarray:
    return np.array([-math.sin(psi), math.cos(psi)], dtype=float)


def deg2rad(deg: float) -> float:
    return deg * np.pi / 180.0


def lambda_from_normalized_action(
    a_norm: float,
    lambda_min: float,
    lambda_max: float,
) -> float:
    """
    Map normalized action a_norm in [-1, 1] to lambda in [lambda_min, lambda_max].
    """
    a_clipped = float(np.clip(a_norm, -1.0, 1.0))
    lam = lambda_min + 0.5 * (a_clipped + 1.0) * (lambda_max - lambda_min)
    return float(lam)


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
    u_weight: np.ndarray = field(default_factory=lambda: np.diag([10.0, 1.0]))


@dataclass
class RouteSpec:
    name: str
    road_half_width: float
    lane_axis: str  # "horizontal"


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
class RandomInitConfig:
    y_min: float = -1.0
    y_max: float = 1.0
    psi_i_min_deg: float = -20.0
    psi_i_max_deg: float = 20.0
    psi_j_offset_deg: float = 20.0
    x_i0: float = -15.0
    x_j0: float = 15.0
    v_i0: float = 6.0
    v_j0: float = 6.0
    delta_i0: float = 0.0
    delta_j0: float = 0.0


@dataclass
class RewardConfig:
    collision_penalty: float = -100.0/100
    infeasible_qp_penalty: float = -350.0/100
    boundary_collision_penalty: float = -70.0/100
    bad_collision_penalty: float = 20.0/100
    early_collision_penalty: float = -350.0/100
    progress_weight: float = 8.0/100
    deviation_weight: float = 0.0
    survival_bonus: float = 0.5/100
    collision_threshold: float = 5.0
    high_speed_reward_weight: float = 0.025
    timeout_one_fail_penalty: float =  0.0
    timeout_both_fail_penalty: float = 0.0
    goal_distance_x: float = 30.0


@dataclass
class LambdaBounds:
    min_val: float = 0.1
    max_val: float = 0.4


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
    s_track: Optional[float] = None,
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
    s_track: Optional[float] = None,
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

    d = c1 - c2
    d_dot = c1_dot - c2_dot

    q = max(np.linalg.norm(d), cbfp.eps_dist)
    n = d / q
    R = 2.0 * circle_radius

    h = q - R
    h_dot = float(n @ d_dot)

    tangential_term = (np.dot(d_dot, d_dot) - (np.dot(n, d_dot))**2) / q

    hddot_const = float(n @ (c1_ddot_0 - c2_ddot_0) + tangential_term)
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
    Returns:
        h, h_dot, [hddot_const, coeff_a, coeff_delta_rate]
    """
    c, c_dot, c_ddot_0, B = circle_center_and_kinematics_affine(ego_state, ego_s, ego_vp)

    half_w = route.road_half_width

    if route.lane_axis != "horizontal":
        raise ValueError("This script uses only horizontal roads.")

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
                + 0.5 * cbfp.lambda_cbf * h
                + s_cbf[cbf_index] >= 0.0
            ]
            cbf_index += 1

    # boundary CBFs
    for boundary_side in ["left", "right"]:
        for s_i in circle_offsets:
            h, h_dot, hddot_aff = road_boundary_cbf_affine(
                ego_state=ego_state,
                ego_s=s_i,
                ego_vp=ego_vp,
                route=ego_route,
                boundary_side=boundary_side,
            )
            hddot_const = hddot_aff[0]
            hddot_u = hddot_aff[1:]
            h_eff = h - circle_radius

            constraints += [
                0.5 * Dt**2 * (hddot_const + hddot_u @ u)
                + Dt * h_dot
                + cbfp.lambda_cbf * h_eff
                + s_cbf[cbf_index] >= 0.0
            ]
            cbf_index += 1

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
        "fallback": False,
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
        try:
            prob.solve(warm_start=True, verbose=False)
        except Exception:
            pass

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
        u_sol = np.array(u.value, dtype=float).reshape(2)

    return u_sol, qp_info


# =========================
# Safety helpers
# =========================

def get_vehicle_circle_centers(
    state: np.ndarray,
    vp: VehicleParams,
    circle_offsets: np.ndarray,
) -> List[np.ndarray]:
    return [circle_center_and_kinematics_affine(state, s, vp)[0] for s in circle_offsets]


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


def compute_time_to_goal_x(
    hist_x: np.ndarray,
    t: np.ndarray,
    direction: str,
    distance_goal: float = 30.0,
) -> float:
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
    return (R @ corners_local.T).T + np.array([px, py], dtype=float)


# =========================
# RL environment
# =========================

class TwoVehicleLambdaEnv:
    """
    Action:
        [lambda_i_norm, lambda_j_norm] in [-1, 1]^2

    Mapped to:
        lambda_i, lambda_j in [0.1, 0.4]

    Observation (13):
        [x_i, y_i, psi_i, v_i, delta_i,
         x_j, y_j, psi_j, v_j, delta_j,
         a_i_prev, delta_rate_i_prev, a_j_prev, delta_rate_j_prev,
         inter_vehicle_clearance, boundary_clearance_i, boundary_clearance_j,
         dx, dy]
    """

    def __init__(
        self,
        sim: Optional[SimParams] = None,
        init_cfg: Optional[RandomInitConfig] = None,
        reward_cfg: Optional[RewardConfig] = None,
        lambda_bounds: Optional[LambdaBounds] = None,
        seed: Optional[int] = None,
    ):
        self.sim = sim or SimParams()
        self.init_cfg = init_cfg or RandomInitConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        self.lambda_bounds = lambda_bounds or LambdaBounds()

        self.vp_i = VehicleParams()
        self.vp_j = VehicleParams()

        self.clfp_i = CLFParams()
        self.clfp_j = CLFParams()

        self.qpw_i = QPWeights(u_weight=np.diag([10.0, 1.0]))
        self.qpw_j = QPWeights(u_weight=np.diag([10.0, 1.0]))

        self.route_i = RouteSpec(name="vehicle_i", road_half_width=3.1, lane_axis="horizontal")
        self.route_j = RouteSpec(name="vehicle_j", road_half_width=3.1, lane_axis="horizontal")

        self.circle_offsets, self.circle_radius = circle_approximation(
            self.vp_i.length, self.vp_i.width, n_circles=3
        )

        self.observation_dim = 13
        self.action_dim = 2

        self.rng = np.random.default_rng(seed)

        self.x_i: Optional[np.ndarray] = None
        self.x_j: Optional[np.ndarray] = None
        self.u_i_prev = np.zeros(2, dtype=float)
        self.u_j_prev = np.zeros(2, dtype=float)
        self.lambda_i = 0.2
        self.lambda_j = 0.2
        self.step_count = 0
        self.done = False
        self.last_info: Dict = {}
        self.episode_min_intervehicle_clearance = np.inf

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def sample_initial_states(self) -> Tuple[np.ndarray, np.ndarray]:
        psi_i_min = deg2rad(self.init_cfg.psi_i_min_deg)
        psi_i_max = deg2rad(self.init_cfg.psi_i_max_deg)
        psi_j_span = deg2rad(self.init_cfg.psi_j_offset_deg)

        y_i = self.rng.uniform(self.init_cfg.y_min, self.init_cfg.y_max)
        y_j = self.rng.uniform(self.init_cfg.y_min, self.init_cfg.y_max)

        psi_i = self.rng.uniform(psi_i_min, psi_i_max)
        psi_j = self.rng.uniform(np.pi - psi_j_span, np.pi + psi_j_span)

        x_i0 = np.array([
            self.init_cfg.x_i0,
            y_i,
            psi_i,
            self.init_cfg.v_i0,
            self.init_cfg.delta_i0,
        ], dtype=float)

        x_j0 = np.array([
            self.init_cfg.x_j0,
            y_j,
            psi_j,
            self.init_cfg.v_j0,
            self.init_cfg.delta_j0,
        ], dtype=float)

        return x_i0, x_j0

    def reset(
        self,
        seed: Optional[int] = None,
        initial_states: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        if seed is not None:
            self.seed(seed)

        if initial_states is None:
            self.x_i, self.x_j = self.sample_initial_states()
        else:
            self.x_i = np.array(initial_states[0], dtype=float).copy()
            self.x_j = np.array(initial_states[1], dtype=float).copy()

        self.u_i_prev = np.zeros(2, dtype=float)
        self.u_j_prev = np.zeros(2, dtype=float)
        self.lambda_i = 0.4
        self.lambda_j = 0.4
        self.step_count = 0
        self.done = False
        self.last_info = {}
        self.episode_min_intervehicle_clearance = np.inf

        return self._get_obs()

    def _goal_points(self) -> Tuple[np.ndarray, np.ndarray]:
        lookahead_i = 8.0
        lookahead_j = 8.0
        goal_i = np.array([self.x_i[0] + lookahead_i, 0.0], dtype=float)
        goal_j = np.array([self.x_j[0] - lookahead_j, 0.0], dtype=float)
        return goal_i, goal_j

    def _get_obs(self) -> np.ndarray:
        inter_clear = min_intervehicle_clearance(
            self.x_i, self.x_j, self.vp_i, self.vp_j, self.circle_offsets, self.circle_radius
        )
        bound_i = min_boundary_clearance(
            self.x_i, self.route_i, self.vp_i, self.circle_offsets, self.circle_radius
        )
        bound_j = min_boundary_clearance(
            self.x_j, self.route_j, self.vp_j, self.circle_offsets, self.circle_radius
        )

        dx = self.x_j[0] - self.x_i[0]
        dy = self.x_j[1] - self.x_i[1]

        obs = np.array([
            self.x_i[0], self.x_i[1], self.x_i[3], 
            self.x_j[0], self.x_j[1], self.x_j[3],
            self.u_i_prev[0], self.u_j_prev[0],
            inter_clear, bound_i, bound_j,
            dx, dy,
        ], dtype=np.float32)

        return obs

    def _progress_terms(
        self,
        prev_x_i: np.ndarray,
        prev_x_j: np.ndarray,
        next_x_i: np.ndarray,
        next_x_j: np.ndarray,
    ) -> Tuple[float, float]:
        progress_i = float(next_x_i[0] - prev_x_i[0])   # vehicle i goes +x
        progress_j = float(prev_x_j[0] - next_x_j[0])   # vehicle j goes -x
        return progress_i, progress_j
    
    def _moved_distances(self) -> Tuple[float, float]:
        moved_i = self.x_i[0] - self.init_cfg.x_i0
        moved_j = self.init_cfg.x_j0 - self.x_j[0]
        return float(moved_i), float(moved_j)

    def _goal_status(self) -> Tuple[bool, bool]:
        xi_progress = self.x_i[0] - self.init_cfg.x_i0
        xj_progress = self.init_cfg.x_j0 - self.x_j[0]

        reached_i = xi_progress >= self.reward_cfg.goal_distance_x
        reached_j = xj_progress >= self.reward_cfg.goal_distance_x
        return bool(reached_i), bool(reached_j)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() before step().")

        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape[0] != 2:
            raise ValueError("Action must have shape (2,) for [lambda_i_norm, lambda_j_norm].")

        # map normalized action to physical lambda values
        self.lambda_i = lambda_from_normalized_action(
            action[0], self.lambda_bounds.min_val, self.lambda_bounds.max_val
        )
        self.lambda_j = lambda_from_normalized_action(
            action[1], self.lambda_bounds.min_val, self.lambda_bounds.max_val
        )

        cbfp_i = CBFParams(lambda_cbf=self.lambda_i, cbf_slack_weight=1e8)
        cbfp_j = CBFParams(lambda_cbf=self.lambda_j, cbf_slack_weight=1e8)

        prev_x_i = self.x_i.copy()
        prev_x_j = self.x_j.copy()

        goal_i, goal_j = self._goal_points()

        u_i, qp_i = solve_vehicle_qp(
            ego_state=self.x_i.copy(),
            other_state=self.x_j.copy(),
            ego_route=self.route_i,
            ego_vp=self.vp_i,
            other_vp=self.vp_j,
            clfp=self.clfp_i,
            cbfp=cbfp_i,
            qpw=self.qpw_i,
            sim=self.sim,
            circle_offsets=self.circle_offsets,
            circle_radius=self.circle_radius,
            goal_point=goal_i,
        )

        u_j, qp_j = solve_vehicle_qp(
            ego_state=self.x_j.copy(),
            other_state=self.x_i.copy(),
            ego_route=self.route_j,
            ego_vp=self.vp_j,
            other_vp=self.vp_i,
            clfp=self.clfp_j,
            cbfp=cbfp_j,
            qpw=self.qpw_j,
            sim=self.sim,
            circle_offsets=self.circle_offsets,
            circle_radius=self.circle_radius,
            goal_point=goal_j,
        )

        infeasible_i = qp_i.get("status") not in ["optimal", "optimal_inaccurate"]
        infeasible_j = qp_j.get("status") not in ["optimal", "optimal_inaccurate"]

        self.step_count += 1

        # terminate on infeasible / non-optimal QP
        if infeasible_i or infeasible_j:
            self.u_i_prev = u_i.copy()
            self.u_j_prev = u_j.copy()
            self.done = True

            obs = self._get_obs()
            reward = float(self.reward_cfg.infeasible_qp_penalty)

            info = {
                "lambda_i": self.lambda_i,
                "lambda_j": self.lambda_j,
                "qp_i": qp_i,
                "qp_j": qp_j,
                "event": "infeasible_qp",
                "infeasible_i": infeasible_i,
                "infeasible_j": infeasible_j,
                "u_i": u_i.copy(),
                "u_j": u_j.copy(),
                "inter_vehicle_clearance": min_intervehicle_clearance(
                    self.x_i, self.x_j, self.vp_i, self.vp_j, self.circle_offsets, self.circle_radius
                ),
                "boundary_clearance_i": min_boundary_clearance(
                    self.x_i, self.route_i, self.vp_i, self.circle_offsets, self.circle_radius
                ),
                "boundary_clearance_j": min_boundary_clearance(
                    self.x_j, self.route_j, self.vp_j, self.circle_offsets, self.circle_radius
                ),
            }
            self.last_info = info
            return obs, reward, True, info

        # normal state transition
        self.x_i = step_vehicle(self.x_i, u_i, self.vp_i, self.sim.dt)
        self.x_j = step_vehicle(self.x_j, u_j, self.vp_j, self.sim.dt)
        self.u_i_prev = u_i.copy()
        self.u_j_prev = u_j.copy()

        # safety checks after propagation
        inter_clear = min_intervehicle_clearance(
            self.x_i, self.x_j, self.vp_i, self.vp_j, self.circle_offsets, self.circle_radius
        )
        bound_i = min_boundary_clearance(
            self.x_i, self.route_i, self.vp_i, self.circle_offsets, self.circle_radius
        )
        bound_j = min_boundary_clearance(
            self.x_j, self.route_j, self.vp_j, self.circle_offsets, self.circle_radius
        )
        prev_episode_min_clear = self.episode_min_intervehicle_clearance
        self.episode_min_intervehicle_clearance = min(self.episode_min_intervehicle_clearance, inter_clear)

        # reward
        moved_i, moved_j = self._moved_distances()
        progress_i, progress_j = self._progress_terms(prev_x_i, prev_x_j, self.x_i, self.x_j)
        progress_reward = self.reward_cfg.progress_weight * (progress_i + progress_j)
        deviation_penalty = self.reward_cfg.deviation_weight * (abs(self.x_i[1]) + abs(self.x_j[1]))
        reward = float(progress_reward - deviation_penalty + self.reward_cfg.survival_bonus)
        if(inter_clear >= 12.0 and inter_clear <= self.episode_min_intervehicle_clearance):
            reward -= abs(min(self.u_i_prev[0] ,self.u_j_prev[0])) * self.reward_cfg.high_speed_reward_weight
        if(moved_i >= 7.5 and moved_j >= 7.5 and inter_clear <= 11.0 and inter_clear >= 5.0 and inter_clear <= self.episode_min_intervehicle_clearance):
            reward += abs(self.u_i_prev[0] - self.u_j_prev[0]) * self.reward_cfg.high_speed_reward_weight

        collision = inter_clear < 0.0
        boundary_collision = (bound_i < 0.0) or (bound_j < 0.0)

        done = False
        event = "running"

        if collision:
            reward = float(self.reward_cfg.collision_penalty +
                0.8 * self.reward_cfg.collision_penalty * max(self.lambda_i, self.lambda_j))
            if(min(moved_i, moved_j) <= 8.0): 
                reward *= 2.5 
            #if(abs(self.x_i[1] - self.x_j[1]) < 0.5):
            #    reward-= self.reward_cfg.bad_collision_penalty
            done = True
            event = "collision"

        elif boundary_collision:

            early_boundary_collision = (
                (max(moved_i, moved_j) < 15.0 and inter_clear >= 3.0) or (min(moved_i, moved_j) <= 8.0)
            )

            if early_boundary_collision:
                reward = float(self.reward_cfg.early_collision_penalty)
                event = "early_boundary_collision"
            else:
                reward = float(self.reward_cfg.boundary_collision_penalty +
                               0.8 * self.reward_cfg.boundary_collision_penalty * max(self.lambda_j, self.lambda_i))
                event = "boundary_collision"

            done = True


        elif self.step_count >= self.sim.steps:
            reached_i, reached_j = self._goal_status()
            if (not reached_i) and (not reached_j):
                reward += float(self.reward_cfg.timeout_both_fail_penalty)
                event = "timeout_both_fail"
            elif (not reached_i) or (not reached_j):
                reward += float(self.reward_cfg.timeout_one_fail_penalty)
                event = "timeout_one_fail"
            else:
                event = "timeout_both_reached"
            done = True

        self.done = done
        obs = self._get_obs()

        info = {
            "lambda_i": self.lambda_i,
            "lambda_j": self.lambda_j,
            "qp_i": qp_i,
            "qp_j": qp_j,
            "u_i": u_i.copy(),
            "u_j": u_j.copy(),
            "progress_i": progress_i,
            "progress_j": progress_j,
            "inter_vehicle_clearance": inter_clear,
            "boundary_clearance_i": bound_i,
            "boundary_clearance_j": bound_j,
            "collision": collision,
            "boundary_collision": boundary_collision,
            "step_count": self.step_count,
            "event": event,
        }
        self.last_info = info

        return obs, reward, done, info


# =========================
# Fixed test rollouts helper
# =========================

def build_fixed_initial_states(
    n_rollouts: int = 10,
    seed: int = 7,
    cfg: Optional[RandomInitConfig] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    cfg = cfg or RandomInitConfig()
    rng = np.random.default_rng(seed)

    states = []
    psi_i_min = deg2rad(cfg.psi_i_min_deg)
    psi_i_max = deg2rad(cfg.psi_i_max_deg)
    psi_j_span = deg2rad(cfg.psi_j_offset_deg)

    for _ in range(n_rollouts):
        y_i = rng.uniform(cfg.y_min, cfg.y_max)
        y_j = rng.uniform(cfg.y_min, cfg.y_max)

        psi_i = rng.uniform(psi_i_min, psi_i_max)
        psi_j = rng.uniform(np.pi - psi_j_span, np.pi + psi_j_span)

        x_i0 = np.array([cfg.x_i0, y_i, psi_i, cfg.v_i0, cfg.delta_i0], dtype=float)
        x_j0 = np.array([cfg.x_j0, y_j, psi_j, cfg.v_j0, cfg.delta_j0], dtype=float)

        states.append((x_i0, x_j0))

    return states

# =========================
# Rollout metrics + evaluation helpers
# =========================

def compute_rollout_metrics(
    hist: Dict[str, np.ndarray],
    vp_i: VehicleParams,
    vp_j: VehicleParams,
    route_i: RouteSpec,
    route_j: RouteSpec,
    circle_offsets: np.ndarray,
    circle_radius: float,
    goal_distance_x: float = 30.0,
) -> Dict:
    x_i = hist["x_i"]
    x_j = hist["x_j"]
    u_i = hist["u_i"]
    u_j = hist["u_j"]
    t = hist["t"]

    dt = t[1] - t[0] if len(t) > 1 else 0.1

    acc_eff_i = float(np.sum(np.abs(u_i[:, 0])) * dt) if len(u_i) > 0 else 0.0
    acc_eff_j = float(np.sum(np.abs(u_j[:, 0])) * dt) if len(u_j) > 0 else 0.0

    steer_eff_i = float(np.sum(np.abs(u_i[:, 1])) * dt) if len(u_i) > 0 else 0.0
    steer_eff_j = float(np.sum(np.abs(u_j[:, 1])) * dt) if len(u_j) > 0 else 0.0

    min_iv_clear = np.inf
    min_bound_clear_i = np.inf
    min_bound_clear_j = np.inf

    intervehicle_collision_i = False
    intervehicle_collision_j = False
    boundary_collision_i = False
    boundary_collision_j = False

    infeasible_qp = int(any(ev == "infeasible_qp" for ev in hist["events"]))

    for k in range(len(t)):
        xi = x_i[k]
        xj = x_j[k]

        iv_clear = min_intervehicle_clearance(
            xi, xj, vp_i, vp_j, circle_offsets, circle_radius
        )
        bi_clear = min_boundary_clearance(
            xi, route_i, vp_i, circle_offsets, circle_radius
        )
        bj_clear = min_boundary_clearance(
            xj, route_j, vp_j, circle_offsets, circle_radius
        )

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
        x_i[:, 0],
        t,
        direction="positive",
        distance_goal=goal_distance_x,
    )

    time_to_goal_j = compute_time_to_goal_x(
        x_j[:, 0],
        t,
        direction="negative",
        distance_goal=goal_distance_x,
    )

    avg_abs_y_i = float(np.mean(np.abs(x_i[:, 1])))
    avg_abs_y_j = float(np.mean(np.abs(x_j[:, 1])))

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
        "infeasible_qp": infeasible_qp,
        "episode_return": float(np.sum(hist["reward"])),
    }


def average_rollout_metrics(all_metrics: List[Dict]) -> Dict:
    avg = {"i": {}, "j": {}}

    for veh in ["i", "j"]:
        avg[veh]["time_to_goal"] = float(
            np.nanmean([m[veh]["time_to_goal"] for m in all_metrics])
        )
        avg[veh]["avg_abs_y"] = float(
            np.mean([m[veh]["avg_abs_y"] for m in all_metrics])
        )
        avg[veh]["acc_effort"] = float(
            np.mean([m[veh]["acc_effort"] for m in all_metrics])
        )
        avg[veh]["steer_rate_effort"] = float(
            np.mean([m[veh]["steer_rate_effort"] for m in all_metrics])
        )
        avg[veh]["intervehicle_collision"] = float(
            np.mean([m[veh]["intervehicle_collision"] for m in all_metrics])
        )
        avg[veh]["boundary_collision"] = float(
            np.mean([m[veh]["boundary_collision"] for m in all_metrics])
        )
        avg[veh]["min_intervehicle_clearance"] = float(
            np.mean([m[veh]["min_intervehicle_clearance"] for m in all_metrics])
        )
        avg[veh]["min_boundary_clearance"] = float(
            np.mean([m[veh]["min_boundary_clearance"] for m in all_metrics])
        )

    avg["deadlock_count"] = int(np.sum([m["deadlock"] for m in all_metrics]))
    avg["deadlock_rate"] = float(np.mean([m["deadlock"] for m in all_metrics]))
    avg["infeasible_qp_rate"] = float(np.mean([m["infeasible_qp"] for m in all_metrics]))
    avg["avg_episode_return"] = float(np.mean([m["episode_return"] for m in all_metrics]))

    return avg


def run_policy_rollout(
    env: TwoVehicleLambdaEnv,
    action_fn,
    initial_states: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[Dict, Dict]:
    obs = env.reset(initial_states=initial_states)

    hist = {
        "x_i": [env.x_i.copy()],
        "x_j": [env.x_j.copy()],
        "u_i": [],
        "u_j": [],
        "reward": [0.0],
        "events": [],
        "t": [0.0],
    }

    done = False

    while not done:
        action = action_fn(obs)
        next_obs, reward, done, info = env.step(action)

        hist["x_i"].append(env.x_i.copy())
        hist["x_j"].append(env.x_j.copy())
        hist["u_i"].append(info["u_i"].copy())
        hist["u_j"].append(info["u_j"].copy())
        hist["reward"].append(reward)
        hist["events"].append(info["event"])
        hist["t"].append(env.step_count * env.sim.dt)

        obs = next_obs

    for key in ["x_i", "x_j", "u_i", "u_j", "reward", "t"]:
        hist[key] = np.array(hist[key])

    metrics = compute_rollout_metrics(
        hist=hist,
        vp_i=env.vp_i,
        vp_j=env.vp_j,
        route_i=env.route_i,
        route_j=env.route_j,
        circle_offsets=env.circle_offsets,
        circle_radius=env.circle_radius,
        goal_distance_x=env.reward_cfg.goal_distance_x,
    )

    return hist, metrics