from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LM_Baseline_Methods import (
    CBFParams,
    CLFParams,
    MergeGeometry,
    QPWeights,
    RolloutConfig,
    SimParams,
    VehicleParams,
    average_rollout_metrics,
    build_random_initial_states,
    circle_approximation,
    compute_rollout_metrics,
    front_circle_center,
    lookahead_goal_from_track_point,
    min_boundary_clearance,
    nominal_control,
    pairwise_vehicle_clearances,
    sample_initial_point_on_lane,
    signed_line_coordinate,
    solve_vehicle_qp,
    step_vehicle,
    vehicle_corners,
    circle_center_and_kinematics_affine,
)


NAMES = ["upper", "mid", "lower"]
PAIR_KEYS = ["upper_lower", "mid_upper", "mid_lower"]
PAIR_TO_NAMES = {
    "upper_lower": ("upper", "lower"),
    "mid_upper": ("mid", "upper"),
    "mid_lower": ("mid", "lower"),
}
OPTIMAL_STATUSES = ["optimal", "optimal_inaccurate"]


def lambda_from_normalized_action(a_norm: float, lambda_min: float, lambda_max: float) -> float:
    a_clipped = float(np.clip(a_norm, -1.0, 1.0))
    return float(lambda_min + 0.5 * (a_clipped + 1.0) * (lambda_max - lambda_min))


@dataclass
class RewardConfig:
    collision_penalty: float = -100.0 / 100
    infeasible_qp_penalty: float = -150.0 / 100
    boundary_collision_penalty: float = -70.0 / 100
    progress_weight: float = 8.0 / 100
    deviation_weight: float = 0.15 / 100
    survival_bonus: float = 0.5 / 100
    goal_x: float = 5.0


@dataclass
class LambdaBounds:
    min_val: float = 0.1
    max_val: float = 0.4


class ThreeVehicleLambdaEnv:
    """
    Action:
        [lambda_upper_norm, lambda_mid_norm, lambda_lower_norm] in [-1, 1]^3

    Observation (26):
        states: upper/mid/lower x, y, psi, speed                         (12)
        previous acceleration inputs: upper/mid/lower                     (3)
        pairwise clearances: upper-lower, mid-upper, mid-lower            (3)
        boundary clearances: upper, lower                                 (2)
        pairwise deltas: upper-lower, mid-upper, mid-lower dx, dy         (6)
    """

    def __init__(
        self,
        sim: Optional[SimParams] = None,
        init_cfg: Optional[RolloutConfig] = None,
        reward_cfg: Optional[RewardConfig] = None,
        lambda_bounds: Optional[LambdaBounds] = None,
        seed: Optional[int] = None,
    ):
        self.sim = sim or SimParams()
        self.init_cfg = init_cfg or RolloutConfig(n_rollouts=1)
        self.reward_cfg = reward_cfg or RewardConfig()
        self.lambda_bounds = lambda_bounds or LambdaBounds()
        self.geom = MergeGeometry()

        self.vps = {name: VehicleParams() for name in NAMES}
        self.clfps = {name: CLFParams() for name in NAMES}
        self.qpws = {name: QPWeights() for name in NAMES}

        self.circle_offsets, self.circle_radius = circle_approximation(
            self.vps["upper"].length, self.vps["upper"].width, n_circles=3
        )

        self.observation_dim = 26
        self.action_dim = 3

        self.rng = np.random.default_rng(seed)
        self.states: Dict[str, Optional[np.ndarray]] = {name: None for name in NAMES}
        self.initial_states: Dict[str, Optional[np.ndarray]] = {name: None for name in NAMES}
        self.u_prev = {name: np.zeros(2, dtype=float) for name in NAMES}
        self.lambdas = {name: 0.2 for name in NAMES}
        self.step_count = 0
        self.done = False
        self.last_info: Dict = {}
        self.episode_min_intervehicle_clearance = np.inf

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def sample_initial_states(self) -> Dict[str, np.ndarray]:
        e_vals = {name: self.rng.uniform(self.init_cfg.y_min, self.init_cfg.y_max) for name in NAMES}
        dpsi_vals = {
            name: np.deg2rad(self.rng.uniform(self.init_cfg.psi_min_deg, self.init_cfg.psi_max_deg))
            for name in NAMES
        }

        p_upper = sample_initial_point_on_lane(
            self.init_cfg.s0, e_vals["upper"], self.geom.theta_upper, self.geom.pref_upper
        )
        p_mid = sample_initial_point_on_lane(
            self.init_cfg.s0, e_vals["mid"], self.geom.theta_mid, self.geom.pref_mid
        )
        p_lower = sample_initial_point_on_lane(
            self.init_cfg.s0, e_vals["lower"], self.geom.theta_lower, self.geom.pref_lower
        )

        return {
            "upper": np.array([p_upper[0], p_upper[1], self.geom.theta_upper + dpsi_vals["upper"], 5.0, 0.0], dtype=float),
            "mid": np.array([p_mid[0], p_mid[1], self.geom.theta_mid + dpsi_vals["mid"], 5.0, 0.0], dtype=float),
            "lower": np.array([p_lower[0], p_lower[1], self.geom.theta_lower + dpsi_vals["lower"], 5.0, 0.0], dtype=float),
        }

    def reset(
        self,
        seed: Optional[int] = None,
        initial_states: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        if seed is not None:
            self.seed(seed)

        if initial_states is None:
            sampled = self.sample_initial_states()
        else:
            sampled = {name: np.array(initial_states[name], dtype=float).copy() for name in NAMES}

        self.states = {name: sampled[name].copy() for name in NAMES}
        self.initial_states = {name: sampled[name].copy() for name in NAMES}
        self.u_prev = {name: np.zeros(2, dtype=float) for name in NAMES}
        self.lambdas = {name: 0.4 for name in NAMES}
        self.step_count = 0
        self.done = False
        self.last_info = {}
        self.episode_min_intervehicle_clearance = np.inf
        return self._get_obs()

    def _goal_points(self) -> Dict[str, np.ndarray]:
        goals = {}
        for name in NAMES:
            fc = front_circle_center(self.states[name], self.vps[name])
            goals[name] = lookahead_goal_from_track_point(fc, name, self.geom)
        return goals

    def _pairwise_clearances(self) -> Dict[str, float]:
        return pairwise_vehicle_clearances(self.states, self.vps, self.circle_offsets, self.circle_radius)

    def _boundary_clearances(self) -> Dict[str, float]:
        return {
            name: min_boundary_clearance(
                self.states[name], name, self.geom, self.vps[name], self.circle_offsets, self.circle_radius
            )
            for name in NAMES
        }

    def _get_obs(self) -> np.ndarray:
        pair_clear = self._pairwise_clearances()
        bound_clear = self._boundary_clearances()

        deltas = []
        for pair_key in PAIR_KEYS:
            a, b = PAIR_TO_NAMES[pair_key]
            deltas.extend([
                self.states[b][0] - self.states[a][0],
                self.states[b][1] - self.states[a][1],
            ])

        obs = np.array([
            self.states["upper"][0], self.states["upper"][1], self.states["upper"][2], self.states["upper"][3],
            self.states["mid"][0], self.states["mid"][1], self.states["mid"][2], self.states["mid"][3],
            self.states["lower"][0], self.states["lower"][1], self.states["lower"][2], self.states["lower"][3],
            self.u_prev["upper"][0], self.u_prev["mid"][0], self.u_prev["lower"][0],
            pair_clear["upper_lower"], pair_clear["mid_upper"], pair_clear["mid_lower"],
            bound_clear["upper"], bound_clear["lower"],
            *deltas,
        ], dtype=np.float32)

        return obs

    def _route_coordinate(self, name: str, state: np.ndarray) -> float:
        point = state[:2]
        if name == "mid":
            return float(point[0])

        if name == "upper":
            tvec = self.geom.t_upper
            pref = self.geom.pref_upper
        else:
            tvec = self.geom.t_lower
            pref = self.geom.pref_lower

        s_switch = self.geom.x_switch / tvec[0]
        if point[0] <= self.geom.x_switch:
            return signed_line_coordinate(point, pref, tvec)
        return float(s_switch + (point[0] - self.geom.x_switch))

    def _moved_distances(self) -> Dict[str, float]:
        return {
            name: self._route_coordinate(name, self.states[name])
            - self._route_coordinate(name, self.initial_states[name])
            for name in NAMES
        }

    def _progress_terms(self, prev_states: Dict[str, np.ndarray]) -> Dict[str, float]:
        return {
            name: self._route_coordinate(name, self.states[name])
            - self._route_coordinate(name, prev_states[name])
            for name in NAMES
        }

    def _lateral_deviations(self) -> Dict[str, float]:
        """Compute lateral deviation from lane centerline for each vehicle."""
        deviations = {}
        for name in NAMES:
            point = self.states[name][:2]
            if name == "upper":
                nvec = self.geom.n_upper
                pref = self.geom.pref_upper
            elif name == "mid":
                nvec = self.geom.n_mid
                pref = self.geom.pref_mid
            else:  # lower
                nvec = self.geom.n_lower
                pref = self.geom.pref_lower
            
            # Lateral deviation: projection onto normal vector
            deviations[name] = float(abs((point - pref) @ nvec))
        return deviations

    def _goal_status(self) -> Dict[str, bool]:
        return {name: bool(self.states[name][0] >= self.reward_cfg.goal_x) for name in NAMES}

    def _fallback_control(self, name: str, goal: np.ndarray) -> Tuple[np.ndarray, Dict]:
        u_nom = nominal_control(self.states[name], self.clfps[name], self.vps[name], goal)
        u = np.array([
            np.clip(u_nom[0], self.vps[name].min_accel, self.vps[name].max_accel),
            np.clip(u_nom[1], self.vps[name].min_steer_rate, self.vps[name].max_steer_rate),
        ], dtype=float)
        return u, {"status": "exception", "u_nom": u_nom.copy(), "clf_V": np.nan, "fallback": True}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() before step().")

        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape[0] != 3:
            raise ValueError("Action must have shape (3,) for upper, mid, lower lambdas.")

        for idx, name in enumerate(NAMES):
            self.lambdas[name] = lambda_from_normalized_action(
                action[idx], self.lambda_bounds.min_val, self.lambda_bounds.max_val
            )

        prev_states = {name: self.states[name].copy() for name in NAMES}
        goals = self._goal_points()
        controls = {}
        qp_infos = {}

        for name in NAMES:
            other_names = [n for n in NAMES if n != name]
            try:
                controls[name], qp_infos[name] = solve_vehicle_qp(
                    ego_state=self.states[name].copy(),
                    other_states=[self.states[n].copy() for n in other_names],
                    ego_name=name,
                    ego_vp=self.vps[name],
                    other_vps=[self.vps[n] for n in other_names],
                    clfp=self.clfps[name],
                    cbfp=CBFParams(lambda_cbf=self.lambdas[name], cbf_slack_weight=1e8),
                    qpw=self.qpws[name],
                    sim=self.sim,
                    geom=self.geom,
                    circle_offsets=self.circle_offsets,
                    circle_radius=self.circle_radius,
                    goal_point=goals[name],
                )
            except Exception as exc:
                controls[name], qp_infos[name] = self._fallback_control(name, goals[name])
                qp_infos[name]["status"] = f"exception:{type(exc).__name__}"
                qp_infos[name]["exception"] = str(exc)

        infeasible_names = [name for name in NAMES if qp_infos[name].get("status") not in OPTIMAL_STATUSES]
        self.step_count += 1

        if infeasible_names:
            self.u_prev = {name: controls[name].copy() for name in NAMES}
            self.done = True
            pair_clear = self._pairwise_clearances()
            bound_clear = self._boundary_clearances()
            reward = float(self.reward_cfg.infeasible_qp_penalty)
            info = self._build_info(
                controls, qp_infos, goals, pair_clear, bound_clear,
                event="infeasible_qp", reward=reward, progress={name: 0.0 for name in NAMES}
            )
            info["infeasible_names"] = infeasible_names
            self.last_info = info
            return self._get_obs(), reward, True, info

        for name in NAMES:
            self.states[name] = step_vehicle(self.states[name], controls[name], self.vps[name], self.sim.dt)
            self.u_prev[name] = controls[name].copy()

        pair_clear = self._pairwise_clearances()
        bound_clear = self._boundary_clearances()
        min_inter_clear = min(pair_clear.values())
        self.episode_min_intervehicle_clearance = min(self.episode_min_intervehicle_clearance, min_inter_clear)

        progress = self._progress_terms(prev_states)
        moved = self._moved_distances()
        lateral_devs = self._lateral_deviations()
        progress_reward = self.reward_cfg.progress_weight * sum(progress.values())
        deviation_penalty = self.reward_cfg.deviation_weight * sum(lateral_devs.values())
        reward = float(progress_reward - deviation_penalty + self.reward_cfg.survival_bonus)


        collision = min_inter_clear < -1e-3
        boundary_collision = min(bound_clear.values()) < -1e-3
        done = False
        event = "running"

        if collision:
            reward = float(self.reward_cfg.collision_penalty)
            done = True
            event = "collision"
        elif boundary_collision:
            reward = float(self.reward_cfg.boundary_collision_penalty)
            event = "boundary_collision"
            done = True
        elif self.step_count >= self.sim.steps:
            done = True

        self.done = done
        info = self._build_info(controls, qp_infos, goals, pair_clear, bound_clear, event, reward, progress)
        info["moved"] = moved
        info["goal_reached"] = self._goal_status()
        self.last_info = info
        return self._get_obs(), reward, done, info

    def _build_info(
        self,
        controls: Dict[str, np.ndarray],
        qp_infos: Dict[str, Dict],
        goals: Dict[str, np.ndarray],
        pair_clear: Dict[str, float],
        bound_clear: Dict[str, float],
        event: str,
        reward: float,
        progress: Dict[str, float],
    ) -> Dict:
        info = {
            "event": event,
            "reward": reward,
            "step_count": self.step_count,
            "pairwise_clearances": pair_clear.copy(),
            "boundary_clearances": bound_clear.copy(),
            "min_intervehicle_clearance": float(min(pair_clear.values())),
            "collision": event == "collision",
            "boundary_collision": "boundary_collision" in event,
        }
        for name in NAMES:
            info[f"lambda_{name}"] = self.lambdas[name]
            info[f"qp_{name}"] = qp_infos[name]
            info[f"u_{name}"] = controls[name].copy()
            info[f"goal_{name}"] = goals[name].copy()
            info[f"progress_{name}"] = progress[name]
        return info


def pair_key_for_names(name_a: str, name_b: str) -> str:
    pair_names = {name_a, name_b}
    if pair_names == {"upper", "lower"}:
        return "upper_lower"
    if pair_names == {"mid", "upper"}:
        return "mid_upper"
    return "mid_lower"


class DecentralizedLaneMergingEnv(ThreeVehicleLambdaEnv):
    """Normalized nearest-first decentralized observation, matching the RI D5 structure without decision IDs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geom.lookahead_dist = 8.0
        self.clfps = {
            name: CLFParams(
                w_v=1.0,
                w_delta=15.0,
                desired_speed=5.0,
                nominal_speed_gain=1.0,
                nominal_heading_gain=1.0,
                nominal_steer_rate_gain=5.0,
                clf_rate=10.0,
                clf_slack_weight=1.0,
            )
            for name in NAMES
        }
        self.qpws = {name: QPWeights(u_weight=np.diag([1.0, 15.0])) for name in NAMES}
        self.observation_dim = 18
        self.action_dim = 1
        self.position_obs_scale = max(abs(float(self.init_cfg.s0)), abs(float(self.reward_cfg.goal_x)), 1e-6)
        self.speed_obs_scale = max(float(self.clfps[NAMES[0]].desired_speed), 1e-6)
        first_vp = self.vps[NAMES[0]]
        self.accel_obs_scale = max(abs(first_vp.max_accel), abs(first_vp.min_accel), 1e-6)

    def _get_obs(self) -> np.ndarray:
        pair_clear = self._pairwise_clearances()
        bound_clear = self._boundary_clearances()
        goals = self._goal_points()
        positions = {name: self.states[name][:2].copy() for name in NAMES}
        velocities = {
            name: self.states[name][3] * np.array([
                np.cos(self.states[name][2]),
                np.sin(self.states[name][2]),
            ], dtype=float)
            for name in NAMES
        }
        local_obs = []

        for name in NAMES:
            state = self.states[name]
            p = positions[name]
            goal = goals[name]
            psi_norm = ((state[2] + np.pi) % (2.0 * np.pi) - np.pi) / np.pi
            heading = np.array([np.cos(state[2]), np.sin(state[2])], dtype=float)
            left = np.array([-np.sin(state[2]), np.cos(state[2])], dtype=float)

            obs = [
                p[0] / self.position_obs_scale,
                p[1] / self.position_obs_scale,
                psi_norm,
                state[3] / self.speed_obs_scale,
                self.u_prev[name][0] / self.accel_obs_scale,
                bound_clear[name] / self.position_obs_scale,
                goal[0] / self.position_obs_scale,
                goal[1] / self.position_obs_scale,
            ]

            other_names = sorted(
                [other for other in NAMES if other != name],
                key=lambda other: pair_clear[pair_key_for_names(name, other)],
            )
            for other in other_names:
                rel = positions[other] - p
                rel_v = velocities[other] - velocities[name]
                obs.extend([
                    float(rel @ heading) / self.position_obs_scale,
                    float(rel @ left) / self.position_obs_scale,
                    float(rel_v @ heading) / self.speed_obs_scale,
                    float(rel_v @ left) / self.speed_obs_scale,
                    pair_clear[pair_key_for_names(name, other)] / self.position_obs_scale,
                ])

            local_obs.append(obs)

        return np.array(local_obs, dtype=np.float32)


def build_fixed_initial_states(
    n_rollouts: int = 10,
    seed: int = 7,
    cfg: Optional[RolloutConfig] = None,
    geom: Optional[MergeGeometry] = None,
) -> List[Dict[str, np.ndarray]]:
    cfg = cfg or RolloutConfig(n_rollouts=n_rollouts, seed=seed)
    cfg.n_rollouts = n_rollouts
    cfg.seed = seed
    return build_random_initial_states(cfg, geom or MergeGeometry())


def _empty_hist() -> Dict[str, List]:
    hist = {"reward": [0.0], "events": [], "t": [0.0]}
    for name in NAMES:
        hist[f"x_{name}"] = []
        hist[f"u_{name}"] = []
        hist[f"goal_{name}"] = []
        hist[f"lambda_{name}"] = []
    for pair_key in PAIR_KEYS:
        hist[f"clearance_{pair_key}"] = []
    return hist


def run_policy_rollout(
    env: ThreeVehicleLambdaEnv,
    action_fn,
    initial_states: Optional[Dict[str, np.ndarray]] = None,
    step_callback: Optional[Callable[[Dict], None]] = None,
) -> Tuple[Dict, Dict]:
    obs = env.reset(initial_states=initial_states)
    hist = _empty_hist()
    pair_clear = env._pairwise_clearances()
    for name in NAMES:
        hist[f"x_{name}"].append(env.states[name].copy())
        hist[f"lambda_{name}"].append(env.lambdas[name])
    for pair_key in PAIR_KEYS:
        hist[f"clearance_{pair_key}"].append(pair_clear[pair_key])

    done = False
    while not done:
        action = action_fn(obs)
        obs, reward, done, info = env.step(action)
        for name in NAMES:
            hist[f"x_{name}"].append(env.states[name].copy())
            hist[f"u_{name}"].append(info[f"u_{name}"].copy())
            hist[f"goal_{name}"].append(info[f"goal_{name}"].copy())
            hist[f"lambda_{name}"].append(info[f"lambda_{name}"])
        for pair_key in PAIR_KEYS:
            hist[f"clearance_{pair_key}"].append(info["pairwise_clearances"][pair_key])
        hist["reward"].append(reward)
        hist["events"].append(info["event"])
        hist["t"].append(env.step_count * env.sim.dt)
        if step_callback is not None:
            step_callback(info)

    for key in list(hist.keys()):
        if key != "events":
            hist[key] = np.array(hist[key])
    for name in NAMES:
        if hist[f"u_{name}"].size == 0:
            hist[f"u_{name}"] = hist[f"u_{name}"].reshape(0, 2)
        if hist[f"goal_{name}"].size == 0:
            hist[f"goal_{name}"] = hist[f"goal_{name}"].reshape(0, 2)

    metrics = compute_rollout_metrics(hist, env.vps, env.geom, env.circle_offsets, env.circle_radius, env.reward_cfg.goal_x)
    return hist, metrics
