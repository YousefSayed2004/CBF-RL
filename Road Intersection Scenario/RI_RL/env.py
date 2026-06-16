from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, WORKSPACE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from RI_Baseline_Methods import (  # noqa: E402
    CBFParams,
    CLFParams,
    IntersectionGeometry,
    QPWeights,
    RouteSpec,
    SimParams,
    VehicleParams,
    centerline_deviation,
    circle_approximation,
    compute_multi_rollout_metrics,
    initial_state_from_route,
    min_boundary_clearance,
    min_intervehicle_clearance,
    min_multi_intervehicle_clearance,
    right_normal,
    route_goal_from_position,
    solve_vehicle_qp_against_many,
    step_vehicle,
)


VEHICLE_NAMES = ("i", "j", "k")
START_BY_NAME = {"i": 1, "j": 2, "k": 3}
DECISION_ID = {"S": 0.0, "R": 1.0, "L": 2.0}
DECISION_TO_REL_EXIT = {"R": 1, "S": 2, "L": 3}
DECISION_ONE_HOT_ORDER = ("S", "R", "L")
OPTIMAL_STATUSES = ["optimal", "optimal_inaccurate"]


def lambda_from_normalized_action(a_norm: float, lambda_min: float, lambda_max: float) -> float:
    a_clipped = float(np.clip(a_norm, -1.0, 1.0))
    return float(lambda_min + 0.5 * (a_clipped + 1.0) * (lambda_max - lambda_min))


def decision_from_route(route: RouteSpec) -> str:
    rel = (route.exit - route.start) % 4
    if rel == 1:
        return "R"
    if rel == 2:
        return "S"
    if rel == 3:
        return "L"
    raise ValueError("Start and exit must be different.")


def route_decision_sequence(routes: Dict[str, RouteSpec]) -> str:
    return "".join(decision_from_route(routes[name]) for name in VEHICLE_NAMES)


ALL_DECISION_SEQUENCES = tuple(
    "".join((d_i, d_j, d_k))
    for d_i in ("R", "L", "S")
    for d_j in ("L", "S", "R")
    for d_k in ("L", "S", "R")
)


def one_hot_decision(decision: str) -> List[float]:
    return [1.0 if decision == option else 0.0 for option in DECISION_ONE_HOT_ORDER]


def normalize_decision_sequence(sequence: str) -> str:
    cleaned = sequence.replace(",", "").replace(" ", "").upper()
    if len(cleaned) != len(VEHICLE_NAMES) or any(ch not in DECISION_TO_REL_EXIT for ch in cleaned):
        names = ", ".join(VEHICLE_NAMES)
        raise ValueError(
            f"Decision sequence must contain one L/R/S decision for each vehicle ({names}), "
            f"for example RLS."
        )
    return cleaned


def route_set_from_decision_sequence(sequence: str, geom: IntersectionGeometry) -> Dict[str, RouteSpec]:
    sequence = normalize_decision_sequence(sequence)
    routes = {}
    for name, decision in zip(VEHICLE_NAMES, sequence):
        start = START_BY_NAME[name]
        exit_lane = ((start - 1 + DECISION_TO_REL_EXIT[decision]) % 4) + 1
        routes[name] = RouteSpec(start=start, exit=exit_lane, geom=geom)
    return routes


def sample_route_set(rng: np.random.Generator, geom: IntersectionGeometry) -> Dict[str, RouteSpec]:
    routes = {}
    for name in VEHICLE_NAMES:
        start = START_BY_NAME[name]
        possible_exits = [lane for lane in (1, 2, 3, 4) if lane != start]
        exit_lane = int(rng.choice(possible_exits))
        routes[name] = RouteSpec(start=start, exit=exit_lane, geom=geom)
    return routes


def build_fixed_initial_states(
    n_rollouts: int,
    seed: int,
    geom: Optional[IntersectionGeometry] = None,
) -> List[Dict[str, RouteSpec]]:
    rng = np.random.default_rng(seed)
    geom = geom or IntersectionGeometry()
    return [sample_route_set(rng, geom) for _ in range(n_rollouts)]


@dataclass
class RewardConfig:
    collision_penalty: float = -100.0 / 100
    infeasible_qp_penalty: float = -150.0 / 100
    boundary_collision_penalty: float = -100.0 / 100
    progress_weight: float = 3.3 / 100
    deviation_weight: float = 0.15 / 100
    survival_bonus: float = 0.5 / 100
    goal_reward: float = 75.0 / 100


@dataclass
class LambdaBounds:
    min_val: float = 0.1
    max_val: float = 0.4


@dataclass
class GeometryConfig:
    road_width: float = 6.0
    corner_radius: float = 2.0
    lookahead_dist: float = 8.0
    start_distance: float = 15.0
    switch_offset: float = 2.0

    def build(self) -> IntersectionGeometry:
        return IntersectionGeometry(
            road_width=self.road_width,
            corner_radius=self.corner_radius,
            lookahead_dist=self.lookahead_dist,
            start_distance=self.start_distance,
            switch_offset=self.switch_offset,
        )


class ThreeVehicleIntersectionEnv:
    """
    Three-vehicle road-intersection lambda environment.

    Starts are fixed:
        i -> lane 1, j -> lane 2, k -> lane 3.

    Exits are random and exclude each vehicle's own start lane, giving 3^3
    decision possibilities.
    """

    def __init__(
        self,
        sim: Optional[SimParams] = None,
        geom_cfg: Optional[GeometryConfig] = None,
        reward_cfg: Optional[RewardConfig] = None,
        lambda_bounds: Optional[LambdaBounds] = None,
        seed: Optional[int] = None,
        cbf_slack_weight: float = 1e8,
        clf_slack_weight: float = 1.0,
    ):
        self.sim = sim or SimParams()
        self.geom_cfg = geom_cfg or GeometryConfig()
        self.geom = self.geom_cfg.build()
        self.reward_cfg = reward_cfg or RewardConfig()
        self.lambda_bounds = lambda_bounds or LambdaBounds()
        self.cbf_slack_weight = cbf_slack_weight
        self.clf_slack_weight = clf_slack_weight

        self.vps = {name: VehicleParams() for name in VEHICLE_NAMES}
        self.clfps = {name: CLFParams(clf_slack_weight=self.clf_slack_weight) for name in VEHICLE_NAMES}
        self.qpws = {name: QPWeights() for name in VEHICLE_NAMES}
        first_vp = self.vps[VEHICLE_NAMES[0]]
        self.circle_offsets, self.circle_radius = circle_approximation(
            first_vp.length, first_vp.width, n_circles=3
        )

        self.observation_dim = 32
        self.action_dim = 3

        self.rng = np.random.default_rng(seed)
        self.routes: Optional[Dict[str, RouteSpec]] = None
        self.states: Dict[str, Optional[np.ndarray]] = {name: None for name in VEHICLE_NAMES}
        self.u_prev = {name: np.zeros(2, dtype=float) for name in VEHICLE_NAMES}
        self.lambdas = {name: self.lambda_bounds.max_val for name in VEHICLE_NAMES}
        self.step_count = 0
        self.done = False
        self.last_info: Dict = {}
        self.episode_min_intervehicle_clearance = np.inf
        self.goal_reward_paid = {name: False for name in VEHICLE_NAMES}

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def sample_routes(self) -> Dict[str, RouteSpec]:
        return sample_route_set(self.rng, self.geom)

    def reset(
        self,
        seed: Optional[int] = None,
        route_set: Optional[Dict[str, RouteSpec]] = None,
        initial_states: Optional[Dict[str, RouteSpec]] = None,
    ) -> np.ndarray:
        if seed is not None:
            self.seed(seed)

        if initial_states is not None:
            route_set = initial_states
        if route_set is None:
            route_set = self.sample_routes()

        self.routes = route_set
        for name in VEHICLE_NAMES:
            self.states[name] = initial_state_from_route(self.routes[name], self.clfps[name].desired_speed)
            self.u_prev[name] = np.zeros(2, dtype=float)
            self.lambdas[name] = self.lambda_bounds.max_val
            self.goal_reward_paid[name] = False

        self.step_count = 0
        self.done = False
        self.last_info = {}
        self.episode_min_intervehicle_clearance = np.inf
        return self._get_obs()

    def _current_clearances(self) -> Tuple[float, Dict[str, float]]:
        inter_clear = min_multi_intervehicle_clearance(
            self.states, self.vps, self.circle_offsets, self.circle_radius
        )
        bound = {
            name: min_boundary_clearance(
                self.states[name],
                self.routes[name],
                self.vps[name],
                self.circle_offsets,
                self.circle_radius,
            )
            for name in VEHICLE_NAMES
        }
        return inter_clear, bound

    def _get_obs(self) -> np.ndarray:
        inter_clear, bound = self._current_clearances()
        decisions = route_decision_sequence(self.routes)
        obs = []
        positions = {}

        for name in VEHICLE_NAMES:
            state = self.states[name]
            route = self.routes[name]
            p = state[:2].copy()
            positions[name] = p
            obs.extend([
                p[0], p[1], state[2], state[3],
                self.u_prev[name][0],
                bound[name],
                float(route.start),
                float(route.exit),
            ])

        first_pos = positions[VEHICLE_NAMES[0]]
        for name in VEHICLE_NAMES[1:]:
            rel = positions[name] - first_pos
            obs.extend([rel[0], rel[1]])

        obs.append(inter_clear)
        for decision in decisions:
            obs.append(DECISION_ID[decision])

        return np.array(obs, dtype=np.float32)

    def _progress_switch_point(self, route: RouteSpec) -> np.ndarray:
        geom = route.geom
        return -geom.corner_offset * route.start_dir + geom.lane_offset * right_normal(route.start_dir)

    def _progress_for_vehicle(self, prev_state: np.ndarray, next_state: np.ndarray, route: RouteSpec) -> float:
        switch_point = self._progress_switch_point(route)
        inbound_s = float((prev_state[:2] - switch_point) @ route.start_dir)
        direction = route.start_dir if inbound_s < 0.0 else route.exit_dir
        return float((next_state[:2] - prev_state[:2]) @ direction)

    def _goal_status(self) -> Dict[str, bool]:
        return {
            name: bool(float((self.states[name][:2] - self.routes[name].exit_point) @ self.routes[name].exit_dir) >= 0.0)
            for name in VEHICLE_NAMES
        }

    def _goal_points(self) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        goals = {}
        modes = {}
        for name in VEHICLE_NAMES:
            goals[name], modes[name] = route_goal_from_position(self.states[name][:2], self.routes[name])
        return goals, modes

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() before step().")

        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape[0] != len(VEHICLE_NAMES):
            raise ValueError("Action must have shape (3,) for [lambda_i, lambda_j, lambda_k].")

        for idx, name in enumerate(VEHICLE_NAMES):
            self.lambdas[name] = lambda_from_normalized_action(
                action[idx], self.lambda_bounds.min_val, self.lambda_bounds.max_val
            )

        cbfps = {
            name: CBFParams(lambda_cbf=self.lambdas[name], cbf_slack_weight=self.cbf_slack_weight)
            for name in VEHICLE_NAMES
        }
        prev_states = {name: self.states[name].copy() for name in VEHICLE_NAMES}
        goals, modes = self._goal_points()
        controls = {}
        qp_infos = {}

        for name in VEHICLE_NAMES:
            other_names = [other for other in VEHICLE_NAMES if other != name]
            controls[name], qp_infos[name] = solve_vehicle_qp_against_many(
                ego_state=self.states[name].copy(),
                other_states=[self.states[other].copy() for other in other_names],
                ego_route=self.routes[name],
                ego_vp=self.vps[name],
                other_vps=[self.vps[other] for other in other_names],
                clfp=self.clfps[name],
                cbfp=cbfps[name],
                qpw=self.qpws[name],
                sim=self.sim,
                circle_offsets=self.circle_offsets,
                circle_radius=self.circle_radius,
                goal_point=goals[name],
            )

        infeasible = {
            name: qp_infos[name].get("status") not in OPTIMAL_STATUSES
            for name in VEHICLE_NAMES
        }
        self.step_count += 1

        if any(infeasible.values()):
            self.done = True
            for name in VEHICLE_NAMES:
                self.u_prev[name] = controls[name].copy()
            inter_clear, bound = self._current_clearances()
            reward = float(self.reward_cfg.infeasible_qp_penalty)
            info = self._build_info(
                controls, qp_infos, goals, modes, inter_clear, bound,
                {name: 0.0 for name in VEHICLE_NAMES},
                event="infeasible_qp", reward=reward,
            )
            info["infeasible"] = infeasible
            self.last_info = info
            return self._get_obs(), reward, True, info

        for name in VEHICLE_NAMES:
            self.states[name] = step_vehicle(self.states[name], controls[name], self.vps[name], self.sim.dt)
            self.u_prev[name] = controls[name].copy()

        inter_clear, bound = self._current_clearances()
        self.episode_min_intervehicle_clearance = min(self.episode_min_intervehicle_clearance, inter_clear)
        progress = {
            name: self._progress_for_vehicle(prev_states[name], self.states[name], self.routes[name])
            for name in VEHICLE_NAMES
        }
        deviations = {
            name: centerline_deviation(self.states[name][:2], self.routes[name])
            for name in VEHICLE_NAMES
        }
        reward = float(
            self.reward_cfg.progress_weight * sum(progress.values())
            - self.reward_cfg.deviation_weight * sum(deviations.values())
            + self.reward_cfg.survival_bonus
        )

        reached = self._goal_status()
        for name in VEHICLE_NAMES:
            if reached[name] and not self.goal_reward_paid[name]:
                reward += self.reward_cfg.goal_reward
                self.goal_reward_paid[name] = True

        collision = inter_clear < -1e-3
        boundary_collision = any(clear < -1e-3 for clear in bound.values())
        done = False
        event = "running"
        if collision:
            reward = float(self.reward_cfg.collision_penalty)
            done = True
            event = "collision"
        elif boundary_collision:
            reward = float(self.reward_cfg.boundary_collision_penalty)
            done = True
            event = "boundary_collision"
        elif self.step_count >= self.sim.steps:
            done = True

        self.done = done
        info = self._build_info(
            controls, qp_infos, goals, modes, inter_clear, bound, progress,
            event=event, reward=reward,
        )
        info["collision"] = collision
        info["boundary_collision"] = boundary_collision
        info["goal_reached"] = reached
        info["goal_reward_paid"] = self.goal_reward_paid.copy()
        info["centerline_deviation"] = deviations
        self.last_info = info
        return self._get_obs(), reward, done, info

    def _build_info(
        self,
        controls: Dict[str, np.ndarray],
        qp_infos: Dict[str, Dict],
        goals: Dict[str, np.ndarray],
        modes: Dict[str, str],
        inter_clear: float,
        bound: Dict[str, float],
        progress: Dict[str, float],
        event: str,
        reward: float,
    ) -> Dict:
        info = {
            "inter_vehicle_clearance": inter_clear,
            "event": event,
            "reward": reward,
            "step_count": self.step_count,
            "decision_sequence": route_decision_sequence(self.routes),
        }
        for name in VEHICLE_NAMES:
            info[f"lambda_{name}"] = self.lambdas[name]
            info[f"qp_{name}"] = qp_infos[name]
            info[f"u_{name}"] = controls[name].copy()
            info[f"goal_{name}"] = goals[name].copy()
            info[f"goal_mode_{name}"] = modes[name]
            info[f"progress_{name}"] = progress[name]
            info[f"boundary_clearance_{name}"] = bound[name]
        return info


class DecentralizedThreeVehicleIntersectionEnv(ThreeVehicleIntersectionEnv):
    """Nearest-first normalized ego-frame observation for shared decentralized training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_dim = 27
        self.action_dim = 1
        self.position_obs_scale = max(float(self.geom.start_distance), 1e-6)
        self.speed_obs_scale = max(float(self.clfps[VEHICLE_NAMES[0]].desired_speed), 1e-6)
        first_vp = self.vps[VEHICLE_NAMES[0]]
        self.accel_obs_scale = max(abs(first_vp.max_accel), abs(first_vp.min_accel), 1e-6)

    def _pair_clearance(self, name: str, other: str) -> float:
        return float(
            min_intervehicle_clearance(
                self.states[name],
                self.states[other],
                self.vps[name],
                self.vps[other],
                self.circle_offsets,
                self.circle_radius,
            )
        )

    def _get_obs(self) -> np.ndarray:
        _, bound = self._current_clearances()
        goals, _ = self._goal_points()
        decisions = route_decision_sequence(self.routes)
        positions = {name: self.states[name][:2].copy() for name in VEHICLE_NAMES}
        velocities = {
            name: self.states[name][3] * np.array([
                math.cos(self.states[name][2]),
                math.sin(self.states[name][2]),
            ], dtype=float)
            for name in VEHICLE_NAMES
        }
        local_obs = []

        for name in VEHICLE_NAMES:
            state = self.states[name]
            p = positions[name]
            goal = goals[name]
            psi_norm = ((state[2] + np.pi) % (2.0 * np.pi) - np.pi) / np.pi
            decision = decisions[VEHICLE_NAMES.index(name)]
            heading = np.array([math.cos(state[2]), math.sin(state[2])], dtype=float)
            left = np.array([-math.sin(state[2]), math.cos(state[2])], dtype=float)

            obs = [
                p[0] / self.position_obs_scale,
                p[1] / self.position_obs_scale,
                psi_norm,
                state[3] / self.speed_obs_scale,
                self.u_prev[name][0] / self.accel_obs_scale,
                bound[name] / self.position_obs_scale,
                goal[0] / self.position_obs_scale,
                goal[1] / self.position_obs_scale,
            ]

            other_names = sorted(
                [other for other in VEHICLE_NAMES if other != name],
                key=lambda other: self._pair_clearance(name, other),
            )
            for other in other_names:
                rel = positions[other] - p
                rel_v = velocities[other] - velocities[name]
                obs.extend([
                    float(rel @ heading) / self.position_obs_scale,
                    float(rel @ left) / self.position_obs_scale,
                    float(rel_v @ heading) / self.speed_obs_scale,
                    float(rel_v @ left) / self.speed_obs_scale,
                ])
                obs.append(self._pair_clearance(name, other) / self.position_obs_scale)
                obs.extend(one_hot_decision(decisions[VEHICLE_NAMES.index(other)]))

            obs.extend(one_hot_decision(decision))
            local_obs.append(obs)

        return np.array(local_obs, dtype=np.float32)


def run_policy_rollout(
    env: ThreeVehicleIntersectionEnv,
    action_fn: Callable[[np.ndarray], np.ndarray],
    initial_states: Optional[Dict[str, RouteSpec]] = None,
    step_callback: Optional[Callable[[Dict], None]] = None,
) -> Tuple[Dict, Dict]:
    obs = env.reset(initial_states=initial_states)
    inter_clear, bound = env._current_clearances()
    hist = {
        "names": list(VEHICLE_NAMES),
        "x": {name: [env.states[name].copy()] for name in VEHICLE_NAMES},
        "u": {name: [] for name in VEHICLE_NAMES},
        "goal": {name: [] for name in VEHICLE_NAMES},
        "lambda": {name: [env.lambdas[name]] for name in VEHICLE_NAMES},
        "boundary": {name: [bound[name]] for name in VEHICLE_NAMES},
        "clearance_iv": [inter_clear],
        "reward": [0.0],
        "reward_cum": [0.0],
        "events": [],
        "t": [0.0],
    }

    done = False
    total_reward = 0.0
    while not done:
        action = action_fn(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        for name in VEHICLE_NAMES:
            hist["x"][name].append(env.states[name].copy())
            hist["u"][name].append(info[f"u_{name}"].copy())
            hist["goal"][name].append(info[f"goal_{name}"].copy())
            hist["lambda"][name].append(info[f"lambda_{name}"])
            hist["boundary"][name].append(info[f"boundary_clearance_{name}"])
        hist["clearance_iv"].append(info["inter_vehicle_clearance"])
        hist["reward"].append(reward)
        hist["reward_cum"].append(total_reward)
        hist["events"].append(info["event"])
        hist["t"].append(env.step_count * env.sim.dt)

        if step_callback is not None:
            step_callback(info)

    hist["t"] = np.array(hist["t"])
    hist["clearance_iv"] = np.array(hist["clearance_iv"])
    hist["reward"] = np.array(hist["reward"])
    hist["reward_cum"] = np.array(hist["reward_cum"])
    for name in VEHICLE_NAMES:
        for key in ["x", "u", "goal", "lambda", "boundary"]:
            hist[key][name] = np.array(hist[key][name])
        if hist["u"][name].size == 0:
            hist["u"][name] = hist["u"][name].reshape(0, 2)
        if hist["goal"][name].size == 0:
            hist["goal"][name] = hist["goal"][name].reshape(0, 2)

    vehicle_data = {
        name: {
            "route": env.routes[name],
            "vp": env.vps[name],
        }
        for name in VEHICLE_NAMES
    }
    metrics = compute_multi_rollout_metrics(
        hist=hist,
        vehicle_data=vehicle_data,
        circle_offsets=env.circle_offsets,
        circle_radius=env.circle_radius,
    )
    metrics["episode_return"] = float(np.sum(hist["reward"]))
    return hist, metrics


def average_rollout_metrics(all_metrics: List[Dict]) -> Dict:
    n_rollouts = len(all_metrics)
    avg = {name: {} for name in VEHICLE_NAMES}
    valid_motion_metrics = [m for m in all_metrics if m.get("valid_motion_metrics", 1)]
    completion_metrics = [m for m in valid_motion_metrics if not m.get("deadlock", 0)]

    def mean_valid(name: str, key: str, nanmean: bool = False) -> float:
        if not valid_motion_metrics:
            return float("nan")
        values = [m[name][key] for m in valid_motion_metrics]
        if nanmean:
            finite_values = [v for v in values if not np.isnan(v)]
            return float(np.mean(finite_values)) if finite_values else float("nan")
        return float(np.mean(values))

    def rollout_completion_time(m: Dict) -> float:
        times = np.array([m[name]["time_to_goal"] for name in VEHICLE_NAMES], dtype=float)
        if np.any(np.isnan(times)):
            return float("nan")
        return float(np.max(times))

    for name in VEHICLE_NAMES:
        avg[name]["time_to_goal"] = mean_valid(name, "time_to_goal", nanmean=True)
        avg[name]["avg_centerline_deviation"] = mean_valid(name, "avg_centerline_deviation")
        avg[name]["acc_effort"] = mean_valid(name, "acc_effort")
        avg[name]["steer_rate_effort"] = mean_valid(name, "steer_rate_effort")
        avg[name]["intervehicle_collision"] = float(np.mean([m[name]["intervehicle_collision"] for m in all_metrics]))
        avg[name]["boundary_collision"] = float(np.mean([m[name]["boundary_collision"] for m in all_metrics]))
        avg[name]["min_intervehicle_clearance"] = mean_valid(name, "min_intervehicle_clearance")
        avg[name]["min_boundary_clearance"] = mean_valid(name, "min_boundary_clearance")

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
        int(any(m[name]["intervehicle_collision"] for name in VEHICLE_NAMES))
        for m in all_metrics
    ]))
    avg["boundary_collision_count"] = int(np.sum([
        int(any(m[name]["boundary_collision"] for name in VEHICLE_NAMES))
        for m in all_metrics
    ]))
    avg["collision_count"] = int(np.sum([
        int(any(m[name]["intervehicle_collision"] or m[name]["boundary_collision"] for name in VEHICLE_NAMES))
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
    import matplotlib.pyplot as plt

    def fmt_nan(x):
        return "NaN" if np.isnan(x) else f"{x:.3f}"

    def fmt_percent(x):
        return "NaN" if np.isnan(x) else f"{x:.1f}%"

    rows = [
        "Task Completion Time [s]",
        "Collision Rate",
        "Deadlock Rate",
    ]
    system = avg_metrics["system"]
    table_data = [
        [fmt_nan(system["completion_time"])],
        [f"{system['collision_count']}/{n_rollouts} ({fmt_percent(system['collision_rate'])})"],
        [f"{system['deadlock_count']}/{n_rollouts} ({fmt_percent(system['deadlock_rate'])})"],
    ]

    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.axis("off")
    ax.set_title(f"Three-Vehicle RI Metrics Summary over {n_rollouts} Rollouts", fontsize=13, pad=10)
    table = ax.table(
        cellText=table_data,
        rowLabels=rows,
        colLabels=["Process"],
        cellLoc="center",
        rowLoc="center",
        loc="center",
        bbox=[0.15, 0.05, 0.80, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    plt.tight_layout()
    plt.show()
    return avg_metrics
