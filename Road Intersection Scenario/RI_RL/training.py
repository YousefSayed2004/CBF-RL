import csv
from itertools import product
import math
import os
import signal
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.distributions import Normal
from torch.utils.data import DataLoader, IterableDataset

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, WORKSPACE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from .env import (
        LambdaBounds,
        RewardConfig,
        RouteSpec,
        SimParams,
        ThreeVehicleIntersectionEnv,
        VEHICLE_NAMES,
        START_BY_NAME,
        average_rollout_metrics,
        build_fixed_initial_states,
        route_decision_sequence,
        run_policy_rollout,
    )
except ImportError:
    from env import (
        LambdaBounds,
        RewardConfig,
        RouteSpec,
        SimParams,
        ThreeVehicleIntersectionEnv,
        VEHICLE_NAMES,
        START_BY_NAME,
        average_rollout_metrics,
        build_fixed_initial_states,
        route_decision_sequence,
        run_policy_rollout,
    )

from RI_Baseline_Methods import min_intervehicle_clearance  # noqa: E402


ALL_DECISION_SEQUENCES = tuple(
    first + "".join(rest)
    for first in ("R", "L", "S")
    for rest in product(("L", "S", "R"), repeat=2)
)
DECISION_TO_REL_EXIT = {"R": 1, "S": 2, "L": 3}
DECISION_ONE_HOT_ORDER = ("S", "R", "L")
DECISION_TO_HEAD_ID = {decision: idx for idx, decision in enumerate(DECISION_ONE_HOT_ORDER)}
N_AGENTS = len(VEHICLE_NAMES)


def route_set_from_decision_sequence(sequence: str, geom) -> Dict[str, RouteSpec]:
    routes = {}
    for name, decision in zip(VEHICLE_NAMES, sequence):
        start = START_BY_NAME[name]
        exit_lane = ((start - 1 + DECISION_TO_REL_EXIT[decision]) % 4) + 1
        routes[name] = RouteSpec(start=start, exit=exit_lane, geom=geom)
    return routes


def one_hot_decision(decision: str) -> list:
    return [1.0 if decision == option else 0.0 for option in DECISION_ONE_HOT_ORDER]


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


@dataclass
class TrainConfig:
    hidden_size: int = 256
    num_envs: int = 8
    samples_per_epoch: int = 128
    epoch_repeat: int = 8
    batch_size: int = 128
    policy_lr: float = 1e-4
    value_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.005
    max_epochs: int = 4000
    eval_every: int = 10
    eval_seed: int = 7
    eval_rollouts: int = 20
    bad_epoch_return_threshold: float = -1.0
    T: float = 10.0
    dt: float = 0.1
    best_score_override: Optional[float] = None
    worst_score_override: Optional[float] = None
    lambda_min: float = 0.1
    lambda_max: float = 0.4
    cbf_slack_weight: float = 1e8
    clf_slack_weight: float = 1.0
    progress_weight: float = 3.3 / 100
    deviation_weight: float = 0.15 / 100
    survival_bonus: float = 0.5 / 100
    goal_reward: float = 75.0 / 100
    collision_penalty: float = -100.0 / 100
    boundary_collision_penalty: float = -100.0 / 100
    infeasible_qp_penalty: float = -150.0 / 100
    project_root: str = str(PACKAGE_ROOT)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PlotState:
    episode_count: float = 0.0
    current_epoch: int = 0
    average_reward: list = None


class GradientPolicy(nn.Module):
    def __init__(self, in_features: int, hidden_size: int = 128, out_dims: int = 1):
        super().__init__()
        self.out_dims = out_dims
        self.features = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict({
            decision: nn.Linear(hidden_size, out_dims)
            for decision in ("R", "L", "S")
        })
        self.log_std = nn.ParameterDict({
            decision: nn.Parameter(torch.full((out_dims,), -1.0))
            for decision in ("R", "L", "S")
        })

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        loc = torch.empty((x.shape[0], self.out_dims), dtype=x.dtype, device=x.device)
        log_std = torch.empty_like(loc)
        decision_ids = torch.argmax(x[:, -len(DECISION_ONE_HOT_ORDER):], dim=1)

        for decision, decision_id in DECISION_TO_HEAD_ID.items():
            mask = decision_ids == decision_id
            if not torch.any(mask):
                continue
            loc[mask] = torch.tanh(self.heads[decision](h[mask]))
            log_std[mask] = self.log_std[decision].expand(int(mask.sum().item()), -1)

        scale = torch.exp(log_std)
        return loc, scale


class ValueNet(nn.Module):
    def __init__(self, in_features: int, hidden_size: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


class RLDataset(IterableDataset):
    def __init__(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        batch_size: int,
        epoch_repeat: int,
    ):
        super().__init__()
        self.obs = obs
        self.actions = actions
        self.old_log_probs = old_log_probs
        self.advantages = advantages
        self.returns = returns
        self.batch_size = batch_size
        self.epoch_repeat = epoch_repeat
        self.n_samples = obs.shape[0]

    def __iter__(self):
        for _ in range(self.epoch_repeat):
            indices = np.random.permutation(self.n_samples)
            for start in range(0, self.n_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                yield (
                    self.obs[batch_idx],
                    self.actions[batch_idx],
                    self.old_log_probs[batch_idx],
                    self.advantages[batch_idx],
                    self.returns[batch_idx],
                )


class PPOTrainer:
    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.training_decision_sequences = ALL_DECISION_SEQUENCES
        self.device = torch.device(config.device)
        self.plot = PlotState(episode_count=0.0, average_reward=[])
        self.running_returns = np.zeros(self.cfg.num_envs, dtype=np.float32)

        self.next_sequence_index = 0

        self.envs = [self.make_env(i) for i in range(config.num_envs)]
        self.obs_dim = self.envs[0].observation_dim
        self.action_dim = self.envs[0].action_dim
        self.n_agents = N_AGENTS

        self.policy = GradientPolicy(self.obs_dim, config.hidden_size, self.action_dim).to(self.device)
        self.value_net = ValueNet(self.obs_dim, config.hidden_size).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=config.value_lr)

        self.project_root = os.fspath(Path(config.project_root).resolve())
        self.checkpoint_dir = os.path.join(self.project_root, "checkpoints")
        self.log_dir = os.path.join(self.project_root, "logs")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.latest_ckpt = os.path.join(self.checkpoint_dir, "latest.pt")
        self.best_ckpt = os.path.join(self.checkpoint_dir, "best.pt")
        self.worst_ckpt = os.path.join(self.checkpoint_dir, "worst.pt")
        self.interrupt_ckpt = os.path.join(self.checkpoint_dir, "interrupt.pt")
        self.csv_log = os.path.join(self.log_dir, "training_history.csv")
        self.best_score = -float("inf")
        self.worst_score = float("inf")
        self.start_epoch = 1
        self.stop_requested = False

        self._register_interrupt_handler()
        self._try_resume()
        self._sync_best_worst_scores_from_checkpoints()
        self.obs_list = [self.reset_env_ordered(env) for env in self.envs]
        print(
            f"Training decentralized three-vehicle multi-head policy on "
            f"{len(self.training_decision_sequences)} ordered scenarios: "
            f"{', '.join(self.training_decision_sequences)}"
        )
        print(f"Evaluation uses {self.cfg.eval_rollouts} random rollouts from seed {self.cfg.eval_seed}.")
        print(f"Outputs: {self.log_dir} and {self.checkpoint_dir}")

    def make_env(self, seed: int) -> DecentralizedThreeVehicleIntersectionEnv:
        reward_cfg = RewardConfig(
            collision_penalty=self.cfg.collision_penalty,
            infeasible_qp_penalty=self.cfg.infeasible_qp_penalty,
            boundary_collision_penalty=self.cfg.boundary_collision_penalty,
            progress_weight=self.cfg.progress_weight,
            deviation_weight=self.cfg.deviation_weight,
            survival_bonus=self.cfg.survival_bonus,
            goal_reward=self.cfg.goal_reward,
        )
        return DecentralizedThreeVehicleIntersectionEnv(
            sim=SimParams(dt=self.cfg.dt, T=self.cfg.T),
            reward_cfg=reward_cfg,
            lambda_bounds=LambdaBounds(min_val=self.cfg.lambda_min, max_val=self.cfg.lambda_max),
            seed=seed,
            cbf_slack_weight=self.cfg.cbf_slack_weight,
            clf_slack_weight=self.cfg.clf_slack_weight,
        )

    def _register_interrupt_handler(self):
        def handle_interrupt(signum, frame):
            self.stop_requested = True
            print("\nInterrupt received. Will save checkpoint at end of this epoch.")
        signal.signal(signal.SIGINT, handle_interrupt)

    def _try_resume(self):
        if os.path.exists(self.latest_ckpt):
            try:
                checkpoint = torch.load(self.latest_ckpt, map_location=self.device, weights_only=False)
                self.policy.load_state_dict(checkpoint["policy_state_dict"])
                self.value_net.load_state_dict(checkpoint["value_state_dict"])
                self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
                self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
            except (RuntimeError, KeyError, ValueError) as exc:
                print(
                    f"Found an incompatible latest checkpoint at {self.latest_ckpt}. "
                    f"Starting a fresh multi-head run instead. Reason: {exc}"
                )
                return

            self.start_epoch = checkpoint["epoch"] + 1
            self.plot.average_reward = checkpoint.get("average_reward", [])
            self.running_returns = checkpoint.get("running_returns", self.running_returns)
            self.next_sequence_index = checkpoint.get("next_sequence_index", self.next_sequence_index)
            print(f"Resuming training from epoch {self.start_epoch}")

    def _load_checkpoint_score(self, path: str, key: str) -> Optional[float]:
        if not os.path.exists(path):
            return None
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as exc:
            print(f"Could not read {key} from {path}: {exc}")
            return None
        value = checkpoint.get(key)
        return None if value is None else float(value)

    def _sync_best_worst_scores_from_checkpoints(self) -> None:
        if self.cfg.best_score_override is not None:
            self.best_score = float(self.cfg.best_score_override)
        else:
            best_score = self._load_checkpoint_score(self.best_ckpt, "best_score")
            if best_score is not None:
                self.best_score = best_score

        if self.cfg.worst_score_override is not None:
            self.worst_score = float(self.cfg.worst_score_override)
        else:
            worst_score = self._load_checkpoint_score(self.worst_ckpt, "worst_score")
            if worst_score is not None:
                self.worst_score = worst_score

    def save_checkpoint(self, path: str, epoch: int):
        torch.save({
            "epoch": epoch,
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value_net.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "best_score": self.best_score,
            "worst_score": self.worst_score,
            "config": self.cfg.__dict__,
            "average_reward": self.plot.average_reward,
            "running_returns": self.running_returns,
            "next_sequence_index": self.next_sequence_index,
            "training_decision_sequences": self.training_decision_sequences,
        }, path)

    def zero_collision_checkpoint_path(self) -> str:
        index = 1
        while True:
            path = os.path.join(self.checkpoint_dir, f"zero_coll_{index}.pt")
            if not os.path.exists(path):
                return path
            index += 1

    def reset_env_ordered(self, env: DecentralizedThreeVehicleIntersectionEnv) -> np.ndarray:
        sequence = self.training_decision_sequences[self.next_sequence_index % len(self.training_decision_sequences)]
        self.next_sequence_index += 1
        return env.reset(initial_states=route_set_from_decision_sequence(sequence, env.geom))

    def collect_rollouts(self):
        batch_obs, batch_actions, batch_log_probs = [], [], []
        batch_rewards, batch_dones, batch_values, batch_next_values = [], [], [], []
        finished_episode_returns = []

        for _ in range(self.cfg.samples_per_epoch):
            obs_np = np.stack(self.obs_list, axis=0)
            obs_flat_np = obs_np.reshape(-1, self.obs_dim)
            obs_t = torch.tensor(obs_flat_np, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                loc, scale = self.policy(obs_t)
                dist = Normal(loc, scale)
                actions_t = torch.clamp(dist.sample(), -1.0, 1.0)
                log_probs_t = dist.log_prob(actions_t).sum(dim=-1)
                values_t = self.value_net(obs_t)

            actions_np = actions_t.cpu().numpy().reshape(self.cfg.num_envs, self.n_agents, self.action_dim)
            log_probs_np = log_probs_t.cpu().numpy().reshape(self.cfg.num_envs, self.n_agents)
            values_np = values_t.cpu().numpy().reshape(self.cfg.num_envs, self.n_agents)
            next_obs_list, rewards, dones = [], [], []

            for env_idx, env in enumerate(self.envs):
                next_obs, reward, done, _ = env.step(actions_np[env_idx])
                self.running_returns[env_idx] += reward
                if done:
                    next_obs_list.append(self.reset_env_ordered(env))
                    self.plot.episode_count += 1.0
                    finished_episode_returns.append(float(self.running_returns[env_idx]))
                    self.running_returns[env_idx] = 0.0
                else:
                    next_obs_list.append(next_obs)
                rewards.append(reward)
                dones.append(float(done))

            next_obs_np = np.stack(next_obs_list, axis=0)
            next_obs_flat_np = next_obs_np.reshape(-1, self.obs_dim)
            next_obs_t = torch.tensor(next_obs_flat_np, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                next_values_np = self.value_net(next_obs_t).cpu().numpy().reshape(self.cfg.num_envs, self.n_agents)

            batch_obs.append(obs_np)
            batch_actions.append(actions_np)
            batch_log_probs.append(log_probs_np)
            batch_rewards.append(
                np.repeat(np.array(rewards, dtype=np.float32)[:, None], self.n_agents, axis=1)
            )
            batch_dones.append(
                np.repeat(np.array(dones, dtype=np.float32)[:, None], self.n_agents, axis=1)
            )
            batch_values.append(values_np)
            batch_next_values.append(next_values_np)
            self.obs_list = next_obs_list

        obs_arr = np.asarray(batch_obs, dtype=np.float32)
        actions_arr = np.asarray(batch_actions, dtype=np.float32)
        log_probs_arr = np.asarray(batch_log_probs, dtype=np.float32)
        rewards_arr = np.asarray(batch_rewards, dtype=np.float32)
        dones_arr = np.asarray(batch_dones, dtype=np.float32)
        values_arr = np.asarray(batch_values, dtype=np.float32)
        next_values_arr = np.asarray(batch_next_values, dtype=np.float32)

        advantages_arr, returns_arr = self.compute_gae_all_envs(rewards_arr, dones_arr, values_arr, next_values_arr)
        flat_advantages = advantages_arr.reshape(-1)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        avg_finished_episode_return = float(np.mean(finished_episode_returns)) if finished_episode_returns else np.nan

        return (
            obs_arr.reshape(-1, self.obs_dim),
            actions_arr.reshape(-1, self.action_dim),
            log_probs_arr.reshape(-1),
            flat_advantages,
            returns_arr.reshape(-1),
            avg_finished_episode_return,
        )

    def compute_gae(self, rewards: np.ndarray, dones: np.ndarray, values: np.ndarray, next_values: np.ndarray):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_values[t] * mask - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * gae
            advantages[t] = gae
        return advantages, advantages + values

    def compute_gae_all_envs(self, rewards_arr, dones_arr, values_arr, next_values_arr):
        advantages_all, returns_all = [], []
        for env_idx in range(self.cfg.num_envs):
            env_advantages, env_returns = [], []
            for agent_idx in range(self.n_agents):
                adv, ret = self.compute_gae(
                    rewards_arr[:, env_idx, agent_idx],
                    dones_arr[:, env_idx, agent_idx],
                    values_arr[:, env_idx, agent_idx],
                    next_values_arr[:, env_idx, agent_idx],
                )
                env_advantages.append(adv)
                env_returns.append(ret)
            advantages_all.append(np.stack(env_advantages, axis=1))
            returns_all.append(np.stack(env_returns, axis=1))
        return np.stack(advantages_all, axis=1).astype(np.float32), np.stack(returns_all, axis=1).astype(np.float32)

    def policy_loss(self, obs, actions, old_log_probs, advantages):
        loc, scale = self.policy(obs)
        dist = Normal(loc, scale)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * advantages
        return -torch.min(surr1, surr2).mean() - self.cfg.entropy_coef * entropy, entropy

    def value_loss(self, obs, returns):
        values_pred = self.value_net(obs)
        return 0.5 * ((values_pred - returns) ** 2).mean()

    def evaluate(self) -> Dict:
        eval_env = self.make_env(self.cfg.eval_seed)
        initial_states = build_fixed_initial_states(
            n_rollouts=self.cfg.eval_rollouts,
            seed=self.cfg.eval_seed,
            geom=eval_env.geom,
        )
        all_metrics = []

        def action_fn(obs_np: np.ndarray) -> np.ndarray:
            obs_t = torch.tensor(obs_np.reshape(-1, self.obs_dim), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                loc, _ = self.policy(obs_t)
            return loc.cpu().numpy()

        for init_states in initial_states:
            _, metrics = run_policy_rollout(eval_env, action_fn=action_fn, initial_states=init_states)
            all_metrics.append(metrics)

        eval_metrics = average_rollout_metrics(all_metrics)
        eval_metrics["intervehicle_collision_count"] = int(np.sum([
            int(any(m[name]["intervehicle_collision"] for name in VEHICLE_NAMES))
            for m in all_metrics
        ]))
        eval_metrics["boundary_collision_count"] = int(np.sum([
            sum(m[name]["boundary_collision"] for name in VEHICLE_NAMES)
            for m in all_metrics
        ]))
        eval_metrics["infeasible_qp_count"] = int(np.sum([m["infeasible_qp"] for m in all_metrics]))
        eval_metrics["deadlock_count"] = int(np.sum([m["deadlock"] for m in all_metrics if m.get("valid_motion_metrics", 1)]))
        return eval_metrics

    def append_csv_row(self, row: Dict):
        file_exists = os.path.exists(self.csv_log)
        fieldnames = list(row.keys())
        if file_exists:
            with open(self.csv_log, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_fieldnames = reader.fieldnames or []
                existing_rows = list(reader)
            if existing_fieldnames != fieldnames:
                merged_fieldnames = list(existing_fieldnames)
                for field in fieldnames:
                    if field not in merged_fieldnames:
                        merged_fieldnames.append(field)
                with open(self.csv_log, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
                    writer.writeheader()
                    for existing_row in existing_rows:
                        writer.writerow(existing_row)
                fieldnames = merged_fieldnames

        with open(self.csv_log, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def train(self):
        if not self.plot.average_reward:
            self.plot.average_reward = []

        for epoch in range(self.start_epoch, self.cfg.max_epochs + 1):
            self.plot.episode_count = 0.0
            self.plot.current_epoch = epoch

            obs_np, actions_np, old_log_probs_np, advantages_np, returns_np, avg_return = self.collect_rollouts()
            if not np.isnan(avg_return) and avg_return <= self.cfg.bad_epoch_return_threshold:
                print(
                    f"Epoch {epoch:03d}: Average Episode Return {avg_return:.6f} "
                    f"is at or below {self.cfg.bad_epoch_return_threshold:.3f}. "
                    "Terminating training before PPO update and without saving latest.pt."
                )
                break

            self.plot.average_reward.append(
                self.plot.average_reward[-1] if np.isnan(avg_return) and self.plot.average_reward else (0.0 if np.isnan(avg_return) else avg_return)
            )

            plt.plot(range(1, len(self.plot.average_reward) + 1), self.plot.average_reward, label="Avg Reward per Episode")
            plt.xlabel("Epochs")
            plt.ylabel("Average Reward per Episode")
            plt.title("Learning Curve")
            plt.savefig(os.path.join(self.log_dir, "learning_curve_epoch_.png"))
            plt.clf()

            dataset = RLDataset(obs_np, actions_np, old_log_probs_np, advantages_np, returns_np, self.cfg.batch_size, self.cfg.epoch_repeat)
            dataloader = DataLoader(dataset, batch_size=None)

            policy_loss_epoch = 0.0
            value_loss_epoch = 0.0
            n_updates = 0
            for batch in dataloader:
                obs_b, actions_b, old_log_probs_b, adv_b, returns_b = batch
                obs_t = torch.tensor(obs_b, dtype=torch.float32, device=self.device)
                actions_t = torch.tensor(actions_b, dtype=torch.float32, device=self.device)
                old_log_probs_t = torch.tensor(old_log_probs_b, dtype=torch.float32, device=self.device)
                adv_t = torch.tensor(adv_b, dtype=torch.float32, device=self.device)
                returns_t = torch.tensor(returns_b, dtype=torch.float32, device=self.device)

                p_loss, _ = self.policy_loss(obs_t, actions_t, old_log_probs_t, adv_t)
                self.policy_optimizer.zero_grad()
                p_loss.backward()
                self.policy_optimizer.step()

                v_loss = self.value_loss(obs_t, returns_t)
                self.value_optimizer.zero_grad()
                v_loss.backward()
                self.value_optimizer.step()

                policy_loss_epoch += float(p_loss.item())
                value_loss_epoch += float(v_loss.item())
                n_updates += 1

            if n_updates > 0:
                policy_loss_epoch /= n_updates
                value_loss_epoch /= n_updates

            print(
                f"Epoch {epoch:03d}/{self.cfg.max_epochs} | "
                f"Policy Loss: {policy_loss_epoch:.6f} | "
                f"Value Loss: {value_loss_epoch:.6f} | "
                f"Average Episode Return: {avg_return:.6f}"
            )

            if epoch % self.cfg.eval_every == 0 or epoch == 1:
                eval_metrics = self.evaluate()
                csv_row = {
                    "epoch": epoch,
                    "avg_reward": round(eval_metrics["avg_episode_return"], 6),
                    "valid_rollouts": int(eval_metrics["valid_motion_rollouts"]),
                    "collision_count": eval_metrics["intervehicle_collision_count"],
                    "boundary_collision_count": eval_metrics["boundary_collision_count"],
                    "infeasible_qp_count": eval_metrics["infeasible_qp_count"],
                    "deadlock_count": eval_metrics["deadlock_count"],
                }
                for name in VEHICLE_NAMES:
                    csv_row[f"avg_time_to_goal_{name}"] = (
                        round(eval_metrics[name]["time_to_goal"], 6)
                        if not math.isnan(eval_metrics[name]["time_to_goal"])
                        else float("nan")
                    )
                self.append_csv_row(csv_row)

                eval_score = float(eval_metrics["avg_episode_return"])
                if eval_score > self.best_score:
                    self.best_score = eval_score
                    self.save_checkpoint(self.best_ckpt, epoch)
                    print(f"  Saved new best evaluated checkpoint ({eval_score:.6f}) -> {self.best_ckpt}")
                if eval_score < self.worst_score:
                    self.worst_score = eval_score
                    self.save_checkpoint(self.worst_ckpt, epoch)
                    print(f"  Saved new worst evaluated checkpoint ({eval_score:.6f}) -> {self.worst_ckpt}")

                if (
                    eval_metrics["intervehicle_collision_count"] == 0
                    and eval_metrics["boundary_collision_count"] == 0
                ):
                    zero_coll_ckpt = self.zero_collision_checkpoint_path()
                    self.save_checkpoint(zero_coll_ckpt, epoch)
                    print(f"  Saved zero-collision checkpoint -> {zero_coll_ckpt}")

            self.save_checkpoint(self.latest_ckpt, epoch)

            if self.stop_requested:
                self.save_checkpoint(self.interrupt_ckpt, epoch)
                print(f"Saved interrupt checkpoint -> {self.interrupt_ckpt}")
                print("Training stopped safely.")
                break

def main():
    cfg = TrainConfig()
    trainer = PPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
