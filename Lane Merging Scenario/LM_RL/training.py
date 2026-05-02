import csv
import math
import os
import signal
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.distributions import Normal
from torch.utils.data import DataLoader, IterableDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .env import NAMES, ThreeVehicleLambdaEnv
except ImportError:
    from env import NAMES, ThreeVehicleLambdaEnv


@dataclass
class TrainConfig:
    hidden_size: int = 128
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
    max_epochs: int = 200
    eval_every: int = 20
    train_seed: int = 42
    eval_seed: int = 7
    eval_rollouts: int = 10
    project_root: str = "."
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PlotState:
    episode_count: float = 0.0
    current_epoch: int = 0
    average_reward: list = None


class GradientPolicy(nn.Module):
    def __init__(self, in_features: int, hidden_size: int = 128, out_dims: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.loc = nn.Linear(hidden_size, out_dims)
        self.log_std = nn.Parameter(torch.zeros(out_dims))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        loc = torch.tanh(self.loc(h))
        scale = torch.exp(self.log_std).expand_as(loc)
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
        self.device = torch.device(config.device)
        self.plot = PlotState(episode_count=0.0, average_reward=[])
        self.running_returns = np.zeros(self.cfg.num_envs, dtype=np.float32)

        torch.manual_seed(config.train_seed)
        np.random.seed(config.train_seed)

        self.envs = [ThreeVehicleLambdaEnv(seed=config.train_seed + i) for i in range(config.num_envs)]
        self.obs_list = [env.reset() for env in self.envs]
        self.obs_dim = self.envs[0].observation_dim
        self.action_dim = self.envs[0].action_dim

        self.policy = GradientPolicy(self.obs_dim, config.hidden_size, self.action_dim).to(self.device)
        self.value_net = ValueNet(self.obs_dim, config.hidden_size).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=config.value_lr)

        self.project_root = config.project_root
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
        self.start_epoch = 1
        self.stop_requested = False
        self.best_testing_count = 0

        self._register_interrupt_handler()
        self._try_resume()

    def _register_interrupt_handler(self):
        def handle_interrupt(signum, frame):
            self.stop_requested = True
            print("\nInterrupt received. Will save checkpoint at end of this epoch.")
        signal.signal(signal.SIGINT, handle_interrupt)

    def _try_resume(self):
        if os.path.exists(self.latest_ckpt):
            checkpoint = torch.load(self.latest_ckpt, map_location=self.device, weights_only=False)
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.value_net.load_state_dict(checkpoint["value_state_dict"])
            self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
            self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_score = checkpoint["best_score"]
            self.plot.average_reward = checkpoint.get("average_reward", [])
            self.running_returns = checkpoint.get("running_returns", self.running_returns)
            self.best_testing_count = checkpoint.get("best_testing_count", 0)
            print(f"Resuming training from epoch {self.start_epoch}")

    def save_checkpoint(self, path: str, epoch: int):
        torch.save({
            "epoch": epoch,
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value_net.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "best_score": self.best_score,
            "config": self.cfg.__dict__,
            "average_reward": self.plot.average_reward,
            "running_returns": self.running_returns,
            "best_testing_count": self.best_testing_count,
        }, path)

    def collect_rollouts(self):
        batch_obs, batch_actions, batch_log_probs = [], [], []
        batch_rewards, batch_dones, batch_values, batch_next_values = [], [], [], []
        finished_episode_returns = []

        for _ in range(self.cfg.samples_per_epoch):
            obs_np = np.stack(self.obs_list, axis=0)
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                loc, scale = self.policy(obs_t)
                dist = Normal(loc, scale)
                actions_t = torch.clamp(dist.sample(), -1.0, 1.0)
                log_probs_t = dist.log_prob(actions_t).sum(dim=-1)
                values_t = self.value_net(obs_t)

            actions_np = actions_t.cpu().numpy()
            next_obs_list, rewards, dones = [], [], []

            for env_idx, env in enumerate(self.envs):
                next_obs, reward, done, _ = env.step(actions_np[env_idx])
                self.running_returns[env_idx] += reward
                if done:
                    next_obs_list.append(env.reset())
                    self.plot.episode_count += 1.0
                    finished_episode_returns.append(float(self.running_returns[env_idx]))
                    self.running_returns[env_idx] = 0.0
                else:
                    next_obs_list.append(next_obs)
                rewards.append(reward)
                dones.append(float(done))

            next_obs_np = np.stack(next_obs_list, axis=0)
            next_obs_t = torch.tensor(next_obs_np, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                next_values_t = self.value_net(next_obs_t)

            batch_obs.append(obs_np)
            batch_actions.append(actions_np)
            batch_log_probs.append(log_probs_t.cpu().numpy())
            batch_rewards.append(np.array(rewards, dtype=np.float32))
            batch_dones.append(np.array(dones, dtype=np.float32))
            batch_values.append(values_t.cpu().numpy())
            batch_next_values.append(next_values_t.cpu().numpy())
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

        avg_finished_episode_return = (
            float(np.mean(finished_episode_returns)) if finished_episode_returns else np.nan
        )

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
            adv, ret = self.compute_gae(
                rewards_arr[:, env_idx],
                dones_arr[:, env_idx],
                values_arr[:, env_idx],
                next_values_arr[:, env_idx],
            )
            advantages_all.append(adv)
            returns_all.append(ret)
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
        from env import average_rollout_metrics, build_fixed_initial_states, run_policy_rollout

        eval_env = ThreeVehicleLambdaEnv(seed=self.cfg.eval_seed)
        initial_states = build_fixed_initial_states(self.cfg.eval_rollouts, self.cfg.eval_seed)
        all_metrics = []

        def action_fn(obs_np: np.ndarray) -> np.ndarray:
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                loc, _ = self.policy(obs_t)
            return loc.squeeze(0).cpu().numpy()

        for init_states in initial_states:
            _, metrics = run_policy_rollout(eval_env, action_fn=action_fn, initial_states=init_states)
            all_metrics.append(metrics)
        return average_rollout_metrics(all_metrics)

    def append_csv_row(self, row: Dict):
        file_exists = os.path.exists(self.csv_log)
        with open(self.csv_log, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
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
                    "collision_rate": round(max(eval_metrics[name]["intervehicle_collision"] for name in NAMES), 6),
                    "boundary_collision_rate": round(sum(eval_metrics[name]["boundary_collision"] for name in NAMES), 6),
                    "infeasible_qp_rate": round(eval_metrics["infeasible_qp_rate"], 6),
                    "deadlock_rate": round(eval_metrics["deadlock_rate"], 6),
                }
                for name in NAMES:
                    t_goal = eval_metrics[name]["time_to_goal"]
                    csv_row[f"avg_time_to_goal_{name}"] = round(t_goal, 6) if not math.isnan(t_goal) else float("nan")
                    csv_row[f"avg_abs_y_{name}"] = round(eval_metrics[name]["avg_abs_y"], 6)
                    csv_row[f"min_intervehicle_clearance_{name}"] = round(eval_metrics[name]["min_intervehicle_clearance"], 6)
                self.append_csv_row(csv_row)

                if (
                    all(eval_metrics[name]["intervehicle_collision"] == 0.0 for name in NAMES)
                    and all(eval_metrics[name]["boundary_collision"] == 0.0 for name in NAMES)
                    and eval_metrics["infeasible_qp_rate"] == 0.0
                ):
                    self.best_testing_count += 1
                    path = os.path.join(self.checkpoint_dir, f"best_testing_{self.best_testing_count}.pt")
                    self.save_checkpoint(path, epoch)
                    print(f"  Saved zero-collision test checkpoint -> {path}")

            self.save_checkpoint(self.latest_ckpt, epoch)
            if self.plot.average_reward[-1] == max(self.plot.average_reward):
                self.save_checkpoint(self.best_ckpt, epoch)
                print(f"  Saved new best checkpoint -> {self.best_ckpt}")
            if self.plot.average_reward[-1] == min(self.plot.average_reward):
                self.save_checkpoint(self.worst_ckpt, epoch)
                print(f"  Saved new worst checkpoint -> {self.worst_ckpt}")

            if self.stop_requested:
                self.save_checkpoint(self.interrupt_ckpt, epoch)
                print(f"Saved interrupt checkpoint -> {self.interrupt_ckpt}")
                print("Training stopped safely.")
                break


def main():
    cfg = TrainConfig(project_root=".")
    trainer = PPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
