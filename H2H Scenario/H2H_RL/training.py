import csv
import math
import os
import signal
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader, IterableDataset
from matplotlib import pyplot as plt

from env import TwoVehicleLambdaEnv


# =========================
# Config
# =========================

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
    #value_coef: float = 0.5
    #max_grad_norm: float = 0.5

    max_epochs: int = 1000
    eval_every: int = 100

    train_seed: int = 42
    eval_seed: int = 7
    eval_rollouts: int = 10

    project_root: str = "."
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class plot:
    episode_count: float = 0.0
    current_epoch: int = 0
    average_reward: list = None

# =========================
# Networks
# =========================

class GradientPolicy(nn.Module):
    def __init__(self, in_features: int, hidden_size: int = 128, out_dims: int = 2):
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
        loc = torch.tanh(self.loc(h))  # keep mean roughly in [-1, 1]
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


# =========================
# Dataset
# =========================

class RLDataset(IterableDataset):
    """
    Similar in spirit to course-style RL datasets:
    stores rollout data already collected, then yields minibatches.
    """
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


# =========================
# PPO trainer
# =========================

class PPOTrainer:
    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        self.plot = plot(episode_count=0.0, average_reward=[])
        self.running_returns = np.zeros(self.cfg.num_envs, dtype=np.float32)

        torch.manual_seed(config.train_seed)
        np.random.seed(config.train_seed)

        self.envs = [TwoVehicleLambdaEnv(seed=config.train_seed + i) for i in range(config.num_envs)]
        self.obs_list = [env.reset() for env in self.envs]

        self.obs_dim = self.envs[0].observation_dim
        self.action_dim = self.envs[0].action_dim

        self.policy = GradientPolicy(
            in_features=self.obs_dim,
            hidden_size=config.hidden_size,
            out_dims=self.action_dim,
        ).to(self.device)

        self.value_net = ValueNet(
            in_features=self.obs_dim,
            hidden_size=config.hidden_size,
        ).to(self.device)

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=config.value_lr)

        self.project_root = config.project_root
        self.checkpoint_dir = os.path.join(self.project_root, "checkpoints")
        self.log_dir = os.path.join(self.project_root, "logs")

        os.makedirs(self.project_root, exist_ok=True)
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

    # -------------------------
    # interrupt + resume
    # -------------------------

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

            if "average_reward" in checkpoint:
                self.plot.average_reward = checkpoint["average_reward"]

            if "running_returns" in checkpoint:
                self.running_returns = checkpoint["running_returns"]

            if "best_testing_count" in checkpoint:
                self.best_testing_count = checkpoint["best_testing_count"]

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

    # -------------------------
    # rollout collection
    # -------------------------

    def collect_rollouts(self):
        """
        Collect experience from multiple environments.
        Returns flattened arrays ready for PPO update.
        """
        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_dones = []
        batch_values = []
        batch_next_values = []
        finished_episode_returns = []

        #self.plot.average_reward = []
        for _ in range(self.cfg.samples_per_epoch):
            obs_np = np.stack(self.obs_list, axis=0)
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                loc, scale = self.policy(obs_t)
                dist = Normal(loc, scale)

                actions_t = dist.sample()
                actions_t = torch.clamp(actions_t, -1.0, 1.0)

                log_probs_t = dist.log_prob(actions_t).sum(dim=-1)
                values_t = self.value_net(obs_t)

            actions_np = actions_t.cpu().numpy()

            next_obs_list = []
            rewards = []
            dones = []


            for env_idx, env in enumerate(self.envs):
                next_obs, reward, done, _ = env.step(actions_np[env_idx])

                self.running_returns[env_idx] += reward

                if done:
                    next_obs_reset = env.reset()
                    next_obs_list.append(next_obs_reset)
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

        obs_arr = np.asarray(batch_obs, dtype=np.float32)                # [T, N, obs_dim]
        actions_arr = np.asarray(batch_actions, dtype=np.float32)        # [T, N, act_dim]
        log_probs_arr = np.asarray(batch_log_probs, dtype=np.float32)    # [T, N]
        rewards_arr = np.asarray(batch_rewards, dtype=np.float32)        # [T, N]
        dones_arr = np.asarray(batch_dones, dtype=np.float32)            # [T, N]
        values_arr = np.asarray(batch_values, dtype=np.float32)          # [T, N]
        next_values_arr = np.asarray(batch_next_values, dtype=np.float32)# [T, N]

        advantages_arr, returns_arr = self.compute_gae_all_envs(
            rewards_arr,
            dones_arr,
            values_arr,
            next_values_arr,
        )

        flat_obs = obs_arr.reshape(-1, self.obs_dim)
        flat_actions = actions_arr.reshape(-1, self.action_dim)
        flat_old_log_probs = log_probs_arr.reshape(-1)
        flat_advantages = advantages_arr.reshape(-1)
        flat_returns = returns_arr.reshape(-1)

        # normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        avg_finished_episode_return = (
            float(np.mean(finished_episode_returns))
            if len(finished_episode_returns) > 0
            else np.nan
        )

        return flat_obs, flat_actions, flat_old_log_probs, flat_advantages, flat_returns, avg_finished_episode_return

    # -------------------------
    # GAE
    # -------------------------

    def compute_gae(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_values[t] * mask - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def compute_gae_all_envs(
        self,
        rewards_arr: np.ndarray,
        dones_arr: np.ndarray,
        values_arr: np.ndarray,
        next_values_arr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        advantages_all = []
        returns_all = []

        for env_idx in range(self.cfg.num_envs):
            adv, ret = self.compute_gae(
                rewards_arr[:, env_idx],
                dones_arr[:, env_idx],
                values_arr[:, env_idx],
                next_values_arr[:, env_idx],
            )
            advantages_all.append(adv)
            returns_all.append(ret)

        advantages_arr = np.stack(advantages_all, axis=1).astype(np.float32)
        returns_arr = np.stack(returns_all, axis=1).astype(np.float32)

        return advantages_arr, returns_arr

    # -------------------------
    # PPO losses
    # -------------------------

    def policy_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loc, scale = self.policy(obs)
        dist = Normal(loc, scale)

        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - self.cfg.clip_epsilon,
            1.0 + self.cfg.clip_epsilon
        ) * advantages

        loss = -torch.min(surr1, surr2).mean() - self.cfg.entropy_coef * entropy
        return loss, entropy

    def value_loss(
        self,
        obs: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        values_pred = self.value_net(obs)
        return 0.5 * ((values_pred - returns) ** 2).mean()

    # -------------------------
    # evaluation
    # -------------------------

    def evaluate(self) -> Dict:
        """
        Light evaluation during training.
        Uses fixed seed = 7 and deterministic mean action.
        """
        from env import build_fixed_initial_states, run_policy_rollout, average_rollout_metrics

        eval_env = TwoVehicleLambdaEnv(seed=self.cfg.eval_seed)
        initial_states = build_fixed_initial_states(
            n_rollouts=self.cfg.eval_rollouts,
            seed=self.cfg.eval_seed,
        )

        all_metrics = []

        def action_fn(obs_np: np.ndarray) -> np.ndarray:
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                loc, _ = self.policy(obs_t)
            return loc.squeeze(0).cpu().numpy()

        for init_states in initial_states:
            _, metrics = run_policy_rollout(
                eval_env,
                action_fn=action_fn,
                initial_states=init_states,
            )
            all_metrics.append(metrics)

        return average_rollout_metrics(all_metrics)

    # -------------------------
    # csv logging
    # -------------------------

    def append_csv_row(self, row: Dict):
        file_exists = os.path.exists(self.csv_log)

        with open(self.csv_log, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # -------------------------
    # train
    # -------------------------

    def train(self):
        
        if not self.plot.average_reward:
            self.plot.average_reward = []  

        for epoch in range(self.start_epoch, self.cfg.max_epochs + 1):
            
            self.plot.episode_count = 0.0
            self.plot.current_epoch = epoch

            obs_np, actions_np, old_log_probs_np, advantages_np, returns_np, avg_finished_episode_return = self.collect_rollouts()
            
            if np.isnan(avg_finished_episode_return):
                if len(self.plot.average_reward) > 0:
                    self.plot.average_reward.append(self.plot.average_reward[-1])
                else:
                    self.plot.average_reward.append(0.0)
            else:
                self.plot.average_reward.append(avg_finished_episode_return)
            
            # Plot the average reward per episode
            plt.plot(range(1, epoch + 1), self.plot.average_reward, label="Avg Reward per Episode")
            plt.xlabel("Epochs")
            plt.ylabel("Average Reward per Episode")
            plt.title("Learning Curve")
            
            # Save the plot as an image
            plot_image_path = os.path.join(self.log_dir, f"learning_curve_epoch_.png")
            plt.savefig(plot_image_path)
            plt.clf()  

            dataset = RLDataset(
                obs=obs_np,
                actions=actions_np,
                old_log_probs=old_log_probs_np,
                advantages=advantages_np,
                returns=returns_np,
                batch_size=self.cfg.batch_size,
                epoch_repeat=self.cfg.epoch_repeat,
            )

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

                # policy update
                p_loss, entropy = self.policy_loss(
                    obs=obs_t,
                    actions=actions_t,
                    old_log_probs=old_log_probs_t,
                    advantages=adv_t,
                )

                self.policy_optimizer.zero_grad()
                p_loss.backward()
                #nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.policy_optimizer.step()

                # value update
                v_loss = self.value_loss(
                    obs=obs_t,
                    returns=returns_t,
                )

                self.value_optimizer.zero_grad()
                v_loss.backward()
                #nn.utils.clip_grad_norm_(self.value_net.parameters(), self.cfg.max_grad_norm)
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
                f"Average Episode Return: {avg_finished_episode_return:.6f}"
            )

            # evaluation
            if epoch % self.cfg.eval_every == 0 or epoch == 1:
                eval_metrics = self.evaluate()
                score = eval_metrics["avg_episode_return"]
                log_std_np = self.policy.log_std.detach().cpu().numpy()
                std_np = np.exp(log_std_np)

            #     print(
            #         f"  Policy std -> "
            #         f"log_std: [{log_std_np[0]:.3f}, {log_std_np[1]:.3f}] | "
            #         f"std: [{std_np[0]:.3f}, {std_np[1]:.3f}]"
            #     )

            #     print(
            #         f"  Eval -> Reward: {eval_metrics['avg_episode_return']:.3f}, "
            #         f"Collision: {eval_metrics['i']['intervehicle_collision']:.3f}, "
            #         f"Boundary: {eval_metrics['i']['boundary_collision']+eval_metrics['j']['boundary_collision']:.3f}, "
            #         f"Infeasible QP: {eval_metrics['infeasible_qp_rate']:.3f}, "
            #         f"Deadlock: {eval_metrics['deadlock_rate']:.3f}"
            #     )

                csv_row = {
                    "epoch": epoch,
                    "avg_reward": round(eval_metrics["avg_episode_return"], 6),
                    "collision_rate": round(eval_metrics["i"]["intervehicle_collision"], 6),
                    "boundary_collision_rate": round(eval_metrics["i"]["boundary_collision"] + eval_metrics["j"]["boundary_collision"], 6),
                    "infeasible_qp_rate": round(eval_metrics["infeasible_qp_rate"], 6),
                    "deadlock_rate": round(eval_metrics["deadlock_rate"], 6),
                    "avg_time_to_goal_i": (
                        round(eval_metrics["i"]["time_to_goal"], 6)
                        if not math.isnan(eval_metrics["i"]["time_to_goal"])
                        else float("nan")
                    ),
                    "avg_time_to_goal_j": (
                        round(eval_metrics["j"]["time_to_goal"], 6)
                        if not math.isnan(eval_metrics["j"]["time_to_goal"])
                        else float("nan")
                    ),
                    "avg_abs_y_i": round(eval_metrics["i"]["avg_abs_y"], 6),
                    "avg_abs_y_j": round(eval_metrics["j"]["avg_abs_y"], 6),
                    "min_intervehicle_clearance": round(eval_metrics["i"]["min_intervehicle_clearance"], 6),
                }
                self.append_csv_row(csv_row)

                if (
                    eval_metrics["i"]["intervehicle_collision"] == 0.0
                    and eval_metrics["j"]["intervehicle_collision"] == 0.0
                    and eval_metrics["i"]["boundary_collision"] == 0.0
                    and eval_metrics["j"]["boundary_collision"] == 0.0
                    and eval_metrics["infeasible_qp_rate"] == 0.0
                ):
                    self.best_testing_count += 1
                    best_testing_path = os.path.join(
                        self.checkpoint_dir,
                        f"best_testing_{self.best_testing_count}.pt"
                    )
                    self.save_checkpoint(best_testing_path, epoch)
                    print(f"  Saved zero-collision test checkpoint -> {best_testing_path}")


            # regular checkpoint
            self.save_checkpoint(self.latest_ckpt, epoch)
            if(self.plot.average_reward[epoch-1] == max(self.plot.average_reward)):
                self.save_checkpoint(self.best_ckpt, epoch)
                print(f"  Saved new best checkpoint -> {self.best_ckpt}")

            if(self.plot.average_reward[epoch-1] == min(self.plot.average_reward)):
                self.save_checkpoint(self.worst_ckpt, epoch)
                print(f"  Saved new worst checkpoint -> {self.worst_ckpt}")
                
            # graceful interrupt
            if self.stop_requested:
                self.save_checkpoint(self.interrupt_ckpt, epoch)
                print(f"Saved interrupt checkpoint -> {self.interrupt_ckpt}")
                print("Training stopped safely.")
                break


# =========================
# Main
# =========================

def main():
    cfg = TrainConfig(project_root=".")
    trainer = PPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
