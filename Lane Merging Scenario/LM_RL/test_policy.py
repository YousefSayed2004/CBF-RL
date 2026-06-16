import argparse
import os
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .env import (
        NAMES,
        DecentralizedLaneMergingEnv,
        ThreeVehicleLambdaEnv,
        average_rollout_metrics,
        build_fixed_initial_states,
        run_policy_rollout as env_run_policy_rollout,
    )
except ImportError:
    from env import (
        NAMES,
        DecentralizedLaneMergingEnv,
        ThreeVehicleLambdaEnv,
        average_rollout_metrics,
        build_fixed_initial_states,
        run_policy_rollout as env_run_policy_rollout,
    )
from LM_Baseline_Methods import animate_simulation, show_average_metrics_table


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

    def forward(self, x: torch.Tensor):
        h = self.features(x)
        loc = torch.tanh(self.loc(h))
        scale = torch.exp(self.log_std).expand_as(loc)
        return loc, scale


def load_policy(checkpoint_path: str, device: torch.device, obs_dim: int, action_dim: int):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_size = checkpoint.get("config", {}).get("hidden_size", 128)
    policy = GradientPolicy(obs_dim, hidden_size=hidden_size, out_dims=action_dim).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy


def resolve_from_package(path_value: str) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = PACKAGE_ROOT / path
    return str(path)


def get_env_class(checkpoint_path: str):
    if "checkpoints" in Path(checkpoint_path).parts:
        return DecentralizedLaneMergingEnv
    return ThreeVehicleLambdaEnv


def run_policy_rollout(
    env: ThreeVehicleLambdaEnv,
    policy: GradientPolicy,
    device: torch.device,
    initial_states: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict, Dict]:
    def action_fn(obs_np: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            loc, _ = policy(obs_t)
        return loc.squeeze(0).cpu().numpy() if obs_np.ndim == 1 else loc.cpu().numpy()

    def print_step_details(info: Dict) -> None:
        t_now = info["step_count"] * env.sim.dt
        pair_clear = info["pairwise_clearances"]
        boundary_clear = info["boundary_clearances"]
        pieces = [
            f"t = {t_now:.1f} s",
            #f"clr_ul = {pair_clear['upper_lower']:.3f}",
            #f"clr_mu = {pair_clear['mid_upper']:.3f}",
            #f"clr_ml = {pair_clear['mid_lower']:.3f}",
            f"bclr_u = {boundary_clear['upper']:.3f}",
            f"bclr_m = {boundary_clear['mid']:.3f}",
            f"bclr_l = {boundary_clear['lower']:.3f}",
        ]
        pieces.extend(f"lambda_{name} = {info[f'lambda_{name}']:.3f}" for name in NAMES)
        print(" | ".join(pieces), flush=True)
        event = info["event"]
        if event != "running":
            print(f"  terminating at t = {t_now:.1f} s: {event.replace('_', ' ')}")

    hist, metrics = env_run_policy_rollout(
        env,
        action_fn=action_fn,
        initial_states=initial_states,
        step_callback=print_step_details,
    )

    print()
    return hist, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(PACKAGE_ROOT / "checkpoints" / "zero_coll_1.pt"))
    parser.add_argument("--n_rollouts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save_dir", type=str, default=str(PACKAGE_ROOT / "evaluation_outputs"))
    parser.add_argument("--make_video", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip_video", action="store_true")
    args = parser.parse_args()

    default_save_dir = str(PACKAGE_ROOT / "evaluation_outputs")
    checkpoint_path = resolve_from_package(args.checkpoint)
    env_class = get_env_class(checkpoint_path)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = env_class(seed=args.seed)
    policy = load_policy(checkpoint_path, device, env.observation_dim, env.action_dim)
    print(
        f"Using {env_class.__name__} with observation_dim = {env.observation_dim}, "
        f"action_dim = {env.action_dim}"
    )

    initial_states = build_fixed_initial_states(args.n_rollouts, args.seed)
    all_metrics = []
    last_hist = None

    for r_idx, init_states in enumerate(initial_states, start=1):
        print(f"\n=== Rollout {r_idx}/{args.n_rollouts} ===")
        for name in NAMES:
            print(
                f"Initial {name}: x={init_states[name][0]:.3f}, "
                f"y={init_states[name][1]:.3f}, psi={init_states[name][2]:.3f} rad"
            )

        hist, metrics = run_policy_rollout(env, policy, device, initial_states=init_states)
        all_metrics.append(metrics)
        last_hist = hist

    avg_metrics = average_rollout_metrics(all_metrics)

    show_average_metrics_table(avg_metrics, n_rollouts=args.n_rollouts)

    if not args.skip_video:
        video_filename = os.path.join(args.save_dir, "rl_policy_rollout.mp4")
        animate_simulation(
            last_hist,
            env.vps,
            env.geom,
            env.circle_offsets,
            env.circle_radius,
            filename=video_filename,
        )


if __name__ == "__main__":
    main()
