import argparse
import os
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch

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
        ALL_DECISION_SEQUENCES,
        DECISION_TO_REL_EXIT,
        DecentralizedThreeVehicleIntersectionEnv,
        START_BY_NAME,
        ThreeVehicleIntersectionEnv,
        VEHICLE_NAMES,
        average_rollout_metrics,
        build_fixed_initial_states,
        route_decision_sequence,
        run_policy_rollout as env_run_policy_rollout,
        show_average_metrics_table,
    )
    from .training import (
        GradientPolicy,
    )
except ImportError:
    from env import (
        LambdaBounds,
        RewardConfig,
        RouteSpec,
        SimParams,
        ALL_DECISION_SEQUENCES,
        DECISION_TO_REL_EXIT,
        DecentralizedThreeVehicleIntersectionEnv,
        START_BY_NAME,
        ThreeVehicleIntersectionEnv,
        VEHICLE_NAMES,
        average_rollout_metrics,
        build_fixed_initial_states,
        route_decision_sequence,
        run_policy_rollout as env_run_policy_rollout,
        show_average_metrics_table,
    )
    from training import (
        GradientPolicy,
    )

from RI_Baseline_Methods import animate_multi_vehicle_simulation  # noqa: E402


def load_policy(checkpoint_path: str, device: torch.device, obs_dim: int, action_dim: int) -> GradientPolicy:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_size = checkpoint.get("config", {}).get("hidden_size", 256)
    policy = GradientPolicy(obs_dim, hidden_size=hidden_size, out_dims=action_dim).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy


def normalize_decision_sequence(sequence: str) -> str:
    cleaned = sequence.replace(",", "").replace(" ", "").upper()
    if len(cleaned) != len(VEHICLE_NAMES) or any(ch not in DECISION_TO_REL_EXIT for ch in cleaned):
        names = ", ".join(VEHICLE_NAMES)
        raise ValueError(
            f"Decision sequence must contain one L/R/S decision for each vehicle ({names}), "
            f"for example RLS."
        )
    return cleaned


def route_set_from_decision_sequence(sequence: str, geom) -> Dict[str, RouteSpec]:
    sequence = normalize_decision_sequence(sequence)
    routes = {}
    for name, decision in zip(VEHICLE_NAMES, sequence):
        start = START_BY_NAME[name]
        exit_lane = ((start - 1 + DECISION_TO_REL_EXIT[decision]) % 4) + 1
        routes[name] = RouteSpec(start=start, exit=exit_lane, geom=geom)
    return routes


def choose_initial_states(args: argparse.Namespace, geom) -> Tuple[list, int]:
    mode = args.mode
    if mode is None:
        print("\nChoose decentralized RI test mode:")
        print("  1: run one custom rollout by decision sequence")
        print(f"  2: run {args.n_rollouts} random rollouts with seed {args.seed}")
        print("  3: run all 27 ordered scenarios")
        mode = input("Mode (1/2/3) [1]: ").strip() or "1"

    if mode == "1":
        sequence = args.decision_sequence
        if sequence is None:
            sequence = input("Decision sequence for i,j,k using L/R/S, e.g. RLS: ").strip()
        return [route_set_from_decision_sequence(sequence, geom)], 1

    if mode == "2":
        return build_fixed_initial_states(args.n_rollouts, args.seed, geom=geom), args.n_rollouts

    if mode == "3":
        initial_states = [
            route_set_from_decision_sequence(sequence, geom)
            for sequence in ALL_DECISION_SEQUENCES
        ]
        return initial_states, len(initial_states)

    raise ValueError("Mode must be 1, 2, or 3.")


def resolve_from_package(path_value: str) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = PACKAGE_ROOT / path
    return str(path)


def make_reward_config(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        collision_penalty=args.collision_penalty,
        infeasible_qp_penalty=args.infeasible_qp_penalty,
        boundary_collision_penalty=args.boundary_collision_penalty,
        progress_weight=args.progress_weight,
        deviation_weight=args.deviation_weight,
        survival_bonus=args.survival_bonus,
        goal_reward=args.goal_reward,
    )


def run_policy_rollout(
    env: ThreeVehicleIntersectionEnv,
    policy: GradientPolicy,
    device: torch.device,
    initial_states: Optional[Dict] = None,
) -> Tuple[Dict, Dict]:
    def action_fn(obs_np: np.ndarray) -> np.ndarray:
        obs_arr = np.asarray(obs_np, dtype=np.float32)
        obs_t = torch.tensor(obs_arr.reshape(-1, env.observation_dim), dtype=torch.float32, device=device)
        with torch.no_grad():
            loc, _ = policy(obs_t)
        return loc.cpu().numpy().reshape(*obs_arr.shape[:-1], env.action_dim)

    def print_step_details(info: Dict) -> None:
        t_now = info["step_count"] * env.sim.dt
        msg = (
            f"t = {t_now:.1f} s | "
            f"clearance = {info['inter_vehicle_clearance']:.3f} m | "
            f"min_boundary = {min(info[f'boundary_clearance_{name}'] for name in VEHICLE_NAMES):.3f} m | "
            + " | ".join(f"lambda_{name} = {info[f'lambda_{name}']:.3f}" for name in VEHICLE_NAMES)
            + " |"
        )
        print(msg.ljust(95), flush=True)
        if info["event"] != "running":
            print(f"  terminating at t = {t_now:.1f} s: {info['event'].replace('_', ' ')}")

    hist, metrics = env_run_policy_rollout(
        env,
        action_fn=action_fn,
        initial_states=initial_states,
        step_callback=print_step_details,
    )
    if metrics.get("deadlock", 0):
        deadlocked = [
            name for name in VEHICLE_NAMES
            if np.isnan(metrics.get(name, {}).get("time_to_goal", float("nan")))
        ]
        detail = f" ({', '.join(deadlocked)})" if deadlocked else ""
        print(f"  deadlock: vehicle(s) did not reach the goal before the time limit{detail}")
    print()
    return hist, metrics


def main():
    parser = argparse.ArgumentParser()
    default_save_dir = str(PACKAGE_ROOT / "evaluation_outputs")
    parser.add_argument("--checkpoint", type=str, default=str(PACKAGE_ROOT / "checkpoints" / "zero_coll_5.pt"))
    parser.add_argument("--n_rollouts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save_dir", type=str, default=default_save_dir)
    parser.add_argument("--mode", choices=("1", "2", "3"), default=None)
    parser.add_argument("--decision_sequence", type=str, default=None)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--lambda_min", type=float, default=0.1)
    parser.add_argument("--lambda_max", type=float, default=0.4)
    parser.add_argument("--cbf_slack_weight", type=float, default=1e8)
    parser.add_argument("--clf_slack_weight", type=float, default=1.0)
    parser.add_argument("--progress_weight", type=float, default=3.3 / 100)
    parser.add_argument("--deviation_weight", type=float, default=0.15 / 100)
    parser.add_argument("--survival_bonus", type=float, default=0.5 / 100)
    parser.add_argument("--goal_reward", type=float, default=75.0 / 100)
    parser.add_argument("--collision_penalty", type=float, default=-100.0 / 100)
    parser.add_argument("--boundary_collision_penalty", type=float, default=-100.0 / 100)
    parser.add_argument("--infeasible_qp_penalty", type=float, default=-150.0 / 100)
    parser.add_argument("--skip_video", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(resolve_from_package(args.checkpoint))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    save_dir = resolve_from_package(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DecentralizedThreeVehicleIntersectionEnv(
        sim=SimParams(dt=args.dt, T=args.T),
        reward_cfg=make_reward_config(args),
        lambda_bounds=LambdaBounds(min_val=args.lambda_min, max_val=args.lambda_max),
        seed=args.seed,
        cbf_slack_weight=args.cbf_slack_weight,
        clf_slack_weight=args.clf_slack_weight,
    )
    policy = load_policy(str(checkpoint_path), device, env.observation_dim, env.action_dim)
    print(
        f"Using decentralized RI policy {checkpoint_path.name} with observation_dim = {env.observation_dim}, "
        f"T = {env.sim.T}, dt = {env.sim.dt}"
    )

    initial_states, args.n_rollouts = choose_initial_states(args, env.geom)
    all_metrics = []
    last_hist = None

    for r_idx, route_set in enumerate(initial_states, start=1):
        print(f"\n=== Rollout {r_idx}/{args.n_rollouts} ===")
        print(f"decision sequence: {route_decision_sequence(route_set)}")
        for name in VEHICLE_NAMES:
            route = route_set[name]
            print(f"vehicle {name}: start {route.start}, exit {route.exit}")

        hist, metrics = run_policy_rollout(env, policy, device, initial_states=route_set)
        all_metrics.append(metrics)
        last_hist = hist

    avg_metrics = average_rollout_metrics(all_metrics)

    show_average_metrics_table(avg_metrics, n_rollouts=args.n_rollouts)

    vehicle_data = {
        name: {
            "route": env.routes[name],
            "vp": env.vps[name],
        }
        for name in VEHICLE_NAMES
    }

    if not args.skip_video:
        video_filename = os.path.join(save_dir, "rl_policy_rollout.mp4")
        animate_multi_vehicle_simulation(
            last_hist,
            vehicle_data,
            env.circle_offsets,
            env.circle_radius,
            filename=video_filename,
        )


if __name__ == "__main__":
    main()
