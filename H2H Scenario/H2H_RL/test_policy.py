import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from env import (
    TwoVehicleLambdaEnv,
    build_fixed_initial_states,
    min_intervehicle_clearance,
    min_boundary_clearance,
    compute_time_to_goal_x,
    circle_center_and_kinematics_affine,
    vehicle_corners,
)


# =========================
# Policy network
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

    def forward(self, x: torch.Tensor):
        h = self.features(x)
        loc = torch.tanh(self.loc(h))
        scale = torch.exp(self.log_std).expand_as(loc)
        return loc, scale


# =========================
# Rollout helpers
# =========================

def load_policy(checkpoint_path: str, device: torch.device, obs_dim: int, action_dim: int):
    policy = GradientPolicy(
        in_features=obs_dim,
        hidden_size=128,
        out_dims=action_dim,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy


def run_policy_rollout(
    env: TwoVehicleLambdaEnv,
    policy: GradientPolicy,
    device: torch.device,
    initial_states: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[Dict, Dict]:
    obs = env.reset(initial_states=initial_states)

    hist = {
        "x_i": [env.x_i.copy()],
        "x_j": [env.x_j.copy()],
        "u_i": [],
        "u_j": [],
        "goal_i": [],
        "goal_j": [],
        "lambda_i": [env.lambda_i],
        "lambda_j": [env.lambda_j],
        "clearance_iv": [
            min_intervehicle_clearance(
                env.x_i, env.x_j,
                env.vp_i, env.vp_j,
                env.circle_offsets, env.circle_radius
            )
        ],
        "reward": [0.0],
        "reward_cum": [0.0],
        "events": [],
        "t": [0.0],
    }

    done = False
    total_reward = 0.0

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            loc, _ = policy(obs_t)

        action = loc.squeeze(0).cpu().numpy()

        next_obs, reward, done, info = env.step(action)
        goal_i, goal_j = env._goal_points()

        current_t = env.step_count * env.sim.dt

        bi_clear = min_boundary_clearance(
            env.x_i, env.route_i, env.vp_i, env.circle_offsets, env.circle_radius
        )
        bj_clear = min_boundary_clearance(
            env.x_j, env.route_j, env.vp_j, env.circle_offsets, env.circle_radius
        )

        msg = (
            f"t = {current_t:.1f} s | "
            f"clearance = {info['inter_vehicle_clearance']:.3f} m | "
            f"bi_clear = {bi_clear:.3f} m | "
            f"bj_clear = {bj_clear:.3f} m | "
            f"lambda_i = {info['lambda_i']:.3f} | "
            f"lambda_j = {info['lambda_j']:.3f}"
        )
        print(msg.ljust(95), flush=True)

        total_reward += reward

        hist["x_i"].append(env.x_i.copy())
        hist["x_j"].append(env.x_j.copy())
        hist["u_i"].append(info["u_i"].copy())
        hist["u_j"].append(info["u_j"].copy())
        hist["lambda_i"].append(info["lambda_i"])
        hist["lambda_j"].append(info["lambda_j"])
        hist["goal_i"].append(goal_i.copy())
        hist["goal_j"].append(goal_j.copy())
        hist["clearance_iv"].append(info["inter_vehicle_clearance"])
        hist["reward"].append(reward)
        hist["reward_cum"].append(total_reward)
        hist["events"].append(info["event"])
        hist["t"].append(current_t)

        obs = next_obs

    print()

    for key in [
        "x_i", "x_j", "u_i", "u_j",
        "goal_i", "goal_j",
        "lambda_i", "lambda_j",
        "clearance_iv", "reward",
        "reward_cum", "t"
    ]:
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


def compute_rollout_metrics(
    hist: Dict,
    vp_i,
    vp_j,
    route_i,
    route_j,
    circle_offsets,
    circle_radius,
    goal_distance_x: float = 30.0,
):
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
        x_i[:, 0], t, direction="positive", distance_goal=goal_distance_x
    )
    time_to_goal_j = compute_time_to_goal_x(
        x_j[:, 0], t, direction="negative", distance_goal=goal_distance_x
    )

    avg_abs_y_i = float(np.mean(np.abs(x_i[:, 1])))
    avg_abs_y_j = float(np.mean(np.abs(x_j[:, 1])))

    deadlock = int(np.isnan(time_to_goal_i) and np.isnan(time_to_goal_j))
    invalid_motion_metrics = (
        infeasible_qp
        or intervehicle_collision_i
        or intervehicle_collision_j
        or boundary_collision_i
        or boundary_collision_j
    )

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
        "valid_motion_metrics": int(not invalid_motion_metrics),
        "episode_return": float(np.sum(hist["reward"])),
    }


def average_rollout_metrics(all_metrics: List[Dict]) -> Dict:
    avg = {"i": {}, "j": {}}
    valid_motion_metrics = [m for m in all_metrics if m.get("valid_motion_metrics", 1)]

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

    for veh in ["i", "j"]:
        avg[veh]["time_to_goal"] = mean_valid(veh, "time_to_goal", nanmean=True)
        avg[veh]["avg_abs_y"] = mean_valid(veh, "avg_abs_y")
        avg[veh]["acc_effort"] = mean_valid(veh, "acc_effort")
        avg[veh]["steer_rate_effort"] = mean_valid(veh, "steer_rate_effort")
        avg[veh]["intervehicle_collision"] = float(
            np.mean([m[veh]["intervehicle_collision"] for m in all_metrics])
        )
        avg[veh]["boundary_collision"] = float(
            np.mean([m[veh]["boundary_collision"] for m in all_metrics])
        )
        avg[veh]["min_intervehicle_clearance"] = mean_valid(veh, "min_intervehicle_clearance")
        avg[veh]["min_boundary_clearance"] = mean_valid(veh, "min_boundary_clearance")

    avg["deadlock_rate"] = (
        float(np.mean([m["deadlock"] for m in valid_motion_metrics]))
        if valid_motion_metrics
        else float("nan")
    )
    avg["infeasible_qp_rate"] = float(np.mean([m["infeasible_qp"] for m in all_metrics]))
    avg["valid_motion_rollouts"] = len(valid_motion_metrics)
    avg["avg_episode_return"] = float(np.mean([m["episode_return"] for m in all_metrics]))

    return avg


# =========================
# Metrics table
# =========================

def show_average_metrics_table(avg_metrics, n_rollouts: int):
    def fmt_nan(x):
        return "NaN" if np.isnan(x) else f"{x:.3f}"

    row_labels = [
        "Time to Goal [s]",
        "Avg |y|",
        "Acceleration Effort",
        "Steering-Rate Effort",
        "Inter-Vehicle Collision Rate",
        "Boundary Collision Rate",
        "Min Inter-Vehicle Clearance [m]",
        "Min Boundary Clearance [m]",
        "Deadlock Rate",
        "Infeasible QP Rate",
        "Valid Motion Rollouts",
        "Avg Episode Return",
    ]

    deadlock_rate_str = fmt_nan(avg_metrics["deadlock_rate"])
    infeasible_rate_str = f"{avg_metrics['infeasible_qp_rate']:.3f}"
    valid_motion_rollouts_str = f"{avg_metrics['valid_motion_rollouts']}/{n_rollouts}"
    avg_return_str = fmt_nan(avg_metrics["avg_episode_return"])

    table_data = [
        [fmt_nan(avg_metrics["i"]["time_to_goal"]), fmt_nan(avg_metrics["j"]["time_to_goal"])],
        [f"{avg_metrics['i']['avg_abs_y']:.3f}", f"{avg_metrics['j']['avg_abs_y']:.3f}"],
        [f"{avg_metrics['i']['acc_effort']:.3f}", f"{avg_metrics['j']['acc_effort']:.3f}"],
        [f"{avg_metrics['i']['steer_rate_effort']:.3f}", f"{avg_metrics['j']['steer_rate_effort']:.3f}"],
        [f"{avg_metrics['i']['intervehicle_collision']:.3f}", f"{avg_metrics['j']['intervehicle_collision']:.3f}"],
        [f"{avg_metrics['i']['boundary_collision']:.3f}", f"{avg_metrics['j']['boundary_collision']:.3f}"],
        [f"{avg_metrics['i']['min_intervehicle_clearance']:.3f}", f"{avg_metrics['j']['min_intervehicle_clearance']:.3f}"],
        [f"{avg_metrics['i']['min_boundary_clearance']:.3f}", f"{avg_metrics['j']['min_boundary_clearance']:.3f}"],
        [deadlock_rate_str, deadlock_rate_str],
        [infeasible_rate_str, infeasible_rate_str],
        [valid_motion_rollouts_str, valid_motion_rollouts_str],
        [avg_return_str, avg_return_str],
    ]

    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.axis("off")
    ax.set_title(f"Average Metrics Summary over {n_rollouts} Rollouts", fontsize=13, pad=12)

    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=["Vehicle i", "Vehicle j"],
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
            cell.set_height(0.080)
        else:
            cell.set_height(0.082)

        if col == -1:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    plt.show()
    return fig


# =========================
# Plots
# =========================

def plot_results(hist, route_i, filename: str = "rl_policy_results.png"):
    t_state = hist["t"]
    t_u = t_state[:-1]

    x_i = hist["x_i"]
    x_j = hist["x_j"]
    u_i = hist["u_i"]
    u_j = hist["u_j"]

    fig, axs = plt.subplots(5, 2, figsize=(12, 16), constrained_layout=True)

    half_w = route_i.road_half_width
    axs[0, 0].axhspan(-half_w, half_w, color="lightgray", alpha=0.4)
    axs[0, 0].axhline(half_w, color="k", linewidth=1.2)
    axs[0, 0].axhline(-half_w, color="k", linewidth=1.2)
    axs[0, 0].plot(x_i[:, 0], x_i[:, 1], label="vehicle i")
    axs[0, 0].plot(x_j[:, 0], x_j[:, 1], label="vehicle j")
    axs[0, 0].scatter(x_i[0, 0], x_i[0, 1], marker="o", s=60, label="i start")
    axs[0, 0].scatter(x_j[0, 0], x_j[0, 1], marker="o", s=60, label="j start")
    axs[0, 0].scatter(x_i[-1, 0], x_i[-1, 1], marker="x", s=70, label="i end")
    axs[0, 0].scatter(x_j[-1, 0], x_j[-1, 1], marker="x", s=70, label="j end")
    axs[0, 0].axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.6)
    axs[0, 0].set_title("Trajectories (Last Rollout)")
    axs[0, 0].set_xlabel("x [m]")
    axs[0, 0].set_ylabel("y [m]")
    axs[0, 0].axis("equal")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

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

    axs[2, 0].plot(t_u, u_i[:, 0], label="vehicle i")
    axs[2, 0].plot(t_u, u_j[:, 0], label="vehicle j")
    axs[2, 0].set_title("Acceleration Input")
    axs[2, 0].set_xlabel("t [s]")
    axs[2, 0].set_ylabel("a [m/s^2]")
    axs[2, 0].grid(True)
    axs[2, 0].legend()

    axs[2, 1].plot(t_u, u_i[:, 1], label="vehicle i")
    axs[2, 1].plot(t_u, u_j[:, 1], label="vehicle j")
    axs[2, 1].set_title("Steering Rate Input")
    axs[2, 1].set_xlabel("t [s]")
    axs[2, 1].set_ylabel("delta_rate [rad/s]")
    axs[2, 1].grid(True)
    axs[2, 1].legend()

    axs[3, 0].plot(t_state, hist["lambda_i"], label="lambda_i")
    axs[3, 0].plot(t_state, hist["lambda_j"], label="lambda_j")
    axs[3, 0].set_title("CBF Lambda")
    axs[3, 0].set_xlabel("t [s]")
    axs[3, 0].set_ylabel("lambda_cbf")
    axs[3, 0].grid(True)
    axs[3, 0].legend()

    axs[3, 1].plot(t_state, hist["clearance_iv"], label="inter-vehicle clearance")
    axs[3, 1].set_title("Inter-Vehicle Clearance")
    axs[3, 1].set_xlabel("t [s]")
    axs[3, 1].set_ylabel("clearance [m]")
    axs[3, 1].grid(True)
    axs[3, 1].legend()

    axs[4, 0].plot(t_state, x_i[:, 1], label="vehicle i y")
    axs[4, 0].plot(t_state, x_j[:, 1], label="vehicle j y")
    axs[4, 0].axhline(half_w, color="k", linewidth=1.0)
    axs[4, 0].axhline(-half_w, color="k", linewidth=1.0)
    axs[4, 0].axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.6)
    axs[4, 0].set_title("Lateral Position")
    axs[4, 0].set_xlabel("t [s]")
    axs[4, 0].set_ylabel("y [m]")
    axs[4, 0].grid(True)
    axs[4, 0].legend()

    axs[4, 1].plot(t_state, hist["reward_cum"], label="cumulative reward")
    axs[4, 1].set_title("Cumulative Reward")
    axs[4, 1].set_xlabel("t [s]")
    axs[4, 1].set_ylabel("return")
    axs[4, 1].grid(True)
    axs[4, 1].legend()

    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {filename}")
    plt.show()


# =========================
# Animation
# =========================

def animate_simulation(
    hist,
    vp_i,
    vp_j,
    route_i,
    circle_offsets,
    circle_radius,
    filename: str = "rl_policy_rollout.mp4",
):
    import os
    from matplotlib import animation
    from matplotlib.patches import Polygon, Circle as PatchCircle

    x_i = hist["x_i"]
    x_j = hist["x_j"]
    goal_i_hist = hist["goal_i"]
    goal_j_hist = hist["goal_j"]
    t = hist["t"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(-30, 30)
    ax.set_ylim(-8, 8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("RL-PPO CLF-CBF-QP on a Straight Road")
    ax.grid(True)

    half_w = route_i.road_half_width
    ax.fill_between([-30, 30], -half_w, half_w, color="lightgray", alpha=0.5)
    ax.plot([-30, 30], [ half_w,  half_w], "k-", linewidth=1.5)
    ax.plot([-30, 30], [-half_w, -half_w], "k-", linewidth=1.5)
    ax.axhline(0.0, linestyle="--", color="gray", alpha=0.6, linewidth=1.0)

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
    lambda_i_text = ax.text(
        0.20, 0.98, "", transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    lambda_j_text = ax.text(
        0.50, 0.98, "", transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    ax.legend(loc="upper right")

    def set_heading_line(line_obj, state, length=1.8):
        px, py, psi, _, _ = state
        ex = length * np.cos(psi)
        ey = length * np.sin(psi)
        line_obj.set_data([px, px + ex], [py, py + ey])

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

        goal_i_scatter.set_offsets(goal_i_hist[0])
        goal_j_scatter.set_offsets(goal_j_hist[0])

        time_text.set_text(f"t = {t[0]:.1f} s")
        lambda_i_text.set_text(f"lambda_i = {hist['lambda_i'][0]:.3f}")
        lambda_j_text.set_text(f"lambda_j = {hist['lambda_j'][0]:.3f}")

        return [
            traj_i_line, traj_j_line, poly_i, poly_j,
            heading_i_line, heading_j_line, goal_i_scatter, goal_j_scatter,
            time_text, lambda_i_text, lambda_j_text,
            *circle_patches_i, *circle_patches_j
        ]

    def update(frame):
        if frame == 0:
            gi = goal_i_hist[0]
            gj = goal_j_hist[0]
        else:
            gi = goal_i_hist[frame - 1]
            gj = goal_j_hist[frame - 1]

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

        goal_i_scatter.set_offsets(gi)
        goal_j_scatter.set_offsets(gj)

        time_text.set_text(f"t = {t[frame]:.1f} s")
        lambda_i_text.set_text(f"lambda_i = {hist['lambda_i'][frame]:.3f}")
        lambda_j_text.set_text(f"lambda_j = {hist['lambda_j'][frame]:.3f}")

        return [
            traj_i_line, traj_j_line, poly_i, poly_j,
            heading_i_line, heading_j_line, goal_i_scatter, goal_j_scatter,
            time_text, *circle_patches_i, *circle_patches_j
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

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--checkpoint", type=str, default="checkpoints/best_testing_23.pt")
    #parser.add_argument("--checkpoint", type=str, default="V0/best_1.pt")
    parser.add_argument("--checkpoint", type=str, default="V1/best_3.pt")
    parser.add_argument("--n_rollouts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save_dir", type=str, default="evaluation_outputs")
    parser.add_argument("--make_video", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = TwoVehicleLambdaEnv(seed=args.seed)
    policy = load_policy(
        checkpoint_path=args.checkpoint,
        device=device,
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
    )

    initial_states = build_fixed_initial_states(
        n_rollouts=args.n_rollouts,
        seed=args.seed,
    )

    all_metrics = []
    last_hist = None

    for r_idx, init_states in enumerate(initial_states, start=1):
        print(f"\n=== Rollout {r_idx}/{args.n_rollouts} ===")
        print(f"Initial vehicle i: y={init_states[0][1]:.3f}, psi={init_states[0][2]:.3f} rad")
        print(f"Initial vehicle j: y={init_states[1][1]:.3f}, psi={init_states[1][2]:.3f} rad")

        hist, metrics = run_policy_rollout(
            env=env,
            policy=policy,
            device=device,
            initial_states=init_states,
        )

        all_metrics.append(metrics)
        last_hist = hist

    avg_metrics = average_rollout_metrics(all_metrics)

    print("\nAverage metrics:")
    print(avg_metrics)

    show_average_metrics_table(avg_metrics, n_rollouts=args.n_rollouts)

    figure_filename = os.path.join(args.save_dir, "rl_policy_results.png")
    plot_results(last_hist, env.route_i, filename=figure_filename)


    video_filename = os.path.join(args.save_dir, "rl_policy_rollout.mp4")
    animate_simulation(
        last_hist,
        env.vp_i,
        env.vp_j,
        env.route_i,
        env.circle_offsets,
        env.circle_radius,
        filename=video_filename,
        )


if __name__ == "__main__":
    main()
