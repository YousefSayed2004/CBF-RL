import csv
import contextlib
from dataclasses import dataclass
import io
from pathlib import Path
import pickle
import signal
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SCENARIO_ROOT = Path(__file__).resolve().parent
if str(SCENARIO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCENARIO_ROOT))

from LM_Baseline_Methods import run_single_rollout as run_attcbf_single_rollout
from RL.env import LambdaBounds, NAMES, DecentralizedLaneMergingEnv, ThreeVehicleLambdaEnv, build_fixed_initial_states


COLLISION_EVENTS = {"collision", "boundary_collision", "early_boundary_collision"}
STOP_REQUESTED = False
MODE_SUFFIX = {
    "1": "uniform",
    "2": "heterogeneous",
    "3": "heuristic",
    "4": "attcbf",
}
HEURISTIC_CLEARANCE_RULE = "upper_lower_for_upper_lower_mid_uses_nearest"


@dataclass
class BOConfig:
    iterations: int = 500
    rollouts_per_iteration: int = 10
    seed: int = 7
    mode: str = "2"
    kappa: float = 2.0
    init_random_points: int = 5
    acq_candidates: int = 20000
    resolution: float = 0.001
    length_scale: float = 0.08
    gp_noise: float = 1e-6
    output_dir: str = str(SCENARIO_ROOT / "LM_BO_outputs")
    collision_penalty: float = -100.0
    infeasible_qp_penalty: float = -100.0
    resume: bool = True


def handle_interrupt(_signum, _frame) -> None:
    global STOP_REQUESTED
    if STOP_REQUESTED:
        raise KeyboardInterrupt
    STOP_REQUESTED = True
    print("\nInterrupt received. Will stop after the current BO iteration and save checkpoint.")


def mode_suffix(mode: str) -> str:
    return MODE_SUFFIX.get(mode, MODE_SUFFIX["2"])


def apply_output_suffix(output_dir: str, mode: str) -> str:
    suffix = f"_{mode_suffix(mode)}"
    path = Path(output_dir)
    if path.name.endswith(suffix):
        return str(path)
    return str(path.with_name(path.name + suffix))


def search_spec(cfg: BOConfig) -> Tuple[List[str], float, float]:
    if cfg.mode == "1":
        return ["lambda"], 0.1, 0.4
    if cfg.mode == "3":
        return [f"gain_{name}" for name in NAMES], 0.01, 0.5
    if cfg.mode == "4":
        return ["eta_weight"], 1.0, 1e6
    return [f"lambda_{name}" for name in NAMES], 0.1, 0.4


def search_resolution(cfg: BOConfig) -> float:
    if cfg.mode == "4":
        return 1000.0
    if cfg.mode == "3":
        return 0.01
    return cfg.resolution


def quantize_values(values: np.ndarray, lower: float, upper: float, resolution: float) -> np.ndarray:
    quantized = np.round(values / resolution) * resolution
    return np.clip(quantized, lower, upper)


def quantize_scalar(value: float, lower: float, upper: float, resolution: float) -> float:
    return float(quantize_values(np.array([value], dtype=float), lower, upper, resolution)[0])


def lambdas_from_params(params: Dict[str, float]) -> Dict[str, float]:
    if "lambda" in params:
        return {name: params["lambda"] for name in NAMES}
    return {name: params[f"lambda_{name}"] for name in NAMES}


def heuristic_lambdas(env: ThreeVehicleLambdaEnv, params: Dict[str, float], bounds: LambdaBounds) -> Dict[str, float]:
    pair_clear = env._pairwise_clearances()
    clearance_upper_lower = pair_clear["upper_lower"]
    clearance_mid_upper = pair_clear["mid_upper"]
    clearance_mid_lower = pair_clear["mid_lower"]

    def adaptive_clearance_for_vehicle(name: str) -> float:
        #return clearance_mid_upper if (name == "upper" or name == "mid") else min(clearance_mid_lower, clearance_upper_lower)
        #return clearance_mid_lower if (name == "lower" or name == "mid") else min(clearance_mid_upper, clearance_upper_lower)
        return clearance_upper_lower if (name == "upper" or name == "lower") else min(clearance_mid_upper, clearance_mid_lower)

    return {
        name: float(np.clip(params[f"gain_{name}"] * adaptive_clearance_for_vehicle(name), bounds.min_val, bounds.max_val))
        for name in NAMES
    }


def normalized_action_from_lambdas(lambdas: Dict[str, float], bounds: LambdaBounds) -> np.ndarray:
    values = np.array([lambdas[name] for name in NAMES], dtype=float)
    action = 2.0 * (values - bounds.min_val) / (bounds.max_val - bounds.min_val) - 1.0
    return np.clip(action, -1.0, 1.0)


def action_for_step(env: ThreeVehicleLambdaEnv, params: Dict[str, float], cfg: BOConfig, bounds: LambdaBounds) -> np.ndarray:
    if cfg.mode == "3":
        lambdas = heuristic_lambdas(env, params, bounds)
    else:
        lambdas = lambdas_from_params(params)
    return normalized_action_from_lambdas(lambdas, bounds)


def route_progress(env: ThreeVehicleLambdaEnv) -> Dict[str, float]:
    moved = env._moved_distances()
    return {
        name: float(moved[name])
        for name in NAMES
    }


def route_coordinate_from_state(name: str, state: np.ndarray, geom) -> float:
    point = state[:2]
    if name == "upper":
        tvec = geom.t_upper
        pref = geom.pref_upper
    elif name == "mid":
        tvec = geom.t_mid
        pref = geom.pref_mid
    else:
        tvec = geom.t_lower
        pref = geom.pref_lower

    s_switch = geom.x_switch / tvec[0]
    if point[0] <= geom.x_switch:
        return float((point - pref) @ tvec)
    return float(s_switch + (point[0] - geom.x_switch))


def evaluate_attcbf_candidate(
    params: Dict[str, float],
    initial_states_list: List[Dict[str, np.ndarray]],
    cfg: BOConfig,
) -> Tuple[Dict, List[Dict]]:
    eta_weight = float(params["eta_weight"])
    rollout_rows = []

    for rollout_idx, initial_states in enumerate(initial_states_list, start=1):
        with contextlib.redirect_stdout(io.StringIO()):
            hist, metrics, _sim, _vps, geom, _circle_offsets, _circle_radius = run_attcbf_single_rollout(
                init_states=initial_states,
                cbfp_schedule="attcbf",
                goal_x=5.0,
                eta_weight_override=eta_weight,
            )

        events = hist.get("events", [])
        if any(event == "infeasible_qp" for event in events):
            event = "infeasible_qp"
        elif any(event == "collision" for event in events):
            event = "collision"
        elif any("boundary_collision" in event for event in events):
            event = "boundary_collision"
        elif metrics.get("deadlock", 0):
            event = "deadlock"
        else:
            event = "running"

        progress = {}
        for name in NAMES:
            progress[name] = (
                route_coordinate_from_state(name, hist[f"x_{name}"][-1], geom)
                - route_coordinate_from_state(name, initial_states[name], geom)
            )

        intervehicle_collision = event == "collision"
        boundary_collision = "boundary_collision" in event
        safety_failure = event in COLLISION_EVENTS
        infeasible_qp = event == "infeasible_qp"
        penalty = 0.0
        if safety_failure:
            penalty += cfg.collision_penalty
        if infeasible_qp:
            penalty += cfg.infeasible_qp_penalty

        reward = sum(progress.values()) + penalty

        row = {
            "rollout": rollout_idx,
            "reward": float(reward),
            "event": event,
            "steps": max(len(hist["t"]) - 1, 0),
            "intervehicle_collision": int(intervehicle_collision),
            "boundary_collision": int(boundary_collision),
            "safety_failure": int(safety_failure),
            "infeasible_qp": int(infeasible_qp),
            "eta_weight": eta_weight,
        }
        for name in NAMES:
            row[f"progress_{name}"] = float(progress[name])
            if f"lambda_{name}" in hist and len(hist[f"lambda_{name}"]) > 0:
                row[f"final_eta_{name}"] = float(hist[f"lambda_{name}"][-1])
        rollout_rows.append(row)

    rewards = np.array([row["reward"] for row in rollout_rows], dtype=float)
    summary = {
        **params,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "intervehicle_collision_rate": float(np.mean([row["intervehicle_collision"] for row in rollout_rows])),
        "boundary_collision_rate": float(np.mean([row["boundary_collision"] for row in rollout_rows])),
        "safety_failure_rate": float(np.mean([row["safety_failure"] for row in rollout_rows])),
        "infeasible_qp_rate": float(np.mean([row["infeasible_qp"] for row in rollout_rows])),
    }
    for name in NAMES:
        summary[f"avg_progress_{name}"] = float(np.mean([row[f"progress_{name}"] for row in rollout_rows]))
        eta_values = [row[f"final_eta_{name}"] for row in rollout_rows if f"final_eta_{name}" in row]
        if eta_values:
            summary[f"avg_final_eta_{name}"] = float(np.mean(eta_values))
    return summary, rollout_rows


def evaluate_candidate(
    params: Dict[str, float],
    initial_states_list: List[Dict[str, np.ndarray]],
    cfg: BOConfig,
) -> Tuple[Dict, List[Dict]]:
    if cfg.mode == "4":
        return evaluate_attcbf_candidate(params, initial_states_list, cfg)

    env = DecentralizedLaneMergingEnv(seed=cfg.seed)
    bounds = LambdaBounds()
    rollout_rows = []

    for rollout_idx, initial_states in enumerate(initial_states_list, start=1):
        env.reset(initial_states=initial_states)
        done = False
        event = "running"

        while not done:
            action = action_for_step(env, params, cfg, bounds)
            _, _, done, info = env.step(action)
            event = info.get("event", "running")

        progress = route_progress(env)

        intervehicle_collision = event == "collision"
        boundary_collision = "boundary_collision" in event
        safety_failure = event in COLLISION_EVENTS
        infeasible_qp = event == "infeasible_qp"
        penalty = 0.0

        if safety_failure:
            penalty += cfg.collision_penalty
        if infeasible_qp:
            penalty += cfg.infeasible_qp_penalty

        reward = sum(progress.values()) + penalty
        row = {
            "rollout": rollout_idx,
            "reward": float(reward),
            "event": event,
            "steps": env.step_count,
            "intervehicle_collision": int(intervehicle_collision),
            "boundary_collision": int(boundary_collision),
            "safety_failure": int(safety_failure),
            "infeasible_qp": int(infeasible_qp),
        }
        row.update(params)
        for name in NAMES:
            row[f"progress_{name}"] = progress[name]
        rollout_rows.append(row)

    rewards = np.array([row["reward"] for row in rollout_rows], dtype=float)
    summary = {
        **params,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "intervehicle_collision_rate": float(np.mean([row["intervehicle_collision"] for row in rollout_rows])),
        "boundary_collision_rate": float(np.mean([row["boundary_collision"] for row in rollout_rows])),
        "safety_failure_rate": float(np.mean([row["safety_failure"] for row in rollout_rows])),
        "infeasible_qp_rate": float(np.mean([row["infeasible_qp"] for row in rollout_rows])),
    }
    for name in NAMES:
        summary[f"avg_progress_{name}"] = float(np.mean([row[f"progress_{name}"] for row in rollout_rows]))
    return summary, rollout_rows


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float) -> np.ndarray:
    diff = x1[:, None, :] - x2[None, :, :]
    sqdist = np.sum(diff * diff, axis=2)
    return np.exp(-0.5 * sqdist / (length_scale**2))


def gp_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    length_scale: float,
    noise: float,
) -> Tuple[np.ndarray, np.ndarray]:
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std < 1e-8:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std

    k_train = rbf_kernel(x_train, x_train, length_scale)
    k_train += (noise + 1e-10) * np.eye(len(x_train))
    k_query = rbf_kernel(x_train, x_query, length_scale)

    chol = np.linalg.cholesky(k_train)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_norm))
    mu_norm = k_query.T @ alpha

    v = np.linalg.solve(chol, k_query)
    var_norm = np.maximum(1.0 - np.sum(v * v, axis=0), 0.0)

    mu = mu_norm * y_std + y_mean
    std = np.sqrt(var_norm) * y_std
    return mu, std


def array_to_params(values: np.ndarray, param_names: List[str]) -> Dict[str, float]:
    return {name: float(values[idx]) for idx, name in enumerate(param_names)}


def initial_points(cfg: BOConfig, rng: np.random.Generator) -> List[Dict[str, float]]:
    param_names, lower, upper = search_spec(cfg)
    resolution = search_resolution(cfg)
    points = []
    if len(param_names) == 1:
        points.extend([
            {param_names[0]: lower},
            {param_names[0]: upper},
            {param_names[0]: quantize_scalar(0.5 * (lower + upper), lower, upper, resolution)},
        ])
    else:
        for first in [lower, upper]:
            for second in [lower, upper]:
                for third in [lower, upper]:
                    points.append({param_names[0]: first, param_names[1]: second, param_names[2]: third})

        center = quantize_scalar(0.5 * (lower + upper), lower, upper, resolution)
        points.append({name: center for name in param_names})
        if cfg.mode == "3":
            points.append({"gain_upper": 0.05, "gain_mid": 0.1, "gain_lower": 0.05})

    for _ in range(cfg.init_random_points):
        values = rng.uniform(lower, upper, size=len(param_names))
        values = quantize_values(values, lower, upper, resolution)
        points.append(array_to_params(values, param_names))

    return points


def candidate_pool(cfg: BOConfig, rng: np.random.Generator) -> np.ndarray:
    param_names, lower, upper = search_spec(cfg)
    resolution = search_resolution(cfg)
    if len(param_names) == 1:
        if cfg.mode == "4":
            values = np.unique(np.concatenate([
                np.array([lower], dtype=float),
                np.arange(resolution, upper + 0.5 * resolution, resolution, dtype=float),
                np.array([upper], dtype=float),
            ]))
            return values.reshape(-1, 1)
        return np.arange(lower, upper + 0.5 * resolution, resolution).reshape(-1, 1)

    candidates = rng.uniform(lower, upper, size=(cfg.acq_candidates, len(param_names)))
    candidates = quantize_values(candidates, lower, upper, resolution)
    candidates = np.unique(candidates, axis=0)

    anchors = []
    for first in [lower, upper]:
        for second in [lower, upper]:
            for third in [lower, upper]:
                anchors.append([first, second, third])
    anchors.append([quantize_scalar(0.5 * (lower + upper), lower, upper, resolution)] * len(param_names))
    if cfg.mode == "3":
        anchors.append([0.05, 0.1, 0.05])
    anchors = np.array(anchors, dtype=float)
    return np.unique(np.vstack([candidates, anchors]), axis=0)


def select_next_candidate(
    evaluated_x: np.ndarray,
    evaluated_y: np.ndarray,
    candidates: np.ndarray,
    cfg: BOConfig,
) -> Dict[str, float]:
    length_scale = 2e5 if cfg.mode == "4" else cfg.length_scale
    mu, std = gp_predict(
        evaluated_x,
        evaluated_y,
        candidates,
        length_scale=length_scale,
        noise=cfg.gp_noise,
    )
    acquisition = mu + cfg.kappa * std

    distances = np.linalg.norm(candidates[:, None, :] - evaluated_x[None, :, :], axis=2)
    available = np.min(distances, axis=1) > 1e-12
    if not np.any(available):
        best_idx = int(np.argmax(acquisition))
    else:
        available_indices = np.where(available)[0]
        best_idx = int(available_indices[np.argmax(acquisition[available])])

    return array_to_params(candidates[best_idx], search_spec(cfg)[0])


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    iteration_rows: List[Dict],
    rollout_rows: List[Dict],
    init_queue: List[Dict[str, float]],
    rng: np.random.Generator,
    best_reward: float,
    best_params: Dict[str, float],
    cfg: BOConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "iteration_rows": iteration_rows,
        "rollout_rows": rollout_rows,
        "init_queue": init_queue,
        "rng_state": rng.bit_generator.state,
        "best_reward": best_reward,
        "best_params": best_params,
        "objective_signature": objective_signature(cfg),
    }
    with path.open("wb") as f:
        pickle.dump(state, f)


def load_checkpoint(path: Path) -> Dict:
    with path.open("rb") as f:
        return pickle.load(f)


def objective_signature(cfg: BOConfig) -> Tuple:
    _, lower, upper = search_spec(cfg)
    if cfg.mode == "3":
        param_rule = HEURISTIC_CLEARANCE_RULE
    elif cfg.mode == "4":
        param_rule = "attcbf_eta_weight"
    else:
        param_rule = "lambda_direct"
    return (
        "solver_cascade_v2_decentralized_training_env",
        mode_suffix(cfg.mode),
        param_rule,
        lower,
        upper,
        cfg.collision_penalty,
        cfg.infeasible_qp_penalty,
        search_resolution(cfg),
    )


def next_available_output_dir(output_dir: Path) -> Path:
    if not output_dir.exists():
        return output_dir
    for idx in range(2, 10000):
        candidate = output_dir.with_name(f"{output_dir.name}_{idx}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available output folder for {output_dir}")


def choose_output_dir(output_dir: Path, cfg: BOConfig, scenario_name: str) -> Tuple[Path, Optional[Dict]]:
    output_dir = Path(apply_output_suffix(str(output_dir), cfg.mode))
    requested_output_dir = output_dir
    output_dir = next_available_output_dir(output_dir)
    if output_dir != requested_output_dir:
        print(f"Output folder exists. Using: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir, None


def plot_learning_curve(iteration_rows: List[Dict], rollout_rows: List[Dict], output_dir: Path) -> None:
    iterations = np.array([row["iteration"] for row in iteration_rows], dtype=int)
    mean_rewards = np.array([row["mean_reward"] for row in iteration_rows], dtype=float)
    best_rewards = np.array([row["best_mean_reward"] for row in iteration_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(11, 6))
    for row in rollout_rows:
        ax.scatter(row["iteration"], row["reward"], color="0.65", alpha=0.35, s=18)
    ax.plot(iterations, mean_rewards, color="tab:blue", linewidth=2.0, label="Mean of rollouts")
    ax.plot(iterations, best_rewards, color="tab:orange", linewidth=2.0, label="Best mean so far")
    ax.set_xlabel("BO iteration")
    ax.set_ylabel("Reward")
    ax.set_title("Lane Merging Bayesian Optimization Learning Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "bo_learning_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_best_summary_to_colorbar(fig, scatter, ax, best_text: str, shrink: float = 1.0, pad: float = 0.05) -> None:
    cbar = fig.colorbar(scatter, ax=ax, label="Mean reward", shrink=shrink, pad=pad)
    best_handle = Line2D(
        [0],
        [0],
        marker="*",
        markersize=13,
        markerfacecolor="tab:red",
        markeredgecolor="k",
        markeredgewidth=0.9,
        linestyle="None",
    )
    cbar.ax.legend(
        handles=[best_handle],
        labels=["Best"],
        loc="upper center",
        bbox_to_anchor=(0.45, -0.08),
        framealpha=0.85,
        facecolor="white",
        edgecolor="0.2",
        borderpad=0.35,
        handlelength=1.55,
        handletextpad=0.35,
    )
    cbar.ax.text(
        0.5,
        -0.34,
        best_text,
        transform=cbar.ax.transAxes,
        va="top",
        ha="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def plot_search_map(iteration_rows: List[Dict], output_dir: Path) -> None:
    param_names = [
        key for key in iteration_rows[0].keys()
        if key in {"lambda", "lambda_upper", "lambda_mid", "lambda_lower", "gain_upper", "gain_mid", "gain_lower"}
    ]
    rewards = np.array([row["mean_reward"] for row in iteration_rows], dtype=float)
    best_idx = int(np.argmax(rewards))
    best_text = "\n".join(
        [f"{name} = {iteration_rows[best_idx][name]:.4f}" for name in param_names]
        + [f"reward = {rewards[best_idx]:.3f}"]
    )

    if len(param_names) == 1:
        x_vals = np.array([row[param_names[0]] for row in iteration_rows], dtype=float)
        fig, ax = plt.subplots(figsize=(7, 6))
        scatter = ax.scatter(x_vals, rewards, c=rewards, cmap="viridis", s=58, edgecolor="k", linewidth=0.4, zorder=1)
        ax.scatter(
            x_vals[best_idx],
            rewards[best_idx],
            marker="*",
            s=340,
            color="white",
            edgecolor="k",
            linewidth=1.2,
            zorder=100,
        )
        ax.scatter(
            x_vals[best_idx],
            rewards[best_idx],
            marker="*",
            s=210,
            color="tab:red",
            edgecolor="k",
            linewidth=1.2,
            zorder=101,
        )
        ax.set_xlabel(param_names[0])
        ax.set_ylabel("Mean reward")
        ax.set_title(f"Evaluated {param_names[0]} Values")
        ax.grid(True, alpha=0.25)
        add_best_summary_to_colorbar(fig, scatter, ax, best_text)
        fig.tight_layout()
        fig.savefig(output_dir / "bo_search_map.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    x_vals = np.array([row[param_names[0]] for row in iteration_rows], dtype=float)
    y_vals = np.array([row[param_names[1]] for row in iteration_rows], dtype=float)
    z_vals = np.array([row[param_names[2]] for row in iteration_rows], dtype=float)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    if hasattr(ax, "computed_zorder"):
        ax.computed_zorder = False
    scatter = ax.scatter(x_vals, y_vals, z_vals, c=rewards, cmap="viridis", s=48, depthshade=False, zorder=1)
    ax.scatter(
        x_vals[best_idx],
        y_vals[best_idx],
        z_vals[best_idx],
        marker="*",
        s=420,
        color="white",
        edgecolor="k",
        linewidth=1.4,
        depthshade=False,
        zorder=100,
    )
    ax.scatter(
        x_vals[best_idx],
        y_vals[best_idx],
        z_vals[best_idx],
        marker="*",
        s=240,
        color="tab:red",
        edgecolor="k",
        linewidth=1.2,
        depthshade=False,
        zorder=101,
    )
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    ax.set_zlabel(param_names[2])
    lower, upper = (0.01, 0.5) if param_names[0].startswith("gain") else (0.1, 0.4)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_zlim(lower, upper)
    ax.set_title("Evaluated Lane Merging BO Parameters")
    add_best_summary_to_colorbar(fig, scatter, ax, best_text, shrink=0.72, pad=0.12)
    fig.tight_layout()
    fig.savefig(output_dir / "bo_search_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_bo(cfg: BOConfig) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    signal.signal(signal.SIGINT, handle_interrupt)

    output_dir = Path(cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = SCENARIO_ROOT / output_dir
    output_dir, checkpoint = choose_output_dir(output_dir, cfg, "Lane Merging")
    checkpoint_file = output_dir / "bo_checkpoint.pkl"

    rng = np.random.default_rng(cfg.seed)
    initial_env = DecentralizedLaneMergingEnv(seed=cfg.seed)
    initial_states = build_fixed_initial_states(
        n_rollouts=cfg.rollouts_per_iteration,
        seed=cfg.seed,
        cfg=initial_env.init_cfg,
        geom=initial_env.geom,
    )
    init_queue = initial_points(cfg, rng)

    iteration_rows = []
    rollout_rows = []
    best_reward = -np.inf
    best_params = {name: np.nan for name in search_spec(cfg)[0]}
    start_iteration = 1

    if checkpoint is not None:
        iteration_rows = checkpoint["iteration_rows"]
        rollout_rows = checkpoint["rollout_rows"]
        init_queue = checkpoint["init_queue"]
        rng.bit_generator.state = checkpoint["rng_state"]
        best_reward = checkpoint["best_reward"]
        best_params = checkpoint["best_params"]
        start_iteration = len(iteration_rows) + 1
        print(f"Resuming Lane Merging BO from iteration {start_iteration}.")

    for iteration in range(start_iteration, cfg.iterations + 1):
        if init_queue:
            params = init_queue.pop(0)
            selected_by = "initial"
        else:
            param_names = search_spec(cfg)[0]
            evaluated_x = np.array([
                [row[name] for name in param_names]
                for row in iteration_rows
            ], dtype=float)
            evaluated_y = np.array([row["mean_reward"] for row in iteration_rows], dtype=float)
            candidates = candidate_pool(cfg, rng)
            params = select_next_candidate(evaluated_x, evaluated_y, candidates, cfg)
            selected_by = "ucb"

        summary, candidate_rollouts = evaluate_candidate(params, initial_states, cfg)

        if summary["mean_reward"] > best_reward:
            best_reward = summary["mean_reward"]
            best_params = params.copy()

        iteration_row = {
            "iteration": iteration,
            "selected_by": selected_by,
            **summary,
            "best_mean_reward": float(best_reward),
        }
        for name in search_spec(cfg)[0]:
            iteration_row[f"best_{name}"] = float(best_params[name])
        iteration_rows.append(iteration_row)

        for row in candidate_rollouts:
            rollout_rows.append({"iteration": iteration, **row})

        write_csv(output_dir / "bo_iterations.csv", iteration_rows)
        save_checkpoint(
            checkpoint_file,
            iteration_rows,
            rollout_rows,
            init_queue,
            rng,
            best_reward,
            best_params,
            cfg,
        )

        lambda_text = ", ".join(f"{name}={params[name]:.4f}" for name in search_spec(cfg)[0])
        best_text = ", ".join(f"{name}={best_params[name]:.4f}" for name in search_spec(cfg)[0])
        print(
            f"iter {iteration:03d} | "
            f"{lambda_text} | "
            f"mean_reward={summary['mean_reward']:.3f} | "
            f"best={best_reward:.3f} at ({best_text}) | "
            f"safety={summary['safety_failure_rate']:.2f}, infeasible={summary['infeasible_qp_rate']:.2f}"
        )

        if STOP_REQUESTED:
            print(f"\nStopped after iteration {iteration}. Checkpoint saved to: {checkpoint_file}")
            print("Re-run the same command to continue, or pass --restart to start over.")
            return

    if not iteration_rows:
        print("No BO iterations completed.")
        return

    plot_learning_curve(iteration_rows, rollout_rows, output_dir)
    plot_search_map(iteration_rows, output_dir)

    print("\nBest result:")
    for name in search_spec(cfg)[0]:
        print(f"  {name} = {best_params[name]:.4f}")
    print(f"  mean reward = {best_reward:.3f}")
    print(f"\nSaved BO outputs to: {output_dir}")


def main() -> None:
    cfg = BOConfig()
    print("Select Lane Merging BO mode:")
    print("  1. Uniform lambda for all vehicles")
    print("  2. Heterogeneous lambdas")
    print("  3. Heuristic clearance gains")
    print("  4. aTTCBF eta_weight")
    mode = input(f"Enter 1, 2, 3, or 4 [default: {cfg.mode}]: ").strip() or cfg.mode
    if mode not in MODE_SUFFIX:
        mode = cfg.mode
    cfg.mode = mode
    run_bo(cfg)


if __name__ == "__main__":
    main()
