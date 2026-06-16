from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import cvxpy as cp
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import LM_Baseline_Methods as lm 


@dataclass
class PACBFWeights(lm.QPWeights):
    p1_initial: float = 1.0
    p1_desired: float = 0.4
    p2_desired: float = 0.4
    p2_min: float = 0.0
    epsilon_clf_p1: float = 10.0
    W_p1_rate: float = 1e-2
    W_slack_clf_p1: float = 1e5
    W_p2_deviation: float = 1e5


def solve_vehicle_qp_pacbf(
    ego_state: np.ndarray,
    other_states: List[np.ndarray],
    ego_name: str,
    ego_vp: lm.VehicleParams,
    other_vps: List[lm.VehicleParams],
    clfp: lm.CLFParams,
    cbfp: lm.CBFParams,
    qpw: PACBFWeights,
    sim: lm.SimParams,
    geom: lm.MergeGeometry,
    circle_offsets: np.ndarray,
    circle_radius: float,
    goal_point: np.ndarray,
    p1_old: float,
) -> Tuple[np.ndarray, Dict]:
    u_nom = lm.nominal_control(ego_state, clfp, ego_vp, goal_point)

    u = cp.Variable(2)
    s_clf = cp.Variable(1, nonneg=True)
    p1_rate = cp.Variable()
    s_clf_p1 = cp.Variable(1, nonneg=True)
    p2 = cp.Variable()

    constraints = [
        u[0] >= ego_vp.min_accel,
        u[0] <= ego_vp.max_accel,
        u[1] >= ego_vp.min_steer_rate,
        u[1] <= ego_vp.max_steer_rate,
    ]

    dt = sim.dt
    constraints += [
        ego_state[4] + dt * u[1] >= ego_vp.min_steer,
        ego_state[4] + dt * u[1] <= ego_vp.max_steer,
        ego_state[3] + dt * u[0] >= ego_vp.min_speed,
        ego_state[3] + dt * u[0] <= ego_vp.max_speed,
    ]

    V, Vdot_const, Vdot_u = lm.clf_terms(ego_state, ego_vp, clfp, goal_point)
    constraints += [Vdot_const + Vdot_u @ u <= -clfp.clf_rate * V + s_clf[0]]

    V_p1 = (p1_old - qpw.p1_desired) ** 2
    constraints += [
        p1_rate * dt + p1_old >= 0.0,
        p2 >= qpw.p2_min,
        2.0 * (p1_old - qpw.p1_desired) * p1_rate
        + qpw.epsilon_clf_p1 * V_p1
        <= s_clf_p1[0],
    ]

    # Pairwise inter-vehicle constraints. The ego control term keeps its full
    # authority, while the non-control/PACBF residual is split 50/50.
    for other_state, other_vp in zip(other_states, other_vps):
        for s_i in circle_offsets:
            for s_j in circle_offsets:
                h, h_dot, hddot_aff = lm.pairwise_circle_cbf_affine(
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
                psi1 = h_dot + p1_old * h
                constraints += [
                    hddot_u @ u
                    + 0.5 * (
                        hddot_const
                        + p1_rate * h
                        + p1_old * h_dot
                        + p2 * psi1
                    )
                    >= 0.0
                ]

    # Boundary constraints are not shared with another vehicle, so no 1/2 split.
    for boundary_side in ["left", "right"]:
        for s_i in circle_offsets:
            h, h_dot, hddot_aff = lm.active_lane_boundary_cbf_affine(
                ego_state=ego_state,
                ego_s=s_i,
                ego_vp=ego_vp,
                geom=geom,
                vehicle_name=ego_name,
                boundary_side=boundary_side,
                circle_radius=circle_radius,
            )
            hddot_const = hddot_aff[0]
            hddot_u = hddot_aff[1:]
            psi1 = h_dot + p1_old * h
            constraints += [
                hddot_const
                + hddot_u @ u
                + p1_rate * h
                + p1_old * h_dot
                + p2 * psi1
                >= 0.0
            ]

    objective_terms = (
        0.5 * cp.quad_form(u - u_nom, qpw.u_weight)
        + clfp.clf_slack_weight * cp.sum_squares(s_clf)
        + qpw.W_p1_rate * p1_rate
        + qpw.W_slack_clf_p1 * cp.sum_squares(s_clf_p1)
        + qpw.W_p2_deviation * cp.square(p2 - qpw.p2_desired)
    )

    prob = cp.Problem(cp.Minimize(objective_terms), constraints)

    qp_info = {
        "status": None,
        "solver": None,
        "solver_attempts": [],
        "u_nom": u_nom.copy(),
        "clf_V": V,
        "fallback": False,
        "p1_old": float(p1_old),
        "p1_rate": None,
        "p1_new": None,
        "p2": None,
        "s_clf_p1": None,
    }

    optimal_statuses = {"optimal", "optimal_inaccurate"}
    solver_attempts = []
    solver_plan = [
        ("OSQP", {
            "solver": cp.OSQP,
            "warm_start": True,
            "verbose": False,
            "eps_abs": 1e-5,
            "eps_rel": 1e-5,
            "max_iter": 20000,
        }),
        ("CLARABEL", {
            "solver": "CLARABEL",
            "warm_start": True,
            "verbose": False,
        }),
        ("SCS", {
            "solver": cp.SCS,
            "warm_start": True,
            "verbose": False,
        }),
    ]

    for solver_name, solve_kwargs in solver_plan:
        try:
            prob.solve(**solve_kwargs)
            solver_attempts.append({"solver": solver_name, "status": prob.status})
        except cp.SolverError as exc:
            solver_attempts.append({
                "solver": solver_name,
                "status": f"exception:{type(exc).__name__}",
                "exception": str(exc),
            })
            continue
        except Exception as exc:
            solver_attempts.append({
                "solver": solver_name,
                "status": f"exception:{type(exc).__name__}",
                "exception": str(exc),
            })
            continue

        if prob.status in optimal_statuses:
            qp_info["solver"] = solver_name
            break

    qp_info["status"] = prob.status
    qp_info["solver_attempts"] = solver_attempts
    if qp_info["status"] is None and solver_attempts:
        qp_info["status"] = solver_attempts[-1]["status"]
        if "exception" in solver_attempts[-1]:
            qp_info["exception"] = solver_attempts[-1]["exception"]

    if u.value is None or prob.status not in optimal_statuses:
        u_sol = np.array([
            np.clip(u_nom[0], ego_vp.min_accel, ego_vp.max_accel),
            np.clip(u_nom[1], ego_vp.min_steer_rate, ego_vp.max_steer_rate),
        ], dtype=float)
        qp_info["fallback"] = True
    else:
        u_sol = np.array(u.value).reshape(2)
        p1_rate_value = float(p1_rate.value)
        p1_new = float(p1_old + sim.dt * p1_rate_value)
        qp_info["p1_rate"] = p1_rate_value
        qp_info["p1_new"] = p1_new
        qp_info["p2"] = float(p2.value)
        qp_info["s_clf_p1"] = float(np.array(s_clf_p1.value).reshape(-1)[0])

    return u_sol, qp_info


def run_single_rollout(init_states: Dict[str, np.ndarray], goal_x: float = 5.0):
    sim = lm.SimParams(dt=0.1, T=10.0)
    geom = lm.MergeGeometry()
    names = ["upper", "mid", "lower"]

    vps = {name: lm.VehicleParams() for name in names}
    clfps = {name: lm.CLFParams() for name in names}
    qpws = {name: PACBFWeights() for name in names}
    cbfps = {name: lm.CBFParams(lambda_cbf=0.0, cbf_slack_weight=0.0) for name in names}

    circle_offsets, circle_radius = lm.circle_approximation(
        vps["upper"].length,
        vps["upper"].width,
        n_circles=3,
    )

    states = {name: init_states[name].copy() for name in names}
    p1_values = {name: qpws[name].p1_initial for name in names}
    p2_values = {name: qpws[name].p2_desired for name in names}

    current_clearances = lm.pairwise_vehicle_clearances(states, vps, circle_offsets, circle_radius)

    hist = {
        "t": [0.0],
        "x_switch": geom.x_switch,
        "events": [],
        "cbf_gain_label": "p2",
    }
    for name in names:
        hist[f"x_{name}"] = [states[name].copy()]
        hist[f"u_{name}"] = []
        hist[f"qp_{name}"] = []
        hist[f"goal_{name}"] = []
        hist[f"epsi_{name}"] = []
        hist[f"lambda_{name}"] = [p2_values[name]]
        hist[f"p1_{name}"] = [p1_values[name]]
        hist[f"p2_{name}"] = [p2_values[name]]
    for pair_name, clearance in current_clearances.items():
        hist[f"clearance_{pair_name}"] = [clearance]

    def print_step_details(t_now, pairwise_clearances, boundary_clearances):
        gain_msg = (
            f"p1_u = {p1_values['upper']:.3f} | p2_u = {p2_values['upper']:.3f} | "
            f"p1_m = {p1_values['mid']:.3f} | p2_m = {p2_values['mid']:.3f} | "
            f"p1_l = {p1_values['lower']:.3f} | p2_l = {p2_values['lower']:.3f}"
        )
        msg = (
            f"t = {t_now:.1f} s | "
            f"clr_ul = {pairwise_clearances['upper_lower']:.3f} | "
            f"clr_mu = {pairwise_clearances['mid_upper']:.3f} | "
            f"clr_ml = {pairwise_clearances['mid_lower']:.3f} | "
            f"{gain_msg}"
        )
        print(msg, flush=True)

    for k in range(sim.steps):
        current_clearances = lm.pairwise_vehicle_clearances(states, vps, circle_offsets, circle_radius)
        boundary_clearances = {
            name: lm.min_boundary_clearance(states[name], name, geom, vps[name], circle_offsets, circle_radius)
            for name in names
        }

        if min(current_clearances.values()) < -1e-3:
            hist["events"].append("collision")
            print_step_details(k * sim.dt, current_clearances, boundary_clearances)
            print(f"  terminating at t = {k * sim.dt:.1f} s: inter-vehicle collision")
            break
        if min(boundary_clearances.values()) < -1e-3:
            hist["events"].append("boundary_collision")
            print_step_details(k * sim.dt, current_clearances, boundary_clearances)
            print(f"  terminating at t = {k * sim.dt:.1f} s: boundary collision")
            break

        goals = {}
        controls = {}
        qp_infos = {}

        for name in names:
            fc = lm.front_circle_center(states[name], vps[name])
            goals[name] = lm.lookahead_goal_from_track_point(fc, name, geom)

            other_names = [n for n in names if n != name]
            try:
                controls[name], qp_infos[name] = solve_vehicle_qp_pacbf(
                    ego_state=states[name].copy(),
                    other_states=[states[n].copy() for n in other_names],
                    ego_name=name,
                    ego_vp=vps[name],
                    other_vps=[vps[n] for n in other_names],
                    clfp=clfps[name],
                    cbfp=cbfps[name],
                    qpw=qpws[name],
                    sim=sim,
                    geom=geom,
                    circle_offsets=circle_offsets,
                    circle_radius=circle_radius,
                    goal_point=goals[name],
                    p1_old=p1_values[name],
                )
            except Exception as exc:
                u_nom = lm.nominal_control(states[name], clfps[name], vps[name], goals[name])
                controls[name] = np.array([
                    np.clip(u_nom[0], vps[name].min_accel, vps[name].max_accel),
                    np.clip(u_nom[1], vps[name].min_steer_rate, vps[name].max_steer_rate),
                ], dtype=float)
                qp_infos[name] = {
                    "status": f"exception:{type(exc).__name__}",
                    "u_nom": u_nom.copy(),
                    "clf_V": np.nan,
                    "fallback": True,
                    "exception": str(exc),
                    "p1_old": p1_values[name],
                    "p1_rate": None,
                    "p1_new": None,
                    "p2": None,
                }

        infeasible_names = [
            name for name in names
            if qp_infos[name].get("status") not in ["optimal", "optimal_inaccurate"]
        ]

        if infeasible_names:
            status_msg = ", ".join(
                f"{name}={qp_infos[name].get('status')}" for name in infeasible_names
            )
            print_step_details(k * sim.dt, current_clearances, boundary_clearances)
            print(f"  terminating rollout at t = {k * sim.dt:.1f} s: infeasible QP ({status_msg})")
            print("  moving to next rollout")

            for name in names:
                hist[f"x_{name}"].append(states[name].copy())
                hist[f"u_{name}"].append(controls[name].copy())
                hist[f"qp_{name}"].append(qp_infos[name])
                hist[f"goal_{name}"].append(goals[name].copy())
                hist[f"lambda_{name}"].append(p2_values[name])
                hist[f"p1_{name}"].append(p1_values[name])
                hist[f"p2_{name}"].append(p2_values[name])
            for pair_name, clearance in current_clearances.items():
                hist[f"clearance_{pair_name}"].append(clearance)
            hist["events"].append("infeasible_qp")
            hist["t"].append((k + 1) * sim.dt)
            break

        for name in names:
            _, epsi = lm.desired_steer_from_heading_error(
                state=states[name],
                goal_point=goals[name],
                vp=vps[name],
                clfp=clfps[name],
            )
            hist[f"epsi_{name}"].append(epsi)

        for name in names:
            if qp_infos[name].get("p1_new") is not None:
                p1_values[name] = qp_infos[name]["p1_new"]
            if qp_infos[name].get("p2") is not None:
                p2_values[name] = qp_infos[name]["p2"]

        print_step_details(k * sim.dt, current_clearances, boundary_clearances)

        for name in names:
            states[name] = lm.step_vehicle(states[name], controls[name], vps[name], sim.dt)

        next_clearances = lm.pairwise_vehicle_clearances(states, vps, circle_offsets, circle_radius)
        next_boundary_clearances = {
            name: lm.min_boundary_clearance(states[name], name, geom, vps[name], circle_offsets, circle_radius)
            for name in names
        }

        if min(next_clearances.values()) < -1e-3:
            event = "collision"
        elif min(next_boundary_clearances.values()) < -1e-3:
            event = "boundary_collision"
        else:
            event = "running"

        for name in names:
            hist[f"x_{name}"].append(states[name].copy())
            hist[f"u_{name}"].append(controls[name].copy())
            hist[f"qp_{name}"].append(qp_infos[name])
            hist[f"goal_{name}"].append(goals[name].copy())
            hist[f"lambda_{name}"].append(p2_values[name])
            hist[f"p1_{name}"].append(p1_values[name])
            hist[f"p2_{name}"].append(p2_values[name])
        for pair_name, clearance in next_clearances.items():
            hist[f"clearance_{pair_name}"].append(clearance)
        hist["events"].append(event)
        hist["t"].append((k + 1) * sim.dt)

        if event != "running":
            print_step_details((k + 1) * sim.dt, next_clearances, next_boundary_clearances)
            print(f"  terminating at t = {(k + 1) * sim.dt:.1f} s: {event.replace('_', ' ')}")
            break

    for key in list(hist.keys()):
        if (
            key.startswith("x_")
            or key.startswith("u_")
            or key.startswith("goal_")
            or key.startswith("epsi_")
            or key.startswith("lambda_")
            or key.startswith("p1_")
            or key.startswith("p2_")
            or key.startswith("clearance_")
            or key == "t"
        ):
            hist[key] = np.array(hist[key])
    for name in names:
        if hist[f"u_{name}"].size == 0:
            hist[f"u_{name}"] = hist[f"u_{name}"].reshape(0, 2)
        if hist[f"goal_{name}"].size == 0:
            hist[f"goal_{name}"] = hist[f"goal_{name}"].reshape(0, 2)

    metrics = lm.compute_rollout_metrics(
        hist=hist,
        vps=vps,
        geom=geom,
        circle_offsets=circle_offsets,
        circle_radius=circle_radius,
        goal_x=goal_x,
    )

    return hist, metrics, sim, vps, geom, circle_offsets, circle_radius


def run_monte_carlo(n_rollouts: int = 20):
    geom = lm.MergeGeometry()
    rollout_cfg = lm.RolloutConfig(n_rollouts=n_rollouts)
    initial_states = lm.build_random_initial_states(rollout_cfg, geom)

    all_metrics = []
    last_hist = None
    last_bundle = None

    for r_idx, init_states in enumerate(initial_states, start=1):
        print(f"\n=== Rollout {r_idx}/{n_rollouts} ===")
        for name in ["upper", "mid", "lower"]:
            print(
                f"Initial {name}: x={init_states[name][0]:.3f}, y={init_states[name][1]:.3f}, "
                f"psi={init_states[name][2]:.3f} rad"
            )

        hist, metrics, sim, vps, geom, circle_offsets, circle_radius = run_single_rollout(
            init_states=init_states,
            goal_x=rollout_cfg.goal_x,
        )

        all_metrics.append(metrics)
        last_hist = hist
        last_bundle = (sim, vps, geom, circle_offsets, circle_radius)

    avg_metrics = lm.average_rollout_metrics(all_metrics)
    return avg_metrics, last_hist, last_bundle


if __name__ == "__main__":
    print("PACBF baseline will be used.")
    print("Edit PACBFWeights near the top of this file to tune p1, p2, and objective weights.")

    n_rollouts = 20
    video_filename = "three_vehicle_merge_pacbf.mp4"

    avg_metrics, last_hist, last_bundle = run_monte_carlo(n_rollouts=n_rollouts)
    lm.show_average_metrics_table(avg_metrics, n_rollouts=n_rollouts)

    sim, vps, geom, circle_offsets, circle_radius = last_bundle
    lm.animate_simulation(
        last_hist,
        vps,
        geom,
        circle_offsets,
        circle_radius,
        filename=video_filename,
    )
