import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import cvxpy as cp
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import RI_Baseline_Methods as ri  # noqa: E402


@dataclass
class PACBFWeights(ri.QPWeights):
    p1_initial: float = 0.1
    p1_desired: float = 0.1
    p2_desired: float = 1.0
    p2_min: float = 0.0
    epsilon_clf_p1: float = 5.0
    W_p1_rate: float = 1e-4
    W_slack_clf_p1: float = 1e3
    W_p2_deviation: float = 1e3


def solve_vehicle_qp_pacbf_against_many(
    ego_state: np.ndarray,
    other_states: List[np.ndarray],
    ego_route: ri.RouteSpec,
    ego_vp: ri.VehicleParams,
    other_vps: List[ri.VehicleParams],
    clfp: ri.CLFParams,
    cbfp: ri.CBFParams,
    qpw: PACBFWeights,
    sim: ri.SimParams,
    circle_offsets: np.ndarray,
    circle_radius: float,
    goal_point: np.ndarray,
    p1_old: float,
) -> Tuple[np.ndarray, Dict]:
    """PACBF-style QP for one ego vehicle against all other vehicles.

    The ego control term is kept with full authority. The non-control/PACBF
    residual of inter-vehicle CBF constraints is split by 1/2, matching the
    decentralized TTCBF split used elsewhere. Boundary constraints are not
    shared with another vehicle, so they are not split.
    """

    u_nom = ri.nominal_control(ego_state, clfp, ego_vp, goal_point)

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

    V, Vdot_const, Vdot_u = ri.clf_terms(ego_state, ego_vp, clfp, goal_point)
    constraints += [Vdot_const + Vdot_u @ u <= -clfp.clf_rate * V + s_clf[0]]

    V_p1 = (p1_old - qpw.p1_desired) ** 2
    constraints += [
        p1_old + p1_rate * dt >= 0.0,
        p2 >= qpw.p2_min,
        2.0 * (p1_old - qpw.p1_desired) * p1_rate
        + qpw.epsilon_clf_p1 * V_p1
        <= s_clf_p1[0],
    ]

    for other_state, other_vp in zip(other_states, other_vps):
        for s_i in circle_offsets:
            for s_j in circle_offsets:
                h, h_dot, hddot_aff = ri.pairwise_circle_cbf_affine(
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

    boundary_specs = []
    for s_i in circle_offsets:
        for h, h_dot, hddot_aff, spec in ri.intersection_boundary_cbf_affines(
            ego_state=ego_state,
            ego_s=s_i,
            ego_vp=ego_vp,
            geom=ego_route.geom,
            circle_radius=circle_radius,
            cbfp=cbfp,
        ):
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
            boundary_specs.append(spec)

    objective = cp.Minimize(
        0.5 * cp.quad_form(u - u_nom, qpw.u_weight)
        + clfp.clf_slack_weight * cp.sum_squares(s_clf)
        + qpw.W_p1_rate * p1_rate
        + qpw.W_slack_clf_p1 * cp.sum_squares(s_clf_p1)
        + qpw.W_p2_deviation * cp.square(p2 - qpw.p2_desired)
    )

    prob = cp.Problem(objective, constraints)
    qp_info = {
        "status": None,
        "solver": None,
        "solver_attempts": [],
        "u_nom": u_nom.copy(),
        "clf_V": V,
        "boundary_specs": boundary_specs,
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
        qp_info["p1_rate"] = p1_rate_value
        qp_info["p1_new"] = float(p1_old + sim.dt * p1_rate_value)
        qp_info["p2"] = float(p2.value)
        qp_info["s_clf_p1"] = float(np.array(s_clf_p1.value).reshape(-1)[0])

    return u_sol, qp_info


def make_pacbf_vehicle_data(routes: Dict[str, ri.RouteSpec], cfg: ri.BaselineConfig) -> Dict[str, Dict]:
    vehicle_data = ri.make_baseline_vehicle_data(
        routes,
        cfg,
        {name: 0.0 for name in ri.BASELINE_VEHICLE_NAMES},
    )
    for name in ri.BASELINE_VEHICLE_NAMES:
        vehicle_data[name]["qpw"] = PACBFWeights()
        vehicle_data[name]["cbfp"].lambda_cbf = 0.0
    return vehicle_data


def run_pacbf_monte_carlo(
    cfg: ri.BaselineConfig,
    n_rollouts: int = 20,
    seed: int = 7,
) -> Tuple[Dict, Dict, Tuple]:
    geom = ri.IntersectionGeometry()
    sim = ri.SimParams()
    rng = np.random.default_rng(seed)

    all_metrics = []
    last_hist = None
    last_bundle = None

    for rollout_idx in range(1, n_rollouts + 1):
        routes = ri.sample_baseline_routes(rng, geom)
        vehicle_data = make_pacbf_vehicle_data(routes, cfg)
        p1_values = {
            name: vehicle_data[name]["qpw"].p1_initial
            for name in ri.BASELINE_VEHICLE_NAMES
        }

        print(f"\n=== PACBF Rollout {rollout_idx}/{n_rollouts} ===")
        for name in ri.BASELINE_VEHICLE_NAMES:
            route = routes[name]
            qpw = vehicle_data[name]["qpw"]
            print(
                f"vehicle {name}: start {route.start}, exit {route.exit}, "
                f"p1 = {p1_values[name]:.3f}, p2_desired = {qpw.p2_desired:.3f}"
            )

        def pacbf_solver(name: str, other_names: List[str], **kwargs):
            qpw = kwargs["qpw"]
            u_sol, qp_info = solve_vehicle_qp_pacbf_against_many(
                **kwargs,
                p1_old=p1_values[name],
            )
            if qp_info.get("p1_new") is not None:
                p1_values[name] = float(qp_info["p1_new"])
            if qp_info.get("p2") is not None:
                vehicle_data[name]["cbfp"].lambda_cbf = float(qp_info["p2"])
            else:
                vehicle_data[name]["cbfp"].lambda_cbf = float(qpw.p2_desired)
            return u_sol, qp_info

        hist, metrics, bundle = ri.run_multi_vehicle_rollout(
            vehicle_data=vehicle_data,
            sim=sim,
            qp_solve_fn=pacbf_solver,
        )
        metrics["episode_return"] = float("nan")
        all_metrics.append(metrics)
        last_hist = hist
        last_bundle = bundle

    avg_metrics = ri.average_rollout_metrics(all_metrics)
    return avg_metrics, last_hist, last_bundle


def main() -> None:
    cfg = ri.BaselineConfig()

    parser = argparse.ArgumentParser(description="Three-vehicle RI PACBF baseline rollouts.")
    parser.add_argument("--n_rollouts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip_animation", action="store_true")
    args = parser.parse_args()

    print("\nRoad Intersection PACBF settings")
    print(f"  n_rollouts = {args.n_rollouts}, seed = {args.seed}")
    print("  p1/p2 parameters are in PACBFWeights.")

    avg_metrics, last_hist, last_bundle = run_pacbf_monte_carlo(
        cfg,
        n_rollouts=args.n_rollouts,
        seed=args.seed,
    )
    ri.show_average_metrics_table(avg_metrics, n_rollouts=args.n_rollouts)

    _, vehicle_data, circle_offsets, circle_radius = last_bundle
    video_filename = "RI_PACBF_simulation.mp4"

    if not args.skip_animation:
        ri.animate_multi_vehicle_simulation(
            last_hist,
            vehicle_data,
            circle_offsets,
            circle_radius,
            filename=video_filename,
        )


if __name__ == "__main__":
    main()
