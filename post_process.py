import argparse, os, sys, json, copy
import numpy as np
import trimesh
import cvxpy as cp
import time

def load_meshes_from_folder(folder):
    obj_paths = sorted([p for p in os.listdir(folder) if p.lower().endswith(".obj")])
    if not obj_paths:
        raise FileNotFoundError("No OBJ files found.")
    if "asset.obj" not in [os.path.basename(p) for p in obj_paths]:
        raise FileNotFoundError("The folder must contain 'asset.obj'.")
    meshes, names = [], []
    for rel in obj_paths:
        path = os.path.join(folder, rel)
        m = trimesh.load(path, process=True)
        if not isinstance(m, trimesh.Trimesh):
            # Merge if Scene
            if isinstance(m, trimesh.Scene):
                m = trimesh.util.concatenate(tuple(g for g in m.geometry.values()))
            else:
                raise TypeError(f"{rel}: must be Trimesh or Scene.")
        meshes.append(m)
        names.append(os.path.basename(rel))
    return meshes, names

def sample_vertices(mesh, max_points):
    V = mesh.vertices
    if len(V) <= max_points:
        return V
    # Uniform sample (by stride order)
    idx = np.linspace(0, len(V)-1, num=max_points, dtype=int)
    return V[idx]

def moved_mesh(mesh, t):
    # Copy applying translation only
    m = mesh.copy()
    m.apply_translation(t)
    return m

def collect_contact_constraints(moved_meshes, max_samples, clearance=1e-4):
    """
    For each pair i != k:
      - If a sampled vertex of i is inside k (penetrating),
        add constraint (n · (Δt_i - Δt_k)) >= d + clearance.
    Returns:
      contacts: list of (i, k, n(3,), rhs_scalar)
    """
    N = len(moved_meshes)
    contacts = []
    for i in range(N):
        Pi = sample_vertices(moved_meshes[i], max_samples)
        if len(Pi) == 0:
            continue
        for k in range(N):
            if i == k:
                continue
            mk = moved_meshes[k]

            # Penetration test: signed distance > 0 (trimesh convention; inside is positive)
            inside_mask = mk.contains(Pi)
            P_in = Pi[inside_mask]
            if len(P_in) == 0:
                continue
            # Also get closest point/triangle to extract normal
            closest_pts, dist_unsigned, tri_id = trimesh.proximity.closest_point(mk, P_in)

            '''
            # Take normals from face_normals (translation does not change normals)
            face_normals = mk.face_normals
            ns = face_normals[tri_id]  # (M, 3)
            ds = dist_unsigned         # (M,)  (penetration depth; always positive)
            # For each contact, (n·(Δt_i - Δt_k)) >= d + clearance
            for n_vec, d in zip(ns, ds):
                n_unit = n_vec / (np.linalg.norm(n_vec) + 1e-12)
                rhs = float(d + clearance)
                contacts.append((i, k, n_unit, rhs))
            '''
            dirs = closest_pts - P_in
            dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)

            # rhs = penetration depth(=dist_unsigned) + clearance
            for g, d in zip(dirs, dist_unsigned):
                rhs = float(d + clearance)
                contacts.append((i, k, g, rhs))
    return contacts

def reduce_contacts(contacts, max_per_pair=50):
    """
    Keep at most `max_per_pair` deepest contacts per (i,k) pair.
    """
    by_pair = {}
    for (i, k, n, rhs) in contacts:
        by_pair.setdefault((i, k), []).append((rhs, n))
    reduced = []
    for (i, k), lst in by_pair.items():
        lst.sort(key=lambda x: x[0], reverse=True)  # deepest first
        for rhs, n in lst[:max_per_pair]:
            reduced.append((i, k, n, rhs))
    return reduced

def solve_qp_deltas(N, asset_index, contacts, step_limit=None, slack_weight=1e4, allow_inaccurate=True):
    """
    Variables: ΔT ∈ R^{N×3}, s ∈ R^{M}_{>=0}
    Objective: minimize ∑ ||Δt_i||^2 + slack_weight * ||s||^2
    Constraints: n_j·(Δt_i - Δt_k) + s_j >= rhs_j   (for all contacts j)
                 Δt_asset = 0
                 (optional) -step_limit <= Δt_i,· <= step_limit
    """
    M = len(contacts)
    dT = cp.Variable((N, 3))
    s  = cp.Variable(M, nonneg=True)
    cons = []

    # fix asset
    cons += [dT[asset_index, :] == 0]

    # contact constraints with slack
    for j, (i, k, n, rhs) in enumerate(contacts):
        cons.append(n @ (dT[i, :] - dT[k, :]) + s[j] >= rhs)

    # trust-region box constraint (optional)
    if step_limit is not None and step_limit > 0:
        cons += [dT <= step_limit, dT >= -step_limit]

    obj = cp.Minimize(cp.sum_squares(dT) + slack_weight * cp.sum_squares(s))
    prob = cp.Problem(obj, cons)

    status_ok = {"optimal", "optimal_inaccurate"} if allow_inaccurate else {"optimal"}

    # Try OSQP first
    try:
        prob.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, max_iter=200000, polish=True, verbose=False)
    except Exception:
        pass

    if prob.status not in status_ok or dT.value is None:
        # Fallback to ECOS
        try:
            prob.solve(solver=cp.ECOS, abstol=1e-7, reltol=1e-7, feastol=1e-7, verbose=False)
        except Exception:
            pass

    if prob.status not in status_ok or dT.value is None:
        raise RuntimeError(f"QP infeasible/failed (status={prob.status}). Try increasing --step_limit or lowering --clearance.")

    # diagnostics
    if s.value is not None:
        active = int((s.value > 1e-8).sum())
        print(f"  slack active: {active}/{M}, max slack: {float(s.value.max()):.3e}")

    return dT.value

def penetration_score(meshes, T, samples_per_mesh):
    moved = [moved_mesh(meshes[i], T[i]) for i in range(len(meshes))]
    score = 0.0
    count = 0
    for i in range(len(moved)):
        Pi = sample_vertices(moved[i], samples_per_mesh)
        for k in range(len(moved)):
            if i == k:
                continue
            mk = moved[k]
            inside = mk.contains(Pi)
            if not inside.any():
                continue
            P_in = Pi[inside]
            _, dist_unsigned, _ = trimesh.proximity.closest_point(mk, P_in)
            score += float(dist_unsigned.sum())
            count += int(dist_unsigned.shape[0])
    return score, count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_folder", type=str, required=True, help="Folder containing OBJ files")
    ap.add_argument("--max_iters", type=int, default=10, help="Number of linearize-QP iterations")
    ap.add_argument("--samples_per_mesh", type=int, default=2000, help="Max number of vertex samples per mesh for penetration test")
    ap.add_argument("--clearance", type=float, default=1e-4, help="Clearance distance after resolving overlaps")
    ap.add_argument("--max_contacts_per_pair", type=int, default=60, help="Cap contacts per (i,k) pair to stabilize QP")
    ap.add_argument("--step_limit", type=float, default=0.02, help="Per-iteration translation limit (same units as meshes)")
    ap.add_argument("--slack_weight", type=float, default=1e4, help="Quadratic weight for contact slacks")
    args = ap.parse_args()

    # Derive output directory next to the input tree
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(args.target_folder)),
        "mesh_outputs_per_obj_opt",
        os.path.basename(args.target_folder)
    )
    os.makedirs(out_dir, exist_ok=True)

    meshes, names = load_meshes_from_folder(args.target_folder)
    N = len(meshes)
    asset_index = names.index("asset.obj")

    # Accumulated translation (absolute pose change)
    T = np.zeros((N, 3), dtype=np.float64)

    for it in range(1, args.max_iters + 1):
        # Copy of meshes with current translation applied
        moved = [moved_mesh(meshes[i], T[i]) for i in range(N)]
        # Collect constraints due to contacts (penetrations)
        contacts = collect_contact_constraints(
            moved_meshes=moved,
            max_samples=args.samples_per_mesh,
            clearance=args.clearance
        )
        # Reduce per-pair contacts to avoid contradictory/duplicated constraints
        contacts = reduce_contacts(contacts, max_per_pair=args.max_contacts_per_pair)
        print(f"[Iter {it}] contacts: {len(contacts)} (after reduce)")
        if len(contacts) == 0:
            print("No penetration detected. Terminating.")
            break

        # Compute ΔT with QP
        dT = solve_qp_deltas(
            N, asset_index, contacts,
            step_limit=args.step_limit,
            slack_weight=args.slack_weight,
            allow_inaccurate=True
        )
        # accumulate
        #T += dT

        # ---- line search ----
        ls_alphas = [1.0, 0.5, 0.25, 0.125, 0.0625]
        ls_samples = min(400, max(50, args.samples_per_mesh // 10))  # 평가용은 적게

        base_score, base_cnt = penetration_score(meshes, T, ls_samples)

        best_alpha = 0.0
        best_score = base_score
        best_cnt = base_cnt

        for a in ls_alphas:
            cand_T = T + a * dT
            s, c = penetration_score(meshes, cand_T, ls_samples)
            if s < best_score - 1e-9:  # 개선된 경우만 채택
                best_score, best_cnt, best_alpha = s, c, a
        
        print(f"  LS: base(score={base_score:.4e}, cnt={base_cnt}) -> best(score={best_score:.4e}, cnt={best_cnt}) with alpha={best_alpha}")

        if best_alpha == 0.0:
            best_alpha = ls_alphas[-1]   # 예: 0.0625
            print(f"  LS: no improvement found; applying smallest alpha={best_alpha} and consider reducing step_limit")

        T += best_alpha * dT

        # Convergence check: terminate if max step displacement is very small
        max_step = np.linalg.norm(best_alpha * dT, axis=1).max()
        print(f"  max |αΔt| = {max_step:.6g}")
        if max_step < 1e-6:
            print("Δt is very small. Terminating.")
            # Check penetration one more time
            moved = [moved_mesh(meshes[i], T[i]) for i in range(N)]
            contacts = collect_contact_constraints(moved, args.samples_per_mesh, args.clearance)
            contacts = reduce_contacts(contacts, max_per_pair=args.max_contacts_per_pair)
            if len(contacts) > 0:
                print("Note: small penetrations may remain. Consider tuning samples_per_mesh/clearance.")
            break

    # Output: save translated OBJs
    for i, (m, name) in enumerate(zip(meshes, names)):
        m_out = moved_mesh(m, T[i])
        out_path = os.path.join(out_dir, name)
        m_out.export(out_path)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"{end - start:.5f} sec")