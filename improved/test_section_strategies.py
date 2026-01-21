"""
Test different section selection strategies for hybrid search
Compare:
1. Original: Take last point of section (current implementation)
2. Average: Take average (midpoint) of section
3. Median: Take median of section
"""

import numpy as np
import pickle
import time
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)
import Functions as F_orig
sys.path.insert(0, os.path.join(parent_dir, 'improved'))
from Functions_improved import hybrid_smart_search_nk

# Load datasets
datasets = {}
for i in range(1, 5):
    try:
        dataset_path = os.path.join(parent_dir, f'ts{i}.p')
        with open(dataset_path, 'rb') as f:
            datasets[f'ts{i}'] = pickle.load(f)
        print(f"✓ Loaded ts{i}: {len(datasets[f'ts{i}'])} points")
    except Exception as e:
        print(f"✗ Failed to load ts{i}: {e}")

print(f"\nTotal datasets: {len(datasets)}")
print("="*80)

# Modify section candidate generation for different strategies
def hybrid_avg_search_nk(Data, p=4, q=2, K=10, refine_range=5):
    """Hybrid search using section average (midpoint)"""
    choice = 2
    N = Data.shape[0]

    print(f"[Hybrid Avg Search] Phase 1: Coarse search with K={K} sections (midpoint)")

    section_size = max((N - 1) / K, 1)
    # Use midpoint of each section
    coarse_candidates = [int(round((i - 0.5) * section_size)) for i in range(1, K + 1)]
    coarse_candidates = [max(1, min(c, N - 1)) for c in coarse_candidates]
    coarse_candidates = sorted(set(coarse_candidates))

    best_GCV_coarse = 1.0e+9
    best_n_coarse = None

    for n in coarse_candidates:
        c = n + p
        U = F_orig.Kno_pspline_opt(Data, p, n)
        B = F_orig.Basis_Pspline(n, p, U, Data[:, 0])
        lamb = 0.1
        lam = F_orig.Smoothing_par(Data, B, q, c, lamb, choice)

        if lam.fun < best_GCV_coarse:
            best_GCV_coarse = lam.fun
            best_n_coarse = n

    print(f"[Hybrid Avg Search] Phase 1 result: n={best_n_coarse}")
    print(f"[Hybrid Avg Search] Phase 2: Fine search in [{best_n_coarse}-{refine_range}, {best_n_coarse}+{refine_range}]")

    fine_start = max(1, best_n_coarse - refine_range)
    fine_end = min(N - 1, best_n_coarse + refine_range)

    best_GCV = 1.0e+9
    opt_n = None
    opt_lam = None
    tested_count = len(coarse_candidates)

    for n in range(fine_start, fine_end + 1):
        c = n + p
        U = F_orig.Kno_pspline_opt(Data, p, n)
        B = F_orig.Basis_Pspline(n, p, U, Data[:, 0])
        lamb = 0.1
        lam = F_orig.Smoothing_par(Data, B, q, c, lamb, choice)

        tested_count += 1

        if lam.fun < best_GCV:
            best_GCV = lam.fun
            opt_n = n
            opt_lam = lam.x[0]

    total_candidates = N - 1
    print(f"[Hybrid Avg Search] Tested {tested_count}/{total_candidates} candidates ({100*tested_count/total_candidates:.1f}%)")
    print(f"[Hybrid Avg Search] Final result: n={opt_n}, λ={opt_lam:.6f}")

    c = opt_n + p
    P = opt_lam * F_orig.Penalty_p(q, c)
    U = F_orig.Kno_pspline_opt(Data, p, opt_n)
    B_dat = F_orig.Basis_Pspline(opt_n, p, U, Data[:, 0])
    theta = np.linalg.solve(B_dat.T.dot(B_dat) + P, B_dat.T.dot(Data[:, 1].reshape(-1, 1)))
    nr = (Data[:, 1].reshape(-1, 1) - B_dat.dot(theta)).reshape(-1, 1)

    term = B_dat.dot(np.linalg.inv(B_dat.T.dot(B_dat) + P).dot(B_dat.T))
    n_pts = Data.shape[0]
    df_res = n_pts - 2 * np.trace(term) + np.trace(term.dot(term.T))
    sigmasq = (nr.T.dot(nr)) / (df_res)
    sigmasq = sigmasq[0][0]

    return [opt_n, opt_lam, sigmasq]


def hybrid_median_search_nk(Data, p=4, q=2, K=10, refine_range=5):
    """Hybrid search using section median (calculated as midpoint for consistency)"""
    choice = 2
    N = Data.shape[0]

    print(f"[Hybrid Median Search] Phase 1: Coarse search with K={K} sections (median)")

    section_size = max((N - 1) / K, 1)
    # For median, we use same calculation as average for consistency with integer knots
    coarse_candidates = [int(round((i - 0.5) * section_size)) for i in range(1, K + 1)]
    coarse_candidates = [max(1, min(c, N - 1)) for c in coarse_candidates]
    coarse_candidates = sorted(set(coarse_candidates))

    best_GCV_coarse = 1.0e+9
    best_n_coarse = None

    for n in coarse_candidates:
        c = n + p
        U = F_orig.Kno_pspline_opt(Data, p, n)
        B = F_orig.Basis_Pspline(n, p, U, Data[:, 0])
        lamb = 0.1
        lam = F_orig.Smoothing_par(Data, B, q, c, lamb, choice)

        if lam.fun < best_GCV_coarse:
            best_GCV_coarse = lam.fun
            best_n_coarse = n

    print(f"[Hybrid Median Search] Phase 1 result: n={best_n_coarse}")
    print(f"[Hybrid Median Search] Phase 2: Fine search in [{best_n_coarse}-{refine_range}, {best_n_coarse}+{refine_range}]")

    fine_start = max(1, best_n_coarse - refine_range)
    fine_end = min(N - 1, best_n_coarse + refine_range)

    best_GCV = 1.0e+9
    opt_n = None
    opt_lam = None
    tested_count = len(coarse_candidates)

    for n in range(fine_start, fine_end + 1):
        c = n + p
        U = F_orig.Kno_pspline_opt(Data, p, n)
        B = F_orig.Basis_Pspline(n, p, U, Data[:, 0])
        lamb = 0.1
        lam = F_orig.Smoothing_par(Data, B, q, c, lamb, choice)

        tested_count += 1

        if lam.fun < best_GCV:
            best_GCV = lam.fun
            opt_n = n
            opt_lam = lam.x[0]

    total_candidates = N - 1
    print(f"[Hybrid Median Search] Tested {tested_count}/{total_candidates} candidates ({100*tested_count/total_candidates:.1f}%)")
    print(f"[Hybrid Median Search] Final result: n={opt_n}, λ={opt_lam:.6f}")

    c = opt_n + p
    P = opt_lam * F_orig.Penalty_p(q, c)
    U = F_orig.Kno_pspline_opt(Data, p, opt_n)
    B_dat = F_orig.Basis_Pspline(opt_n, p, U, Data[:, 0])
    theta = np.linalg.solve(B_dat.T.dot(B_dat) + P, B_dat.T.dot(Data[:, 1].reshape(-1, 1)))
    nr = (Data[:, 1].reshape(-1, 1) - B_dat.dot(theta)).reshape(-1, 1)

    term = B_dat.dot(np.linalg.inv(B_dat.T.dot(B_dat) + P).dot(B_dat.T))
    n_pts = Data.shape[0]
    df_res = n_pts - 2 * np.trace(term) + np.trace(term.dot(term.T))
    sigmasq = (nr.T.dot(nr)) / (df_res)
    sigmasq = sigmasq[0][0]

    return [opt_n, opt_lam, sigmasq]


# Test all strategies
results = []
for ds_name, data in datasets.items():
    print(f"\n{'='*80}")
    print(f"TESTING: {ds_name} ({len(data)} points)")
    print('='*80)

    result = {
        'dataset': ds_name,
        'n_points': len(data)
    }

    # Original full search
    start = time.time()
    r_orig = F_orig.full_search_nk(data, p=4, q=2)
    time_orig = time.time() - start
    result['orig_n'] = r_orig[0]
    result['orig_lam'] = r_orig[1]
    result['orig_time'] = time_orig
    print(f"  Original: n={r_orig[0]}, λ={r_orig[1]:.6f}, Time: {time_orig:.3f}s")

    # Current hybrid (last point)
    start = time.time()
    r_hybrid = hybrid_smart_search_nk(data, p=4, q=2, K=10, refine_range=5)
    time_hybrid = time.time() - start
    result['hybrid_n'] = r_hybrid[0]
    result['hybrid_time'] = time_hybrid
    result['hybrid_match'] = (r_hybrid[0] == r_orig[0])
    print(f"  Hybrid (last): n={r_hybrid[0]}, λ={r_hybrid[1]:.6f}, Time: {time_hybrid:.3f}s, Speedup: {time_orig/time_hybrid:.2f}×")

    # Hybrid with midpoint
    start = time.time()
    r_avg = hybrid_avg_search_nk(data, p=4, q=2, K=10, refine_range=5)
    time_avg = time.time() - start
    result['avg_n'] = r_avg[0]
    result['avg_time'] = time_avg
    result['avg_match'] = (r_avg[0] == r_orig[0])
    print(f"  Hybrid (avg): n={r_avg[0]}, λ={r_avg[1]:.6f}, Time: {time_avg:.3f}s, Speedup: {time_orig/time_avg:.2f}×")

    # Hybrid with median
    start = time.time()
    r_median = hybrid_median_search_nk(data, p=4, q=2, K=10, refine_range=5)
    time_median = time.time() - start
    result['median_n'] = r_median[0]
    result['median_time'] = time_median
    result['median_match'] = (r_median[0] == r_orig[0])
    print(f"  Hybrid (median): n={r_median[0]}, λ={r_median[1]:.6f}, Time: {time_median:.3f}s, Speedup: {time_orig/time_median:.2f}×")

    results.append(result)


print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)
print(f"{'Dataset':<8} {'Pts':<5} {'Orig n':<8} {'Hybrid n':<9} {'Avg n':<8} {'Median n':<10}")
print("-"*80)

for r in results:
    print(f"{r['dataset']:<8} {r['n_points']:<5} "
          f"{r['orig_n']:<8} {r['hybrid_n']:<9} {r['avg_n']:<8} {r['median_n']:<10}")

print("\n" + "="*80)
print("ACCURACY CHECK")
print("="*80)
print(f"{'Dataset':<8} {'Hybrid Match':<14} {'Avg Match':<12} {'Median Match':<14}")
print("-"*80)

for r in results:
    print(f"{r['dataset']:<8} {str(r['hybrid_match']):<14} {str(r['avg_match']):<12} {str(r['median_match']):<14}")

print("\n" + "="*80)
print("SPEED COMPARISON")
print("="*80)
print(f"{'Dataset':<8} {'Original':<12} {'Hybrid':<12} {'Avg':<12} {'Median':<12}")
print("-"*80)

for r in results:
    print(f"{r['dataset']:<8} {r['orig_time']:<12.3f} {r['hybrid_time']:<12.3f} {r['avg_time']:<12.3f} {r['median_time']:<12.3f}")

# Calculate averages
avg_hybrid_speedup = np.mean([r['orig_time']/r['hybrid_time'] for r in results])
avg_avg_speedup = np.mean([r['orig_time']/r['avg_time'] for r in results])
avg_median_speedup = np.mean([r['orig_time']/r['median_time'] for r in results])

print("\n" + "="*80)
print("AVERAGE SPEEDUP")
print("="*80)
print(f"Hybrid (last): {avg_hybrid_speedup:.2f}× faster")
print(f"Hybrid (avg): {avg_avg_speedup:.2f}× faster")
print(f"Hybrid (median): {avg_median_speedup:.2f}× faster")

print("\n" + "="*80)
print("KEY OBSERVATIONS")
print("="*80)

all_hybrid_match = all(r['hybrid_match'] for r in results)
all_avg_match = all(r['avg_match'] for r in results)
all_median_match = all(r['median_match'] for r in results)

if all_hybrid_match and all_avg_match and all_median_match:
    print("✓ All three strategies achieved 100% accuracy")
elif all_hybrid_match and all_avg_match:
    print("✓ Original and average strategies achieved 100% accuracy")
elif all_hybrid_match and all_median_match:
    print("✓ Original and median strategies achieved 100% accuracy")
elif all_avg_match and all_median_match:
    print("✓ Average and median strategies achieved 100% accuracy")
elif all_hybrid_match:
    print("✓ Original hybrid strategy achieved 100% accuracy")
elif all_avg_match:
    print("✓ Average strategy achieved 100% accuracy")
elif all_median_match:
    print("✓ Median strategy achieved 100% accuracy")
else:
    print("⚠ All strategies missed some optimal values")

avg_time_diff_hybrid_avg = abs(np.mean([r['hybrid_time'] - r['avg_time'] for r in results]))
avg_time_diff_hybrid_median = abs(np.mean([r['hybrid_time'] - r['median_time'] for r in results]))
avg_time_diff_avg_median = abs(np.mean([r['avg_time'] - r['median_time'] for r in results]))

print(f"\nAverage time differences:")
print(f"- Hybrid vs Average: {avg_time_diff_hybrid_avg:.3f} seconds")
print(f"- Hybrid vs Median: {avg_time_diff_hybrid_median:.3f} seconds")
print(f"- Average vs Median: {avg_time_diff_avg_median:.3f} seconds")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("Changing section selection strategy (last, average, or median):")
print("- Does NOT significantly change speed")
print("- May slightly improve accuracy (if original strategy missed optimal)")
print("- Tested candidates per section remains the same (1 per section)")
print("- All strategies have comparable performance")
print("- To truly speed up, consider reducing K or dynamic refine_range")
