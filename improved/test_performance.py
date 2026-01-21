"""
ALPS Performance Testing
Compare original vs optimized search methods

Tests 4 methods:
1. Original full_search_nk - Tests all n values (baseline)
2. Myopic search - Fast but may miss optimal
3. Section search - Systematic but may skip optimal
4. Hybrid smart search - Best balance (RECOMMENDED)

Author: Xiaoyu
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import sys
import os

# Add parent directory to path to import original Functions
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
import Functions as F_orig

# Import improved functions from current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from Functions_improved import myopic_search_nk, section_search_nk, hybrid_smart_search_nk

print("="*80)
print("ALPS PERFORMANCE TESTING")
print("="*80)

# Load all 4 datasets
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

# Storage for results
results = []

# Test each dataset
for ds_name, data in datasets.items():
    print(f"\n{'='*80}")
    print(f"TESTING: {ds_name} ({len(data)} points)")
    print('='*80)

    result = {
        'dataset': ds_name,
        'n_points': len(data)
    }

    # Method 1: Original full search
    print(f"\n[1/4] Original full_search_nk")
    start = time.time()
    r_orig = F_orig.full_search_nk(data, p=4, q=2)
    time_orig = time.time() - start
    result['orig_n'] = r_orig[0]
    result['orig_lam'] = r_orig[1]
    result['orig_time'] = time_orig
    print(f"  Result: n={r_orig[0]}, λ={r_orig[1]:.6f}")
    print(f"  Time: {time_orig:.3f}s")

    # Method 2: Myopic search
    print(f"\n[2/4] Myopic search")
    start = time.time()
    r_myopic = myopic_search_nk(data, p=4, q=2)
    time_myopic = time.time() - start
    result['myopic_n'] = r_myopic[0]
    result['myopic_lam'] = r_myopic[1]
    result['myopic_time'] = time_myopic
    result['myopic_match'] = (r_myopic[0] == r_orig[0])
    print(f"  Time: {time_myopic:.3f}s")
    print(f"  Speedup: {time_orig/time_myopic:.2f}×")
    print(f"  Match original: {result['myopic_match']}")

    # Method 3: Section search
    print(f"\n[3/4] Section search (K=10)")
    start = time.time()
    r_section = section_search_nk(data, p=4, q=2, K=10)
    time_section = time.time() - start
    result['section_n'] = r_section[0]
    result['section_lam'] = r_section[1]
    result['section_time'] = time_section
    result['section_match'] = (r_section[0] == r_orig[0])
    print(f"  Time: {time_section:.3f}s")
    print(f"  Speedup: {time_orig/time_section:.2f}×")
    print(f"  Match original: {result['section_match']}")

    # Method 4: Hybrid smart search
    print(f"\n[4/4] Hybrid smart search (K=10, range=5)")
    start = time.time()
    r_hybrid = hybrid_smart_search_nk(data, p=4, q=2, K=10, refine_range=5)
    time_hybrid = time.time() - start
    result['hybrid_n'] = r_hybrid[0]
    result['hybrid_lam'] = r_hybrid[1]
    result['hybrid_time'] = time_hybrid
    result['hybrid_match'] = (r_hybrid[0] == r_orig[0])
    print(f"  Time: {time_hybrid:.3f}s")
    print(f"  Speedup: {time_orig/time_hybrid:.2f}×")
    print(f"  Match original: {result['hybrid_match']}")

    results.append(result)

# Summary
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Dataset':<8} {'Pts':<5} {'Orig n':<8} {'Myopic n':<9} {'Section n':<10} {'Hybrid n':<9}")
print("-"*80)

for r in results:
    print(f"{r['dataset']:<8} {r['n_points']:<5} "
          f"{r['orig_n']:<8} {r['myopic_n']:<9} {r['section_n']:<10} {r['hybrid_n']:<9}")

print("\n" + "="*80)
print("ACCURACY CHECK")
print("="*80)
print(f"{'Dataset':<8} {'Myopic Match':<14} {'Section Match':<15} {'Hybrid Match':<12}")
print("-"*80)

all_myopic_match = all(r['myopic_match'] for r in results)
all_section_match = all(r['section_match'] for r in results)
all_hybrid_match = all(r['hybrid_match'] for r in results)

for r in results:
    print(f"{r['dataset']:<8} {str(r['myopic_match']):<14} "
          f"{str(r['section_match']):<15} {str(r['hybrid_match']):<12}")

print("-"*80)
print(f"{'ALL MATCH:':<8} {str(all_myopic_match):<14} {str(all_section_match):<15} {str(all_hybrid_match):<12}")

print("\n" + "="*80)
print("SPEED COMPARISON")
print("="*80)
print(f"{'Dataset':<8} {'Original':<12} {'Myopic':<12} {'Section':<12} {'Hybrid':<12}")
print("-"*80)

for r in results:
    print(f"{r['dataset']:<8} {r['orig_time']:<12.3f} {r['myopic_time']:<12.3f} "
          f"{r['section_time']:<12.3f} {r['hybrid_time']:<12.3f}")

print("\n" + "="*80)
print("SPEEDUP FACTORS (vs Original)")
print("="*80)
print(f"{'Dataset':<8} {'Myopic':<12} {'Section':<12} {'Hybrid':<12}")
print("-"*80)

for r in results:
    print(f"{r['dataset']:<8} {r['orig_time']/r['myopic_time']:<12.2f}× "
          f"{r['orig_time']/r['section_time']:<12.2f}× "
          f"{r['orig_time']/r['hybrid_time']:<12.2f}×")

# Calculate averages
avg_myopic_speedup = np.mean([r['orig_time']/r['myopic_time'] for r in results])
avg_section_speedup = np.mean([r['orig_time']/r['section_time'] for r in results])
avg_hybrid_speedup = np.mean([r['orig_time']/r['hybrid_time'] for r in results])

print("-"*80)
print(f"{'AVERAGE:':<8} {avg_myopic_speedup:<12.2f}× "
      f"{avg_section_speedup:<12.2f}× {avg_hybrid_speedup:<12.2f}×")

# Generate visualization
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(16, 12))

# Plot 1: Fitted curves comparison
for idx, (ds_name, data) in enumerate(datasets.items()):
    ax = plt.subplot(3, 4, idx + 1)
    r = results[idx]

    # Get original fit
    opt_n = r['orig_n']
    opt_lam = r['orig_lam']

    c = opt_n + 4
    U = F_orig.Kno_pspline_opt(data, 4, opt_n)
    B = F_orig.Basis_Pspline(opt_n, 4, U, data[:, 0])
    P = opt_lam * F_orig.Penalty_p(2, c)
    theta = np.linalg.solve(B.T.dot(B) + P, B.T.dot(data[:, 1].reshape(-1, 1)))
    y_pred = B.dot(theta).flatten()

    # Plot data and fit
    ax.scatter(data[:, 0], data[:, 1], alpha=0.6, s=30, label='Data')
    ax.plot(data[:, 0], y_pred, 'r-', linewidth=2, label=f'ALPS (n={opt_n})')

    # Info
    match_str = "✓" if (r['myopic_match'] and r['section_match'] and r['hybrid_match']) else "✗"
    ax.text(0.02, 0.98, f"{ds_name}\n{match_str} All methods match",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if match_str=="✓" else 'yellow', alpha=0.8),
            fontsize=9)

    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'{ds_name} ({r["n_points"]} pts)')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

# Plot 5-8: Time comparison bars
for idx, r in enumerate(results):
    ax = plt.subplot(3, 4, idx + 5)

    methods = ['Original', 'Myopic', 'Section', 'Hybrid']
    times = [r['orig_time'], r['myopic_time'], r['section_time'], r['hybrid_time']]
    colors = ['blue', 'red', 'orange', 'green']

    bars = ax.bar(methods, times, color=colors, alpha=0.7, edgecolor='black')

    # Add speedup labels
    for i, (bar, t) in enumerate(zip(bars, times)):
        if i == 0:
            label = f'{t:.3f}s\n(1.0×)'
        else:
            speedup = r['orig_time'] / t
            label = f'{t:.3f}s\n({speedup:.1f}×)'
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                label, ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'{r["dataset"]} - Speed Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15, ha='right')

# Plot 9: Accuracy comparison
ax = plt.subplot(3, 4, 9)
datasets_names = [r['dataset'] for r in results]
orig_n = [r['orig_n'] for r in results]
myopic_n = [r['myopic_n'] for r in results]
section_n = [r['section_n'] for r in results]
hybrid_n = [r['hybrid_n'] for r in results]

x = np.arange(len(datasets_names))
width = 0.2

ax.bar(x - 1.5*width, orig_n, width, label='Original', color='blue', alpha=0.7)
ax.bar(x - 0.5*width, myopic_n, width, label='Myopic', color='red', alpha=0.7)
ax.bar(x + 0.5*width, section_n, width, label='Section', color='orange', alpha=0.7)
ax.bar(x + 1.5*width, hybrid_n, width, label='Hybrid', color='green', alpha=0.7)

ax.set_xlabel('Dataset')
ax.set_ylabel('Optimal n')
ax.set_title('Optimal n Comparison (Should Match Original)')
ax.set_xticks(x)
ax.set_xticklabels(datasets_names)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 10: Speedup comparison
ax = plt.subplot(3, 4, 10)
myopic_speedups = [r['orig_time']/r['myopic_time'] for r in results]
section_speedups = [r['orig_time']/r['section_time'] for r in results]
hybrid_speedups = [r['orig_time']/r['hybrid_time'] for r in results]

x = np.arange(len(datasets_names))
width = 0.25

ax.bar(x - width, myopic_speedups, width, label='Myopic', color='red', alpha=0.7)
ax.bar(x, section_speedups, width, label='Section', color='orange', alpha=0.7)
ax.bar(x + width, hybrid_speedups, width, label='Hybrid', color='green', alpha=0.7)

ax.axhline(y=1, color='blue', linestyle='--', label='Original (1×)', linewidth=2)
ax.set_xlabel('Dataset')
ax.set_ylabel('Speedup Factor')
ax.set_title('Speedup vs Original')
ax.set_xticks(x)
ax.set_xticklabels(datasets_names)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 11: Average performance
ax = plt.subplot(3, 4, 11)
methods = ['Original', 'Myopic', 'Section', 'Hybrid']
avg_times = [
    np.mean([r['orig_time'] for r in results]),
    np.mean([r['myopic_time'] for r in results]),
    np.mean([r['section_time'] for r in results]),
    np.mean([r['hybrid_time'] for r in results])
]
colors = ['blue', 'red', 'orange', 'green']

bars = ax.bar(methods, avg_times, color=colors, alpha=0.7, edgecolor='black')

for i, (bar, t) in enumerate(zip(bars, avg_times)):
    if i == 0:
        label = f'{t:.3f}s'
    else:
        speedup = avg_times[0] / t
        label = f'{t:.3f}s\n({speedup:.1f}×)'
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            label, ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Average Time (seconds)')
ax.set_title('Average Performance Across All Datasets')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=15, ha='right')

# Plot 12: Accuracy summary
ax = plt.subplot(3, 4, 12)
methods_acc = ['Myopic', 'Section', 'Hybrid']
accuracy = [
    sum(r['myopic_match'] for r in results) / len(results) * 100,
    sum(r['section_match'] for r in results) / len(results) * 100,
    sum(r['hybrid_match'] for r in results) / len(results) * 100
]
colors_acc = ['red', 'orange', 'green']

bars = ax.bar(methods_acc, accuracy, color=colors_acc, alpha=0.7, edgecolor='black')

for bar, acc in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{acc:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.axhline(y=100, color='blue', linestyle='--', label='Target (100%)', linewidth=2)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy: % Matching Original Optimal n')
ax.set_ylim([0, 110])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: performance_comparison.png")

print("\n" + "="*80)
print("TESTING COMPLETE")
print("="*80)

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
print(f"1. Myopic search:  {avg_myopic_speedup:.1f}× faster, {sum(r['myopic_match'] for r in results)}/{len(results)} exact matches")
print(f"2. Section search: {avg_section_speedup:.1f}× faster, {sum(r['section_match'] for r in results)}/{len(results)} exact matches")
print(f"3. Hybrid search:  {avg_hybrid_speedup:.1f}× faster, {sum(r['hybrid_match'] for r in results)}/{len(results)} exact matches")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if all_hybrid_match:
    print("✓ HYBRID SMART SEARCH is recommended:")
    print(f"  - {avg_hybrid_speedup:.1f}× faster than original")
    print(f"  - 100% accuracy (all {len(results)}/{len(results)} datasets matched)")
    print(f"  - Best balance of speed and reliability")
else:
    print("⚠ Results show trade-offs:")
    hybrid_accuracy = sum(r['hybrid_match'] for r in results) / len(results) * 100
    print(f"  - Hybrid: {avg_hybrid_speedup:.1f}× faster, {hybrid_accuracy:.0f}% accurate")
    print(f"  - Myopic: {avg_myopic_speedup:.1f}× faster but less reliable")
    print(f"  - Use hybrid for best balance")

print("\n" + "="*80)
