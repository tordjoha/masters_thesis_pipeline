import os
import json
import argparse
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt

from config import BASE_DATA_DIR
from shap_explainer import compute_faithfulness

# LIME hyperparameters, reported in the thesis (exact Shapley has none).
KERNEL_WIDTH = 0.75   # sigma in exp(-D^2 / sigma^2); D = fraction of masked players
RIDGE_ALPHA = 0.01    # small L2 penalty for numerical stability


def discover_bug_ids(data_dir):
    """Find every bug directory under data_dir that has shap_results.json."""
    ids = []
    if not os.path.isdir(data_dir):
        return ids
    for name in sorted(os.listdir(data_dir)):
        if os.path.exists(os.path.join(data_dir, name, 'shap_results.json')):
            ids.append(name)
    return ids


def load_coalitions(shap_results):
    """
    Rebuild (players, v) from the coalition_values cached in
    shap_results.json. Keys are '+'-joined sorted segment names, with
    'EMPTY' for the empty coalition.
    """
    players = shap_results['players']
    v = {}
    for key, value in shap_results['coalition_values'].items():
        S = frozenset() if key == 'EMPTY' else frozenset(key.split('+'))
        v[S] = value
    return players, v


def lime_attributions(players, v, kernel_width=KERNEL_WIDTH, alpha=RIDGE_ALPHA):
    """
    LIME attribution as a weighted ridge regression over the full coalition
    enumeration:

        v(S) ~ intercept + sum_i w_i * z_i(S)

    where z_i(S) = 1 if player i is unmasked in S. Sample weights follow
    LIME's locality kernel around the full (all-unmasked) prompt:

        pi(S) = exp(-D(S)^2 / kernel_width^2),  D(S) = fraction masked

    All 2^k coalitions are already cached, so this is LIME's best case: no
    sampling noise and zero additional LLM calls. LIME and exact Shapley
    therefore share the identical value function; only the aggregation
    rule differs.

    Returns (coefficients per player, intercept).
    """
    k = len(players)
    rows, targets, weights = [], [], []
    for r in range(k + 1):
        for combo in combinations(players, r):
            S = frozenset(combo)
            if S not in v:
                continue
            distance = 1.0 - len(S) / k
            rows.append([1.0 if p in S else 0.0 for p in players])
            targets.append(v[S])
            weights.append(np.exp(-(distance ** 2) / (kernel_width ** 2)))

    Z = np.column_stack([np.ones(len(rows)), np.array(rows)])  # intercept column
    y = np.array(targets)
    W = np.diag(weights)

    reg = alpha * np.eye(Z.shape[1])
    reg[0, 0] = 0.0  # don't penalise the intercept
    beta = np.linalg.solve(Z.T @ W @ Z + reg, Z.T @ W @ y)

    coeffs = {p: float(beta[i + 1]) for i, p in enumerate(players)}
    return coeffs, float(beta[0])


def spearman_rho(a, b):
    """Spearman rank correlation without a scipy dependency."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def analyse_bug(bug_id, data_dir):
    """
    Fit LIME on one bug's cached coalitions and compare it with the stored
    exact Shapley values. Faithfulness uses shap_explainer.compute_faithfulness,
    so both methods are scored by the same implementation.
    """
    path = os.path.join(data_dir, bug_id, 'shap_results.json')
    with open(path) as f:
        shap_results = json.load(f)

    players, v = load_coalitions(shap_results)
    if len(players) < 2:
        return None

    lime, intercept = lime_attributions(players, v)
    shapley = {p: shap_results['attributions'][p] for p in players}

    rho = spearman_rho([shapley[p] for p in players],
                       [lime[p] for p in players])
    top_agree = (max(players, key=lambda p: shapley[p])
                 == max(players, key=lambda p: lime[p]))

    result = {
        'bug_id': bug_id,
        'players': players,
        'baseline_tier': shap_results.get('baseline_tier'),
        'lime': {p: round(lime[p], 4) for p in players},
        'lime_intercept': round(intercept, 4),
        'shapley': shapley,
        'lime_faithfulness': compute_faithfulness(v, lime, players),
        'shapley_faithfulness_gain':
            (shap_results.get('faithfulness') or {}).get('faithfulness_gain'),
        'spearman_rho_lime_vs_shapley': None if rho is None else round(rho, 4),
        'top_segment_agreement': top_agree,
        'kernel_width': KERNEL_WIDTH,
        'ridge_alpha': RIDGE_ALPHA,
    }

    with open(os.path.join(data_dir, bug_id, 'lime_results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return result


def plot_faithfulness_comparison(results, output_path=None):
    """
    Per-bug deletion-curve faithfulness gain for exact Shapley vs LIME,
    both computed from the identical coalition data.
    """
    bugs = [r['bug_id'] for r in results]
    shap_gain = [r['shapley_faithfulness_gain'] or 0.0 for r in results]
    lime_gain = [r['lime_faithfulness']['faithfulness_gain'] for r in results]

    x = np.arange(len(bugs))
    plt.figure(figsize=(max(9, len(bugs) * 0.6), 5))
    plt.bar(x - 0.2, shap_gain, width=0.4, label='Exact Shapley', color='#1f6f43')
    plt.bar(x + 0.2, lime_gain, width=0.4, label='LIME', color='#7c4dbe')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(x, bugs, rotation=60, ha='right', fontsize=9)
    plt.ylabel('Deletion-curve faithfulness gain')
    plt.title('Faithfulness: exact Shapley vs LIME on identical coalition data',
              fontsize=12, pad=12)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_attribution_scatter(results, output_path=None):
    """Per-segment LIME coefficient vs exact Shapley value (one point per
    segment per bug); points on the diagonal mean the two methods agree."""
    xs = [r['shapley'][p] for r in results for p in r['players']]
    ys = [r['lime'][p] for r in results for p in r['players']]

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, s=22, alpha=0.6, color='#33658a')
    lims = plt.gca().get_xlim() + plt.gca().get_ylim()
    lo, hi = min(lims), max(lims)
    plt.plot([lo, hi], [lo, hi], '--', color='grey', linewidth=1)
    plt.xlabel('Exact Shapley value')
    plt.ylabel('LIME coefficient')
    plt.title('Per-segment attribution: LIME vs exact Shapley\n'
              '(each point = one segment of one bug)', fontsize=12, pad=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def print_summary(results):
    rhos = [r['spearman_rho_lime_vs_shapley'] for r in results
            if r['spearman_rho_lime_vs_shapley'] is not None]
    agree = sum(1 for r in results if r['top_segment_agreement'])
    lime_gains = [r['lime_faithfulness']['faithfulness_gain'] for r in results]
    shap_gains = [r['shapley_faithfulness_gain'] for r in results
                  if r['shapley_faithfulness_gain'] is not None]

    print(f"\nSummary over {len(results)} bug(s):")
    if rhos:
        print(f"  mean Spearman rho (LIME vs Shapley): {np.mean(rhos):.3f}")
    print(f"  top-segment agreement: {agree}/{len(results)}")
    print(f"  mean faithfulness gain: LIME = {np.mean(lime_gains):.3f}, "
          f"Shapley = {np.mean(shap_gains):.3f}" if shap_gains else
          f"  mean faithfulness gain: LIME = {np.mean(lime_gains):.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', action='store_true',
                        help='analyse the description-augmented experiment '
                             '(data_desc, sibling of BASE_DATA_DIR)')
    args = parser.parse_args()

    if args.desc:
        data_dir = os.path.join(
            os.path.dirname(os.path.normpath(BASE_DATA_DIR)), 'data_desc'
        )
    else:
        data_dir = BASE_DATA_DIR

    bug_ids = discover_bug_ids(data_dir)
    print(f"Discovered {len(bug_ids)} bug(s) with cached coalition data")

    results = []
    for bug_id in bug_ids:
        r = analyse_bug(bug_id, data_dir)
        if r:
            results.append(r)
            print(f"{bug_id}: rho = {r['spearman_rho_lime_vs_shapley']}, "
                  f"top-segment agree = {r['top_segment_agreement']}, "
                  f"faithfulness gain LIME = "
                  f"{r['lime_faithfulness']['faithfulness_gain']} "
                  f"vs Shapley = {r['shapley_faithfulness_gain']}")

    if not results:
        print("No results to compare.")
    else:
        n = len(results)
        output_dir = os.path.join(data_dir, 'plots')
        os.makedirs(output_dir, exist_ok=True)

        print_summary(results)
        plot_faithfulness_comparison(
            results,
            output_path=os.path.join(output_dir,
                                     f'lime_vs_shap_faithfulness_{n}bugs.png')
        )
        plot_attribution_scatter(
            results,
            output_path=os.path.join(output_dir,
                                     f'lime_vs_shap_scatter_{n}bugs.png')
        )

        summary_path = os.path.join(data_dir, 'lime_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nPer-bug results in each bug dir (lime_results.json); "
              f"summary at {summary_path}")