import json
import os
import numpy as np
import shap
import matplotlib.pyplot as plt
from config import BASE_DATA_DIR

SEGMENT_LABELS = ['imports', 'fields', 'constructors', 'other_methods', 'target_method']
DISPLAY_LABELS = ['Imports', 'Fields', 'Constructors', 'Other Methods', 'Target Method']


def discover_bug_ids():
    """Find every bug directory under BASE_DATA_DIR that has shap_results.json,
    so plots automatically scale with the experiment."""
    ids = []
    if not os.path.isdir(BASE_DATA_DIR):
        return ids
    for name in sorted(os.listdir(BASE_DATA_DIR)):
        if os.path.exists(os.path.join(BASE_DATA_DIR, name, 'shap_results.json')):
            ids.append(name)
    return ids


def load_shap_results(bug_ids):
    """Load SHAP results from disk for a list of bug IDs."""
    results = []
    for bug_id in bug_ids:
        path = os.path.join(BASE_DATA_DIR, bug_id, 'shap_results.json')
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
            print(f"Loaded: {bug_id}")
        else:
            print(f"WARNING: No SHAP results found for {bug_id}")
    return results


def plot_aggregated_shap(results, output_path=None):
    """
    Aggregated bar plot across all bugs, using the mean exact Shapley value
    per segment. Segments excluded for a given bug (empty/'none' content,
    stored as None) are simply left out of that bug's contribution.
    """
    all_scores = {seg: [] for seg in SEGMENT_LABELS}
    for result in results:
        attributions = result.get('attributions', {})
        for seg in SEGMENT_LABELS:
            score = attributions.get(seg)
            if score is not None:
                all_scores[seg].append(score)

    mean_scores = {
        seg: np.mean(scores) if scores else 0.0
        for seg, scores in all_scores.items()
    }
    counts = {seg: len(scores) for seg, scores in all_scores.items()}

    feature_names = [
        f"{label} (n={counts[seg]})"
        for seg, label in zip(SEGMENT_LABELS, DISPLAY_LABELS)
    ]
    values = np.array([mean_scores[seg] for seg in SEGMENT_LABELS])

    explanation = shap.Explanation(values=values, feature_names=feature_names)

    plt.figure(figsize=(10, 5))
    shap.plots.bar(explanation, show=False, max_display=len(SEGMENT_LABELS))

    plt.title(
        f"Mean exact Shapley value per prompt segment across {len(results)} bug(s)",
        fontsize=13, pad=12
    )
    plt.xlabel("Mean Shapley value (similarity-scale characteristic function)",
               fontsize=11)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_per_bug_shap(results, output_dir=None):
    """Individual Shapley bar plots per bug."""
    for result in results:
        bug_id = result['bug_id']
        attributions = result.get('attributions', {})

        present = [(seg, label) for seg, label in zip(SEGMENT_LABELS, DISPLAY_LABELS)
                   if attributions.get(seg) is not None]
        if not present:
            print(f"Skipping {bug_id}: no attributions")
            continue

        values = np.array([attributions[seg] for seg, _ in present])
        feature_names = [label for _, label in present]

        explanation = shap.Explanation(values=values, feature_names=feature_names)

        plt.figure(figsize=(10, 5))
        shap.plots.bar(explanation, show=False, max_display=len(present))

        plt.title(
            f"Exact Shapley attribution — {bug_id} "
            f"({result['modified_class'].split('.')[-1]}.{result['target_method']})",
            fontsize=12, pad=12
        )
        plt.xlabel("Shapley value", fontsize=11)
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"{bug_id}_shap.png")
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        else:
            plt.show()
        plt.close()


def plot_heatmap(results, output_path=None):
    """
    Heatmap of Shapley values per segment per bug. Segments excluded for a
    bug are shown as grey (NaN) rather than a misleading 0.0.
    """
    bug_labels = []
    matrix = []

    for result in results:
        bug_id = result['bug_id']
        class_name = result['modified_class'].split('.')[-1]
        method = result['target_method']
        bug_labels.append(f"{bug_id}\n({class_name}.{method})")

        attributions = result.get('attributions', {})
        row = [
            attributions.get(seg) if attributions.get(seg) is not None else np.nan
            for seg in SEGMENT_LABELS
        ]
        matrix.append(row)

    matrix = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(10, len(results) * 1.2 + 1.5))

    cmap = plt.get_cmap('RdYlGn_r').copy()
    cmap.set_bad(color='lightgrey')

    vmax = max(np.nanmax(matrix), 1e-6)
    vmin = min(np.nanmin(matrix), 0.0)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(DISPLAY_LABELS)))
    ax.set_xticklabels(DISPLAY_LABELS, fontsize=11)
    ax.set_yticks(range(len(bug_labels)))
    ax.set_yticklabels(bug_labels, fontsize=10)

    span = vmax - vmin if vmax > vmin else 1.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                ax.text(j, i, '—', ha='center', va='center',
                        fontsize=10, color='dimgrey')
                continue
            rel = (value - vmin) / span
            text_color = 'white' if rel > 0.75 or rel < 0.2 else 'black'
            ax.text(j, i, f"{value:.2f}", ha='center', va='center',
                    fontsize=10, color=text_color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Exact Shapley value', fontsize=10)

    ax.set_title(
        'Shapley attribution heatmap per prompt segment\n'
        'Grey = segment empty for this bug (excluded from the game)',
        fontsize=12, pad=12
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_faithfulness_summary(results, output_path=None):
    """
    Per-bug faithfulness: AOPC under attribution-ordered deletion vs the mean
    over all orderings. Bars above zero indicate the attribution ordering
    deletes influential content earlier than chance -> faithful.
    """
    bugs, gains = [], []
    for result in results:
        f = result.get('faithfulness')
        if f and f.get('faithfulness_gain') is not None:
            bugs.append(result['bug_id'])
            gains.append(f['faithfulness_gain'])

    if not bugs:
        print("No faithfulness data to plot.")
        return

    plt.figure(figsize=(max(8, len(bugs) * 0.5), 5))
    colors = ['#2b8a3e' if g > 0 else '#c92a2a' for g in gains]
    plt.bar(range(len(bugs)), gains, color=colors)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(range(len(bugs)), bugs, rotation=60, ha='right', fontsize=9)
    plt.ylabel('AOPC(attribution order) − AOPC(mean over all orders)')
    plt.title(f'Deletion-curve faithfulness gain per bug (n={len(bugs)})',
              fontsize=12, pad=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Faithfulness plot saved to {output_path}")
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    bug_ids = discover_bug_ids()
    print(f"Discovered {len(bug_ids)} bug(s) with SHAP results")
    results = load_shap_results(bug_ids)

    if not results:
        print("No results to visualise.")
    else:
        n = len(results)
        output_dir = os.path.join(BASE_DATA_DIR, 'plots')
        os.makedirs(output_dir, exist_ok=True)

        plot_aggregated_shap(
            results,
            output_path=os.path.join(output_dir, f'aggregated_shap_{n}bugs.png')
        )
        plot_per_bug_shap(results, output_dir=output_dir)
        plot_heatmap(
            results,
            output_path=os.path.join(output_dir, f'heatmap_{n}bugs.png')
        )
        plot_faithfulness_summary(
            results,
            output_path=os.path.join(output_dir, f'faithfulness_{n}bugs.png')
        )

        print(f"\nAll plots saved to {output_dir}")
