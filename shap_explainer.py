import os
import json
import math
from itertools import combinations, permutations

import Levenshtein

from generator import call_llm, clean_generated_test
from prompt_parser import assemble_segment_prompt
from models import Bug
from config import STABILITY_TEMPERATURE, STABILITY_RUNS

NEUTRAL_PLACEHOLDER = "[MASKED]"
SEGMENTS = ['imports', 'fields', 'constructors', 'other_methods', 'target_method']


def normalized_edit_distance(a, b):
    """
    Normalized Levenshtein distance in [0, 1]; 0 = identical.
    """
    if not a and not b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    return Levenshtein.distance(a, b) / max_len


def output_similarity(baseline, other):
    """1 = identical output, 0 = completely different."""
    return 1.0 - normalized_edit_distance(baseline, other)


def get_players(segments):
    """
    Segments that actually carry content. Segments trimmed to 'none' by
    build_prompt are excluded from the game: masking them would only measure
    the textual difference between the strings 'none' and '[MASKED]', which
    is not useful. They are reported as None in results.
    """
    return [
        s for s in SEGMENTS
        if segments.get(s)
        and segments[s].strip()
        and segments[s].strip().lower() != 'none'
    ]


def coalition_prompt(segments, players, coalition, modified_class):
    """
    Build the prompt for a coalition S: players inside S keep their content,
    players outside S are masked. Non-player segments keep their (empty)
    content unchanged.
    """
    masked = {}
    for seg in SEGMENTS:
        if seg in players and seg not in coalition:
            masked[seg] = NEUTRAL_PLACEHOLDER
        else:
            masked[seg] = segments[seg]
    return assemble_segment_prompt(masked, modified_class)


def evaluate_coalitions(segments, players, baseline_clean, modified_class,
                        temperature=0.0, verbose=True):
    """
    Evaluate the characteristic function v(S) for every coalition S of players.

    v(S) = similarity(clean(LLM(prompt with only S unmasked)), baseline)

    Returns (v, outputs): dicts keyed by frozenset of segment names.
    Cost: 2^k - 1 LLM calls at temperature 0 (the full coalition reuses the
    saved baseline sample, since decoding is deterministic).

    Masked outputs are passed through clean_generated_test so they are
    compared on the same footing as the (already cleaned) baseline sample --
    otherwise markdown fences in raw output would inflate every distance.
    """
    n = len(players)
    full = frozenset(players)
    v, outputs = {}, {}

    total = 2 ** n
    done = 0
    for r in range(n + 1):
        for combo in combinations(players, r):
            S = frozenset(combo)
            done += 1

            if S == full and temperature == 0.0:
                v[S], outputs[S] = 1.0, baseline_clean
                continue

            prompt = coalition_prompt(segments, players, S, modified_class)
            out = call_llm(prompt, temperature=temperature)
            if out is None:
                out = call_llm(prompt, temperature=temperature)  # one retry
            if out is None:
                raise RuntimeError(
                    f"LLM call failed twice for coalition {sorted(S)} - "
                    f"aborting this bug's explanation"
                )

            cleaned, _ = clean_generated_test(out)
            v[S] = output_similarity(baseline_clean, cleaned)
            outputs[S] = cleaned

            if verbose:
                label = '+'.join(sorted(S)) if S else '(empty)'
                print(f"    coalition {done}/{total} [{label}]: v = {v[S]:.4f}")

    return v, outputs


def exact_shapley(v, players):
    """
    Exact Shapley values from the fully enumerated characteristic function:

        phi_i = sum over S subset of N\\{i} of
                |S|! (n-|S|-1)! / n!  *  (v(S u {i}) - v(S))

    By the efficiency axiom, sum(phi) = v(N) - v(empty) = 1 - v(empty).
    Positive phi = the segment moves the output toward the baseline
    (i.e. the model relies on it) - values can legitimately be negative.
    """
    n = len(players)
    phi = {}
    for i in players:
        others = [p for p in players if p != i]
        total = 0.0
        for r in range(len(others) + 1):
            for combo in combinations(others, r):
                S = frozenset(combo)
                weight = (math.factorial(len(S))
                          * math.factorial(n - len(S) - 1)
                          / math.factorial(n))
                total += weight * (v[S | {i}] - v[S])
        phi[i] = total
    return phi


def leave_one_out(v, players):
    """
    Occlusion (leave-one-out) attribution: v(N) - v(N \\ {i}).
    This is the single-masking measure the pipeline used previously; it is
    kept, for free, from the same coalition cache so Shapley vs. occlusion
    can be compared empirically.
    """
    full = frozenset(players)
    return {i: v[full] - v[full - {i}] for i in players}


def compute_faithfulness(v, phi, players):
    """
    Deletion-curve faithfulness (AOPC style), computed entirely from the
    coalition cache (no extra LLM calls).

    Segments are masked cumulatively in descending attribution order and the
    output change (1 - v) is recorded at each step. The area under this curve
    (aopc_attribution) is compared with the mean area over ALL k! orderings
    (aopc_random). A faithful attribution deletes influential content first,
    so faithfulness_gain = aopc_attribution - aopc_random should be > 0.
    Note: the deletion curve climbs almost no matter what you delete --
    masking more segments naturally pushes the output further from the
    baseline. That alone proves nothing. What matters is whether deleting
    in attribution order (highest first) pushes it away FASTER than a
    random deletion order would -- that comparison is the actual signal,
    which is why aopc_random (not the raw curve) is the meaningful number.
    """
    if len(players) < 2:
        return None

    def deletion_curve(order):
        remaining = set(players)
        changes = []
        for p in order:
            remaining.discard(p)
            changes.append(1.0 - v[frozenset(remaining)])
        return changes

    attr_order = sorted(players, key=lambda p: phi[p], reverse=True)
    attr_curve = deletion_curve(attr_order)
    aopc_attr = sum(attr_curve) / len(attr_curve)

    all_aopcs = []
    for perm in permutations(players):
        curve = deletion_curve(list(perm))
        all_aopcs.append(sum(curve) / len(curve))
    aopc_random = sum(all_aopcs) / len(all_aopcs)

    return {
        'deletion_order': attr_order,
        'deletion_curve': [round(c, 4) for c in attr_curve],
        'aopc_attribution': round(aopc_attr, 4),
        'aopc_random': round(aopc_random, 4),
        'faithfulness_gain': round(aopc_attr - aopc_random, 4),
    }


def compute_stability(segments, players, baseline_clean, modified_class):
    """
    Stability of Shapley values across repeated runs.
    Skipped if STABILITY_TEMPERATURE <= 0.0, since testing shows determinism at temp 0
    """
    if STABILITY_TEMPERATURE <= 0.0:
        return {
            'skipped': True,
            'temperature': 0.0,
            'reason': ('decoding is deterministic at temperature 0; '
                       'attributions are exactly reproducible by construction'),
        }

    runs = []
    for run in range(STABILITY_RUNS):
        print(f"  Stability run {run + 1}/{STABILITY_RUNS} "
              f"(T={STABILITY_TEMPERATURE})...")
        v_run, _ = evaluate_coalitions(
            segments, players, baseline_clean, modified_class,
            temperature=STABILITY_TEMPERATURE, verbose=False
        )
        runs.append(exact_shapley(v_run, players))

    per_segment = {}
    for p in players:
        values = [r[p] for r in runs]
        mean = sum(values) / len(values)
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        per_segment[p] = {
            'mean': round(mean, 4),
            'std': round(std, 4),
            'values': [round(x, 4) for x in values],
        }

    stds = [d['std'] for d in per_segment.values()]
    return {
        'skipped': False,
        'temperature': STABILITY_TEMPERATURE,
        'n_runs': STABILITY_RUNS,
        'per_segment': per_segment,
        'mean_std': round(sum(stds) / len(stds), 4),
    }


def explain_bug(bug: Bug, work_dir: str, sample_id: int,
                baseline_tier: str = 'unspecified'):
    """
    Run the full exact-Shapley explanation for a single bug.

    sample_id is selected by the pipeline from evaluation_results.json with
    tiered eligibility (effective > valid > compiles); baseline_tier records
    which tier applied so analyses can be stratified or filtered to the
    effective-only subset.
    """
    print(f"\nRunning exact Shapley explanation for {bug.bug_id} "
          f"(baseline: sample {sample_id}, tier: {baseline_tier})...")

    segments_path = os.path.join(work_dir, 'segments.json')
    if not os.path.exists(segments_path):
        print(f"ERROR: segments.json not found at {segments_path}")
        return None
    with open(segments_path) as f:
        segments = json.load(f)

    sample_path = os.path.join(
        work_dir, 'generated_tests', f'sample_{sample_id}.java'
    )
    if not os.path.exists(sample_path):
        print(f"ERROR: sample_{sample_id}.java not found")
        return None
    with open(sample_path) as f:
        baseline_clean = f.read()  # already cleaned by generator at write time

    players = get_players(segments)
    empty_segments = [s for s in SEGMENTS if s not in players]

    if len(players) < 2:
        reason = (f"only {len(players)} non-empty segment(s); "
                  f"attribution over a single player is degenerate")
        with open(os.path.join(work_dir, 'shap_skipped.json'), 'w') as f:
            json.dump({'bug_id': bug.bug_id, 'reason': reason}, f, indent=2)
        print(f"  SKIPPED: {reason}")
        return None

    n_calls = 2 ** len(players) - 1
    print(f"  Players: {players}"
          + (f" (empty, excluded: {empty_segments})" if empty_segments else ""))
    print(f"  Evaluating {2 ** len(players)} coalitions "
          f"({n_calls} LLM calls at T=0)...")

    try:
        v, coalition_outputs = evaluate_coalitions(
            segments, players, baseline_clean, bug.modified_class
        )
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return None

    phi = exact_shapley(v, players)
    loo = leave_one_out(v, players)
    faithfulness = compute_faithfulness(v, phi, players)
    stability = compute_stability(
        segments, players, baseline_clean, bug.modified_class
    )

    v_empty = v[frozenset()]
    efficiency_target = 1.0 - v_empty
    efficiency_residual = abs(sum(phi.values()) - efficiency_target)

    attributions = {
        s: (round(phi[s], 4) if s in players else None) for s in SEGMENTS
    }
    loo_attributions = {
        s: (round(loo[s], 4) if s in players else None) for s in SEGMENTS
    }

    results = {
        'bug_id': bug.bug_id,
        'modified_class': bug.modified_class,
        'target_method': bug.target_method,
        'sample_id': sample_id,
        'baseline_tier': baseline_tier,
        'baseline_length': len(baseline_clean),
        'method': 'exact_shapley',
        'players': players,
        'empty_segments': empty_segments,
        'attributions': attributions,
        'leave_one_out': loo_attributions,
        'v_empty': round(v_empty, 4),
        'efficiency_check': {
            'sum_phi': round(sum(phi.values()), 4),
            'target_1_minus_v_empty': round(efficiency_target, 4),
            'residual': round(efficiency_residual, 6),
        },
        'faithfulness': faithfulness,
        'stability': stability,
        'coalition_values': {
            ('+'.join(sorted(S)) if S else 'EMPTY'): round(val, 4)
            for S, val in v.items()
        },
    }

    with open(os.path.join(work_dir, 'shap_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # coalition outputs kept in a separate file coalition_outputs.json for every bug
    with open(os.path.join(work_dir, 'coalition_outputs.json'), 'w') as f:
        json.dump(
            {('+'.join(sorted(S)) if S else 'EMPTY'): out
             for S, out in coalition_outputs.items()},
            f, indent=2
        )

    print(f"\n  Results saved to {os.path.join(work_dir, 'shap_results.json')}")

    # summary
    print(f"\n  Exact Shapley values (sum = {sum(phi.values()):.4f}, "
          f"efficiency residual = {efficiency_residual:.6f}):")
    for seg in SEGMENTS:
        score = attributions[seg]
        if score is None:
            print(f"    {seg:20s} (empty segment, excluded)")
        else:
            bar = '█' * int(max(score, 0.0) * 40)
            print(f"    {seg:20s} {score:+.4f} {bar}")

    if faithfulness:
        print(f"\n  Faithfulness: AOPC(attr) = {faithfulness['aopc_attribution']} "
              f"vs AOPC(random) = {faithfulness['aopc_random']} "
              f"(gain = {faithfulness['faithfulness_gain']:+.4f})")
    if stability.get('skipped'):
        print(f"  Stability:    skipped ({stability['reason']})")
    else:
        print(f"  Stability:    mean std = {stability['mean_std']} "
              f"at T={stability['temperature']}")

    return results
