"""
Description-augmented replication of the main pipeline.

Research question: when the issue-tracker bug description is supplied as a
sixth prompt segment, (a) does the generated test change and become more
targeted at the fault, and (b) does the description outrank target_method
in Shapley attribution?

Design constraints that make this a single-variable experiment:
  * The five original segments are pinned from each bug's existing
    segments.json. They are never re-parsed, so no tree-sitter, encoding or
    token-fallback difference can be mistaken for an effect of the
    description.
  * The five-segment part of the prompt, instruction included, is produced
    by prompt_parser.assemble_segment_prompt itself; the description block
    is only prepended. The instruction does not mention the description,
    which would change a second variable.
  * When the six-segment prompt exceeds the token cap, only the description
    is truncated. build_prompt's own fallback drops 'fields', which would
    also change a second variable.

The main pipeline is not modified. The Shapley math is imported from
shap_explainer, so both experiments are computed by one implementation;
only the parts that hardcode five segment names are reimplemented here.

Usage:
    python desc_pipeline.py --pilot --dry-run   # prompts + token counts only
    python desc_pipeline.py --pilot             # 3 characterised bugs
    python desc_pipeline.py --all               # full run
"""

import os
import re
import sys
import json
import argparse
from itertools import combinations

from config import BASE_DATA_DIR, MAX_PROMPT_TOKENS, DESCRIPTIONS_JSON
from models import Bug
from prompt_parser import assemble_segment_prompt
from generator import call_llm, clean_generated_test, count_tokens, generate_tests
from evaluator import evaluate_samples, ensure_version_ready
from shap_explainer import (
    NEUTRAL_PLACEHOLDER,
    output_similarity,
    exact_shapley,
    leave_one_out,
    compute_faithfulness,
)

try:
    from verified_bugs import bugs
except ModuleNotFoundError as e:
    if e.name != 'verified_bugs':
        raise
    bugs = None

# Read-only source of pinned segments and the original baselines.
SOURCE_DATA_DIR = BASE_DATA_DIR

# All output of this experiment goes to a sibling tree, so visualise.py
# pointed at BASE_DATA_DIR keeps producing the original five-segment figures.
DESC_DATA_DIR = os.path.join(
    os.path.dirname(os.path.normpath(BASE_DATA_DIR)), 'data_desc'
)

# Lang_50's original baseline was an empty class shell with no @Test method,
# producing degenerate uniform attributions. Excluded here for the same reason.
EXCLUDE = {'Lang_50'}

# Bugs already characterised in the five-segment run, used as the pilot:
#   Lang_57  the one effective test (LANG-304, NPE precondition)
#   Lang_47  happy-path bias (passes 123 where the ticket is about null)
#   Codec_9  fabricated expected values, normal-looking attribution profile
PILOT_BUGS = ['Lang_57', 'Lang_47', 'Codec_9']

# The title carries the densest statement of the fault and is the only
# content for tickets with an empty body; leaving it out would make those
# bugs' description segment structurally different from the rest.
INCLUDE_TITLE = True

# Replace code blocks in the description. Some tickets contain a reproducing
# test or the patch itself, so leaving them in measures copying rather than
# comprehension. Kept False; the choice is recorded in every results file.
STRIP_CODE_BLOCKS = False

DESC_SEGMENTS = [
    'bug_description',
    'imports',
    'fields',
    'constructors',
    'other_methods',
    'target_method',
]


def assemble_desc_prompt(segments, modified_class):
    """
    Six-segment prompt: the [BUG_DESCRIPTION] block prepended to the exact
    five-segment prompt from prompt_parser.assemble_segment_prompt, so tags,
    order and instruction cannot drift from the main pipeline.
    """
    return (
        f"[BUG_DESCRIPTION]\n{segments['bug_description']}\n[/BUG_DESCRIPTION]\n\n"
        + assemble_segment_prompt(segments, modified_class)
    )


def format_description(record):
    """
    Render the description segment. The structure is identical for every
    bug, including 'none' for an empty body, so a sparse ticket differs from
    a rich one in content only, not in shape.
    """
    title = (record.get('title') or '').strip()
    body = (record.get('description') or record.get('body')
            or record.get('raw_body') or '').strip()

    # descriptions.json stores title + body joined in 'description';
    # avoid printing the title twice.
    if title and body.startswith(title):
        body = body[len(title):].strip()

    if STRIP_CODE_BLOCKS:
        body = re.sub(
            r"\{code[^}]*\}.*?\{code\}|\{noformat\}.*?\{noformat\}|```.*?```",
            "[code omitted]", body, flags=re.DOTALL,
        )

    if INCLUDE_TITLE:
        return f"Title: {title or 'none'}\nDetails: {body or 'none'}"
    return body or 'none'


def fit_description(segments, desc_text, modified_class):
    """
    Make the six-segment prompt fit MAX_PROMPT_TOKENS by truncating only the
    description body (the Title: line is kept), binary-searching the longest
    prefix that fits.

    Returns (desc_text, n_tokens, truncated, original_chars).
    """
    original_chars = len(desc_text)
    n = count_tokens(assemble_desc_prompt(
        {**segments, 'bug_description': desc_text}, modified_class
    ))
    if n <= MAX_PROMPT_TOKENS:
        return desc_text, n, False, original_chars

    marker = "\n[description truncated]"
    head, sep, tail = desc_text.partition('\nDetails: ')
    prefix = head + sep if sep else ''
    body = tail if sep else desc_text

    lo, hi, best = 0, len(body), None
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = prefix + body[:mid].rstrip() + marker
        n = count_tokens(assemble_desc_prompt(
            {**segments, 'bug_description': candidate}, modified_class
        ))
        if n <= MAX_PROMPT_TOKENS:
            best, lo = (candidate, n), mid + 1
        else:
            hi = mid - 1

    if best is None:
        raise RuntimeError(
            "even an empty description exceeds the token cap; the pinned "
            "five segments alone no longer fit"
        )
    return best[0], best[1], True, original_chars


def get_players(segments):
    """
    Six-segment version of shap_explainer.get_players. Segments trimmed to
    'none' are excluded: masking them would only measure the textual
    difference between 'none' and '[MASKED]'.
    """
    return [
        s for s in DESC_SEGMENTS
        if segments.get(s)
        and segments[s].strip()
        and segments[s].strip().lower() != 'none'
    ]


def coalition_prompt(segments, players, coalition, modified_class):
    """Prompt for coalition S: players outside S are masked."""
    masked = {}
    for seg in DESC_SEGMENTS:
        if seg in players and seg not in coalition:
            masked[seg] = NEUTRAL_PLACEHOLDER
        else:
            masked[seg] = segments[seg]
    return assemble_desc_prompt(masked, modified_class)


def evaluate_coalitions(segments, players, baseline_clean, modified_class,
                        verbose=True):
    """
    v(S) = similarity(clean(LLM(prompt with only S unmasked)), baseline)

    The baseline is the six-segment full-prompt output, not the original
    five-segment one: v(N) must be 1.0 by construction for the efficiency
    axiom to hold on this game.

    Returns (v, outputs): dicts keyed by frozenset of segment names.
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

            if S == full:
                v[S], outputs[S] = 1.0, baseline_clean
                continue

            prompt = coalition_prompt(segments, players, S, modified_class)
            out = call_llm(prompt, temperature=0.0)
            if out is None:
                out = call_llm(prompt, temperature=0.0)  # one retry
            if out is None:
                raise RuntimeError(
                    f"LLM call failed twice for coalition {sorted(S)}"
                )

            cleaned, _ = clean_generated_test(out)
            v[S] = output_similarity(baseline_clean, cleaned)
            outputs[S] = cleaned

            if verbose:
                label = '+'.join(sorted(S)) if S else '(empty)'
                print(f"    coalition {done}/{total} [{label}]: v = {v[S]:.4f}")

    return v, outputs


def load_descriptions():
    """Fetched descriptions keyed by bug_id; records with errors are dropped."""
    with open(DESCRIPTIONS_JSON) as f:
        data = json.load(f)
    records = data.values() if isinstance(data, dict) else data
    return {r['bug_id'].strip(): r for r in records if 'error' not in r}


def load_pinned_segments(bug_id):
    path = os.path.join(SOURCE_DATA_DIR, bug_id, 'segments.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_original_result(bug_id):
    """
    Original five-segment run for comparison.
    Returns (shap_results, baseline test source, evaluation of that sample).
    """
    base = os.path.join(SOURCE_DATA_DIR, bug_id)
    shap_path = os.path.join(base, 'shap_results.json')
    if not os.path.exists(shap_path):
        return None, None, None
    with open(shap_path) as f:
        shap = json.load(f)

    test = None
    sample_path = os.path.join(
        base, 'generated_tests', f"sample_{shap['sample_id']}.java"
    )
    if os.path.exists(sample_path):
        with open(sample_path) as f:
            test = f.read()

    evals = None
    eval_path = os.path.join(base, 'evaluation_results.json')
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            all_evals = json.load(f)
        evals = next((e for e in all_evals
                      if e['sample_id'] == shap['sample_id']), None)

    return shap, test, evals


def select_bugs(which):
    """
    Bugs that have both a completed five-segment run (segments.json and
    shap_results.json) and a fetched description. Discovered from disk, so
    the set is defined by what actually ran.
    """
    descriptions = load_descriptions()
    by_id = {b.bug_id: b for b in bugs}

    if which == 'pilot':
        candidates = PILOT_BUGS
    else:
        candidates = sorted(
            d for d in os.listdir(SOURCE_DATA_DIR)
            if os.path.isdir(os.path.join(SOURCE_DATA_DIR, d))
        )

    selected, missing_desc, missing_seg = [], [], []
    for bug_id in candidates:
        if bug_id in EXCLUDE or bug_id not in by_id:
            continue
        if not os.path.exists(os.path.join(SOURCE_DATA_DIR, bug_id,
                                           'shap_results.json')):
            continue
        if load_pinned_segments(bug_id) is None:
            missing_seg.append(bug_id)
            continue
        if bug_id not in descriptions:
            missing_desc.append(bug_id)
            continue
        selected.append((by_id[bug_id], descriptions[bug_id]))

    if missing_seg:
        print(f"WARNING: no segments.json for {missing_seg}")
    if missing_desc:
        print(f"WARNING: no description for {missing_desc}")
    return selected


def compare_to_original(bug_id, baseline_clean, new_eval, phi, players,
                        shares, ranks):
    """
    Compare against the five-segment run. Shapley values sum to
    v(N) - v(empty), and v(empty) differs between the five- and six-player
    games, so raw phi is not comparable across experiments; normalised
    shares and ranks are.
    """
    old_shap, old_test, old_eval = load_original_result(bug_id)
    if not old_shap:
        return None

    old_players = old_shap['players']
    old_phi = {p: old_shap['attributions'][p] for p in old_players}
    old_total = sum(abs(x) for x in old_phi.values()) or 1.0

    return {
        'test_identical': old_test == baseline_clean if old_test else None,
        'test_similarity': (round(output_similarity(old_test, baseline_clean), 4)
                            if old_test else None),
        'old_effective': old_eval.get('effective') if old_eval else None,
        'new_effective': new_eval.get('effective') if new_eval else None,
        'old_compiles': old_eval.get('compiles') if old_eval else None,
        'new_compiles': new_eval.get('compiles') if new_eval else None,
        'old_shares': {p: round(abs(old_phi[p]) / old_total, 4)
                       for p in old_players},
        'new_shares': shares,
        'old_rank': {p: i + 1 for i, p in enumerate(
            sorted(old_players, key=lambda q: old_phi[q], reverse=True))},
        'new_rank': ranks,
        'old_top_segment': max(old_players, key=lambda p: old_phi[p]),
        'new_top_segment': max(players, key=lambda p: phi[p]),
    }


def skip_bug(bug_id, work_dir, reason):
    with open(os.path.join(work_dir, 'shap_skipped.json'), 'w') as f:
        json.dump({'bug_id': bug_id, 'reason': reason}, f, indent=2)
    print(f"  SKIPPED: {reason}")
    return None


def run_bug(bug: Bug, desc_record, dry_run=False):
    """
    Run the description-augmented experiment for one bug: build the
    six-segment prompt, generate and evaluate one test at T=0, then compute
    exact Shapley values over all 2^k coalitions.

    Bugs with shap_results.json or shap_skipped.json in their work dir are
    skipped, so an interrupted run can be resumed.
    """
    work_dir = os.path.join(DESC_DATA_DIR, bug.bug_id)

    if not dry_run:
        os.makedirs(work_dir, exist_ok=True)
        if (os.path.exists(os.path.join(work_dir, 'shap_results.json'))
                or os.path.exists(os.path.join(work_dir, 'shap_skipped.json'))):
            print(f"\nSKIPPING {bug.bug_id} - already processed")
            return None

    print(f"\n{'='*60}")
    print(f"Running: {bug.bug_id} | {bug.modified_class} | {bug.target_method}")
    print(f"{'='*60}")

    pinned = load_pinned_segments(bug.bug_id)
    desc_text, n_tokens, truncated, original_chars = fit_description(
        pinned, format_description(desc_record), bug.modified_class
    )
    segments = {**pinned, 'bug_description': desc_text}
    prompt = assemble_desc_prompt(segments, bug.modified_class)
    players = get_players(segments)

    print(f"  Description: {len(desc_text)} chars"
          + (f" (truncated from {original_chars})" if truncated else ""))
    print(f"  Prompt tokens: {n_tokens}/{MAX_PROMPT_TOKENS}")
    print(f"  Players ({len(players)}): {players}")
    print(f"  Coalitions: {2 ** len(players)} "
          f"({2 ** len(players) - 1} LLM calls at T=0)")

    if dry_run:
        print(f"\n--- prompt ---\n{prompt}\n--- end ---")
        return {'bug_id': bug.bug_id, 'tokens': n_tokens,
                'truncated': truncated, 'players': players}

    if len(players) < 2:
        return skip_bug(bug.bug_id, work_dir,
                        f"only {len(players)} non-empty segment(s)")

    with open(os.path.join(work_dir, 'prompt.txt'), 'w') as f:
        f.write(prompt)
    with open(os.path.join(work_dir, 'segments.json'), 'w') as f:
        json.dump(segments, f, indent=2)

    # same checkout/compile requirement as the main pipeline
    ensure_version_ready(bug.project, f"{bug.bug_number}b", bug.bug_dir)
    ensure_version_ready(bug.project, f"{bug.bug_number}f", bug.fixed_dir)

    print(f"\nGenerating description-augmented test for {bug.bug_id}...")
    generated = generate_tests(prompt, bug.bug_id,
                               os.path.join(work_dir, 'generated_tests'),
                               n_samples=1, temperature=0.0)
    if not generated:
        return skip_bug(bug.bug_id, work_dir, 'generation failed')

    baseline_clean = generated[0].content

    # Lang_50 lesson: a compilable class shell with no @Test method yields
    # uniform, meaningless attributions. Catch it before spending 2^k calls.
    if '@Test' not in baseline_clean:
        return skip_bug(bug.bug_id, work_dir,
                        'generated baseline contains no @Test method')

    print(f"\nEvaluating test for {bug.bug_id}...")
    eval_results = evaluate_samples(generated, bug.bug_id, bug.bug_dir,
                                    bug.fixed_dir, work_dir,
                                    bug.modified_class)
    new_eval = eval_results[0].__dict__ if eval_results else None

    print(f"\nRunning exact Shapley explanation for {bug.bug_id}...")
    try:
        v, coalition_outputs = evaluate_coalitions(
            segments, players, baseline_clean, bug.modified_class
        )
    except RuntimeError as e:
        return skip_bug(bug.bug_id, work_dir, str(e))

    phi = exact_shapley(v, players)
    loo = leave_one_out(v, players)
    faithfulness = compute_faithfulness(v, phi, players)

    v_empty = v[frozenset()]
    efficiency_target = 1.0 - v_empty
    efficiency_residual = abs(sum(phi.values()) - efficiency_target)

    total_abs = sum(abs(x) for x in phi.values()) or 1.0
    shares = {p: round(abs(phi[p]) / total_abs, 4) for p in players}
    ranks = {p: i + 1 for i, p in
             enumerate(sorted(players, key=lambda q: phi[q], reverse=True))}

    comparison = compare_to_original(bug.bug_id, baseline_clean, new_eval,
                                     phi, players, shares, ranks)

    results = {
        'bug_id': bug.bug_id,
        'experiment': 'description_augmented',
        'modified_class': bug.modified_class,
        'target_method': bug.target_method,
        'sample_id': 1,
        'baseline_tier': 'description_augmented_single_sample',
        'baseline_length': len(baseline_clean),
        'method': 'exact_shapley',
        'players': players,
        'empty_segments': [s for s in DESC_SEGMENTS if s not in players],
        'attributions': {s: (round(phi[s], 4) if s in players else None)
                         for s in DESC_SEGMENTS},
        'leave_one_out': {s: (round(loo[s], 4) if s in players else None)
                          for s in DESC_SEGMENTS},
        'normalized_shares': shares,
        'ranks': ranks,
        'v_empty': round(v_empty, 4),
        'efficiency_check': {
            'sum_phi': round(sum(phi.values()), 4),
            'target_1_minus_v_empty': round(efficiency_target, 4),
            'residual': round(efficiency_residual, 6),
        },
        'faithfulness': faithfulness,
        'stability': {'skipped': True, 'temperature': 0.0,
                      'reason': 'deterministic decoding at temperature 0'},
        'description_meta': {
            'report_id': desc_record.get('report_id'),
            'report_url': desc_record.get('report_url'),
            'tracker': desc_record.get('tracker'),
            'chars_used': len(desc_text),
            'chars_original': original_chars,
            'truncated': truncated,
            'body_empty': desc_record.get('body_empty'),
            'contains_repro_code': desc_record.get('contains_repro_code'),
            'include_title': INCLUDE_TITLE,
            'code_stripped': STRIP_CODE_BLOCKS,
            'prompt_tokens': n_tokens,
        },
        'evaluation': new_eval,
        'comparison_to_original': comparison,
        'coalition_values': {
            ('+'.join(sorted(S)) if S else 'EMPTY'): round(val, 4)
            for S, val in v.items()
        },
    }

    with open(os.path.join(work_dir, 'shap_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(work_dir, 'coalition_outputs.json'), 'w') as f:
        json.dump(
            {('+'.join(sorted(S)) if S else 'EMPTY'): out
             for S, out in coalition_outputs.items()},
            f, indent=2
        )

    print(f"\n  Results saved to {os.path.join(work_dir, 'shap_results.json')}")

    print(f"\n  Exact Shapley values (sum = {sum(phi.values()):.4f}, "
          f"efficiency residual = {efficiency_residual:.6f}):")
    for seg in DESC_SEGMENTS:
        score = results['attributions'][seg]
        if score is None:
            print(f"    {seg:20s} (empty segment, excluded)")
        else:
            bar = '█' * int(max(score, 0.0) * 40)
            print(f"    {seg:20s} {score:+.4f}  share={shares[seg]:.3f}  "
                  f"rank={ranks[seg]}  {bar}")

    if comparison:
        print(f"\n  vs original: top segment "
              f"{comparison['old_top_segment']} -> {comparison['new_top_segment']}"
              f" | effective {comparison['old_effective']} -> "
              f"{comparison['new_effective']}"
              f" | test similarity {comparison['test_similarity']}")

    return results


def print_summary(done, dry_run):
    print("\n" + "=" * 60)
    if dry_run:
        truncated = [r['bug_id'] for r in done if r['truncated']]
        print(f"Dry run:            {len(done)} bug(s)")
        print(f"Truncated:          {truncated or 'none'}")
        print(f"Total LLM calls:    "
              f"{sum(2 ** len(r['players']) - 1 for r in done)}")
    else:
        effective = [r['bug_id'] for r in done
                     if (r.get('evaluation') or {}).get('effective')]
        top_desc = [r['bug_id'] for r in done
                    if (r.get('comparison_to_original') or {})
                    .get('new_top_segment') == 'bug_description']
        print(f"Completed:              {len(done)} bug(s)")
        print(f"Effective tests:        {len(effective)} {effective}")
        print(f"Description ranked top: {len(top_desc)} {top_desc}")
        print(f"Results in {DESC_DATA_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pilot', action='store_true',
                       help=f'run only the pilot bugs {PILOT_BUGS}')
    group.add_argument('--all', action='store_true',
                       help='run every eligible bug')
    parser.add_argument('--dry-run', action='store_true',
                        help='print prompts and token counts only; '
                             'no generation or coalition calls')
    args = parser.parse_args()

    if bugs is None:
        print("FATAL: verified_bugs.py not found.")
        print("Run bug_selector.py and then bug_verifier.py first to generate it.")
        sys.exit(1)

    try:
        count_tokens("warmup")
    except RuntimeError as e:
        print(f"FATAL: {e}")
        print("Start llama-server before running the experiment.")
        sys.exit(1)

    selected = select_bugs('pilot' if args.pilot else 'all')
    print(f"\n{len(selected)} bug(s) selected"
          + (" [DRY RUN]" if args.dry_run else ""))

    if not args.dry_run:
        os.makedirs(DESC_DATA_DIR, exist_ok=True)

    done = []
    for bug, record in selected:
        result = run_bug(bug, record, dry_run=args.dry_run)
        if result is not None:
            done.append(result)

    print_summary(done, args.dry_run)
