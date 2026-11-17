import os
import json
import sys
import subprocess
from shap_explainer import explain_bug
from config import BASE_BUGS_DIR, BASE_DATA_DIR, JAVA_HOME, DEFECTS4J_ENV
from models import Bug, BugResult
from prompt_parser import build_prompt
from generator import generate_tests, count_tokens
from evaluator import evaluate_samples, ensure_version_ready

try:
    from verified_bugs import bugs
except ModuleNotFoundError as e:
    if e.name!= 'verified_bugs':
        raise
    bugs = None

# toggle SHAP explanation stage on/off
RUN_SHAP = True

# --- resume / skip helpers ---
def shap_stage_done(work_dir):
    """The SHAP stage is done if it either produced results or was
    deliberately skipped with a recorded reason."""
    return (
        os.path.exists(os.path.join(work_dir, 'shap_results.json'))
        or os.path.exists(os.path.join(work_dir, 'shap_skipped.json'))
    )


def pick_baseline_sample(work_dir):
    """
    Choose the explanation baseline with tiered eligibility, preferring the
    best available sample:
        effective -> compiles, passes fixed, fails buggy
        valid -> compiles and passes fixed (a correct test that does
                 not detect this particular fault)
        compiles -> syntactically/semantically valid Java only

    Effectiveness is recorded as the baseline tier so analyses can be
    split (attribution patterns of effective vs. ineffective
    tests) or filtered to the effective-only subset.

    Returns (sample_id, tier) or (None, None).
    """
    path = os.path.join(work_dir, 'evaluation_results.json')
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        eval_results = json.load(f)

    tiers = [
        ('effective', lambda r: r.get('effective')),
        ('valid', lambda r: r.get('compiles') and r.get('passes_fixed')),
        ('compiles', lambda r: r.get('compiles')),
    ]
    for tier, predicate in tiers:
        for r in eval_results:
            if predicate(r):
                return r['sample_id'], tier
    return None, None


def run_shap_stage(bug, work_dir):
    sample_id, tier = pick_baseline_sample(work_dir)
    if sample_id is None:
        reason = 'no compiling test available as explanation baseline'
        with open(os.path.join(work_dir, 'shap_skipped.json'), 'w') as f:
            json.dump({'bug_id': bug.bug_id, 'reason': reason}, f, indent=2)
        print(f"SHAP skipped for {bug.bug_id}: {reason}")
        return None
    print(f"Explanation baseline for {bug.bug_id}: sample {sample_id} "
          f"(tier: {tier})")
    return explain_bug(bug, work_dir, sample_id=sample_id, baseline_tier=tier)


def run_bug(bug: Bug):
    """
    Run the pipeline for one bug. Returns a BugResult for freshly processed
    bugs, or None when the bug was skipped / only resumed for SHAP.

    Resume behaviour:
      - exclusion.json present            -> skip entirely
      - pipeline_complete.json present    -> generation/evaluation are done;
        if RUN_SHAP and the SHAP stage isn't done yet, run ONLY the SHAP
        stage from the saved artifacts (this covers bugs processed while
        RUN_SHAP was False), otherwise skip.
    """
    work_dir = os.path.join(BASE_DATA_DIR, bug.bug_id)
    os.makedirs(work_dir, exist_ok=True)

    if os.path.exists(os.path.join(work_dir, 'exclusion.json')):
        print(f"\nSKIPPING {bug.bug_id} - previously excluded")
        return None

    if os.path.exists(os.path.join(work_dir, 'pipeline_complete.json')):
        if RUN_SHAP and not shap_stage_done(work_dir):
            print(f"\nRESUMING {bug.bug_id} - generation/evaluation done, "
                  f"running SHAP stage only")
            run_shap_stage(bug, work_dir)
        else:
            print(f"\nSKIPPING {bug.bug_id} - already fully processed")
        return None

    print(f"\n{'='*60}")
    print(f"Running: {bug.bug_id} | {bug.modified_class} | {bug.target_method}")
    print(f"{'='*60}")

    # both versions must be checked out AND compiled: tests are compiled
    # against the fixed version and executed against both. An uncompiled
    # buggy checkout has an empty classpath, which turns every run into a
    # NoClassDefFoundError 'failure' and fabricates effective tests.
    ensure_version_ready(bug.project, f"{bug.bug_number}b", bug.bug_dir)
    ensure_version_ready(bug.project, f"{bug.bug_number}f", bug.fixed_dir)

    prompt, components, segments, exclusion_reason = build_prompt(
        bug.source_file,
        bug.target_method,
        bug.modified_class
    )

    result = BugResult(
        bug=bug,
        prompt=prompt,
        segments=segments,
        exclusion_reason=exclusion_reason
    )

    if exclusion_reason:
        print(f"EXCLUDED {bug.bug_id}: {exclusion_reason}")
        with open(os.path.join(work_dir, 'exclusion.json'), 'w') as f:
            json.dump({"bug_id": bug.bug_id, "reason": exclusion_reason}, f)
        return result

    print(f"Prompt tokens: {count_tokens(prompt)}")

    with open(os.path.join(work_dir, 'prompt.txt'), 'w') as f:
        f.write(prompt)
    with open(os.path.join(work_dir, 'segments.json'), 'w') as f:
        json.dump(segments, f, indent=2)

    print(f"\nGenerating tests for {bug.bug_id}...")
    generated = generate_tests(
        prompt,
        bug.bug_id,
        os.path.join(work_dir, 'generated_tests')
    )
    result.generated_tests = generated
    print(f"Generated {len(generated)} samples")

    print(f"\nEvaluating tests for {bug.bug_id}...")
    eval_results = evaluate_samples(
        generated,
        bug.bug_id,
        bug.bug_dir,
        bug.fixed_dir,
        work_dir,
        bug.modified_class
    )
    result.evaluation_results = eval_results

    # SHAP explanation
    if RUN_SHAP:
        result.shap_results = run_shap_stage(bug, work_dir)
    else:
        print(f"\nSkipping SHAP explanation (RUN_SHAP=False)")

    # completion marker so reruns skip this bug if a crash occurs later
    with open(os.path.join(work_dir, 'pipeline_complete.json'), 'w') as f:
        json.dump({
            "bug_id": bug.bug_id,
            "n_generated": len(generated),
            "n_effective": sum(1 for r in eval_results if r.effective),
            "shap_run": RUN_SHAP,
        }, f, indent=2)

    return result


def check_java():
    """Fail early if Java 11 isn't available for Defects4J."""
    if not JAVA_HOME:
        print("FATAL: Java 11 not found. Install it, or set "
              "THESIS_JAVA_HOME=/path/to/jdk11.")
        sys.exit(1)

    r = subprocess.run(['java', '-version'], capture_output=True,
                       text=True, env=DEFECTS4J_ENV)
    version_line = r.stderr.splitlines()[0] if r.stderr else 'unknown'
    print(f"Using Java: {version_line}")   # java -version prints to stderr
    if '"11.' not in r.stderr:
        print("FATAL: Expected Java 11. Set THESIS_JAVA_HOME to a JDK 11.")
        sys.exit(1)


def print_summary(results):
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    for r in results:
        print(r.summary())
    print("=" * 60)
    total_effective = sum(r.n_effective for r in results)
    total_tests = sum(r.n_total for r in results)
    total_excluded = sum(1 for r in results if r.exclusion_reason)
    print(f"Total bugs run:     {len(results)}")
    print(f"Total excluded:     {total_excluded}")
    print(f"Effective tests:    {total_effective}/{total_tests}")
    print("=" * 60)


if __name__ == '__main__':

    if bugs is None:
        print("FATAL: verified_bugs.py not found.")
        print("Run bug_selector.py and then bug_verifier.py first to generate it.")
        sys.exit(1)

    check_java()

    try:
        count_tokens("warmup")
    except RuntimeError as e:
        print(f"FATAL: {e}")
        print("Start llama-server before running the pipeline.")
        sys.exit(1)

    print(f"Loaded {len(bugs)} verified bugs | RUN_SHAP = {RUN_SHAP}")

    all_results = []
    for bug in bugs:
        result = run_bug(bug)
        if result is not None:
            all_results.append(result)

    if all_results:
        print_summary(all_results)
    else:
        print("\nNo bugs were freshly processed this run "
              "(all skipped or resumed for SHAP only).")
