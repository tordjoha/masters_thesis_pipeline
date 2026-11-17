import subprocess
import os
import json
from config import DEFECTS4J, DEFECTS4J_ENV, JUNIT_JAR
from models import EvaluationResult, GeneratedTest
from typing import List


def get_bug_classpath(bug_dir):
    """
    Find compiled class directories for a checked out bug.
    Tries both (target/) and (build/).
    Always appends the JUnit jar.
    """
    candidates = [
        os.path.join(bug_dir, 'target', 'classes'),
        os.path.join(bug_dir, 'target', 'tests'),
        os.path.join(bug_dir, 'target', 'test-classes'),
        os.path.join(bug_dir, 'build', 'classes'),
        os.path.join(bug_dir, 'build', 'tests'),
        os.path.join(bug_dir, 'build', 'test-classes'),
    ]
    existing = [c for c in candidates if os.path.exists(c)]
    return existing + [JUNIT_JAR]


def wrap_test_in_class(test_code, class_name, modified_class):
    """
    Wrap a raw test method in a compilable Java class.
    Uses a fixed set of common JDK imports applied identically
    to every bug, to avoid per-bug tailoring of the evaluation harness.
    """
    package_import = f"import {modified_class};"
    static_import = f"import static {modified_class}.*;"

    wrapped = (
        f"import org.junit.*;\n"
        f"import org.junit.Test;\n"
        f"import static org.junit.Assert.*;\n"
        f"import static org.junit.Assume.*;\n"
        f"{package_import}\n"
        f"{static_import}\n"
        f"import java.math.BigDecimal;\n"
        f"import java.math.BigInteger;\n"
        f"import java.util.*;\n"
        f"\n"
        f"public class {class_name} {{\n"
        f"\n"
        f"{test_code}\n"
        f"\n"
        f"}}"
    )
    return wrapped


def compile_test(test_code, class_name, bug_dir, work_dir, modified_class):
    """
    Wrap, write, and compile a generated test against the given bug directory.
    """
    wrapped = wrap_test_in_class(test_code, class_name, modified_class)
    temp_dir = os.path.join(work_dir, f"compile_{class_name}")
    os.makedirs(temp_dir, exist_ok=True)
    java_file = os.path.join(temp_dir, f"{class_name}.java")
    with open(java_file, 'w', encoding='utf-8') as f:
        f.write(wrapped)
    classpath = ':'.join(get_bug_classpath(bug_dir))
    result = subprocess.run(
        ['javac', '-cp', classpath, '-d', temp_dir, java_file],
        capture_output=True,
        text=True,
        env=DEFECTS4J_ENV
    )

    success = result.returncode == 0
    compile_error = result.stderr if not success else ""
    return success, temp_dir, compile_error


def run_test(class_name, temp_dir, bug_dir):
    """
    Run a compiled test class against the given bug directory.
    Returns pass, fail, or timeout.
    """
    classpath = ':'.join([temp_dir] + get_bug_classpath(bug_dir))

    try:
        result = subprocess.run(
            ['java', '-cp', classpath,
             'org.junit.runner.JUnitCore', class_name],
            capture_output=True,
            text=True,
            timeout=30,
            env=DEFECTS4J_ENV
        )
    except subprocess.TimeoutExpired:
        return 'timeout', 'TIMEOUT'

    output = result.stdout + result.stderr
    outcome = 'pass' if result.returncode == 0 else 'fail'
    return outcome, output


def is_compiled(bug_dir):
    """
    True if compiled class dirs exist.
    """
    return len(get_bug_classpath(bug_dir)) > 1


def ensure_version_ready(project, version_id, target_dir):
    """
    Guarantee a Defects4J version is both checked out AND compiled.
    version_id should be in the format '5f' or '5b'.

    Idempotent: skips whichever steps are already done. This matters because
    bug_verifier checks out buggy versions WITHOUT compiling them; running
    tests against an uncompiled checkout yields an empty classpath, so every
    test crashes with NoClassDefFoundError and would previously be counted
    as 'failing on buggy' -- i.e. spuriously effective.
    """
    if not os.path.exists(target_dir):
        print(f"Checking out {project}-{version_id}...")
        subprocess.run([
            DEFECTS4J, 'checkout',
            '-p', project,
            '-v', version_id,
            '-w', target_dir
        ], check=True, env=DEFECTS4J_ENV)

    if not is_compiled(target_dir):
        print(f"Compiling {project}-{version_id}...")
        subprocess.run(
            [DEFECTS4J, 'compile', '-w', target_dir],
            check=True, env=DEFECTS4J_ENV
        )
        print(f"Done: {target_dir}\n")


def evaluate_samples(
    generated: List[GeneratedTest],
    bug_id: str,
    bug_dir_buggy: str,
    bug_dir_fixed: str,
    work_dir: str,
    modified_class: str
) -> List[EvaluationResult]:
    """
    For each generated sample:
    1. Compile against fixed version
    2. Run against fixed version - should pass (valid test)
    3. Run against buggy version - should fail (detects bug)

    A test is effective if it compiles, passes fixed, and fails buggy.
    Non-compiling samples are excluded from the explanation stage, the
    tier of the sample that is explained is chosen in pick_baseline_sample.
    """
    results = []

    for sample in generated:
        sample_id = sample.sample_id
        class_name = f"LLMTest_{bug_id}_{sample_id}"
        print(f"\n  Evaluating sample {sample_id}...")

        result = EvaluationResult(sample_id=sample_id)

        # Compile against fixed version
        success, temp_dir, compile_error = compile_test(
            sample.content, class_name, bug_dir_fixed, work_dir, modified_class
        )

        if not success:
            print(f"    COMPILE FAILED")
            result.compile_error = compile_error[:300]
            results.append(result)
            continue

        result.compiles = True
        print(f"    Compiled OK")

        # Run against fixed version (must pass)
        fixed_outcome, fixed_output = run_test(class_name, temp_dir, bug_dir_fixed)
        result.fixed_outcome = fixed_outcome
        result.passes_fixed = fixed_outcome == 'pass'
        result.fixed_output = fixed_output[:500]
        print(f"    Fixed version: {fixed_outcome.upper()}")

        if fixed_outcome == 'fail':
            for line in fixed_output.splitlines():
                if 'AssertionError' in line or 'Exception' in line:
                    print(f"    --> {line.strip()}")
                    break

        # Run against buggy version
        # Only real failure (a timeout is not counted as a failure)
        buggy_outcome, buggy_output = run_test(class_name, temp_dir, bug_dir_buggy)
        result.buggy_outcome = buggy_outcome
        result.fails_buggy = buggy_outcome == 'fail'
        result.buggy_output = buggy_output[:500]
        print(f"    Buggy version: "
              f"{'FAIL (bug detected)' if result.fails_buggy else buggy_outcome.upper() + ' (bug not detected)'}")

        result.effective = result.passes_fixed and result.fails_buggy
        print(f"    Effective: {'YES' if result.effective else 'NO'}")

        results.append(result)

    # save results to disk
    results_path = os.path.join(work_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump([r.__dict__ for r in results], f, indent=2)

    n_effective = sum(1 for r in results if r.effective)
    print(f"\n  Summary: {n_effective}/{len(results)} samples effective")

    return results
