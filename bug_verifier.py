import os
import json
import subprocess
from config import (
    DEFECTS4J, DEFECTS4J_ENV, BASE_BUGS_DIR,
    BUG_CANDIDATES_PATH, VERIFIED_BUGS_PATH, EXCLUSION_LOG_PATH,
)
from models import Bug
from prompt_parser import parse_java_file, extract_class_components


def checkout_buggy(project, bug_number, target_dir):
    """Checkout the buggy version."""
    if os.path.exists(target_dir):
        return True, None

    result = subprocess.run(
        [DEFECTS4J, 'checkout', '-p', project,
         '-v', f'{bug_number}b', '-w', target_dir],
        capture_output=True, text=True, env=DEFECTS4J_ENV
    )
    if result.returncode != 0:
        return False, result.stderr[:300]
    return True, None


def find_source_file(bug_dir, modified_class):
    """
    Locate the actual source file on disk for a given fully qualified class name.
    Tries common Defects4J layouts (src/main/java, src/java) rather than assuming one.
    """
    class_path = modified_class.replace('.', '/') + '.java'

    candidates = [
        os.path.join(bug_dir, 'src', 'main', 'java', class_path),
        os.path.join(bug_dir, 'src', 'java', class_path),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    # fallback -> search the whole tree
    target_filename = os.path.basename(class_path)
    for root, _, files in os.walk(bug_dir):
        if target_filename in files:
            return os.path.join(root, target_filename)

    return None


def verify_method_exists(source_file, target_method):
    """
    Confirm the target method is a genuine, externally-testable method
    declaration in the file. Requires the method to be 'public', since
    package-private or protected methods cannot be called from an
    external test class in a different package.
    """
    try:
        tree, source = parse_java_file(source_file)
    except Exception:
        return False, "failed to parse source file"

    components = extract_class_components(tree, source, target_method)
    target = components['target_method']

    if target is None:
        return False, f"method '{target_method}' not found"

    first_line = target.strip().splitlines()[0]
    if 'public' not in first_line:
        return False, f"method '{target_method}' found but not public"

    return True, None


def verify_bug(candidate):
    project = candidate['project']
    bug_number = candidate['bug_id']
    bug_key = candidate['bug_key']
    modified_class = candidate['modified_class']
    target_method = candidate.get('target_method')

    if not target_method:
        return None, "no target method identified by bug_selector"

    bug_dir = os.path.join(BASE_BUGS_DIR, bug_key)
    fixed_dir = os.path.join(BASE_BUGS_DIR, f"{bug_key}_fixed")

    ok, err = checkout_buggy(project, bug_number, bug_dir)
    if not ok:
        return None, f"checkout failed: {err}"

    source_file = find_source_file(bug_dir, modified_class)
    if not source_file:
        return None, f"source file not found for {modified_class}"

    method_ok, method_reason = verify_method_exists(source_file, target_method)
    if not method_ok:
        return None, method_reason

    bug = Bug(
        bug_id=bug_key,
        project=project,
        bug_number=bug_number,
        modified_class=modified_class,
        target_method=target_method,
        source_file=source_file,
        bug_dir=bug_dir,
        fixed_dir=fixed_dir
    )
    return bug, None


def verify_candidates(candidates, max_bugs=None):
    """
    Verify a list of candidates, returning (confirmed_bugs, excluded_log).
    excluded_log is a list of dicts documenting why each rejected bug failed
    """
    confirmed = []
    excluded = []

    for candidate in candidates:
        if max_bugs and len(confirmed) >= max_bugs:
            break

        bug_key = candidate['bug_key']
        print(f"Verifying {bug_key}...")
        bug, reason = verify_bug(candidate)

        if bug:
            confirmed.append(bug)
            print(f"  CONFIRMED: {bug.modified_class}.{bug.target_method}")
        else:
            excluded.append({'bug_key': bug_key, 'reason': reason})
            print(f"  EXCLUDED: {reason}")

    return confirmed, excluded


def save_confirmed_bugs(bugs, output_path):
    """
    Save confirmed bugs as a ready-to-import Python file for pipeline.py.
    Paths are stored relative to BASE_BUGS_DIR so the file stays portable
    across machines and can be committed.
    """
    def rel(path):
        return repr(os.path.relpath(path, BASE_BUGS_DIR))

    lines = [
        "import os",
        "from models import Bug",
        "from config import BASE_BUGS_DIR",
        "",
        "bugs = [",
    ]
    for b in bugs:
        lines.append(
            f"    Bug(\n"
            f"        bug_id={b.bug_id!r},\n"
            f"        project={b.project!r},\n"
            f"        bug_number={b.bug_number},\n"
            f"        modified_class={b.modified_class!r},\n"
            f"        target_method={b.target_method!r},\n"
            f"        source_file=os.path.join(BASE_BUGS_DIR, {rel(b.source_file)}),\n"
            f"        bug_dir=os.path.join(BASE_BUGS_DIR, {rel(b.bug_dir)}),\n"
            f"        fixed_dir=os.path.join(BASE_BUGS_DIR, {rel(b.fixed_dir)})\n"
            f"    ),"
        )
    lines.append("]")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"\nSaved {len(bugs)} confirmed bugs to {output_path}")
    
    
def save_exclusion_log(excluded, output_path):
    with open(output_path, 'w') as f:
        json.dump(excluded, f, indent=2)
    print(f"Saved {len(excluded)} exclusions to {output_path}")


if __name__ == '__main__':
    # Amount of bugs to run the experiment on.
    # For the thesis, 200 was used, but for testing out the pipeline less can be used
    MAX_BUGS = 10

    with open(BUG_CANDIDATES_PATH) as f:
        candidates = json.load(f)

    confirmed, excluded = verify_candidates(candidates, max_bugs=MAX_BUGS)

    save_confirmed_bugs(confirmed, VERIFIED_BUGS_PATH)
    save_exclusion_log(excluded, EXCLUSION_LOG_PATH)

    print(f"\n{len(confirmed)} confirmed, {len(excluded)} excluded out of {len(confirmed) + len(excluded)} attempted")
