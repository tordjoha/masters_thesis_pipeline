import subprocess
import os
import json
import csv
import re
from collections import Counter

import tree_sitter_java
from tree_sitter import Language, Parser

from config import (
    DEFECTS4J, DEFECTS4J_ENV, BASE_BUGS_DIR, DEFECTS4J_HOME,
    DEFECTS4J_PROJECT_REPOS, BUG_CANDIDATES_PATH,
)

_TS_LANGUAGE = Language(tree_sitter_java.language())
_TS_PARSER = Parser(_TS_LANGUAGE)

REPO_NAME_MAP = {
    'Lang': 'commons-lang.git',
    'Math': 'commons-math.git',
    'Chart': 'jfreechart',
    'Time': 'joda-time.git',
    'Closure': 'closure-compiler.git',
    'Cli': 'commons-cli.git',
    'Codec': 'commons-codec.git',
    'Csv': 'commons-csv.git',
    'Jsoup': 'jsoup.git',
    'Gson': 'gson.git',
    'JacksonCore': 'jackson-core.git'
}

TEST_METHOD_PATTERN = re.compile(r'::\s*test(\w+)')

# set on the first 'defects4j info' failure so get_modified_classes warns
# once instead of once per bug (hundreds of identical lines otherwise)
_INFO_FAILURE_WARNED = False


def _repo_name(project):
    return REPO_NAME_MAP.get(project, project.lower())


def _repo_path(project):
    return os.path.join(DEFECTS4J_PROJECT_REPOS, _repo_name(project))


def get_active_bugs(project):
    """Read active bugs from Defects4J commit database."""
    csv_path = os.path.join(
        DEFECTS4J_HOME, 'framework', 'projects',
        project, 'active-bugs.csv'
    )
    if not os.path.exists(csv_path):
        print(f"WARNING: no active-bugs.csv for {project}")
        return []

    with open(csv_path) as f:
        return [
            {
                'project': project,
                'bug_id': int(row['bug.id']),
                'revision_buggy': row['revision.id.buggy'],
                'revision_fixed': row['revision.id.fixed'],
            }
            for row in csv.DictReader(f)
        ]


def get_modified_classes(project, bug_id):
    """Get list of modified source classes for a bug using defects4j info."""
    result = subprocess.run(
        [DEFECTS4J, 'info', '-p', project, '-b', str(bug_id)],
        capture_output=True, text=True, env=DEFECTS4J_ENV
    )
    if result.returncode != 0:
        global _INFO_FAILURE_WARNED
        if not _INFO_FAILURE_WARNED:
            print(f"WARNING: 'defects4j info' failed for {project}-{bug_id} "
                  f"(exit {result.returncode}); further failures of this kind "
                  f"will be silent. stderr:\n{result.stderr.strip()[:500]}")
            _INFO_FAILURE_WARNED = True
        return []

    classes = []
    in_modified = False
    for line in result.stdout.splitlines():
        if 'List of modified sources' in line:
            in_modified = True
            continue
        if in_modified:
            stripped = line.strip()
            if stripped.startswith('-'):
                class_name = stripped.lstrip('-').strip()
                if class_name:
                    classes.append(class_name)
            elif stripped:
                in_modified = False
    return classes


def get_diff_size(project, bug_id):
    """
    Count changed lines in source files only (test files excluded), via
    git diff on the bare project repository -- no checkout required.
    Returns None if the repo or bug revision info isn't available.
    """
    repo = _repo_path(project)
    if not os.path.exists(repo):
        return None

    bugs = get_active_bugs(project)
    bug_info = next((b for b in bugs if b['bug_id'] == bug_id), None)
    if not bug_info:
        return None

    result = subprocess.run(
        ['git', 'diff',
         bug_info['revision_buggy'],
         bug_info['revision_fixed'],
         '--', '*.java'],
        capture_output=True, text=True, errors='replace',
        cwd=repo, env=DEFECTS4J_ENV
    )
    if result.returncode != 0:
        return None

    lines_changed = 0
    in_test_file = False
    for line in result.stdout.splitlines():
        if line.startswith('diff --git'):
            in_test_file = 'test' in line.lower() or 'Test' in line
        if in_test_file:
            continue
        if line.startswith('+') and not line.startswith('+++'):
            lines_changed += 1
        elif line.startswith('-') and not line.startswith('---'):
            lines_changed += 1

    return lines_changed


HUNK_HEADER = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@')


def _list_files_at_revision(repo, revision):
    result = subprocess.run(
        ['git', 'ls-tree', '-r', '--name-only', revision],
        capture_output=True, text=True, errors='replace',
        cwd=repo, env=DEFECTS4J_ENV
    )
    if result.returncode != 0:
        return []
    return result.stdout.splitlines()


def _find_class_path_at_revision(repo, revision, class_path):
    """Locate the actual repo path of the class file at a given revision
    (handles src/java vs src/main/java layout differences)."""
    for path in _list_files_at_revision(repo, revision):
        if path == class_path or path.endswith('/' + class_path):
            return path
    return None


def _git_show(repo, revision, path):
    # errors='replace': old Apache Commons revisions contain Latin-1 bytes
    # (e.g. 0xA9 '(c)' in license headers) that are not valid UTF-8. Replacement
    # only affects comment text; line numbers and the AST are unaffected.
    result = subprocess.run(
        ['git', 'show', f'{revision}:{path}'],
        capture_output=True, text=True, errors='replace',
        cwd=repo, env=DEFECTS4J_ENV
    )
    return result.stdout if result.returncode == 0 else None


def _changed_new_lines(diff_text):
    """
    Parse unified diff output and return the line numbers (in the NEW file)
    touched by the change: added lines directly, and for pure deletions the
    hunk's new-side start line as the location anchor.
    """
    lines_changed = []
    new_line = None
    hunk_start = None
    hunk_had_addition = False

    for line in diff_text.splitlines():
        m = HUNK_HEADER.match(line)
        if m:
            if hunk_start is not None and not hunk_had_addition:
                lines_changed.append(hunk_start)
            new_line = int(m.group(1))
            hunk_start = new_line
            hunk_had_addition = False
            continue
        if new_line is None:
            continue
        if line.startswith('+') and not line.startswith('+++'):
            lines_changed.append(new_line)
            hunk_had_addition = True
            new_line += 1
        elif line.startswith('-') and not line.startswith('---'):
            pass  # old-file line: does not advance new-file counter
        elif not line.startswith('\\'):
            new_line += 1

    if hunk_start is not None and not hunk_had_addition:
        lines_changed.append(hunk_start)
    return lines_changed


def _parse_source(source):
    """Parse source and return (tree, source_bytes). Tree-sitter node offsets
    are BYTE offsets, so all text extraction must slice the bytes buffer and
    decode per slice -- slicing the str is only correct for pure-ASCII files."""
    data = source.encode('utf-8')
    return _TS_PARSER.parse(data), data


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def _method_name(node, source_bytes):
    ident = next((c for c in node.children if c.type == 'identifier'), None)
    if not ident:
        return None
    return source_bytes[ident.start_byte:ident.end_byte].decode('utf-8', errors='replace')


def _method_enclosing_line(tree, source_bytes, line):
    """Name of the innermost method_declaration containing a 1-based line."""
    best, best_span = None, None
    for node in _walk(tree.root_node):
        if node.type != 'method_declaration':
            continue
        start, end = node.start_point[0] + 1, node.end_point[0] + 1
        if start <= line <= end:
            span = end - start
            if best_span is None or span < best_span:
                name = _method_name(node, source_bytes)
                if name:
                    best, best_span = name, span
    return best


def _declared_methods(tree, source_bytes):
    return {
        _method_name(node, source_bytes)
        for node in _walk(tree.root_node)
        if node.type == 'method_declaration' and _method_name(node, source_bytes)
    }


def get_target_method(project, bug_id, modified_class):
    """
    Infer the buggy method: map the diff's changed line numbers onto the
    fixed-revision AST and majority-vote over the enclosing methods.
    Falls back to the triggering test name ONLY if the resulting name is
    validated against the methods actually declared in the class.
    Returns None when nothing can be validated (changes in fields,
    constructors, static initialisers, or spread over no single method).
    """
    repo = _repo_path(project)
    if not os.path.exists(repo):
        return None

    bugs = get_active_bugs(project)
    bug_info = next((b for b in bugs if b['bug_id'] == bug_id), None)
    if not bug_info:
        return None

    class_path = modified_class.replace('.', '/') + '.java'
    rev_fixed = bug_info['revision_fixed']

    repo_path = _find_class_path_at_revision(repo, rev_fixed, class_path)
    if not repo_path:
        return None
    source = _git_show(repo, rev_fixed, repo_path)
    if not source:
        return None

    tree, source_bytes = _parse_source(source)
    declared = _declared_methods(tree, source_bytes)

    result = subprocess.run(
        ['git', 'diff', bug_info['revision_buggy'], rev_fixed,
         '--', repo_path],
        capture_output=True, text=True, errors='replace',
        cwd=repo, env=DEFECTS4J_ENV
    )

    if result.returncode == 0 and result.stdout:
        names = []
        for line_no in _changed_new_lines(result.stdout):
            name = _method_enclosing_line(tree, source_bytes, line_no)
            if name:
                names.append(name)
        if names:
            return Counter(names).most_common(1)[0][0]

    return _get_target_method_from_test_name(project, bug_id, declared)


def _get_target_method_from_test_name(project, bug_id, declared_methods):
    """
    Fallback: infer target method from the triggering test name, accepted
    ONLY if the candidate is a method actually declared in the class
    (e.g. testToBoolean -> toBoolean). Guessed-but-nonexistent names were
    the dominant cause of verifier exclusions previously.
    """
    if not declared_methods:
        return None

    result = subprocess.run(
        [DEFECTS4J, 'info', '-p', project, '-b', str(bug_id)],
        capture_output=True, text=True, env=DEFECTS4J_ENV
    )
    if result.returncode != 0:
        return None

    for match in TEST_METHOD_PATTERN.findall(result.stdout):
        stem = match.split('_')[0].lstrip('_')
        if not stem:
            continue
        for candidate in (stem[0].lower() + stem[1:], stem):
            if candidate in declared_methods:
                return candidate

    return None


def rank_bugs_by_simplicity(projects, max_bugs_per_project=None):
    """
    Rank bugs across projects by simplicity:
    - Must have exactly one modified source class
    - Ranked by diff size ascending (fewer changed lines = simpler bug)

    Does not require any bug checkout -- uses git history directly.
    """
    candidates = []

    for project in projects:
        print(f"\nScanning {project}...")
        bugs = get_active_bugs(project)
        if max_bugs_per_project:
            bugs = bugs[:max_bugs_per_project]

        for bug in bugs:
            bug_id = bug['bug_id']
            classes = get_modified_classes(project, bug_id)

            if len(classes) != 1:
                continue

            modified_class = classes[0]
            diff_size = get_diff_size(project, bug_id)
            target_method = get_target_method(project, bug_id, modified_class)

            candidates.append({
                'project': project,
                'bug_id': bug_id,
                'bug_key': f"{project}_{bug_id}",
                'modified_class': modified_class,
                'target_method': target_method,
                'diff_size': diff_size if diff_size is not None else 9999,
            })

    candidates.sort(key=lambda x: x['diff_size'])
    return candidates


def print_top_candidates(candidates, n=30):
    print(f"\n{'='*80}")
    print(f"TOP {n} SIMPLEST BUGS")
    print(f"{'='*80}")
    print(f"{'Rank':<6}{'Bug':<15}{'Class':<35}{'Method':<25}{'Diff':<8}")
    print(f"{'-'*80}")
    for i, c in enumerate(candidates[:n]):
        class_short = c['modified_class'].split('.')[-1]
        method = c['target_method'] or 'unknown'
        print(
            f"{i+1:<6}"
            f"{c['bug_key']:<15}"
            f"{class_short:<35}"
            f"{method:<25}"
            f"{c['diff_size']:<8}"
        )


def save_candidates(candidates, output_path):
    with open(output_path, 'w') as f:
        json.dump(candidates, f, indent=2)
    print(f"\nSaved {len(candidates)} candidates to {output_path}")


if __name__ == '__main__':
    candidates = rank_bugs_by_simplicity(
        projects=['Lang', 'Math', 'Time', 'Cli', 'Codec', 'Csv',
                  'Jsoup', 'Gson', 'JacksonCore'],
    )

    print_top_candidates(candidates, n=30)
    save_candidates(candidates, BUG_CANDIDATES_PATH)
