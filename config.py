import os
import sys
import subprocess

THESIS_ROOT = os.environ.get(
    'THESIS_ROOT',
    os.path.dirname(os.path.abspath(__file__))
)

DEFECTS4J_HOME = os.environ.get(
    'DEFECTS4J_HOME',
    os.path.join(THESIS_ROOT, 'defects4j')
)

DEFECTS4J = os.path.join(DEFECTS4J_HOME, 'framework', 'bin', 'defects4j')

JUNIT_JAR = os.path.join(
    DEFECTS4J_HOME, 'framework', 'projects', 'lib', 'junit-4.12-hamcrest-1.3.jar'
)

DEFECTS4J_PROJECT_REPOS = os.path.join(DEFECTS4J_HOME, 'project_repos')

# --- working dirs ---
BASE_BUGS_DIR = os.environ.get('BASE_BUGS_DIR', os.path.join(THESIS_ROOT, 'bugs'))
BASE_DATA_DIR = os.environ.get('BASE_DATA_DIR', os.path.join(THESIS_ROOT, 'data'))

# make dirs for bugs and data if they don't exist
os.makedirs(BASE_BUGS_DIR, exist_ok=True)
os.makedirs(BASE_DATA_DIR, exist_ok=True)

# --- paths ---
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DESCRIPTIONS_JSON = os.path.join(PIPELINE_DIR, 'bug_descriptions', 'descriptions.json')
BUG_CANDIDATES_PATH = os.path.join(BASE_DATA_DIR, 'bug_candidates.json')
VERIFIED_BUGS_PATH = os.path.join(PIPELINE_DIR, 'verified_bugs.py')
EXCLUSION_LOG_PATH = os.path.join(BASE_DATA_DIR, 'exclusion_log.json')

# --- llm ---
LLAMA_URL = os.environ.get('LLAMA_URL', 'http://localhost:8080/completion')
MODEL_NAME = os.environ.get('MODEL_NAME', 'codellama-13b-instruct')
DEFAULT_TEMPERATURE = float(os.environ.get('DEFAULT_TEMPERATURE', 0.0))
N_SAMPLES = 1
LLM_TIMEOUT = 600        # in seconds, needs a bit of buffer
CTX_SIZE = 4096          # must match --ctx-size passed to llama-server
N_PREDICT = 1024         # must match n_predict in generator.call_llm's payload
CHATML_OVERHEAD = 57     # measured empirically via generator.count_tokens on wrapped vs raw prompt
SAFETY_MARGIN = 32       # buffer for estimation error

MAX_PROMPT_TOKENS = CTX_SIZE - N_PREDICT - CHATML_OVERHEAD - SAFETY_MARGIN

# --- java ---
def _find_java11():
    """
    Locate a Java 11 JDK, independent of the shell's default java/JAVA_HOME
    (which version managers like SDKMAN, jenv or asdf may point elsewhere).
    Order: THESIS_JAVA_HOME override -> macOS java_home -> common install paths.
    Returns '' if not found; pipeline.py checks this before running, so
    scripts that don't need Java (e.g. visualise.py) still import fine.
    """
    override = os.environ.get('THESIS_JAVA_HOME')
    if override:
        return override

    if sys.platform == 'darwin' and os.path.exists('/usr/libexec/java_home'):
        r = subprocess.run(['/usr/libexec/java_home', '-v', '11'],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()

    for path in [
        '/opt/homebrew/opt/openjdk@11',            # Homebrew, Apple Silicon
        '/usr/local/opt/openjdk@11',               # Homebrew, Intel Mac
        '/usr/lib/jvm/java-11-openjdk-amd64',      # Debian/Ubuntu
        '/usr/lib/jvm/java-11-openjdk',            # Fedora/RHEL
    ]:
        if os.path.isdir(path):
            return path
    return ''


JAVA_HOME = _find_java11()

# --- perl ---
# Override these for your OS if defaults dont work.
PERL5LIB = os.environ.get('PERL5LIB', os.path.expanduser('~/perl5/lib/perl5'))
PERL_LOCAL_LIB_ROOT = os.environ.get('PERL_LOCAL_LIB_ROOT', os.path.expanduser('~/perl5'))

DEFECTS4J_ENV = {
    **os.environ,
    'JAVA_HOME': JAVA_HOME,
    'PATH': (f"{JAVA_HOME}/bin:" if JAVA_HOME else "") + os.environ.get('PATH', ''),
    'PERL5LIB': PERL5LIB,
    'PERL_LOCAL_LIB_ROOT': PERL_LOCAL_LIB_ROOT,
    'PERL_MB_OPT': f"--install_base {PERL_LOCAL_LIB_ROOT}",
    'PERL_MM_OPT': f"INSTALL_BASE={PERL_LOCAL_LIB_ROOT}",
}

# --- shap / explanation stage ---
# higher temperature should yield more diverse output
STABILITY_TEMPERATURE = float(os.environ.get('STABILITY_TEMPERATURE', 0.0))
# number of runs to perform for stability analysis
STABILITY_RUNS = int(os.environ.get('STABILITY_RUNS', 5))
