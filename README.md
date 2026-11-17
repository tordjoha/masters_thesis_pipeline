# Masters thesis pipeline
This pipeline generates Java unit tests for Defects4J bugs with a local LLM and explains each generated test with segment-level Shapley values.

## Architecture

Pipeline is split into two phases: bug curation and main experiment. First the bugs are selected and verified, then generation, evaluation and explanation per bug.

**Bug curation**
- `bug_selector.py` - ranks Defects4J bugs by diff simplicity, finds the buggy method and generates `bug_candidates.json`
- `bug_verifier.py` - checks out and confirms each bug (the method exists and is public) and generates `verified_bugs.py`, which `pipeline.py` uses for the generation step

**Main experiment**
- `prompt_parser.py` - parses the class file and builds the segmented prompt
- `generator.py` - calls the local LLM to generate test cases
- `evaluator.py` - compiles the generated tests and runs them against the buggy and fixed versions. A test is effective if it passes on the fixed version and fails on the buggy one.
- `shap_explainer.py` - computes exact Shapley attributions per prompt segment by coalition masking

**Supporting files**
- `config.py` - paths and settings
- `models.py` - shared dataclasses
- `visualise.py` - generates plots from the Shapley values produced by `shap_explainer.py`

`pipeline.py` ties all of this together.

## Requirements
- Java 11
- Python 3.9.6+
- Perl
- llama.cpp
- ~20 GB of disk space, mostly Defects4J bug data

### Java 11
Defects4J requires Java 11. The pipeline finds and uses Java 11 automatically, whatever your system default `java` or `JAVA_HOME` is (including when they're set by version managers like SDKMAN). It checks the macOS `java_home` registry, Homebrew, and common Linux install locations.

If your Java 11 is installed somewhere else, point to it explicitly:
```
export THESIS_JAVA_HOME=/path/to/jdk-11
```
Your global `java` and `JAVA_HOME` are left untouched. When `pipeline.py` starts, it prints the Java version in use and stops if it isn't Java 11.

## Example output
The repository includes the full output for Lang_57, the bug discussed in the thesis, in `examples/data/Lang_57`: the prompt, the generated test, the evaluation results and the Shapley values. The generated plots for this bug is also included in `examples/data/plots`



## Setup

1. Install and initialize Defects4J inside the project folder

    ```
    git clone https://github.com/rjust/defects4j
    cd defects4j
    cpanm --installdeps .
    ./init.sh
    cd ..
    ```
        
2. Create a virtual environment and install the Python dependencies:

    ```
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3. Download the model:

    ```
    huggingface-cli download TheBloke/CodeLlama-13B-Instruct-GGUF \
      codellama-13b-instruct.Q4_K_M.gguf --local-dir ~/models/codellama-13b
    ```

4. Configure paths (optional)

    No configuration is needed by default. To store the `defects4j`, `bugs` and `data` folders elsewhere, set `export THESIS_ROOT=/path/to/elsewhere`. If your Perl modules are not in `~/perl5`, set `PERL5LIB` and `PERL_LOCAL_LIB_ROOT` in `config.py`.

## Running the pipeline

1. Select candidate bugs:

    ```
    python bug_selector.py
    ```

2. Verify the candidates. The maximum number of bugs is set by `MAX_BUGS` in the main block of `bug_verifier.py` (default: 10).

    ```
    python bug_verifier.py
    ```

3. Start llama-server (change the model path if you stored it elsewhere):

    ```
    llama-server -m ~/models/codellama-13b/codellama-13b-instruct.Q4_K_M.gguf \
      --port 8080 \
      --ctx-size 4096 \
      --n-gpu-layers 99
    ```    
4. Run the pipeline:

    ```
    python pipeline.py
    ```

5. Generate plots from the Shapley values:

    ```
    python visualise.py
    ```

## Extended experiment: bug descriptions

`desc_pipeline.py` adds the issue-tracker bug description as a sixth prompt
segment. It reuses the five segments from the main run, so it requires
`pipeline.py` to have been run first.

1. `python pipeline.py`
2. `python desc_pipeline.py --all --dry-run`  (check prompts and token counts)
3. `python desc_pipeline.py --all`

Descriptions are in `bug_descriptions/descriptions.json`. Results are written
to `data_desc/`. Use `python compare_lime_shapley.py --desc` to compare LIME
with Shapley on the extended experiment.

## Known limitations
- **Platform.** Tested on macOS (Apple Silicon) with Java 11. The Linux default paths in `config.py` are untested.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@mastersthesis{johansson2026xai,
  author = {Tord Johansson},
  title  = {Investigating Explainable AI Techniques for Software Testing: Attributing and Analyzing LLM-Generated Unit Tests},
  school = {University of Bergen and Western Norway University of Applied Sciences},
  year   = {2026},
  type   = {Master's thesis},
  url    = {https://github.com/tordjoha/masters_thesis_pipeline}
}
```
