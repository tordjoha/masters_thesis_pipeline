from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class Bug:
    bug_id: str              # e.g. "Math_5"
    project: str             # e.g. "Math"
    bug_number: int          # e.g. 5
    modified_class: str      # e.g. "org.apache.commons.math3.complex.Complex"
    target_method: str       # e.g. "reciprocal"
    source_file: str         # full path to the Java source file
    bug_dir: str             # path to checked out buggy version
    fixed_dir: str           # path to checked out fixed version


@dataclass
class GeneratedTest:
    sample_id: int
    content: str
    path: str
    was_wrapped: bool = False


@dataclass
class EvaluationResult:
    sample_id: int
    compiles: bool = False
    passes_fixed: Optional[bool] = None
    fails_buggy: Optional[bool] = None
    fixed_outcome: Optional[str] = None   # 'pass' | 'fail' | 'timeout'
    buggy_outcome: Optional[str] = None   # 'pass' | 'fail' | 'timeout'
    effective: bool = False
    compile_error: Optional[str] = None
    fixed_output: Optional[str] = None
    buggy_output: Optional[str] = None


@dataclass
class BugResult:
    bug: Bug
    prompt: Optional[str]
    segments: Optional[Dict]
    exclusion_reason: Optional[str]
    generated_tests: List[GeneratedTest] = field(default_factory=list)
    evaluation_results: List[EvaluationResult] = field(default_factory=list)
    shap_results: Optional[Dict] = field(default=None)

    @property
    def effective_tests(self):
        return [
            (t, r) for t, r in zip(self.generated_tests, self.evaluation_results)
            if r.effective
        ]

    @property
    def n_effective(self):
        return len(self.effective_tests)

    @property
    def n_total(self):
        return len(self.generated_tests)

    def summary(self):
        if self.exclusion_reason:
            return f"{self.bug.bug_id}: EXCLUDED ({self.exclusion_reason})"
        return (
            f"{self.bug.bug_id}: "
            f"{self.n_effective}/{self.n_total} effective, "
            f"{sum(1 for r in self.evaluation_results if r.compiles)}/{self.n_total} compiled"
        )
