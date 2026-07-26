"""The notebooks must run on a clean checkout, and must not lie about what they measure.

Every check here corresponds to a defect found by auditing the committed notebooks:

- One notebook loaded `/home/aswani/automl/data/processed/...`, so it ran on exactly one
  machine (finding R1, notebook half).
- One imported `aif360`, which `requirements.txt` deliberately excludes because it pins
  numpy < 2. The notebook was unrunnable in the project's own environment.
- One read `config.data` and `config.fairness`, which the config reduction removed, so it
  raised AttributeError on its second cell.
- One shipped a stored `ModuleNotFoundError` traceback from a conda environment that no
  longer exists, in a notebook titled "Complete Pipeline Walkthrough".
- One regenerated `data/processed/german_credit_complete_analysis.csv`, the duplicate
  deleted under finding R7.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.paths import PROJECT_ROOT

pytestmark = pytest.mark.unit

NOTEBOOKS = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))

# Modules removed during the rewrite. Matched as imports, not as bare substrings: the TabFM
# prototype defines a local helper called `fairness_metrics`, which is not a reference to the
# deleted module of that name.
DELETED_MODULES = (
    "automl_tuner",
    "fairness_metrics",
    "fairness_postprocessing",
    "data_processor",
    "fair_model_wrapper",
)

# Files removed during the rewrite. A notebook that reads or writes one is broken, but a
# comment explaining why it was removed is not, so comments are stripped before matching.
DELETED_FILES = (
    "fair_model_complete.joblib",
    "best_automl_model.joblib",
    "german_credit_complete_analysis.csv",
)

# Attributes the config reduction removed.
DELETED_CONFIG_ATTRS = ("config.data", "config.fairness")

# Packages the project does not install. requirements.txt states why for aif360.
ABSENT_DEPENDENCIES = ("aif360",)

# Colab mount points are correct inside Colab. A notebook may use them only if it says so
# in its first cell, so a reader knows before running it.
COLAB_PREFIX = "/content/drive"

MACHINE_PATHS = ("C:\\Users\\", "C:/Users/", "/home/aswani", "/Users/")


def cells(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def code_text(path: Path) -> str:
    return "\n".join(
        "".join(c["source"]) for c in cells(path) if c["cell_type"] == "code"
    )


def executable_code(path: Path) -> str:
    """Code with comment text removed, so an explanatory comment cannot fail a check."""
    lines = []
    for line in code_text(path).splitlines():
        head = line.split("#", 1)[0]
        if head.strip():
            lines.append(head)
    return "\n".join(lines)


def all_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_there_is_at_least_one_notebook():
    """Guards against the checks below passing vacuously."""
    assert NOTEBOOKS, "no notebooks found"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_machine_specific_paths(path: Path):
    """Covers stored outputs as well as code: a printed absolute path is equally a fact
    about one machine, and this notebook set previously carried both."""
    text = all_text(path)
    for needle in MACHINE_PATHS:
        assert needle not in text.replace("\\\\", "\\"), (
            f"{path.name} contains the machine-specific path {needle!r}; "
            "resolve and print paths relative to the repository root instead"
        )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_colab_paths_are_declared_in_the_first_cell(path: Path):
    if COLAB_PREFIX not in all_text(path):
        pytest.skip("no Colab paths")

    first = "".join(cells(path)[0]["source"]).lower()
    assert "colab" in first, (
        f"{path.name} mounts Google Drive but its first cell does not say it is Colab-only"
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_imports_of_deleted_modules(path: Path):
    code = executable_code(path)
    for module in DELETED_MODULES:
        for statement in (f"import {module}", f"from {module}", f".{module} import"):
            assert statement not in code, (
                f"{path.name} imports {module}, which was deleted in the rewrite"
            )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_references_to_deleted_files_or_config_attributes(path: Path):
    code = executable_code(path)
    for needle in DELETED_FILES + DELETED_CONFIG_ATTRS:
        assert needle not in code, (
            f"{path.name} references {needle!r}, which no longer exists"
        )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_imports_of_packages_the_project_does_not_install(path: Path):
    text = code_text(path)
    for package in ABSENT_DEPENDENCIES:
        assert f"import {package}" not in text and f"from {package}" not in text, (
            f"{path.name} imports {package}, which requirements.txt excludes"
        )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_stored_error_outputs(path: Path):
    """A committed traceback means the notebook was published in a failed state."""
    errors = [
        output.get("ename")
        for cell in cells(path)
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not errors, f"{path.name} ships stored error outputs: {errors}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_kernel_is_the_current_interpreter(path: Path):
    kernel = json.loads(all_text(path)).get("metadata", {}).get("kernelspec", {})
    assert kernel.get("name") == "python3", (
        f"{path.name} declares kernel {kernel.get('name')!r}; the stale 'automlenv' kernel "
        "cannot be selected on any other machine"
    )
