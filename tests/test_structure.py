"""Static check that ground truth never reaches the pipeline.

Rules: nothing under src/ imports evaluation/, and nothing under src/ except the platform that
defines it refers to _ground_truth_pose or the evaluation.ground_truth module.
"""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
FORBIDDEN_ATTRS = {"_ground_truth_pose", "true_pose"}


def test_src_does_not_touch_ground_truth():
    offenders = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                names = []
            if any(n == "evaluation" or n.startswith("evaluation.") for n in names):
                offenders.append(f"{path.name}: imports evaluation")
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
                offenders.append(f"{path.name}: uses {node.attr}")
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_ATTRS:
                offenders.append(f"{path.name}: uses {node.id}")
    assert not offenders, offenders
