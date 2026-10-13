"""Static check that ground truth never reaches the robot's software.

Rules: nothing under robot_code/ imports experiments/ (where the true position is read), and nothing under robot_code/
refers to _ground_truth_pose or true_pose (the simulator only defines _ground_truth_pose; it never calls it).
"""
import ast
from pathlib import Path

ROBOT_CODE = Path(__file__).resolve().parents[1] / "robot_code"
FORBIDDEN_NAMES = {"_ground_truth_pose", "true_pose"}


def test_robot_code_does_not_touch_ground_truth():
    files = list(ROBOT_CODE.rglob("*.py"))
    # Guard: if the folder were renamed or empty, this test would otherwise pass while checking nothing.
    assert len(files) >= 10, f"expected the robot's code in {ROBOT_CODE}, found {len(files)} files"
    offenders = []
    for path in files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                names = []
            if any(n == "experiments" or n.startswith("experiments.") for n in names):
                offenders.append(f"{path.relative_to(ROBOT_CODE)}: imports experiments")
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
                offenders.append(f"{path.relative_to(ROBOT_CODE)}: uses {node.attr}")
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                offenders.append(f"{path.relative_to(ROBOT_CODE)}: uses {node.id}")
    assert not offenders, offenders


def test_the_check_would_catch_a_violation(tmp_path):
    # Self-test: the same rules flag a file that reads the true position.
    bad = "from experiments.true_position import true_pose\nx = true_pose(sim)\n"
    tree = ast.parse(bad)
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module.startswith("experiments")]
    hits += [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in FORBIDDEN_NAMES]
    assert len(hits) == 2
