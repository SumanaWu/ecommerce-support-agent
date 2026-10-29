"""Run isolated online restart and concurrency regressions with scripted models."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
result = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_recovery.py',
    'tests/test_online_planning.py::test_independent_process_commit_checkpoint_gap_replays_proposal_once',
    'tests/test_online_planning.py::test_new_checkpoint_reuses_model_plan_and_real_tool_id_after_restart',
    'tests/test_online_presentation.py::test_deleted_profile_is_not_reintroduced_from_a_saved_model_checkpoint'], cwd=root)
raise SystemExit(result.returncode)
