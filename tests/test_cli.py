"""The command line, end to end."""

from pathlib import Path

from typer.testing import CliRunner

import sfg.cli as cli
from sfg.mock import MockProvider

EXAMPLE = str(Path(__file__).resolve().parents[1] / "examples" / "oat-milk-concept.yaml")
runner = CliRunner()


def test_validate_and_personas():
    assert runner.invoke(cli.app, ["validate", EXAMPLE]).exit_code == 0
    out = runner.invoke(cli.app, ["personas", EXAMPLE, "--n", "5"])
    assert out.exit_code == 0 and "Sample vs. spec" in out.output


def test_invalid_study_fails_cleanly(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("title: x\n")
    out = runner.invoke(cli.app, ["validate", str(bad)])
    assert out.exit_code == 1 and "problems" in out.output


def test_mock_run_writes_a_report(tmp_path):
    out = runner.invoke(cli.app, ["run", EXAMPLE, "--mock", "--groups", "1", "--size", "4", "--out", str(tmp_path)])
    assert out.exit_code == 0, out.output
    reports = list(tmp_path.glob("*/report.html"))
    assert len(reports) == 1


def test_failed_run_keeps_its_call_log(tmp_path, monkeypatch):
    class DiesMidRun(MockProvider):
        """Behaves like the mock, then starts rejecting every call as if the API key were revoked."""

        def __init__(self):
            self.n = 0

        def complete(self, **kw):
            self.n += 1
            if self.n > 10:
                class AuthenticationError(Exception):
                    pass

                raise AuthenticationError("key revoked")
            return super().complete(**kw)

    monkeypatch.setattr(cli, "make_provider", lambda name: DiesMidRun())
    out = runner.invoke(cli.app, ["run", EXAMPLE, "--mock", "--groups", "1", "--size", "4", "--out", str(tmp_path)])
    assert out.exit_code == 1 and "Run stopped" in out.output
    logs = list(tmp_path.glob("*-failed/calls.jsonl"))
    assert len(logs) == 1 and len(logs[0].read_text().splitlines()) == 10


def test_dotenv_is_found_next_to_the_project_not_just_the_cwd(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "examples").mkdir(parents=True)
    (project / ".env").write_text("SFG_TEST_KEY=found\n")
    study = project / "examples" / "s.yaml"
    study.write_text("x")
    monkeypatch.chdir(tmp_path)  # run from somewhere else
    monkeypatch.delenv("SFG_TEST_KEY", raising=False)
    cli._load_dotenv(cli._find_dotenv(study))
    import os

    assert os.environ.get("SFG_TEST_KEY") == "found"
    monkeypatch.delenv("SFG_TEST_KEY")


def test_call_log_is_written_to_the_run_folder(tmp_path):
    out = runner.invoke(cli.app, ["run", EXAMPLE, "--mock", "--groups", "1", "--size", "4", "--out", str(tmp_path)])
    assert out.exit_code == 0
    run_dir = next(tmp_path.iterdir())
    assert len((run_dir / "calls.jsonl").read_text().splitlines()) > 50


def test_quick_preflight_is_small_and_complete(tmp_path):
    import json

    out = runner.invoke(cli.app, ["run", EXAMPLE, "--mock", "--quick", "--out", str(tmp_path)])
    assert out.exit_code == 0, out.output
    data = json.loads(next(tmp_path.glob("*/data.json")).read_text())
    assert len(data["personas"]) == 3
    assert {t["topic"] for t in data["turns"]} == {0}
    assert data["analysis"]["headline"]
