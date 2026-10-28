import importlib.util
from pathlib import Path


_UPLOAD_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "upload_reports_to_yandex.py"
)
_spec = importlib.util.spec_from_file_location(
    "upload_reports_to_yandex", _UPLOAD_PATH
)
upload = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(upload)


def test_report_files_include_transcript_and_coverage(tmp_path):
    expected = [
        "report.pdf",
        "report.md",
        "report.html",
        "transcript.md",
        "coverage.json",
        "youtrack_context.json",
    ]
    for filename in expected:
        (tmp_path / filename).write_text(filename, encoding="utf-8")
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")

    files = upload._report_files(tmp_path)

    assert [path.name for path in files] == expected


def test_report_files_skip_missing_optional_artifacts(tmp_path):
    (tmp_path / "report.md").write_text("# Report\n", encoding="utf-8")
    (tmp_path / "transcript.md").write_text("# Transcript\n", encoding="utf-8")

    files = upload._report_files(tmp_path)

    assert [path.name for path in files] == ["report.md", "transcript.md"]
