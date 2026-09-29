import importlib.util
from pathlib import Path


def test_kit_does_not_include_private_feedback_or_recursively_copy_nested_output(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("production_install", Path(__file__).parents[1] /
                                               "benchmarks/qualify_production_install.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    docs = tmp_path / "docs/pilot"
    docs.mkdir(parents=True)
    for name in installer.PILOT_FILES:
        (docs / name).write_text("public fixture")
    (docs / "completed-private-feedback.json").write_text("PRIVATE_SENTINEL")
    output = docs / "local-output"
    (output / "origin-env").mkdir(parents=True)
    (output / "origin-env/private-file").write_text("PRIVATE_SENTINEL")
    target = output / "portable-pilot/docs/pilot"
    installer.copy_pilot_docs(target)
    assert {p.name for p in target.iterdir()} == set(installer.PILOT_FILES)
    assert all(p.is_file() and "PRIVATE_SENTINEL" not in p.read_text() for p in target.iterdir())
