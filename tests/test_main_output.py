import sys

import polars as pl
import pytest

import main


class FakeLoader:
    def fetch_boxcell(self, *args, **kwargs):
        assert args[4] == ["value"]
        assert kwargs["streaming"] is True
        return pl.DataFrame({"value": [1.0, 2.0]})


def test_main_writes_csv_when_format_is_selected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    output_path = tmp_path / "boxcell.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--dataset",
            "test",
            "--output",
            str(output_path),
            "--format",
            "csv",
            "0",
            "0",
            "0",
            "2",
            "value",
        ],
    )
    monkeypatch.setattr(main, "load_dataset_config", lambda _: {"data_size": 2})
    monkeypatch.setattr(main, "init_loader", lambda *_: FakeLoader())

    main.main()

    assert pl.read_csv(output_path).to_dict(as_series=False) == {"value": [1.0, 2.0]}


def test_main_writes_vtk_when_format_is_selected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    output_path = tmp_path / "boxcell.vtk"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--dataset",
            "test",
            "--output",
            str(output_path),
            "--format",
            "vtk",
            "0",
            "0",
            "0",
            "2",
            "value",
        ],
    )
    monkeypatch.setattr(main, "load_dataset_config", lambda _: {"data_size": 2})
    monkeypatch.setattr(main, "init_loader", lambda *_: FakeLoader())

    main.main()

    assert output_path.read_bytes().startswith(b"# vtk DataFile Version 3.0\n")


def test_main_rejects_output_extension_that_conflicts_with_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--dataset",
            "test",
            "--output",
            "boxcell.vtk",
            "--format",
            "csv",
            "0",
            "0",
            "0",
            "2",
            "value",
        ],
    )

    with pytest.raises(SystemExit) as error:
        main.main()

    assert error.value.code == 2
