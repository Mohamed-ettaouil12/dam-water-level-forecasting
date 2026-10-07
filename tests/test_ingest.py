"""Tests pour l'ingestion des données barrages."""

import pandas as pd
import pandera as pa
import pytest

from src.data.ingest import (
    BarrageSchema,
    normalize_dam_name,
    process_all_files,
    read_tensift_excel,
    validate_and_save,
)


def test_normalize_dam_name():
    """Test la normalisation des noms de barrages."""
    assert normalize_dam_name("  Yacoub El Mansour  ") == "Yacoub El Mansour"
    assert normalize_dam_name("Bge My abdrhmane") == "Moulay Abderrahmane"
    assert normalize_dam_name("Al Massira") == "Al Massira"
    assert normalize_dam_name("Inconnu") == "Inconnu"


def test_barrage_schema_validation():
    """Test la validation pandera du schéma."""
    df_valid = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "barrage": ["Test Dam", "Test Dam"],
            "niveau_pct": [50.0, 51.0],
            "reserve_m3": [1e6, 1.02e6],
        }
    )
    validated = BarrageSchema.validate(df_valid)
    assert len(validated) == 2

    df_invalid_pct = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "barrage": ["Test Dam"],
            "niveau_pct": [150.0],
            "reserve_m3": [1e6],
        }
    )
    with pytest.raises(pa.errors.SchemaErrors):
        BarrageSchema.validate(df_invalid_pct, lazy=True)


def test_read_tensift_excel_structure(tmp_path):
    """Test la lecture d'un fichier Excel Tensift factice (structure réelle)."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active

    ws.cell(row=1, column=1, value="Test Dam 1")
    ws.cell(row=1, column=2, value="")
    ws.cell(row=1, column=3, value="Test Dam 2")
    ws.cell(row=1, column=4, value="")

    ws.cell(row=2, column=1, value="CAPACITÉ NORMALE\n(Mm³)")
    ws.cell(row=2, column=2, value=57.45)
    ws.cell(row=2, column=3, value=57.45)
    ws.cell(row=2, column=4, value=50.82)
    ws.cell(row=2, column=5, value=50.82)

    ws.cell(row=3, column=1, value="JOUR")
    ws.cell(row=3, column=2, value="RESERVE (Mm3)")
    ws.cell(row=3, column=3, value="TAUX DE REMPLISSAGE\n ( % )")
    ws.cell(row=3, column=4, value="RESERVE (Mm3)")
    ws.cell(row=3, column=5, value="TAUX DE REMPLISSAGE\n ( % )")

    ws.cell(row=4, column=1, value=1)
    ws.cell(row=4, column=2, value=49.205)
    ws.cell(row=4, column=3, value=85.64839)
    ws.cell(row=4, column=4, value=42.172)
    ws.cell(row=4, column=5, value=82.983078)

    ws.cell(row=5, column=1, value=2)
    ws.cell(row=5, column=2, value=49.100)
    ws.cell(row=5, column=3, value=85.46)
    ws.cell(row=5, column=4, value=42.100)
    ws.cell(row=5, column=5, value=82.84)

    test_file = tmp_path / "test_tensift_aout.xlsx"
    wb.save(test_file)

    df = read_tensift_excel(test_file)
    assert df is not None
    assert len(df) == 4
    assert set(df["barrage"].unique()) == {"Test Dam 1", "Test Dam 2"}
    assert df["niveau_pct"].between(0, 100).all()
    assert df["reserve_m3"].ge(0).all()
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_validate_and_save(tmp_path):
    """Test la validation et sauvegarde."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "barrage": ["Test Dam", "Test Dam"],
            "niveau_pct": [50.0, 51.0],
            "reserve_m3": [1e6, 1.02e6],
        }
    )
    output_path = tmp_path / "test_output.parquet"
    result_path = validate_and_save(df)
    assert result_path.exists()
    loaded = pd.read_parquet(result_path)
    assert len(loaded) == 2


def test_process_all_files_empty(tmp_path, monkeypatch):
    """Test process_all_files avec dossier vide."""
    monkeypatch.setattr("src.data.ingest.DATA_RAW", tmp_path)
    df = process_all_files()
    assert df.empty
    assert list(df.columns) == ["date", "barrage", "niveau_pct", "reserve_m3"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])