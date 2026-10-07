"""Ingestion et normalisation des données brutes de barrages."""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

from src.config import DATA_INTERIM, DATA_RAW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_INTERIM.mkdir(parents=True, exist_ok=True)


BarrageSchema = pa.DataFrameSchema(
    {
        "date": pa.Column(pa.DateTime, nullable=False),
        "barrage": pa.Column(pa.String, nullable=False),
        "niveau_pct": pa.Column(pa.Float, pa.Check.between(-1, 101), nullable=False),
        "reserve_m3": pa.Column(pa.Float, pa.Check.ge(0), nullable=True),
    },
    strict=False,
)


DAM_NAME_MAPPING = {
    "Yacoub El Mansour": "Yacoub El Mansour",
    "Lalla Takerkoust": "Lalla Takerkoust",
    "Sd Mohamed Ben Slimane El Jazouli": "Mohamed Ben Slimane El Jazouli",
    "Abou El Abess Sebti": "Abou El Abess Sebti",
    "Bge My abdrhmane": "Moulay Abderrahmane",
    "Mansour Eddahbi": "Mansour Eddahbi",
    "Al Massira": "Al Massira",
    "Bin el Ouidane": "Bin el Ouidane",
    "Idriss Ier": "Idriss Ier",
    "Sidi Chahed": "Sidi Chahed",
}


def normalize_dam_name(name: str) -> str:
    """Normalise le nom d'un barrage."""
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    name = name.replace("\n", " ")
    return DAM_NAME_MAPPING.get(name, name)


def read_tensift_excel(filepath: Path) -> Optional[DataFrame]:
    """Lit un fichier Excel Tensift et le convertit en format long."""
    try:
        df = pd.read_excel(filepath, header=None)
        if df.shape[1] < 3:
            logger.warning("Fichier %s : trop peu de colonnes", filepath.name)
            return None

        name_row = df.iloc[0]
        capacity_row = df.iloc[1]
        header_row = df.iloc[2]

        dam_columns = []
        for i, col_name in enumerate(header_row):
            if isinstance(col_name, str) and "RESERVE" in col_name.upper() and i > 0:
                dam_name = str(name_row[i - 1]).strip()
                if dam_name.lower() not in ["bassin", ""]:
                    dam_columns.append((i, i + 1, normalize_dam_name(dam_name)))

        if not dam_columns:
            logger.warning("Aucune colonne RESERVE trouvée dans %s", filepath.name)
            return None

        data_rows = df.iloc[3:].copy()
        data_rows.columns = range(len(data_rows.columns))

        long_rows = []
        fname_lower = filepath.name.lower()
        month = 8 if "aout" in fname_lower or "08" in fname_lower or "aout" in str(filepath).lower() else 9
        if "sept" in fname_lower or "09" in fname_lower:
            month = 9

        for _, row in data_rows.iterrows():
            try:
                day = int(row[0])
                for reserve_idx, pct_idx, dam_name in dam_columns:
                    reserve_val = row[reserve_idx]
                    pct_val = row[pct_idx]
                    if pd.notna(reserve_val) and pd.notna(pct_val):
                        year = 2026
                        date = datetime(year, month, day)
                        long_rows.append(
                            {
                                "date": date,
                                "barrage": dam_name,
                                "niveau_pct": float(pct_val),
                                "reserve_m3": float(reserve_val) * 1e6,
                            }
                        )
            except (ValueError, TypeError):
                continue

        if not long_rows:
            logger.warning("Aucune donnée valide extraite de %s", filepath.name)
            return None

        result = pd.DataFrame(long_rows)
        logger.info("Lu %s : %d lignes, %d barrages", filepath.name, len(result), result["barrage"].nunique())
        return result

    except Exception as e:
        logger.error("Erreur lecture %s : %s", filepath.name, e)
        return None


def read_souss_massa_excel(filepath: Path) -> Optional[DataFrame]:
    """Lit la liste des barrages Souss Massa (pas de données temporelles)."""
    try:
        df = pd.read_excel(filepath)
        logger.info("Lu %s : %d barrages (liste statique)", filepath.name, len(df))
        return None
    except Exception as e:
        logger.error("Erreur lecture %s : %s", filepath.name, e)
        return None


def read_generic_excel(filepath: Path) -> Optional[DataFrame]:
    """Tente de lire un Excel générique avec détection automatique."""
    try:
        df = pd.read_excel(filepath)
        logger.info("Lu %s : shape=%s, cols=%s", filepath.name, df.shape, df.columns.tolist())
        return df
    except Exception as e:
        logger.error("Erreur lecture générique %s : %s", filepath.name, e)
        return None


def read_csv_file(filepath: Path) -> Optional[DataFrame]:
    """Lit un fichier CSV."""
    try:
        df = pd.read_csv(filepath)
        logger.info("Lu CSV %s : shape=%s", filepath.name, df.shape)
        return df
    except Exception as e:
        logger.error("Erreur lecture CSV %s : %s", filepath.name, e)
        return None


def process_all_files() -> DataFrame:
    """Traite tous les fichiers dans data/raw/."""
    all_data = []

    for filepath in sorted(DATA_RAW.glob("*")):
        if filepath.suffix.lower() in [".xlsx", ".xls"]:
            if "tensift" in filepath.name.lower():
                df = read_tensift_excel(filepath)
            elif "souss_massa" in filepath.name.lower():
                df = read_souss_massa_excel(filepath)
            else:
                df = read_generic_excel(filepath)
            if df is not None and not df.empty:
                all_data.append(df)

        elif filepath.suffix.lower() == ".csv":
            df = read_csv_file(filepath)
            if df is not None and not df.empty:
                all_data.append(df)

        elif filepath.suffix.lower() == ".pdf":
            logger.info("PDF ignoré (déjà extrait en CSV) : %s", filepath.name)

    if not all_data:
        logger.warning("Aucune donnée trouvée dans data/raw/")
        return pd.DataFrame(columns=["date", "barrage", "niveau_pct", "reserve_m3"])

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "barrage"]).sort_values(["barrage", "date"])
    combined = combined.reset_index(drop=True)

    logger.info("Total combiné : %d lignes, %d barrages uniques", len(combined), combined["barrage"].nunique())
    logger.info("Période : %s à %s", combined["date"].min(), combined["date"].max())
    logger.info("Barrages : %s", sorted(combined["barrage"].unique()))

    return combined


def validate_and_save(df: DataFrame) -> Path:
    """Valide avec pandera et sauvegarde en Parquet."""
    if df.empty:
        logger.warning("DataFrame vide, rien à sauvegarder")
        return DATA_INTERIM / "barrages_raw.parquet"

    try:
        validated = BarrageSchema.validate(df, lazy=True)
        logger.info("Validation pandera : OK")
    except pa.errors.SchemaErrors as e:
        logger.warning("Validation pandera : %d erreurs", len(e.failure_cases))
        logger.debug("Détails erreurs :\n%s", e.failure_cases)
        validated = df

    output_path = DATA_INTERIM / "barrages_raw.parquet"
    validated.to_parquet(output_path, index=False)
    logger.info("Sauvegardé : %s (%.2f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path


def main() -> int:
    """Point d'entrée principal."""
    logger.info("=== Début ingestion données barrages ===")
    df = process_all_files()
    output_path = validate_and_save(df)
    logger.info("=== Fin ingestion : %s ===", output_path)
    return 0


if __name__ == "__main__":
    exit(main())