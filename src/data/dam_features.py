"""Chargement des caractéristiques statiques des barrages."""

import pandas as pd
import yaml
from src.config import CONFIGS_DIR


def load_dam_features() -> pd.DataFrame:
    """Lit configs/barrages.yaml et retourne un DataFrame des caractéristiques."""
    yaml_path = CONFIGS_DIR / "barrages.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    df = pd.DataFrame(data["barrages"])
    return df


if __name__ == "__main__":
    df = load_dam_features()
    print(df.to_string())