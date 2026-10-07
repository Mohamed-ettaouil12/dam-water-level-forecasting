"""Téléchargement automatique des niveaux de barrages depuis les sources officielles marocaines."""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src.config import DATA_RAW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 5

DATA_RAW.mkdir(parents=True, exist_ok=True)


def make_session() -> requests.Session:
    """Crée une session requests avec User-Agent et retry logic."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def download_with_retry(
    session: requests.Session,
    url: str,
    output_path: Path,
    max_retries: int = MAX_RETRIES,
) -> bool:
    """Télécharge un fichier avec retry."""
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Tentative %d/%d : %s", attempt, max_retries, url)
            resp = session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            resp.raise_for_status()

            total_size = int(resp.headers.get("content-length", 0))
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = output_path.stat().st_size
            logger.info("Succès : %s (%.2f MB)", output_path.name, file_size / 1e6)
            return True

        except requests.RequestException as e:
            logger.warning("Échec tentative %d : %s", attempt, e)
            if attempt < max_retries:
                time.sleep(RETRY_DELAY)
            else:
                logger.error("Échec définitif après %d tentatives : %s", max_retries, url)
                if output_path.exists():
                    output_path.unlink()
                return False
    return False


def download_from_data_gov_ma(session: requests.Session) -> list[Path]:
    """Télécharge les fichiers Excel du dataset Tensift sur data.gov.ma."""
    downloaded = []
    base_url = "https://data.gov.ma/data/fr/dataset/0aff0dd1-3c7e-4e14-9cbf-18782902a6e4/resource/"

    resources = [
        ("57580e59-8b9c-4bde-bf9f-2a25107bbf6e", "tensift_2026_09.xlsx"),
        ("0b69e96f-6c95-40ff-8c91-3721984b2613", "tensift_2026_08.xlsx"),
    ]

    for resource_id, filename in resources:
        url = f"{base_url}{resource_id}/download/{filename}"
        output_path = DATA_RAW / f"{datetime.now():%Y%m%d}_{filename}"
        if download_with_retry(session, url, output_path):
            downloaded.append(output_path)

    return downloaded


def download_from_ministere_equipement(session: requests.Session) -> list[Path]:
    """Tente de télécharger les bulletins PDF depuis le site du Ministère."""
    downloaded = []

    urls_to_try = [
        "https://www.equipement.gov.ma/eau/Ressources-en-eau/Pages/situation-ressources-eau.aspx",
        "https://www.equipement.gov.ma/eau/Documentation/Pages/accueil.aspx",
    ]

    for page_url in urls_to_try:
        try:
            logger.info("Exploration page : %s", page_url)
            resp = session.get(page_url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.content, "html.parser")

            for link in soup.find_all("a", href=True):
                href = link["href"]
                if any(
                    ext in href.lower()
                    for ext in [".pdf", ".xlsx", ".xls", ".csv"]
                ):
                    if "barrage" in href.lower() or "situation" in href.lower():
                        full_url = urljoin(page_url, href)
                        filename = f"ministere_{datetime.now():%Y%m%d}_{Path(href).name}"
                        output_path = DATA_RAW / filename
                        if download_with_retry(session, full_url, output_path):
                            downloaded.append(output_path)

        except requests.RequestException as e:
            logger.warning("Impossible d'accéder à %s : %s", page_url, e)

    return downloaded


def download_from_souss_massa(session: requests.Session) -> list[Path]:
    """Télécharge la liste des barrages Souss Massa depuis data.gov.ma."""
    downloaded = []
    url = (
        "https://data.gov.ma/data/fr/dataset/88d28afd-bf15-4c0e-a97e-25c050f6f256/"
        "resource/9f82d649-ddd9-4f7e-8a09-70122c2ca1eb/download/barrage_souss_massa.xlsx"
    )
    filename = f"souss_massa_barrages_{datetime.now():%Y%m%d}.xlsx"
    output_path = DATA_RAW / filename
    if download_with_retry(session, url, output_path):
        downloaded.append(output_path)
    return downloaded


def extract_pdf_tables(pdf_path: Path) -> list[Path]:
    """Extrait les tableaux d'un PDF avec pdfplumber."""
    import pdfplumber

    extracted = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for j, table in enumerate(tables):
                    if table and len(table) > 1:
                        import pandas as pd

                        df = pd.DataFrame(table[1:], columns=table[0])
                        out_path = pdf_path.with_suffix(f".p{i}_t{j}.csv")
                        df.to_csv(out_path, index=False)
                        extracted.append(out_path)
                        logger.info("Tableau extrait : %s (%d lignes)", out_path.name, len(df))
    except Exception as e:
        logger.error("Erreur extraction PDF %s : %s", pdf_path, e)
    return extracted


def main() -> int:
    """Point d'entrée principal."""
    logger.info("=== Début téléchargement données barrages ===")
    session = make_session()
    all_downloaded = []

    logger.info("--- Source 1: data.gov.ma (Tensift) ---")
    all_downloaded.extend(download_from_data_gov_ma(session))

    logger.info("--- Source 2: Ministère Équipement ---")
    all_downloaded.extend(download_from_ministere_equipement(session))

    logger.info("--- Source 3: data.gov.ma (Souss Massa) ---")
    all_downloaded.extend(download_from_souss_massa(session))

    for pdf_file in DATA_RAW.glob("*.pdf"):
        logger.info("--- Extraction PDF : %s ---", pdf_file.name)
        extract_pdf_tables(pdf_file)

    logger.info("=== Fin téléchargement : %d fichiers ===", len(all_downloaded))
    for f in all_downloaded:
        logger.info("  - %s (%.2f MB)", f.name, f.stat().st_size / 1e6)

    return 0 if all_downloaded else 1


if __name__ == "__main__":
    exit(main())