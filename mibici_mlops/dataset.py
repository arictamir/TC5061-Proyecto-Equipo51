
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from loguru import logger
import pandas as pd
import requests
import typer

from mibici_mlops.config import (
    END_MONTH,
    END_YEAR,
    MIBICI_OPEN_DATA_URL,
    NOMENCLATURE_FILE,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RAW_DATA_DIR,
    START_MONTH,
    START_YEAR,
)

app = typer.Typer()


def discover_download_urls() -> dict[str, str]:
    """Discover CSV download URLs from the MiBici open data page."""
    logger.info("Discovering files from MiBici open data portal...")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/154.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
    }

    response = requests.get(
        MIBICI_OPEN_DATA_URL,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    csv_urls = {}

    for link in soup.find_all("a", href=True):
        href = link["href"]

        if ".csv" not in href.lower():
            continue

        url = urljoin(MIBICI_OPEN_DATA_URL, href)
        filename = Path(url).name

        csv_urls[filename] = url

    logger.info(f"Found {len(csv_urls)} CSV files.")

    return csv_urls


def build_trip_filenames(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> list[str]:
    """Build the expected monthly MiBici trip filenames."""
    filenames = []

    year = start_year
    month = start_month

    while (year, month) <= (end_year, end_month):
        filenames.append(f"datos_abiertos_{year}_{month:02d}.csv")

        month += 1

        if month > 12:
            month = 1
            year += 1

    return filenames


def select_required_files(csv_urls: dict[str, str]) -> dict[str, str]:
    """Select and validate the files required for the project."""
    required_files = build_trip_filenames(
        START_YEAR,
        START_MONTH,
        END_YEAR,
        END_MONTH,
    )

    required_files.append(NOMENCLATURE_FILE)

    missing_files = [
        filename
        for filename in required_files
        if filename not in csv_urls
    ]

    if missing_files:
        raise ValueError(
            f"Required files not found in MiBici portal: {missing_files}"
        )

    selected_urls = {
        filename: csv_urls[filename]
        for filename in required_files
    }

    logger.success(
        f"Validated {len(selected_urls)} required files "
        f"({len(required_files) - 1} trip files + nomenclature)."
    )

    return selected_urls


def download_file(
    url: str,
    destination: Path,
    force: bool = False,
) -> Path:
    """Download a file from a URL using streaming.

    Existing files are skipped unless force=True.
    """
    if destination.exists() and not force:
        logger.info(f"Skipping existing file: {destination.name}")
        return destination

    logger.info(f"Downloading {destination.name}...")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/154.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }

    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=60,
    ) as response:
        response.raise_for_status()

        destination.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = destination.with_suffix(destination.suffix + ".part")

        try:
            with open(temporary_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)

            temporary_path.replace(destination)

        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()

            raise

    logger.success(f"Downloaded {destination.name}")

    return destination


def download_required_files(
    selected_urls: dict[str, str],
) -> list[Path]:
    """Download all required MiBici files to the raw data directory."""
    downloaded_files = []

    for filename, url in selected_urls.items():
        destination = RAW_DATA_DIR / filename
        file_path = download_file(url, destination)
        downloaded_files.append(file_path)

    logger.success(
        f"Download step completed for {len(downloaded_files)} required files."
    )

    return downloaded_files


def normalize_column_name(column: str) -> str:
    """Normalize a column name to lowercase snake_case without accents."""
    column = unicodedata.normalize("NFKD", column)
    column = column.encode("ascii", "ignore").decode("ascii")
    column = column.strip().lower()
    column = re.sub(r"[^a-z0-9]+", "_", column)

    return column.strip("_")


def read_mibici_csv(file_path: Path) -> pd.DataFrame:
    """Read a MiBici CSV trying the expected character encodings."""
    for encoding in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info(f"{file_path.name}: encoding={encoding}")
            return df
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode {file_path}")


def parse_datetime_column(series: pd.Series) -> pd.Series:
    """Parse MiBici datetime columns supporting known source formats."""
    values = series.astype("string").str.strip()

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    iso_mask = values.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
    dayfirst_mask = values.str.match(r"^\d{2}/\d{2}/\d{4}", na=False)

    parsed.loc[iso_mask] = pd.to_datetime(
        values.loc[iso_mask],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    parsed.loc[dayfirst_mask] = pd.to_datetime(
        values.loc[dayfirst_mask],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )

    return parsed


def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize schema and date columns of MiBici trip data."""
    df = df.copy()

    df.columns = [normalize_column_name(column) for column in df.columns]

    # February and May 2024 contain a malformed source header that
    # normalizes to ``a_ae_nacimiento`` instead of ``ano_de_nacimiento``.
    df = df.rename(
        columns={"a_ae_nacimiento": "ano_de_nacimiento"}
    )

    date_columns = ["inicio_del_viaje", "fin_del_viaje"]

    for column in date_columns:
        if column in df.columns:
            df[column] = parse_datetime_column(df[column])

    return df


def deduplicate_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate trip rows while preserving the first copy."""
    duplicate_count = int(df.duplicated().sum())

    if duplicate_count == 0:
        return df

    logger.warning(
        f"Removing {duplicate_count:,} exact duplicate trip rows."
    )

    return df.drop_duplicates().reset_index(drop=True)


def validate_trips(df: pd.DataFrame) -> None:
    """Validate the minimum schema and critical fields of MiBici trip data."""
    required_columns = {
        "viaje_id",
        "inicio_del_viaje",
        "fin_del_viaje",
        "origen_id",
        "destino_id",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    critical_columns = [
        "viaje_id",
        "inicio_del_viaje",
        "fin_del_viaje",
        "origen_id",
        "destino_id",
    ]

    null_counts = df[critical_columns].isna().sum()

    if (null_counts > 0).any():
        raise ValueError(
            f"Null values found in critical columns:\n{null_counts[null_counts > 0]}"
        )

    logger.success("Dataset validation passed.")


def clean_nomenclature(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the MiBici station nomenclature schema."""
    df = df.copy()

    df.columns = [normalize_column_name(column) for column in df.columns]

    df = df.rename(
        columns={
            "id": "station_id",
            "name": "station_name",
            "latitude": "latitude",
            "longitude": "longitude",
            "dpcapacity": "capacity",
        }
    )

    return df


def validate_nomenclature(df: pd.DataFrame) -> None:
    """Validate required MiBici station nomenclature fields."""
    required_columns = [
        "station_id",
        "station_name",
        "latitude",
        "longitude",
        "capacity",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required nomenclature columns: {missing_columns}"
        )

    null_counts = df[required_columns].isna().sum()

    if null_counts.any():
        raise ValueError(
            f"Null values found in nomenclature:\n{null_counts}"
        )

    if df["station_id"].duplicated().any():
        raise ValueError("Duplicate station IDs found in nomenclature.")

    if (df["capacity"] < 0).any():
        raise ValueError("Negative station capacity values found.")

    zero_capacity_count = (df["capacity"] == 0).sum()

    if zero_capacity_count > 0:
        logger.warning(
            f"Nomenclature contains {zero_capacity_count} stations "
            "with capacity equal to zero."
        )

    logger.success("Nomenclature validation passed.")


def remove_unexpected_trip_parquets(
    output_dir: Path,
    expected_filenames: list[str],
) -> None:
    """Remove stale Parquet files from the pipeline-managed trip directory."""
    managed_output_dir = (PROCESSED_DATA_DIR / "trips").resolve()

    if output_dir.resolve() != managed_output_dir:
        raise ValueError(
            f"Refusing to clean unexpected directory: {output_dir}"
        )

    expected_parquet_names = {
        filename.replace(".csv", ".parquet")
        for filename in expected_filenames
    }

    stale_files = sorted(
        file_path
        for file_path in output_dir.glob("*.parquet")
        if file_path.name not in expected_parquet_names
    )

    for file_path in stale_files:
        logger.warning(f"Removing stale Parquet file: {file_path.name}")
        file_path.unlink()


def process_trip_files() -> list[Path]:
    """Clean, validate, and convert monthly trip files to Parquet."""
    trip_filenames = build_trip_filenames(
        START_YEAR,
        START_MONTH,
        END_YEAR,
        END_MONTH,
    )

    output_dir = PROCESSED_DATA_DIR / "trips"
    output_dir.mkdir(parents=True, exist_ok=True)

    remove_unexpected_trip_parquets(output_dir, trip_filenames)

    processed_files = []

    for filename in trip_filenames:
        input_path = RAW_DATA_DIR / filename
        output_path = output_dir / filename.replace(".csv", ".parquet")

        logger.info(f"Processing {filename}...")

        df = read_mibici_csv(input_path)
        df = clean_trips(df)
        df = deduplicate_trips(df)
        validate_trips(df)

        df.to_parquet(
            output_path,
            index=False,
        )

        processed_files.append(output_path)

        logger.success(
            f"Processed {filename}: {len(df):,} rows."
        )

    return processed_files


def process_nomenclature() -> Path:
    """Clean, validate, and convert station nomenclature to Parquet."""
    input_path = RAW_DATA_DIR / NOMENCLATURE_FILE

    output_dir = PROCESSED_DATA_DIR / "nomenclature"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / NOMENCLATURE_FILE.replace(
        ".csv",
        ".parquet",
    )

    logger.info(f"Processing {NOMENCLATURE_FILE}...")

    df = read_mibici_csv(input_path)
    df = clean_nomenclature(df)
    validate_nomenclature(df)

    df.to_parquet(
        output_path,
        index=False,
    )

    logger.success(
        f"Processed nomenclature: {len(df):,} stations."
    )

    return output_path


def audit_trip_file(file_path: Path) -> dict:
    """Inspect one monthly MiBici trip file and return quality metrics."""
    logger.info(f"Auditing {file_path.name}...")

    df = read_mibici_csv(file_path)
    df = clean_trips(df)

    critical_columns = [
        "viaje_id",
        "inicio_del_viaje",
        "fin_del_viaje",
        "origen_id",
        "destino_id",
    ]

    missing_columns = [
        column
        for column in critical_columns
        if column not in df.columns
    ]

    null_counts = {
        column: int(df[column].isna().sum())
        for column in critical_columns
        if column in df.columns
    }

    return {
        "file": file_path.name,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "missing_columns": missing_columns,
        "critical_nulls": null_counts,
        "min_start_date": (
            df["inicio_del_viaje"].min()
            if "inicio_del_viaje" in df.columns
            else None
        ),
        "max_start_date": (
            df["inicio_del_viaje"].max()
            if "inicio_del_viaje" in df.columns
            else None
        ),
    }


def audit_trip_dataset() -> pd.DataFrame:
    """Audit all selected monthly MiBici trip files."""
    trip_filenames = build_trip_filenames(
        START_YEAR,
        START_MONTH,
        END_YEAR,
        END_MONTH,
    )

    audit_results = []

    for filename in trip_filenames:
        file_path = RAW_DATA_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required raw file not found: {file_path}"
            )

        result = audit_trip_file(file_path)
        audit_results.append(result)

    audit_df = pd.DataFrame(audit_results)

    logger.success(
        f"Audit completed for {len(audit_df)} monthly trip files."
    )

    return audit_df


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def calculate_dataset_version(manifest_files: list[dict]) -> str:
    """Create a deterministic version hash from all raw file hashes."""
    hash_input = "\n".join(
        f"{item['filename']}:{item['sha256']}"
        for item in sorted(manifest_files, key=lambda item: item["filename"])
    )

    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


def create_data_manifest(selected_urls: dict[str, str]) -> Path:
    """Create a manifest with metadata and SHA-256 hashes for raw MiBici files."""
    manifest_files = []

    for filename, source_url in sorted(selected_urls.items()):
        file_path = RAW_DATA_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Missing raw file: {file_path}")

        if filename.startswith("datos_abiertos_"):
            df = clean_trips(read_mibici_csv(file_path))
            rows = len(df)
        elif filename == NOMENCLATURE_FILE:
            df = clean_nomenclature(read_mibici_csv(file_path))
            rows = len(df)
        else:
            rows = None

        manifest_files.append(
            {
                "filename": filename,
                "source_url": source_url,
                "size_bytes": file_path.stat().st_size,
                "rows": rows,
                "sha256": calculate_sha256(file_path),
            }
        )

    dataset_version = calculate_dataset_version(manifest_files)

    manifest = {
        "dataset_version": dataset_version,
        "source": MIBICI_OPEN_DATA_URL,
        "trip_period": {
            "start": f"{START_YEAR}-{START_MONTH:02d}",
            "end": f"{END_YEAR}-{END_MONTH:02d}",
        },
        "nomenclature_file": NOMENCLATURE_FILE,
        "files": manifest_files,
    }

    manifest_path = PROJ_ROOT / "data_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    logger.success(
        f"Data manifest created: {manifest_path} "
        f"(dataset_version={dataset_version[:12]})"
    )

    return manifest_path


@app.command()
def main() -> None:
    """Run the complete MiBici data ingestion pipeline."""
    logger.info("Starting MiBici data ingestion pipeline...")

    # 1. Discover and select required source files
    available_urls = discover_download_urls()
    selected_urls = select_required_files(available_urls)

    # 2. Download raw data
    downloaded_files = download_required_files(selected_urls)

    # 3. Process trip data
    processed_trip_files = process_trip_files()

    # 4. Process station nomenclature
    processed_nomenclature = process_nomenclature()

    # 5. Create reproducible data manifest
    manifest_path = create_data_manifest(selected_urls)

    logger.success(
        "MiBici data ingestion pipeline completed successfully. "
        f"Raw files: {len(downloaded_files)} | "
        f"Trip Parquet files: {len(processed_trip_files)} | "
        f"Nomenclature: {processed_nomenclature.name} | "
        f"Manifest: {manifest_path.name}"
    )


if __name__ == "__main__":
    app()
