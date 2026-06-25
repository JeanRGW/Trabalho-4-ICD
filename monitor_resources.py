import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen
from zipfile import ZipFile

DATASET_PAGE_URL = "https://dados.gov.br/dados/conjuntos-dados/arboviroses-dengue"
DEFAULT_STATE_FILE = Path("resource_monitor_state.json")
MIN_RESOURCE_YEAR = 2024
S3_CSV_URL_TEMPLATE = (
    "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SINAN/Dengue/csv/DENGBR{year_short}.csv.zip"
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class CsvResource:
    key: str
    title: str
    update_date: Optional[str]
    catalog_date: Optional[str]
    download_url: str


def _filename(url: str) -> str:
    return Path(unquote(urlparse(url).path)).name


def _format_http_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).strftime("%d/%m/%Y")
    except Exception:
        return None


def _head_check(download_url: str) -> tuple[bool, Optional[str]]:
    request = Request(download_url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    try:
        with urlopen(request, timeout=30) as response:
            return True, _format_http_date(response.headers.get("Last-Modified"))
    except Exception:
        return False, None


def _collect_s3_csv_resources() -> List[CsvResource]:
    current_year = datetime.utcnow().year
    resources: List[CsvResource] = []

    for year in range(current_year, MIN_RESOURCE_YEAR - 1, -1):
        year_short = str(year)[-2:]
        download_url = S3_CSV_URL_TEMPLATE.format(year_short=year_short)
        exists, update_date = _head_check(download_url)
        if not exists:
            continue
        resources.append(CsvResource(
            key=f"dengue-{year}",
            title=f"Dengue - {year}",
            update_date=update_date,
            catalog_date=None,
            download_url=download_url,
        ))

    return resources


def _load_state(state_file: Path) -> Dict[str, Any]:
    if not state_file.exists():
        return {"resources": {}}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return {"resources": {}}


def _save_state(state_file: Path, resources: List[CsvResource]) -> None:
    state = {
        "source_url": DATASET_PAGE_URL,
        "checked_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "resources": {
            r.key: {
                "title": r.title,
                "update_date": r.update_date,
                "catalog_date": r.catalog_date,
                "download_url": r.download_url,
            }
            for r in resources
        },
    }
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _local_file_exists(download_url: str, data_dir: Path) -> bool:
    filename = _filename(download_url)
    if not filename:
        return False
    if (data_dir / filename).exists():
        return True
    return filename.lower().endswith(".zip") and (data_dir / filename[:-4]).exists()


def _download_and_extract(resource: CsvResource, data_dir: Path) -> List[Path]:
    filename = _filename(resource.download_url)
    if not filename:
        raise ValueError(f"Nao foi possivel identificar o nome do arquivo: {resource.download_url}")

    request = Request(resource.download_url, headers={"User-Agent": USER_AGENT})
    downloaded_path = data_dir / filename
    temp_path = downloaded_path.with_suffix(downloaded_path.suffix + ".part")

    with urlopen(request, timeout=180) as response, temp_path.open("wb") as temp_file:
        shutil.copyfileobj(response, temp_file)

    temp_path.replace(downloaded_path)
    if downloaded_path.suffix.lower() != ".zip":
        return [downloaded_path]

    extracted: List[Path] = []
    with ZipFile(downloaded_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".csv"):
                continue
            output_path = data_dir / Path(member.filename).name
            with zip_ref.open(member) as source, output_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted.append(output_path)

    try:
        downloaded_path.unlink()
    except OSError as exc:
        print(f"Aviso: nao foi possivel remover o zip {downloaded_path.name}: {exc}")

    return extracted


def _update_reason(resource: CsvResource, previous: Optional[Dict[str, Any]], data_dir: Path) -> str:
    if previous is None:
        return "novo recurso"
    if previous.get("download_url") != resource.download_url:
        return "link de download mudou"

    current_update = resource.update_date or ""
    previous_update = previous.get("update_date") or ""
    if current_update and current_update != previous_update:
        return "data de atualizacao mudou"
    if not _local_file_exists(resource.download_url, data_dir):
        return "arquivo local ausente"
    return ""


def sync_dengue_csv_resources(
    data_dir: Path = Path("./"),
    state_file: Path = DEFAULT_STATE_FILE,
) -> List[Path]:
    data_dir = Path(data_dir)
    state_file = Path(state_file)

    resources = _collect_s3_csv_resources()
    print(f"Recursos CSV encontrados: {len(resources)}")
    if not resources:
        print("Nenhum recurso CSV encontrado no S3 publico.")
        return []

    previous_resources = _load_state(state_file).get("resources", {})
    changed_count = 0
    updated_files: List[Path] = []

    for resource in resources:
        reason = _update_reason(resource, previous_resources.get(resource.key), data_dir)
        if not reason:
            continue

        changed_count += 1
        print(f"Atualizacao detectada: {resource.title} ({reason}).")
        updated_files.extend(_download_and_extract(resource, data_dir))

    _save_state(state_file, resources)
    print(f"Sincronizacao concluida. Recursos alterados={changed_count}.")
    return updated_files


if __name__ == "__main__":
    print("Iniciando monitoramento de recursos CSV do conjunto Arboviroses/Dengue...")
    sync_dengue_csv_resources()
