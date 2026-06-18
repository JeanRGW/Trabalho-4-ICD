import csv
from pathlib import Path
from tempfile import NamedTemporaryFile

KEEP_COLUMNS = [
    "dt_notific",
    "sem_not",
    "sg_uf_not",
    "dt_nasc",
    "nu_idade_n",
    "cs_sexo",
    "cs_gestant",
    "cs_raca",
    "cs_escol_n",
    "sg_uf",
    "febre",
    "mialgia",
    "cefaleia",
    "exantema",
    "vomito",
    "nausea",
    "dor_costas",
    "conjuntvit",
    "artrite",
    "artralgia",
    "petequia_n",
    "leucopenia",
    "laco",
    "dor_retro",
    "diabetes",
    "hematolog",
    "hepatopat",
    "renal",
    "hipertensa",
    "acido_pept",
    "auto_imune",
    "dt_sin_pri",
    "hospitaliz",
    "classi_fin",
    "criterio",
    "evolucao",
    "dt_encerra",
]

ENCODING_CANDIDATES = ('utf-8-sig', 'utf-8', 'latin-1', 'cp1252')


def _detect_encoding(file_path):
    for enc in ENCODING_CANDIDATES:
        try:
            with file_path.open('r', encoding=enc) as f:
                f.read(8192)
            return enc
        except UnicodeDecodeError:
            continue
    return 'latin-1'


def detect_dialect(file_path, encoding):
    with file_path.open('r', encoding=encoding, newline='') as f:
        sample = f.read(8192)

    try:
        return csv.Sniffer().sniff(sample, delimiters=',;\t|')
    except csv.Error:
        return csv.get_dialect('excel')


def resolve_columns(fieldnames):
    normalized = {name.strip().lower(): name for name in fieldnames}

    available = [col for col in KEEP_COLUMNS if col in normalized]
    source_map = {col: normalized[col] for col in available}
    missing = [col for col in KEEP_COLUMNS if col not in normalized]

    return available, source_map, missing


def process_file_in_place(file_path):
    encoding = _detect_encoding(file_path)
    dialect = detect_dialect(file_path, encoding)

    with file_path.open('r', encoding=encoding, newline='') as src:
        reader = csv.DictReader(src, dialect=dialect)

        if not reader.fieldnames:
            raise ValueError('No header found')

        available, source_map, missing = resolve_columns(reader.fieldnames)

        with NamedTemporaryFile('w', delete=False, encoding='utf-8', newline='', dir=file_path.parent) as tmp:
            tmp_path = Path(tmp.name)
            writer = csv.DictWriter(tmp, fieldnames=available, dialect=dialect, extrasaction='ignore')
            writer.writeheader()

            rows = 0
            for row in reader:
                out_row = {col: row.get(source_map[col], '') for col in available}
                writer.writerow(out_row)
                rows += 1

    tmp_path.replace(file_path)
    return rows, len(available), missing


def process_all_files(data_dir=Path('./')):
    csv_files = sorted(data_dir.glob('DENGBR*.csv'))

    for file_path in csv_files:
        try:
            rows, kept_count, missing = process_file_in_place(file_path)
            print(f'OK {file_path.name}: linhas={rows}, colunas={kept_count}')
            if missing:
                print(f'   ausentes: {", ".join(missing)}')
        except Exception as exc:
            print(f'ERRO {file_path.name}: {exc}')


if __name__ == '__main__':
    process_all_files()
