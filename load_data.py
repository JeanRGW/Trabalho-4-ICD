import csv
from datetime import datetime
from pathlib import Path

from config import close_connection, get_connection

BATCH_SIZE = 50_000
DATE_COLUMNS = {'dt_notific', 'dt_nasc', 'dt_invest', 'dt_sin_pri', 'dt_encerra'}


def _parse_date(value):
    if not value or not value.strip():
        return None
    text = value.strip()
    for fmt in ('%Y-%m-%d', '%d/%m/%Y'):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _convert_value(value):
    if not value or not value.strip():
        return None
    text = value.strip()
    if text.lstrip('-').replace('.', '', 1).isdigit():
        if '.' in text:
            return int(float(text))
        return int(text)
    return text


def _normalize_row(row):
    return {
        key.lower(): (_parse_date(value) if key.lower() in DATE_COLUMNS else _convert_value(value))
        for key, value in row.items()
    }


def _load_csv(connection, csv_file_path):
    cursor = connection.cursor()
    rows_inserted = 0
    rows_failed = 0
    try:
        with open(csv_file_path, 'r', encoding='utf-8-sig', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames:
                return 0, 0

            table_columns = None
            insert_sql = None
            batch = []

            for row in reader:
                processed = _normalize_row(row)
                if processed.get('dt_notific') is None:
                    rows_failed += 1
                    continue

                if table_columns is None:
                    table_columns = list(processed.keys())
                    cols = ', '.join(table_columns)
                    placeholders = ', '.join(['%s'] * len(table_columns))
                    insert_sql = f'INSERT INTO casos_dengue ({cols}) VALUES ({placeholders})'

                batch.append([processed[col] for col in table_columns])

                if len(batch) >= BATCH_SIZE:
                    cursor.executemany(insert_sql, batch)
                    connection.commit()
                    rows_inserted += len(batch)
                    print(f'  {rows_inserted} registros processados...')
                    batch.clear()

            if batch:
                cursor.executemany(insert_sql, batch)
                connection.commit()
                rows_inserted += len(batch)
                print(f'  {rows_inserted} registros processados (lote final).')
        return rows_inserted, rows_failed
    finally:
        cursor.close()


def load_all_files(data_dir='./', truncate=False, batch_size=BATCH_SIZE):
    csv_files = sorted(Path(data_dir).glob('DENGBR*.csv'))
    if not csv_files:
        print('Nenhum arquivo DENGBR*.csv encontrado.')
        return

    print(f'Encontrados {len(csv_files)} arquivos CSV. Batch={batch_size}.')

    connection = get_connection()
    if not connection:
        return

    try:
        cursor = connection.cursor()
        if truncate:
            print('  TRUNCATE TABLE casos_dengue')
            cursor.execute('TRUNCATE TABLE casos_dengue')
            connection.commit()
        cursor.close()

        total_inserted = 0
        total_failed = 0
        for csv_file in csv_files:
            print(f'\nCarregando {csv_file.name}...')
            inserted, failed = _load_csv(connection, csv_file)
            total_inserted += inserted
            total_failed += failed
            print(f'  Inseridos: {inserted} | Falhos: {failed}')

        print(f'\n--- Resumo Final ---')
        print(f'Inseridos: {total_inserted} | Falhos: {total_failed}')
    finally:
        close_connection(connection)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Carga de CSVs de dengue no MySQL.')
    parser.add_argument('--data-dir', default='./')
    parser.add_argument('--truncate', action='store_true')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    load_all_files(data_dir=args.data_dir, truncate=args.truncate, batch_size=args.batch_size)
