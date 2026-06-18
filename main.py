from pathlib import Path

from analytics import run_analysis
from database_schema import setup_database
from load_data import load_all_files
from monitor_resources import sync_dengue_csv_resources
from preprocess_csv import process_file_in_place

BASE_DIR = Path(__file__).resolve().parent


def step_0_monitor_resources():
    print('PASSO 0: Monitorando atualizacoes no portal dados.gov.br')
    try:
        updated_files = sync_dengue_csv_resources(
            data_dir=BASE_DIR,
            state_file=BASE_DIR / 'resource_monitor_state.json',
        )
        if updated_files:
            print(f'  Arquivos baixados/atualizados: {len(updated_files)}')
        else:
            print('  Nenhuma atualizacao detectada para download.')
    except Exception as exc:
        print(f'  Aviso: nao foi possivel monitorar recursos automaticamente: {exc}')
        print('  Continuando com os arquivos locais existentes.')
    print()


def step_1_get_data():
    print('PASSO 1: Validando dados')
    csv_files = sorted(BASE_DIR.glob('DENGBR*.csv'))
    print(f'  Encontrados {len(csv_files)} arquivos CSV\n')
    return len(csv_files) > 0


def step_2_process_data():
    print('PASSO 2: Processamento e Limpeza')
    csv_files = sorted(BASE_DIR.glob('DENGBR*.csv'))

    for file_path in csv_files:
        try:
            rows, kept_count, missing = process_file_in_place(file_path)
            print(f'  OK {file_path.name}: linhas={rows}, colunas={kept_count}')
            if missing:
                print(f'     ausentes: {", ".join(missing)}')
        except Exception as exc:
            print(f'  ERRO {file_path.name}: {exc}')
    print()


def step_3_create_schema():
    print('PASSO 3: Conexao e Schema MySQL')
    setup_database()
    print()


def step_4_load_data(truncate=False):
    print('PASSO 4: Carga de Dados')
    load_all_files(BASE_DIR, truncate=truncate)
    print()


def step_5_run_analysis():
    print('PASSO 5: Graficos e Estatisticas Descritivas')
    try:
        run_analysis(graficos_dir=BASE_DIR / 'graficos')
    except Exception as exc:
        print(f'  Erro no passo de analises descritivas: {exc}')
    print()


def step_6_run_ml_models():
    print('PASSO 6: Modelos de Machine Learning')
    try:
        from ml_models import run_ml_models
        run_ml_models()
    except Exception as exc:
        print(f'  Erro no passo de ML: {exc}')
    print()


def _confirm(prompt):
    answer = input(prompt).strip().lower()
    return answer in ('s', 'sim', 'y', 'yes')


def main():
    print('=== Sistema de Carregamento de Dados de Dengue ===\n')

    step_0_monitor_resources()

    if not step_1_get_data():
        print('Nenhum arquivo de dados encontrado.')
        return

    step_2_process_data()

    if _confirm('Deseja inserir/atualizar os dados no banco de dados? (s/n): '):
        step_3_create_schema()
        step_4_load_data(truncate=_confirm('Deseja limpar a tabela casos_dengue antes da carga? (s/n): '))
    else:
        print('Carga no banco ignorada pelo usuario. Seguindo sem inserir dados.')

    step_5_run_analysis()
    step_6_run_ml_models()

    print('=== Processo concluido com sucesso ===')


if __name__ == '__main__':
    main()
