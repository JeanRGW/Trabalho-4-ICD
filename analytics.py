import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from config import close_connection, get_connection
from helpers import convert_age_to_years, normalize_uf_short

SEX_LABEL_MAP = {
    'M': 'Masculino', 'F': 'Feminino', 'I': 'Ignorado', 'IGN': 'Ignorado',
}


def _normalize_filter(value):
    value = value.strip()
    return value or None


def _build_where_clause(uf_filter, year_filter):
    clauses = []
    params = []
    if uf_filter:
        clauses.append('sg_uf_not = %s')
        params.append(uf_filter)
    if year_filter:
        clauses.append('YEAR(dt_notific) = %s')
        params.append(year_filter)
    if not clauses:
        return '', params
    return 'WHERE ' + ' AND '.join(clauses), params


def _fetch(connection, query, params):
    cursor = connection.cursor()
    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return pd.DataFrame(rows, columns=columns)
    finally:
        cursor.close()


def show_line_chart(df_line, output_path=None):
    max_value = df_line['total_casos'].max()
    if max_value >= 1_000_000:
        scale, unit = 1_000_000, 'milhoes de casos'
    elif max_value >= 1_000:
        scale, unit = 1_000, 'mil casos'
    else:
        scale, unit = 1, 'casos'

    df_plot = df_line.assign(valor_escalado=df_line['total_casos'] / scale)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_plot['periodo'], df_plot['valor_escalado'], marker='o')
    ax.set_title(f'Evolucao mensal de casos de dengue (medida: {unit})')
    ax.set_xlabel('Periodo (AAAA-MM)')
    ax.set_ylabel(f'Total ({unit})')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'.rstrip('0').rstrip('.')))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _autopct_with_count(values):
    total = values.sum()

    def _fmt(pct):
        if pct < 2:
            return ''
        count = int(round((pct / 100.0) * total))
        return f'{pct:.1f}%\n(casos={count})'

    return _fmt


def _map_sex_label(value):
    if pd.isna(value):
        return 'Ignorado'
    return SEX_LABEL_MAP.get(str(value).strip().upper(), str(value).strip().upper())


def show_pie_chart(df_sex, output_path=None):
    df_plot = df_sex.assign(sexo_legivel=df_sex['sexo'].apply(_map_sex_label))
    values = df_plot['total_casos']
    labels = df_plot['sexo_legivel']

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, _, _ = ax.pie(
        values,
        labels=None,
        autopct=_autopct_with_count(values),
        startangle=90,
        pctdistance=0.65,
        textprops={'fontsize': 11},
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
    )
    legend_labels = [f'{label} - {int(count)} casos' for label, count in zip(labels, values)]
    ax.legend(wedges, legend_labels, title='Sexo (medida: casos)', loc='center left', bbox_to_anchor=(1, 0.5))
    ax.axis('equal')
    plt.title('Distribuicao de casos por sexo (medida: percentual e contagem)')
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def show_statistics(df_stats):
    if df_stats.empty:
        print('Nenhum dado encontrado para calcular estatisticas.')
        return

    print('\n--- Estatisticas (pandas) ---')
    print(f'Total de registros na consulta: {len(df_stats)}')

    if 'nu_idade_n' in df_stats.columns:
        idade_anos = df_stats['nu_idade_n'].apply(convert_age_to_years).dropna()
        if not idade_anos.empty:
            print('\nIdade (anos) - resumo:')
            print(f'  media: {idade_anos.mean():.1f}')
            print(f'  mediana: {idade_anos.median():.1f}')
            print(f'  minimo: {idade_anos.min():.1f}')
            print(f'  maximo: {idade_anos.max():.1f}')
        else:
            print('\nIdade (anos) - sem dados validos para resumo.')

    if 'sem_not' in df_stats.columns:
        sem_validas = pd.to_numeric(df_stats['sem_not'], errors='coerce').dropna().astype(int)
        if not sem_validas.empty:
            print('\nSemanas epidemiologicas:')
            print(f'  primeira: {sem_validas.min()}')
            print(f'  ultima: {sem_validas.max()}')
            print(f'  quantidade de semanas distintas: {sem_validas.nunique()}')

    if 'cs_sexo' in df_stats.columns:
        print('\nContagem por sexo:')
        sexo_legivel = df_stats['cs_sexo'].apply(_map_sex_label)
        sexo_counts = sexo_legivel.value_counts(dropna=False)
        sexo_pct = (sexo_counts / sexo_counts.sum() * 100).round(2)
        print(pd.DataFrame({'casos': sexo_counts, 'percentual_%': sexo_pct}).to_string())

    if 'sg_uf_not' in df_stats.columns:
        print('\nTop 10 UFs por casos:')
        uf_legivel = df_stats['sg_uf_not'].apply(normalize_uf_short)
        print(uf_legivel.value_counts(dropna=False).head(10).to_string())


def run_analysis(graficos_dir=None):
    print('PASSO 5: Graficos e Estatisticas')

    if graficos_dir is not None:
        graficos_dir.mkdir(parents=True, exist_ok=True)

    uf_filter = _normalize_filter(input('Filtro por UF (ex: SP) [Enter para todas]: '))
    year_input = _normalize_filter(input('Filtro por ano (ex: 2024) [Enter para todos]: '))

    year_filter = None
    if year_input:
        if not year_input.isdigit() or len(year_input) != 4:
            print('Ano invalido. Use 4 digitos, exemplo 2024.')
            print()
            return
        year_filter = int(year_input)

    connection = get_connection()
    if not connection:
        print('Nao foi possivel conectar no banco para gerar analises.')
        print()
        return

    try:
        where_clause, params = _build_where_clause(uf_filter, year_filter)
        df_line = _fetch(connection, f'''
            SELECT DATE_FORMAT(dt_notific, '%Y-%m') AS periodo, COUNT(*) AS total_casos
            FROM casos_dengue
            {where_clause}
            GROUP BY DATE_FORMAT(dt_notific, '%Y-%m')
            ORDER BY periodo
        ''', params)
        df_sex = _fetch(connection, f'''
            SELECT COALESCE(cs_sexo, 'IGN') AS sexo, COUNT(*) AS total_casos
            FROM casos_dengue
            {where_clause}
            GROUP BY COALESCE(cs_sexo, 'IGN')
            ORDER BY total_casos DESC
        ''', params)
        df_stats = _fetch(connection, f'''
            SELECT sem_not, nu_idade_n, cs_sexo, sg_uf_not
            FROM casos_dengue
            {where_clause}
        ''', params)

        if df_line.empty or df_sex.empty:
            print('Nenhum dado encontrado para os filtros informados.')
            print()
            return

        show_line_chart(
            df_line,
            output_path=graficos_dir / 'evolucao_mensal.png' if graficos_dir else None,
        )
        show_pie_chart(
            df_sex,
            output_path=graficos_dir / 'distribuicao_sexo.png' if graficos_dir else None,
        )
        show_statistics(df_stats)
    except Exception as exc:
        print(f'Erro ao gerar analises: {exc}')
    finally:
        close_connection(connection)

    print()
