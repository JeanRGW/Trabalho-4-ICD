from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, recall_score, r2_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import close_connection, get_connection
from helpers import (
    UF_CODE_TO_SIGLA, UF_TO_REGION, clean_binary_code, convert_age_to_years,
    normalize_uf,
)

BASE_DIR = Path(__file__).resolve().parent
GRAFICOS_DIR = BASE_DIR / 'graficos'
METRICAS_FILE = BASE_DIR / 'metricas_modelos.txt'
SEED = 42
CASE_SAMPLE_POSITIVE_LIMIT = 30_000
CASE_SAMPLE_NEGATIVE_LIMIT = 90_000

NUMERIC_FEATURES = ['idade_anos', 'dias_ate_notificacao']
CATEGORICAL_FEATURES = ['cs_sexo', 'cs_raca', 'cs_escol_n', 'sg_uf', 'cs_gestant']
BINARY_SYMPTOMS = [
    'febre', 'mialgia', 'cefaleia', 'exantema', 'vomito', 'nausea',
    'dor_costas', 'conjuntvit', 'artrite', 'artralgia',
    'petequia_n', 'leucopenia', 'laco', 'dor_retro',
]
BINARY_COMORBIDITIES = [
    'diabetes', 'hematolog', 'hepatopat', 'renal',
    'hipertensa', 'acido_pept', 'auto_imune',
]
BINARY_FEATURES = BINARY_SYMPTOMS + BINARY_COMORBIDITIES
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

BRAZIL_STATES_GEOJSON_URL = (
    'https://raw.githubusercontent.com/codeforamerica/click_that_hood/'
    'master/public/data/brazil-states.geojson'
)
REGION_ORDER = ['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul']


def _log_metrics(block_title, lines):
    METRICAS_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with METRICAS_FILE.open('a', encoding='utf-8') as fp:
        fp.write(f'\n=== {block_title} ({timestamp}) ===\n')
        for line in lines:
            fp.write(f'{line}\n')
        fp.write('\n')


def _save_figure(fig, name):
    GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRAFICOS_DIR / name, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _to_dense_if_needed(matrix):
    if hasattr(matrix, 'toarray'):
        return matrix.toarray()
    return matrix


def _run_query(connection, query, params=()):
    cursor = connection.cursor()
    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        cols = [c[0] for c in cursor.description]
        return pd.DataFrame(rows, columns=cols)
    finally:
        cursor.close()


def fetch_total_record_count():
    connection = get_connection()
    if not connection:
        raise RuntimeError('Nao foi possivel conectar ao MySQL para contar registros.')
    try:
        df = _run_query(
            connection,
            'SELECT COUNT(*) AS n FROM casos_dengue WHERE dt_notific IS NOT NULL',
        )
        return int(df['n'].iloc[0])
    finally:
        close_connection(connection)


def fetch_case_modeling_dataframe_sampled(
    positive_limit=CASE_SAMPLE_POSITIVE_LIMIT,
    negative_limit=CASE_SAMPLE_NEGATIVE_LIMIT,
):
    connection = get_connection()
    if not connection:
        raise RuntimeError('Nao foi possivel conectar ao MySQL para carregar amostra de classificacao.')

    select_cols = '''
        dt_notific, dt_sin_pri, sg_uf, cs_sexo, cs_gestant, cs_raca, cs_escol_n,
        nu_idade_n, febre, mialgia, cefaleia, exantema, vomito, nausea,
        dor_costas, conjuntvit, artrite, artralgia, petequia_n, leucopenia,
        laco, dor_retro, diabetes, hematolog, hepatopat, renal, hipertensa,
        acido_pept, auto_imune, hospitaliz
    '''
    query = f'''
        SELECT * FROM (
            SELECT {select_cols}
            FROM casos_dengue
            WHERE dt_notific IS NOT NULL AND hospitaliz = '1'
            LIMIT %s
        ) positivos
        UNION ALL
        SELECT * FROM (
            SELECT {select_cols}
            FROM casos_dengue
            WHERE dt_notific IS NOT NULL AND hospitaliz = '2'
            LIMIT %s
        ) negativos
    '''
    try:
        return _run_query(connection, query, (positive_limit, negative_limit))
    finally:
        close_connection(connection)


def fetch_monthly_aggregate_dataframe():
    connection = get_connection()
    if not connection:
        raise RuntimeError('Nao foi possivel conectar ao MySQL para carregar agregado mensal.')

    age_expr = '''
        CASE
            WHEN nu_idade_n IS NULL OR nu_idade_n < 0 THEN NULL
            WHEN nu_idade_n < 1000 THEN nu_idade_n
            WHEN FLOOR(nu_idade_n / 1000) = 1 THEN MOD(nu_idade_n, 1000) / (24 * 365)
            WHEN FLOOR(nu_idade_n / 1000) = 2 THEN MOD(nu_idade_n, 1000) / 365
            WHEN FLOOR(nu_idade_n / 1000) = 3 THEN MOD(nu_idade_n, 1000) / 12
            WHEN FLOOR(nu_idade_n / 1000) = 4 THEN MOD(nu_idade_n, 1000)
            ELSE NULL
        END
    '''

    def pct_expr(col):
        return f"AVG(CASE WHEN {col} = '1' THEN 1 WHEN {col} = '2' THEN 0 ELSE NULL END) * 100"

    query = f'''
        SELECT
            YEAR(dt_notific) AS ano,
            MONTH(dt_notific) AS mes,
            COALESCE(sg_uf, 'IGN') AS sg_uf,
            COUNT(*) AS total_casos,
            AVG(CASE WHEN ({age_expr}) BETWEEN 0 AND 120 THEN ({age_expr}) ELSE NULL END) AS idade_media,
            {pct_expr('febre')} AS pct_febre,
            {pct_expr('mialgia')} AS pct_mialgia,
            {pct_expr('cefaleia')} AS pct_cefaleia,
            {pct_expr('vomito')} AS pct_vomito,
            {pct_expr('nausea')} AS pct_nausea,
            {pct_expr('hospitaliz')} AS pct_hospitaliz,
            SUM(CASE WHEN febre = '1' THEN 1 ELSE 0 END) AS casos_com_febre
        FROM casos_dengue
        WHERE dt_notific IS NOT NULL
        GROUP BY YEAR(dt_notific), MONTH(dt_notific), COALESCE(sg_uf, 'IGN')
        ORDER BY ano, mes, sg_uf
    '''

    try:
        df = _run_query(connection, query)
        if not df.empty:
            numeric_cols = [
                'ano', 'mes', 'total_casos', 'idade_media', 'pct_febre', 'pct_mialgia',
                'pct_cefaleia', 'pct_vomito', 'pct_nausea', 'pct_hospitaliz', 'casos_com_febre',
            ]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['periodo'] = (df['ano'] - df['ano'].min()) * 12 + df['mes']
            df['mes_indice'] = (df['periodo'] - df['periodo'].min()).astype(int)
        return df
    finally:
        close_connection(connection)


def prepare_case_level_data(df_raw):
    total_inicial = len(df_raw)
    df = df_raw.copy()

    df['hospitaliz_num'] = df['hospitaliz'].apply(clean_binary_code)
    removidos_alvo = total_inicial - df['hospitaliz_num'].notna().sum()
    df = df.dropna(subset=['hospitaliz_num'])

    df['dt_notific'] = pd.to_datetime(df['dt_notific'], errors='coerce')
    df['dt_sin_pri'] = pd.to_datetime(df['dt_sin_pri'], errors='coerce')
    df['dias_ate_notificacao'] = (df['dt_notific'] - df['dt_sin_pri']).dt.days
    df.loc[
        (df['dias_ate_notificacao'] < 0) | (df['dias_ate_notificacao'] > 365),
        'dias_ate_notificacao',
    ] = np.nan

    df['idade_anos'] = df['nu_idade_n'].apply(convert_age_to_years)
    for col in BINARY_FEATURES:
        df[col] = df[col].apply(clean_binary_code)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna('IGN').astype(str)

    df_features = df[ALL_FEATURES + ['hospitaliz_num']]
    df_clean = df_features.dropna()
    removidos_features = len(df_features) - len(df_clean)

    y = df_clean['hospitaliz_num'].astype(int).values
    X = df_clean[ALL_FEATURES].copy()

    proporcao_positivo = float(y.mean()) if len(y) else 0.0
    log_lines = [
        f'Total inicial de registros: {total_inicial}',
        f'Removidos por alvo ausente (hospitaliz invalido): {removidos_alvo}',
        f'Removidos por features NaN: {removidos_features}',
        f'Dataset final: {len(X)} registros',
        f'Proporcao da classe positiva (hospitalizado): {proporcao_positivo:.4f}',
        f'Features numericas: {NUMERIC_FEATURES}',
        f'Features categoricas: {CATEGORICAL_FEATURES}',
        f'Features binarias (sintomas): {BINARY_SYMPTOMS}',
        f'Features binarias (comorbidades): {BINARY_COMORBIDITIES}',
        'UF usada nos modelos: sg_uf (residencia do paciente)',
    ]
    return X, y, log_lines


def _build_preprocessor():
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
    ])
    return ColumnTransformer([
        ('num', numeric_pipeline, NUMERIC_FEATURES + BINARY_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def _split(X, y, test_size=0.3):
    return train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y,
    )


def _split_regression(X, y, test_size=0.3):
    return train_test_split(X, y, test_size=test_size, random_state=SEED)


def _build_knn_geo_summary(X_test, y_test, y_pred):
    if 'sg_uf' not in X_test.columns:
        return pd.DataFrame(), pd.DataFrame()

    df_geo = pd.DataFrame({
        'uf': X_test['sg_uf'].apply(normalize_uf).values,
        'real': np.asarray(y_test, dtype=float),
        'previsto': np.asarray(y_pred, dtype=float),
    }).dropna(subset=['uf'])

    if df_geo.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_geo['regiao'] = df_geo['uf'].map(UF_TO_REGION)

    by_uf = df_geo.groupby('uf').agg(
        total_registros=('real', 'size'),
        taxa_real=('real', 'mean'),
        taxa_prevista_knn=('previsto', 'mean'),
    ).reset_index()
    by_uf['erro_abs'] = (by_uf['taxa_real'] - by_uf['taxa_prevista_knn']).abs()
    by_uf['regiao'] = by_uf['uf'].map(UF_TO_REGION)

    by_region = df_geo.groupby('regiao').agg(
        total_registros=('real', 'size'),
        taxa_real=('real', 'mean'),
        taxa_prevista_knn=('previsto', 'mean'),
    ).reset_index()
    by_region['erro_abs'] = (by_region['taxa_real'] - by_region['taxa_prevista_knn']).abs()
    by_region['ordem'] = by_region['regiao'].map({r: i for i, r in enumerate(REGION_ORDER)})
    by_region = by_region.sort_values('ordem').drop(columns='ordem')

    return by_uf, by_region


def _plot_knn_region_predictions(by_region):
    if by_region.empty:
        return False

    df_plot = by_region.copy()
    df_plot['taxa_real_pct'] = df_plot['taxa_real'] * 100
    df_plot['taxa_prevista_pct'] = df_plot['taxa_prevista_knn'] * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_plot))
    width = 0.35
    ax.bar(x - width / 2, df_plot['taxa_real_pct'], width, label='Taxa real', color='#4C72B0')
    ax.bar(x + width / 2, df_plot['taxa_prevista_pct'], width, label='Taxa prevista KNN', color='#DD8452')
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['regiao'])
    ax.set_ylabel('Hospitalizacao (%)')
    ax.set_title('KNN: taxa real vs prevista de hospitalizacao por regiao')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for i, row in df_plot.iterrows():
        ax.text(i - width / 2, row['taxa_real_pct'] + 0.15, f"{row['taxa_real_pct']:.1f}%", ha='center', fontsize=8)
        ax.text(i + width / 2, row['taxa_prevista_pct'] + 0.15, f"{row['taxa_prevista_pct']:.1f}%", ha='center', fontsize=8)
    plt.tight_layout()
    _save_figure(fig, 'knn_previsao_por_regiao.png')
    return True


def _plot_knn_brazil_map(by_uf, metric, title, output_name, cmap):
    if by_uf.empty:
        return False

    try:
        import geopandas as gpd
    except ImportError:
        print('  Aviso: geopandas nao instalado; mapas do KNN nao foram gerados.')
        return False

    try:
        states = gpd.read_file(BRAZIL_STATES_GEOJSON_URL)
        states['uf'] = states['sigla'].astype(str).str.upper()
        plot_data = states.merge(by_uf, on='uf', how='left')
        plot_data[f'{metric}_pct'] = plot_data[metric] * 100

        fig, ax = plt.subplots(figsize=(10, 10))
        plot_data.plot(
            column=f'{metric}_pct',
            cmap=cmap,
            linewidth=0.6,
            edgecolor='white',
            legend=True,
            missing_kwds={'color': '#eeeeee', 'label': 'Sem dados'},
            ax=ax,
        )
        plot_data.boundary.plot(ax=ax, color='#555555', linewidth=0.3)

        for _, row in plot_data.dropna(subset=[f'{metric}_pct']).iterrows():
            point = row.geometry.representative_point()
            ax.text(point.x, point.y, row['uf'], ha='center', va='center', fontsize=7, color='black')

        ax.set_title(title)
        ax.set_axis_off()
        plt.tight_layout()
        _save_figure(fig, output_name)
        return True
    except Exception as exc:
        print(f'  Aviso: mapa {output_name} nao gerado: {exc}')
        return False


def _build_knn_geographic_outputs(X_test, y_test, y_pred):
    by_uf, by_region = _build_knn_geo_summary(X_test, y_test, y_pred)
    if by_uf.empty:
        print('  Aviso: sem UFs validas para gerar visualizacoes geograficas do KNN.')
        return ['Visualizacoes geograficas KNN: sem UFs validas no conjunto de teste.']

    generated_outputs = []
    if _plot_knn_region_predictions(by_region):
        generated_outputs.append('  graficos/knn_previsao_por_regiao.png')
    if _plot_knn_brazil_map(
        by_uf,
        metric='taxa_prevista_knn',
        title='KNN: taxa prevista de hospitalizacao por UF',
        output_name='knn_mapa_brasil_taxa_prevista.png',
        cmap='OrRd',
    ):
        generated_outputs.append('  graficos/knn_mapa_brasil_taxa_prevista.png')
    if _plot_knn_brazil_map(
        by_uf,
        metric='erro_abs',
        title='KNN: erro absoluto entre taxa real e prevista por UF',
        output_name='knn_mapa_brasil_erro_abs.png',
        cmap='Purples',
    ):
        generated_outputs.append('  graficos/knn_mapa_brasil_erro_abs.png')

    top_previstas = by_uf.sort_values('taxa_prevista_knn', ascending=False).head(5)
    top_erros = by_uf.sort_values('erro_abs', ascending=False).head(5)
    return [
        'Visualizacoes geograficas KNN:',
        *(generated_outputs or ['  Nenhum arquivo geografico gerado; ver avisos no console.']),
        'Top 5 UFs por taxa prevista KNN:',
        top_previstas[['uf', 'total_registros', 'taxa_real', 'taxa_prevista_knn', 'erro_abs']].to_string(index=False),
        'Top 5 UFs por erro absoluto:',
        top_erros[['uf', 'total_registros', 'taxa_real', 'taxa_prevista_knn', 'erro_abs']].to_string(index=False),
        'Resumo por regiao:',
        by_region[['regiao', 'total_registros', 'taxa_real', 'taxa_prevista_knn', 'erro_abs']].to_string(index=False),
    ]


def _classification_metrics(y_test, y_pred, y_proba=None):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = float('nan')
    if y_proba is not None and len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_proba)
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc}


def run_knn_model(X, y, dataset_info):
    print('\n[KNN] Iniciando treinamento...')

    X_train, X_test, y_train, y_test = _split(X, y)

    pipe = Pipeline([
        ('pre', _build_preprocessor()),
        ('knn', KNeighborsClassifier()),
    ])

    param_grid = {
        'knn__n_neighbors': [5, 9, 15],
        'knn__weights': ['uniform', 'distance'],
    }
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='f1', n_jobs=1)
    grid.fit(X_train, y_train)

    best_k = grid.best_params_['knn__n_neighbors']
    best_weights = grid.best_params_['knn__weights']
    print(f'  Melhor k={best_k}, weights={best_weights}, F1 CV={grid.best_score_:.4f}')

    y_pred = grid.predict(X_test)
    geo_lines = _build_knn_geographic_outputs(X_test, y_test, y_pred)

    try:
        y_proba = grid.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    m = _classification_metrics(y_test, y_pred, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f'  Acuracia={m["acc"]:.4f}  Precisao={m["prec"]:.4f}  Recall={m["rec"]:.4f}  F1={m["f1"]:.4f}  AUC={m["auc"]:.4f}')

    scores = pd.DataFrame(grid.cv_results_)
    scores_por_k = (
        scores[scores['param_knn__weights'] == best_weights]
        .groupby('param_knn__n_neighbors')['mean_test_score'].mean().sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(scores_por_k.index.astype(str), scores_por_k.values, color='steelblue')
    ax.set_xlabel('k (n_neighbors)')
    ax.set_ylabel('F1 medio (3-fold CV)')
    ax.set_title(f'KNN: F1 por k (weights={best_weights})')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(scores_por_k.values):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    _save_figure(fig, 'knn_acuracia_por_k.png')

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Nao hospitalizado', 'Hospitalizado'],
        yticklabels=['Nao hospitalizado', 'Hospitalizado'],
        ax=ax,
    )
    ax.set_xlabel('Previsto')
    ax.set_ylabel('Real')
    ax.set_title(f'Matriz de confusao KNN (k={best_k}, weights={best_weights})')
    plt.tight_layout()
    _save_figure(fig, 'knn_matriz_confusao.png')

    try:
        preprocessor = grid.best_estimator_.named_steps['pre']
        X_train_trans = _to_dense_if_needed(preprocessor.transform(X_train))
        X_test_trans = _to_dense_if_needed(preprocessor.transform(X_test))
        pca = PCA(n_components=2, random_state=SEED)
        pca.fit(X_train_trans)
        X_test_pca = pca.transform(X_test_trans)

        fig, ax = plt.subplots(figsize=(9, 7))
        scatter = ax.scatter(
            X_test_pca[:, 0], X_test_pca[:, 1],
            c=y_test, cmap='coolwarm', alpha=0.5, s=10, edgecolors='none',
        )
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('PCA 2D das features (colorido por classe real)')
        handles, _ = scatter.legend_elements()
        ax.legend(handles, ['Nao hospitalizado', 'Hospitalizado'], title='Classe real')
        plt.tight_layout()
        _save_figure(fig, 'knn_pca_classes.png')
    except Exception as exc:
        print(f'  Aviso: PCA nao gerado: {exc}')

    _log_metrics('KNN', [
        *dataset_info,
        f'Treino: {len(X_train)} | Teste: {len(X_test)}',
        f'Melhor k: {best_k}',
        f'Melhor weights: {best_weights}',
        f'Melhor F1 (CV): {grid.best_score_:.4f}',
        f'Acuracia (teste): {m["acc"]:.4f}',
        f'Precisao (teste): {m["prec"]:.4f}',
        f'Recall (teste): {m["rec"]:.4f}',
        f'F1-score (teste): {m["f1"]:.4f}',
        f'AUC (teste): {m["auc"]:.4f}',
        f'Matriz de confusao:\n{cm}',
        *geo_lines,
    ])
    return {
        'nome': f'KNN (k={best_k}, {best_weights})',
        'accuracy': m['acc'], 'precision': m['prec'], 'recall': m['rec'],
        'f1': m['f1'], 'auc': m['auc'],
    }


def run_logistic_regression_model(X, y, dataset_info):
    print('\n[Regressao Logistica] Iniciando treinamento...')

    X_train, X_test, y_train, y_test = _split(X, y)

    pipe = Pipeline([
        ('pre', _build_preprocessor()),
        ('lr', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=SEED,
            solver='lbfgs',
        )),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    m = _classification_metrics(y_test, y_pred, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f'  Acuracia={m["acc"]:.4f}  Precisao={m["prec"]:.4f}  Recall={m["rec"]:.4f}  F1={m["f1"]:.4f}  AUC={m["auc"]:.4f}')

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Greens',
        xticklabels=['Nao hospitalizado', 'Hospitalizado'],
        yticklabels=['Nao hospitalizado', 'Hospitalizado'],
        ax=ax,
    )
    ax.set_xlabel('Previsto')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de confusao - Regressao Logistica')
    plt.tight_layout()
    _save_figure(fig, 'logistica_matriz_confusao.png')

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {m["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Aleatorio')
    ax.set_xlabel('Falso positivo')
    ax.set_ylabel('Verdadeiro positivo')
    ax.set_title('Curva ROC - Regressao Logistica')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_figure(fig, 'logistica_curva_roc.png')

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(y_proba[y_test == 0], bins=40, alpha=0.6, label='Nao hospitalizado', color='steelblue')
    ax.hist(y_proba[y_test == 1], bins=40, alpha=0.6, label='Hospitalizado', color='indianred')
    ax.set_xlabel('Probabilidade prevista de hospitalizacao')
    ax.set_ylabel('Frequencia')
    ax.set_title('Distribuicao das probabilidades por classe real')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_figure(fig, 'logistica_probabilidades.png')

    try:
        feature_names = pipe.named_steps['pre'].get_feature_names_out()
        coefs = pipe.named_steps['lr'].coef_[0]
        top = pd.DataFrame({'feature': feature_names, 'coef': coefs})
        top['abs_coef'] = top['coef'].abs()
        top = top.sort_values('abs_coef', ascending=False).head(15).sort_values('coef')

        fig, ax = plt.subplots(figsize=(9, 7))
        colors = ['indianred' if v < 0 else 'seagreen' for v in top['coef']]
        ax.barh(top['feature'], top['coef'], color=colors)
        ax.set_xlabel('Coeficiente (log-odds)')
        ax.set_title('Top 15 coeficientes - Regressao Logistica')
        ax.axvline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        _save_figure(fig, 'logistica_coeficientes.png')
    except Exception as exc:
        print(f'  Aviso: grafico de coeficientes nao gerado: {exc}')

    _log_metrics('Regressao Logistica', [
        *dataset_info,
        f'Treino: {len(X_train)} | Teste: {len(X_test)}',
        'Modelo: LogisticRegression(max_iter=1000, class_weight=balanced)',
        f'Acuracia: {m["acc"]:.4f}',
        f'Precisao: {m["prec"]:.4f}',
        f'Recall: {m["rec"]:.4f}',
        f'F1-score: {m["f1"]:.4f}',
        f'AUC: {m["auc"]:.4f}',
        f'Matriz de confusao:\n{cm}',
    ])
    return {
        'nome': 'Regressao Logistica',
        'accuracy': m['acc'], 'precision': m['prec'], 'recall': m['rec'],
        'f1': m['f1'], 'auc': m['auc'],
    }


def run_simple_linear_regression(df_agg, dataset_info):
    print('\n[Regressao Linear Simples] Iniciando treinamento...')

    cols_needed = ['total_casos', 'mes_indice']
    df_clean = df_agg.dropna(subset=cols_needed).copy()
    removidos = len(df_agg) - len(df_clean)

    X = df_clean[['mes_indice']].values
    y = df_clean['total_casos'].values

    X_train, X_test, y_train, y_test = _split_regression(X, y)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f'  R2={r2:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}')
    print(f'  Intercepto={model.intercept_:.2f}  Coef={model.coef_[0]:.4f}')

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(X_train[:, 0], y_train, alpha=0.4, label='Treino', color='steelblue')
    ax.scatter(X_test[:, 0], y_test, alpha=0.4, label='Teste', color='indianred')
    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 100).reshape(-1, 1)
    ax.plot(xs, model.predict(xs), color='black', lw=2, label='Regressao')
    ax.set_xlabel('Indice temporal do mes/UF')
    ax.set_ylabel('Total de casos no mes/UF')
    ax.set_title(f'Regressao Linear Simples (R2={r2:.3f})')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_figure(fig, 'regressao_linear_simples.png')

    residuos = y_test - y_pred
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(y_pred, residuos, alpha=0.5, color='steelblue')
    ax.axhline(0, color='black', lw=1, linestyle='--')
    ax.set_xlabel('Valor previsto')
    ax.set_ylabel('Residuo (real - previsto)')
    ax.set_title('Residuos vs Previsto - Regressao Linear Simples')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_figure(fig, 'regressao_linear_residuos.png')

    _log_metrics('Regressao Linear Simples', [
        *dataset_info,
        f'Registros totais (agregado): {len(df_agg)}',
        f'Removidos por NaN em features/alvo: {removidos}',
        f'Treino: {len(X_train)} | Teste: {len(X_test)}',
        'Modelo: y = total_casos ~ mes_indice',
        f'Intercepto: {model.intercept_:.4f}',
        f'Coeficiente (mes_indice): {model.coef_[0]:.4f}',
        f'R2 (teste): {r2:.4f}',
        f'MAE (teste): {mae:.4f}',
        f'RMSE (teste): {rmse:.4f}',
    ])


def run_multiple_linear_regression(df_agg, dataset_info):
    print('\n[Regressao Linear Multipla] Iniciando treinamento...')

    numeric_features = [
        'idade_media', 'pct_febre', 'pct_mialgia', 'pct_cefaleia',
        'pct_vomito', 'pct_nausea', 'pct_hospitaliz',
    ]
    categorical_features = ['sg_uf', 'mes']
    feature_cols = numeric_features + categorical_features

    df_clean = df_agg.dropna(subset=['total_casos'] + numeric_features).copy()
    for col in categorical_features:
        df_clean[col] = df_clean[col].fillna('IGN').astype(str)
    removidos = len(df_agg) - len(df_clean)

    X = df_clean[feature_cols].copy()
    y = df_clean['total_casos'].values

    X_train, X_test, y_train, y_test = _split_regression(X, y)

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features),
    ])

    pipe_ols = Pipeline([('pre', preprocessor), ('lr', LinearRegression())])
    pipe_ridge = Pipeline([
        ('pre', preprocessor),
        ('lr', Ridge(alpha=1.0, random_state=SEED)),
    ])

    pipe_ols.fit(X_train, y_train)
    pipe_ridge.fit(X_train, y_train)

    y_pred_ols = pipe_ols.predict(X_test)
    y_pred_ridge = pipe_ridge.predict(X_test)

    r2_ols = r2_score(y_test, y_pred_ols)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    n = len(X_test)
    n_features_out = pipe_ols.named_steps['pre'].transform(X_train.iloc[:1]).shape[1]
    r2_adj_ols = 1 - (1 - r2_ols) * (n - 1) / (n - n_features_out - 1) if n > n_features_out + 1 else float('nan')
    mae_ols = mean_absolute_error(y_test, y_pred_ols)
    rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))

    print(f'  OLS R2={r2_ols:.4f}  R2_adj={r2_adj_ols:.4f}  MAE={mae_ols:.2f}  RMSE={rmse_ols:.2f}')
    print(f'  Ridge R2={r2_ridge:.4f}')

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_test, y_pred_ols, alpha=0.5, color='steelblue')
    lim_min = min(y_test.min(), y_pred_ols.min())
    lim_max = max(y_test.max(), y_pred_ols.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1)
    ax.set_xlabel('Total de casos (real)')
    ax.set_ylabel('Total de casos (previsto OLS)')
    ax.set_title(f'Regressao Multipla: Real vs Previsto (R2={r2_ols:.3f})')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_figure(fig, 'regressao_multipla_real_vs_previsto.png')

    residuos = y_test - y_pred_ols
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(y_pred_ols, residuos, alpha=0.5, color='steelblue')
    ax.axhline(0, color='black', lw=1, linestyle='--')
    ax.set_xlabel('Valor previsto (OLS)')
    ax.set_ylabel('Residuo (real - previsto)')
    ax.set_title('Residuos vs Previsto - Regressao Multipla')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_figure(fig, 'regressao_multipla_residuos.png')

    try:
        feature_names = pipe_ols.named_steps['pre'].get_feature_names_out()
        coefs = pipe_ols.named_steps['lr'].coef_
        top = pd.DataFrame({'feature': feature_names, 'coef': coefs})
        top['abs_coef'] = top['coef'].abs()
        top = top.sort_values('abs_coef', ascending=False).head(15).sort_values('coef')

        fig, ax = plt.subplots(figsize=(9, 7))
        colors = ['indianred' if v < 0 else 'seagreen' for v in top['coef']]
        ax.barh(top['feature'], top['coef'], color=colors)
        ax.set_xlabel('Coeficiente padronizado')
        ax.set_title('Top 15 coeficientes - Regressao Linear Multipla (OLS)')
        ax.axvline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        _save_figure(fig, 'regressao_multipla_coeficientes.png')
    except Exception as exc:
        print(f'  Aviso: grafico de coeficientes nao gerado: {exc}')

    _log_metrics('Regressao Linear Multipla', [
        *dataset_info,
        f'Registros totais (agregado): {len(df_agg)}',
        f'Removidos por NaN em features/alvo: {removidos}',
        f'Treino: {len(X_train)} | Teste: {len(X_test)}',
        f'Features: {feature_cols}',
        'Modelos: LinearRegression e Ridge(alpha=1.0)',
        f'R2 OLS (teste): {r2_ols:.4f}',
        f'R2 ajustado OLS: {r2_adj_ols:.4f}',
        f'MAE OLS: {mae_ols:.4f}',
        f'RMSE OLS: {rmse_ols:.4f}',
        f'R2 Ridge (teste): {r2_ridge:.4f}',
    ])


def _build_comparativo_classificacao(knn_resultado, logistica_resultado):
    print('\n[Comparativo Classificacao] Gerando grafico KNN vs Regressao Logistica...')

    df_cmp = pd.DataFrame([
        {
            'modelo': knn_resultado['nome'],
            'acuracia': knn_resultado['accuracy'],
            'precisao': knn_resultado['precision'],
            'recall': knn_resultado['recall'],
            'f1': knn_resultado['f1'],
            'auc': knn_resultado['auc'],
        },
        {
            'modelo': logistica_resultado['nome'],
            'acuracia': logistica_resultado['accuracy'],
            'precisao': logistica_resultado['precision'],
            'recall': logistica_resultado['recall'],
            'f1': logistica_resultado['f1'],
            'auc': logistica_resultado['auc'],
        },
    ])

    print('\n  Tabela final - Classificacao (hospitalizacao):')
    print(df_cmp.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    metricas = ['acuracia', 'precisao', 'recall', 'f1', 'auc']
    x = np.arange(len(metricas))
    width = 0.35
    cores = ['#4C72B0', '#DD8452']
    for i, row in df_cmp.iterrows():
        vals = [row[m] for m in metricas]
        ax.bar(x + (i - len(df_cmp)/2) * width, vals, width,
               label=row['modelo'], color=cores[i])
    ax.set_xticks(x)
    ax.set_xticklabels(['Acuracia', 'Precisao', 'Recall', 'F1', 'AUC'])
    ax.set_ylabel('Score')
    ax.set_title('Comparativo: KNN vs Regressao Logistica')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save_figure(fig, 'comparativo_classificacao.png')

    _log_metrics('Comparativo Classificacao (KNN x Logistica)', [
        df_cmp.to_string(index=False),
        'Vencedor geral (F1): ' + df_cmp.sort_values('f1', ascending=False).iloc[0]['modelo'],
        'Vencedor recall (sensibilidade): ' + df_cmp.sort_values('recall', ascending=False).iloc[0]['modelo'],
        'Vencedor precisao: ' + df_cmp.sort_values('precisao', ascending=False).iloc[0]['modelo'],
    ])


def run_ml_models():
    if METRICAS_FILE.exists():
        METRICAS_FILE.unlink()

    total_raw = fetch_total_record_count()
    print(f'\nTotal de registros brutos no MySQL: {total_raw}')

    df_cases = fetch_case_modeling_dataframe_sampled()
    print(
        'Amostra carregada para classificacao: '
        f'{len(df_cases)} registros '
        f'(limites: positivos={CASE_SAMPLE_POSITIVE_LIMIT}, negativos={CASE_SAMPLE_NEGATIVE_LIMIT})'
    )

    X, y, case_log = prepare_case_level_data(df_cases)
    for line in case_log:
        print(f'  {line}')
    _log_metrics('Preparacao - Dados por caso', [
        f'Total bruto no MySQL: {total_raw}',
        f'Amostra SQL carregada: {len(df_cases)}',
        f'Limite de positivos na amostra: {CASE_SAMPLE_POSITIVE_LIMIT}',
        f'Limite de negativos na amostra: {CASE_SAMPLE_NEGATIVE_LIMIT}',
        *case_log,
    ])

    df_agg = fetch_monthly_aggregate_dataframe()
    agg_log = [
        f'Registros totais (agregado mensal/UF): {len(df_agg)}',
        f'Colunas do agregado: {list(df_agg.columns)}',
        'Agregado calculado diretamente no MySQL para evitar carregar todos os casos em memoria.',
    ]
    for line in agg_log:
        print(f'  {line}')
    _log_metrics('Preparacao - Agregado mensal', agg_log)

    dataset_info = [
        f'Total de registros brutos do MySQL: {total_raw}',
        f'Amostra de classificacao carregada: {len(df_cases)}',
    ]

    knn_resultado = run_knn_model(X, y, dataset_info)
    logistica_resultado = run_logistic_regression_model(X, y, dataset_info)
    run_simple_linear_regression(df_agg, dataset_info)
    run_multiple_linear_regression(df_agg, dataset_info)

    _build_comparativo_classificacao(knn_resultado, logistica_resultado)

    print('\nModelos concluidos. Saidas em graficos/ e metricas_modelos.txt.')


if __name__ == '__main__':
    run_ml_models()
