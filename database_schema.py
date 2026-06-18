import mysql.connector
from mysql.connector import Error
from config import DB_CONFIG, get_connection, close_connection

CREATE_TABLE_SQL = '''
CREATE TABLE IF NOT EXISTS casos_dengue (
    id INT AUTO_INCREMENT PRIMARY KEY,
    dt_notific DATE,
    sem_not INT,
    sg_uf_not VARCHAR(2),
    dt_nasc DATE,
    nu_idade_n INT,
    cs_sexo VARCHAR(1),
    cs_gestant VARCHAR(1),
    cs_raca VARCHAR(1),
    cs_escol_n VARCHAR(2),
    sg_uf VARCHAR(2),
    febre VARCHAR(1),
    mialgia VARCHAR(1),
    cefaleia VARCHAR(1),
    exantema VARCHAR(1),
    vomito VARCHAR(1),
    nausea VARCHAR(1),
    dor_costas VARCHAR(1),
    conjuntvit VARCHAR(1),
    artrite VARCHAR(1),
    artralgia VARCHAR(1),
    petequia_n VARCHAR(1),
    leucopenia VARCHAR(1),
    laco VARCHAR(1),
    dor_retro VARCHAR(1),
    diabetes VARCHAR(1),
    hematolog VARCHAR(1),
    hepatopat VARCHAR(1),
    renal VARCHAR(1),
    hipertensa VARCHAR(1),
    acido_pept VARCHAR(1),
    auto_imune VARCHAR(1),
    dt_sin_pri DATE,
    hospitaliz VARCHAR(1),
    classi_fin VARCHAR(2),
    criterio VARCHAR(1),
    evolucao VARCHAR(1),
    dt_encerra DATE,
    INDEX idx_data (dt_notific),
    INDEX idx_estado_notificacao (sg_uf_not),
    INDEX idx_estado_residencia (sg_uf),
    INDEX idx_hospitaliz (hospitaliz),
    INDEX idx_hospitaliz_dt (hospitaliz, dt_notific),
    INDEX idx_uf_not_dt (sg_uf_not, dt_notific)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
'''


def setup_database():
    print('Criando banco de dados (se necessario)...')
    try:
        connection = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            charset=DB_CONFIG.get('charset', 'utf8mb4'),
            collation=DB_CONFIG.get('collation', 'utf8mb4_unicode_ci'),
        )
        cursor = connection.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']} "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        cursor.close()
        close_connection(connection)
    except Error as exc:
        print(f'Erro ao criar banco de dados: {exc}')
        return False

    print('Criando tabela casos_dengue (se necessario)...')
    connection = get_connection()
    if not connection:
        return False
    try:
        cursor = connection.cursor()
        cursor.execute(CREATE_TABLE_SQL)
        connection.commit()
        cursor.close()
        print('Schema pronto.')
        return True
    except Error as exc:
        print(f'Erro ao criar tabela: {exc}')
        return False
    finally:
        close_connection(connection)


if __name__ == '__main__':
    setup_database()
