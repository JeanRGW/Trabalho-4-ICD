import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    'host': 'localhost',
    'user': 'jean',
    'password': 'jean123',
    'database': 'trab4_icd',
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_unicode_ci'
}

def get_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f'Erro ao conectar ao banco de dados: {e}')
        return None

def close_connection(connection):
    if connection and connection.is_connected():
        connection.close()
