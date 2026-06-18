import numpy as np
import pandas as pd

UF_CODE_TO_SIGLA = {
    '11': 'RO', '12': 'AC', '13': 'AM', '14': 'RR', '15': 'PA', '16': 'AP', '17': 'TO',
    '21': 'MA', '22': 'PI', '23': 'CE', '24': 'RN', '25': 'PB', '26': 'PE', '27': 'AL',
    '28': 'SE', '29': 'BA', '31': 'MG', '32': 'ES', '33': 'RJ', '35': 'SP', '41': 'PR',
    '42': 'SC', '43': 'RS', '50': 'MS', '51': 'MT', '52': 'GO', '53': 'DF',
}

UF_TO_REGION = {
    'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte',
    'RR': 'Norte', 'TO': 'Norte',
    'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste',
    'PB': 'Nordeste', 'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
    'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
    'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
    'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul',
}


def convert_age_to_years(value):
    if pd.isna(value):
        return np.nan
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return np.nan
    if number < 0:
        return np.nan
    if number < 1000:
        years = float(number)
    else:
        unit = number // 1000
        amount = number % 1000
        if unit == 1:
            years = amount / (24 * 365)
        elif unit == 2:
            years = amount / 365
        elif unit == 3:
            years = amount / 12
        elif unit == 4:
            years = float(amount)
        else:
            return np.nan
    if years < 0 or years > 120:
        return np.nan
    return years


def clean_binary_code(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text == '':
        return np.nan
    if text == '1':
        return 1.0
    if text == '2':
        return 0.0
    return np.nan


def normalize_uf(value):
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text in UF_CODE_TO_SIGLA:
        return UF_CODE_TO_SIGLA[text]
    return text if text in UF_TO_REGION else None


def normalize_uf_short(value):
    if pd.isna(value):
        return 'IGN'
    text = str(value).strip().upper()
    if text in UF_CODE_TO_SIGLA:
        return UF_CODE_TO_SIGLA[text]
    return text if len(text) == 2 else 'IGN'
