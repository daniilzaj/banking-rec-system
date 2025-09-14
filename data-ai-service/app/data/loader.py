import pandas as pd
import glob
import os
from typing import Tuple
from fuzzywuzzy import process

# --- DICTIONARY FOR STATUS NORMALIZATION ---
# Translates raw statuses from the CSV into a clean, internal format.
STATUS_MAP = {
    'Зарплатный клиент': 'payroll',
    'Премиальный клиент': 'premium',
    'Студент': 'student',
    'Стандартный клиент': 'standard'
}

# --- CANONICAL LIST FOR CATEGORY NORMALIZATION ---
# The "golden list" of correct category names our system will use.
CANONICAL_CATEGORIES = [
    "Одежда и обувь", "Продукты питания", "Кафе и рестораны", "Медицина", "Авто",
    "Спорт", "Развлечения", "АЗС", "Кино", "Питомцы", "Книги", "Цветы",
    "Едим дома", "Смотрим дома", "Играем дома", "Косметика и Парфюмерия",
    "Подарки", "Ремонт дома", "Мебель", "Спа и массаж",
    "Ювелирные украшения", "Такси", "Отели", "Путешествия"
]
CATEGORY_MAP = {
    "Ювелирные изделия": "Ювелирные украшения",
    "Парфюмерия": "Косметика и Парфюмерия",
    "Рестораны": "Кафе и рестораны",
    "Игры": "Играем дома",
    "Доставка": "Едим дома",
    "Онлайн-сервисы": "Смотрим дома"
}


def normalize_categories(category_series: pd.Series) -> pd.Series:
    unique_categories = category_series.dropna().unique()
    mapping = {}

    for cat in unique_categories:
        if cat in CATEGORY_MAP:
            mapping[cat] = CATEGORY_MAP[cat]
        else:
            best_match, score = process.extractOne(cat, CANONICAL_CATEGORIES)
            mapping[cat] = best_match if score >= 80 else cat
    
    return category_series.map(mapping)


def load_all_data(base_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads and normalizes all data from CSV files, ensuring data consistency.
    """
    print(f"Загрузка данных из папки: {base_path}")

    # 1. Load and normalize client data
    clients_path = os.path.join(base_path, 'clients.csv')
    try:
        # Load, ensuring client_code is a string from the start.
        clients_df = pd.read_csv(clients_path, dtype={'client_code': str})
        
        # Create a new, clean column for statuses using the map.
        clients_df['status_normalized'] = clients_df['status'].map(STATUS_MAP).fillna('standard')
        
        print(f"Успешно загружено и нормализовано {len(clients_df)} клиентов.")
    except FileNotFoundError:
        print(f"ОШИБКА: Файл 'clients.csv' не найден в '{base_path}'")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 2. Load and normalize transaction data
    transaction_files = glob.glob(os.path.join(base_path, '*_transactions_3m.csv'))
    if not transaction_files:
        transactions_df = pd.DataFrame()
        print("Внимание: Файлы транзакций не найдены.")
    else:
        transactions_list = []
        for file in transaction_files:
            client_code = os.path.basename(file).split('_')[1]
            temp_df = pd.read_csv(file, dtype={'client_code': str})
            temp_df['client_code'] = str(client_code) # Ensure code from filename is also a string
            transactions_list.append(temp_df)
        transactions_df = pd.concat(transactions_list, ignore_index=True)
        
        # Apply intelligent category normalization
        transactions_df['category'] = normalize_categories(transactions_df['category'])
        
        print(f"Успешно загружено и нормализовано {len(transactions_df)} транзакций.")

    # 3. Load transfer data
    transfer_files = glob.glob(os.path.join(base_path, '*_transfers_3m.csv'))
    if not transfer_files:
        transfers_df = pd.DataFrame()
        print("Внимание: Файлы переводов не найдены.")
    else:
        transfers_list = []
        for file in transfer_files:
            client_code = os.path.basename(file).split('_')[1]
            temp_df = pd.read_csv(file, dtype={'client_code': str})
            temp_df['client_code'] = str(client_code)
            transfers_list.append(temp_df)
        transfers_df = pd.concat(transfers_list, ignore_index=True)
        print(f"Успешно загружено {len(transfers_df)} переводов.")

    return clients_df, transactions_df, transfers_df

if __name__ == '__main__':
    # Block for independent testing of this script
    input_path = '../../data_input'
    
    clients, transactions, transfers = load_all_data(input_path)
    
    print("\n--- Проверка загруженных данных ---")
    if not clients.empty:
        print("\nКлиенты (info):")
        clients.info()
        print("\nПервые 5 клиентов:")
        print(clients.head())
    
    if not transactions.empty:
        print("\nТранзакции (info):")
        transactions.info()
        print("\nПервые 5 транзакций:")
        print(transactions.head())
        print("\nУникальные нормализованные категории:")
        print(transactions['category'].unique())