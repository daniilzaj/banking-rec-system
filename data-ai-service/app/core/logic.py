import pandas as pd
import yaml
import random
from typing import List, Dict, Any, Tuple

# --- ML and Math ---
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# --- Local Imports ---
from .models import ProductRecommendation
from ..data.loader import load_all_data

# --------------------------------------------------------------------------
# --- 1. CONFIGURATION AND CONSTANTS ---
# --------------------------------------------------------------------------
# --- ДОБАВЬТЕ ЭТОТ СЛОВАРЬ В БЛОК КОНСТАНТ ---

MONTH_MAP_GENITIVE = {
    1: 'января', 2: 'февраля', 3: 'марта', 4: 'апреля', 5: 'мая', 6: 'июня',
    7: 'июля', 8: 'августа', 9: 'сентября', 10: 'октября', 11: 'ноября', 12: 'декабря'
}

def load_config(path: str) -> Dict:
    """Universal loader for YAML configurations."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception:
        return {}

PRODUCT_CATALOG = load_config("config/products.yml").get("products", [])
PUSH_TEMPLATES = load_config("config/pushes.yml").get("templates", {})
NON_COMMERCIAL_CATEGORIES = {"Коммунальные платежи", "Гос. услуги", "Налоги"}

weights_config = load_config("config/scoring_weights.yml")
W_BENEFIT = weights_config.get('benefit_weight', 0.5)
W_PROPENSITY = weights_config.get('propensity_weight', 0.3)
W_COUNTERFACTUAL = weights_config.get('counterfactual_weight', 0.2)

# --------------------------------------------------------------------------
# --- 2. UTILITIES ---
# --------------------------------------------------------------------------
def format_currency(value: float) -> str:
    """Formats a number into a currency string with spaces, e.g., '10 000 ₸'."""
    return f"{value:,.0f}".replace(",", " ") + " ₸"

def generate_push_text(client_name: str, product_name: str, benefit_value: float, **kwargs) -> str:
    """Selects the best template, prioritizing those with a CTA, and formats it."""
    first_name = client_name.split()[0]
    templates = PUSH_TEMPLATES.get(product_name, PUSH_TEMPLATES.get("Generic", []))
    if not templates:
        return f"Здравствуйте, {first_name}. У нас есть предложение для вас."
    
    templates_copy = templates.copy()
    random.shuffle(templates_copy)
    
    kwargs.update({'first_name': first_name, 'benefit_value': format_currency(benefit_value)})
    
    CTA_VERBS = {"Открыть", "Оформить", "Настроить", "Узнать", "Подключить", "Посмотреть", "Попробовать"}
    templates_with_cta = [tpl for tpl in templates_copy if any(verb.lower() in tpl.lower() for verb in CTA_VERBS)]
    
    chosen_list = templates_with_cta if templates_with_cta else templates_copy

    for template in chosen_list:
        try:
            return template.format(**kwargs)
        except KeyError:
            continue
            
    # Fallback to the first available generic template if all else fails
    return PUSH_TEMPLATES.get("Generic", ["Специальное предложение!"])[0].format(**kwargs)

def get_base_propensity_score(client: pd.Series, product: Dict[str, Any]) -> float:
    """Heuristic estimation of the client's base propensity for a product."""
    propensity = 0.5
    status = client.get('status_normalized', 'standard')
    
    if status == 'premium':
        if 'Премиум' in product.get('name', '') or 'Золото' in product.get('name', ''): propensity += 0.4
        elif 'Кредит' in product.get('name', ''): propensity -= 0.2
    
    if status == 'student' and 'Кредит' in product.get('name', ''): 
        propensity += 0.3
    
    if client['age'] < 30 and ('Инвестиции' in product.get('name', '') or 'Кредит' in product.get('name', '')): 
        propensity += 0.2
        
    if client['age'] > 55 and 'Депозит' in product.get('name', ''): 
        propensity += 0.2
        
    return max(0.1, min(1.0, propensity))

def vectorize_clients(clients_df: pd.DataFrame, spend_by_category: pd.DataFrame) -> pd.DataFrame:
    """Converts client data into numerical vectors for finding neighbors."""
    # Use a wider range of categories to capture more signals
    top_categories = spend_by_category.sum().nlargest(20).index.tolist()
    client_vectors = spend_by_category[top_categories].copy()
    clients_indexed = clients_df.set_index('client_code')
    
    # Align indices to prevent data mismatch
    clients_aligned, vectors_aligned = clients_indexed.align(client_vectors, join='inner', axis=0)
    
    vectors_aligned['age'] = clients_aligned['age']
    vectors_aligned['avg_monthly_balance_KZT'] = clients_aligned['avg_monthly_balance_KZT']
    
    vectors_aligned.fillna(0, inplace=True)
    scaler = StandardScaler()
    scaled_vectors = scaler.fit_transform(vectors_aligned)
    
    return pd.DataFrame(scaled_vectors, index=vectors_aligned.index, columns=vectors_aligned.columns)

# --------------------------------------------------------------------------
# --- 3. MAIN RECOMMENDATION LOGIC ---
# --------------------------------------------------------------------------
def generate_recommendations(clients_df: pd.DataFrame, transactions_df: pd.DataFrame, transfers_df: pd.DataFrame) -> Tuple[List[ProductRecommendation], pd.DataFrame]:
    if not PRODUCT_CATALOG: return [], pd.DataFrame()
    
    # --- STEP 1: FEATURE ENGINEERING ---
    transactions_df['date'] = pd.to_datetime(transactions_df['date'], errors='coerce')
    transfers_df['date'] = pd.to_datetime(transfers_df['date'], errors='coerce')
    
    spend_by_category_3m = transactions_df.groupby(['client_code', 'category'])['amount'].sum().unstack(fill_value=0)
    spend_by_category = spend_by_category_3m / 3.0
    total_spend = spend_by_category.sum(axis=1)
    
    avg_transaction_amount = transactions_df.groupby('client_code')['amount'].mean()
    large_purchases = pd.merge(transactions_df, avg_transaction_amount.rename('avg_amount'), on='client_code')
    large_purchases = large_purchases[large_purchases['amount'] > large_purchases['avg_amount'] * 5]
    signal_large_purchase = large_purchases.groupby('client_code').size() > 0
    
    deposit_in_ops = transfers_df[transfers_df['type'] == 'deposit_in']
    signal_has_deposits = deposit_in_ops.groupby('client_code').size() > 0
    
    fx_ops = transfers_df[transfers_df['type'].isin(['fx_buy', 'fx_sell'])]
    fx_ops_count = fx_ops.groupby('client_code')['type'].count() / 3.0
    top_fx_currency = fx_ops.groupby('client_code')['currency'].agg(lambda x: x.mode().get(0, 'USD'))
    
    transactions_df['month_num'] = transactions_df['date'].dt.month
    transactions_df['month'] = transactions_df['month_num'].map(MONTH_MAP_GENITIVE)
    travel_transactions = transactions_df[transactions_df['category'].str.contains('Путешеств|Такси|Отели|Авиа', case=False, na=False)].copy()
    if not travel_transactions.empty:
        max_travel_indices = travel_transactions.loc[travel_transactions.groupby('client_code')['amount'].idxmax()]
        top_travel_month = max_travel_indices[['client_code', 'month']].set_index('client_code')
    else:
        top_travel_month = pd.DataFrame(columns=['month'])

    # --- STEP 2: VECTORIZATION & NEIGHBOR FINDING ---
    client_vectors = vectorize_clients(clients_df, spend_by_category)
    nn_model = NearestNeighbors(n_neighbors=6).fit(client_vectors)
    indices = nn_model.kneighbors(client_vectors, return_distance=False)
    client_code_map = client_vectors.index
    
    # --- STEP 3: RAW OFFER CALCULATION ---
    all_offers_raw = []
    for _, client in clients_df.iterrows():
        client_code = client['client_code']
        
        if client_code not in spend_by_category.index or total_spend.get(client_code, 0) == 0:
            product_name = "Депозит Накопительный"
            product_config = next((p for p in PRODUCT_CATALOG if p.get('name') == product_name), None)
            if product_config and client['avg_monthly_balance_KZT'] > 0:
                benefit = (client['avg_monthly_balance_KZT'] * product_config.get('interest_rate', 0)) / 12
                all_offers_raw.append({'client_code': client_code, 'product_name': product_name, 'benefit': benefit, 'uncapped_benefit': benefit, 'base_propensity': 0.7, 'counterfactual_signal': 0, 'category_weight': product_config.get('category_weight', 1.0)})
            continue

        client_spend_cat = spend_by_category.loc[client_code]
        for product in PRODUCT_CATALOG:
            benefit, uncapped_benefit = 0.0, 0.0
            product_name = product.get('name')
            
            if product.get('benefit_type') == 'cashback':
                if product_name == "Карта для путешествий":
                    travel_spend = client_spend_cat.reindex(product.get('categories', []), fill_value=0).sum()
                    uncapped_benefit = travel_spend * product.get('rate', 0)
                elif product_name == "Премиальная карта":
                    client_balance = client['avg_monthly_balance_KZT']
                    base_rate = next((tier['rate'] for tier in sorted(product.get('tiered_rates', []), key=lambda x: x['threshold'], reverse=True) if client_balance >= tier['threshold']), 0)
                    base_cashback = total_spend.get(client_code, 0) * base_rate
                    special_spend = client_spend_cat.reindex(product.get('special_categories', []), fill_value=0).sum()
                    special_cashback = special_spend * (product.get('special_rate', 0) - base_rate)
                    uncapped_benefit = base_cashback + special_cashback
                elif product_name == "Кредитная карта":
                    commercial_spends = client_spend_cat.drop(labels=NON_COMMERCIAL_CATEGORIES, errors='ignore')
                    top_cats_spend = commercial_spends.nlargest(product.get('top_n_categories', 0)).sum()
                    online_spend = client_spend_cat.reindex(product.get('online_categories', []), fill_value=0).sum()
                    uncapped_benefit = (top_cats_spend * product.get('top_category_rate', 0)) + (online_spend * product.get('online_rate', 0))
                benefit = min(uncapped_benefit, product.get('cashback_limit_monthly', float('inf')))
            elif product.get('benefit_type') == 'interest':
                if client['avg_monthly_balance_KZT'] > 50000:
                    benefit = uncapped_benefit = (client['avg_monthly_balance_KZT'] * product.get('interest_rate', 0)) / 12
            elif product_name == "Обмен валют":
                client_fx_count = fx_ops_count.get(client_code, 0)
                if client_fx_count >= product.get('min_fx_ops_monthly', 99):
                    benefit = uncapped_benefit = client_fx_count * product.get('saved_fee_per_op', 0)
            elif product_name == "Инвестиции":
                idle_money = client['avg_monthly_balance_KZT'] - 50000
                if idle_money > 0 and not signal_has_deposits.get(client_code, False):
                    benefit = uncapped_benefit = idle_money * 0.05
            elif product_name == "Кредит наличными":
                if signal_large_purchase.get(client_code, False): benefit = uncapped_benefit = 5000
            elif product_name == "Золотые слитки":
                if client.get('status_normalized') == 'premium' and client['avg_monthly_balance_KZT'] > 5000000: benefit = uncapped_benefit = 6000
            
            if benefit <= 0: continue
            
            base_propensity = get_base_propensity_score(client, product)
            counterfactual_signal = 0.0
            try:
                client_idx = client_code_map.get_loc(client_code)
                neighbor_indices = indices[client_idx][1:]
                neighbor_codes = client_code_map[neighbor_indices]
                if 'categories' in product and product.get('categories'):
                    valid_cats = list(set(product['categories']) & set(spend_by_category.columns))
                    if valid_cats:
                        neighbor_spends_in_cat = spend_by_category.loc[neighbor_codes, valid_cats].sum(axis=1)
                        neighbor_total_spends = total_spend.loc[neighbor_codes]
                        with np.errstate(divide='ignore', invalid='ignore'):
                            share_of_wallet = np.nan_to_num(neighbor_spends_in_cat / neighbor_total_spends)
                        counterfactual_signal = np.mean(share_of_wallet)
            except (KeyError, IndexError): pass
            
            all_offers_raw.append({'client_code': client_code, 'product_name': product_name, 'benefit': benefit, 'uncapped_benefit': uncapped_benefit, 'base_propensity': base_propensity, 'counterfactual_signal': counterfactual_signal, 'category_weight': product.get('category_weight', 1.0)})
    
    if not all_offers_raw: return [], pd.DataFrame()
    
    offers_df = pd.DataFrame(all_offers_raw)
    
    # --- STEP 4: ENHANCED PROPENSITY & FINAL SCORE ---
    propensity_map = offers_df.set_index(['client_code', 'product_name'])['base_propensity']
    neighbor_propensities = []
    for _, offer in offers_df.iterrows():
        try:
            client_idx = client_code_map.get_loc(offer['client_code'])
            neighbor_indices = indices[client_idx][1:]
            neighbor_codes = client_code_map[neighbor_indices]
            avg_neighbor_prop = propensity_map.loc[neighbor_codes, offer['product_name']].mean()
            neighbor_propensities.append(avg_neighbor_prop)
        except (KeyError, IndexError):
            neighbor_propensities.append(offer['base_propensity'])

    offers_df['neighbor_propensity'] = neighbor_propensities
    offers_df.fillna({'neighbor_propensity': offers_df['base_propensity']}, inplace=True)
    offers_df['final_propensity'] = 0.7 * offers_df['base_propensity'] + 0.3 * offers_df['neighbor_propensity']
    
    scaler = MinMaxScaler()
    offers_df['norm_benefit'] = scaler.fit_transform(offers_df[['uncapped_benefit']])
    offers_df['norm_counterfactual'] = scaler.fit_transform(offers_df[['counterfactual_signal']])
    offers_df['final_score'] = ((W_BENEFIT * offers_df['norm_benefit'] + W_PROPENSITY * offers_df['final_propensity'] + W_COUNTERFACTUAL * offers_df['norm_counterfactual']) * offers_df['category_weight'])
    
    # --- STEP 5: FINAL OUTPUT FORMATTING ---
    recommendations_output = []
    final_recs_df = offers_df.loc[offers_df.groupby('client_code')['final_score'].idxmax()]

    for _, offer in final_recs_df.iterrows():
        client_code = offer['client_code']
        client_info = clients_df[clients_df['client_code'] == client_code].iloc[0]
        
        extra_params = {'cat1': '', 'cat2': '', 'cat3': ''}
        if client_code in spend_by_category.index:
            commercial_spends = spend_by_category.loc[client_code].drop(labels=NON_COMMERCIAL_CATEGORIES, errors='ignore')
            for i, (cat, _) in enumerate(commercial_spends.nlargest(3).items()): extra_params[f"cat{i+1}"] = cat
        
        if client_code in top_travel_month.index: extra_params['month'] = top_travel_month.loc[client_code, 'month']
        if client_code in top_fx_currency.index: extra_params['fx_curr'] = top_fx_currency.loc[client_code]
        
        push_text = generate_push_text(client_info['name'], offer['product_name'], offer['benefit'], **extra_params)
        recommendations_output.append(ProductRecommendation(client_code=client_code, product_name=offer['product_name'], benefit=offer['benefit'], push_notification_text=push_text))
            
    return recommendations_output, offers_df