import os
import logging
from datetime import datetime
from app.data.loader import load_all_data
from app.core.logic import generate_recommendations
from app.core.config import settings
from app.core.logger import setup_logger
import pandas as pd

def main():
    # --- DYNAMIC OUTPUT DIRECTORY ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(settings.OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- SETUP LOGGER ---
    log_filepath = os.path.join(output_dir, "run.log.json")
    setup_logger(log_filepath)
    logger = logging.getLogger(__name__)

    logger.info("--- Recommendation analysis started ---", extra={'event': 'start'})

    # Define final output paths
    final_output_filename = os.path.join(output_dir, 'recommendations_final.csv')
    analysis_output_filename = os.path.join(output_dir, 'recommendations_full_analysis.csv')

    # --- Step 1: Data Loading ---
    logger.info(f"Loading data from '{settings.INPUT_DIR}'...")
    clients_df, transactions_df, transfers_df = load_all_data(settings.INPUT_DIR)
    if clients_df.empty:
        logger.error("Client data failed to load. Halting process.", extra={'event': 'data_load_failed'})
        return

    # --- Step 2: Recommendation Generation ---
    logger.info("Generating and ranking recommendations...")
    final_recs_list, analysis_df = generate_recommendations(clients_df, transactions_df, transfers_df)
    
    if not final_recs_list:
        logger.warning("No recommendations were generated.", extra={'event': 'no_recs_generated'})
        return
    logger.info(f"Generated {len(final_recs_list)} final recommendations.", extra={'count': len(final_recs_list)})

    # --- Step 3: Saving final (Top-1) results ---
    logger.info(f"Saving final (Top-1) results to '{final_output_filename}'")
    final_recs_data = [rec.model_dump() for rec in final_recs_list]
    final_df = pd.DataFrame(final_recs_data)
    output_df = final_df[['client_code', 'product_name', 'push_notification_text']]
    output_df = output_df.rename(columns={
        'product_name': 'product',
        'push_notification_text': 'push_notification'
    })
    output_df.to_csv(final_output_filename, index=False, encoding='utf-8-sig')

    # --- Step 4: Saving full analysis results ---
    if not analysis_df.empty:
        logger.info(f"Saving full analysis results to '{analysis_output_filename}'")
        numeric_cols = [
            'benefit', 'uncapped_benefit', 'base_propensity', 'counterfactual_signal',
            'neighbor_propensity', 'final_propensity', 'final_score'
        ]
        for col in numeric_cols:
            if col in analysis_df.columns:
                analysis_df[col] = analysis_df[col].round(4)
        analysis_df.to_csv(analysis_output_filename, index=False, encoding='utf-8')

    logger.info(
        "--- Analysis finished successfully! ---", 
        extra={'event': 'finish', 'final_file': final_output_filename, 'analysis_file': analysis_output_filename}
    )

if __name__ == "__main__":
    main()
