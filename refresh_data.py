"""
Weekly Data Refresh Script
==========================
Fetches fresh data from Yahoo Finance / FRED, runs feature engineering,
persists everything to MongoDB (and CSV as fallback), and optionally
re-trains the ML model.

Usage:
    python refresh_data.py                  # incremental update
    python refresh_data.py --force          # full re-download from 2008
    python refresh_data.py --retrain        # update data + retrain model
    python refresh_data.py --force --retrain

Scheduling (Windows Task Scheduler):
    Program: python
    Arguments: "C:\\Users\\eomis\\SP500 Project\\S-P-500-Prediction\\refresh_data.py"
    Trigger: Weekly (e.g. every Sunday at 8 AM)
"""

import argparse
import datetime
import logging
import os
import sys
import traceback

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = datetime.datetime.now().strftime("refresh_%Y-%m-%d_%H-%M.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, log_filename)),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_fetch_and_engineer(force: bool = False):
    """
    Fetch data from APIs and run the feature engineering pipeline.

    If *force* is False, tries an incremental update: queries MongoDB for
    the last stored feature date and only re-downloads from 2008 (since
    rolling windows need long history) but the feature pipeline is still
    end-to-end.  A full pipeline re-run is cheap (~30 s) so we always
    re-engineer all features from scratch to keep things simple and correct.

    Returns the features DataFrame (or None on failure).
    """
    from main import main as run_pipeline

    if not force:
        try:
            from db_helpers import get_last_feature_date
            last_date = get_last_feature_date()
            if last_date:
                logger.info(f"Last feature date in MongoDB: {last_date}")
            else:
                logger.info("No features in MongoDB yet — running full pipeline.")
        except Exception:
            logger.info("Could not query MongoDB for last date (may not be configured). Running full pipeline.")

    logger.info("Running full data fetch + feature engineering pipeline...")
    features_df = run_pipeline(save_to_mongodb=True)
    return features_df


def step_retrain():
    """Re-train the walk-forward model with the latest data."""
    logger.info("Starting model re-training (walk-forward)...")
    from ML.train_walkforward import main as train_wf
    train_wf()
    logger.info("Model re-training complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Weekly data refresh for S&P 500 prediction pipeline."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a full re-download of all historical data.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Re-train the ML model after refreshing data.",
    )
    args = parser.parse_args()

    start_time = datetime.datetime.now()
    logger.info("=" * 60)
    logger.info(f"Data refresh started at {start_time.isoformat()}")
    logger.info(f"  force={args.force}, retrain={args.retrain}")
    logger.info("=" * 60)

    success = True

    # 1. Fetch + Feature Engineering
    try:
        features_df = step_fetch_and_engineer(force=args.force)
        if features_df is not None:
            logger.info(
                f"Features refreshed: {len(features_df)} rows, "
                f"{features_df.index.min().date()} to {features_df.index.max().date()}"
            )
        else:
            logger.warning("Feature pipeline returned None.")
            success = False
    except Exception:
        logger.error(f"Feature pipeline failed:\n{traceback.format_exc()}")
        success = False

    # 2. Retrain (optional)
    if args.retrain and success:
        try:
            step_retrain()
        except Exception:
            logger.error(f"Re-training failed:\n{traceback.format_exc()}")
            success = False

    # Summary
    elapsed = datetime.datetime.now() - start_time
    status = "SUCCESS" if success else "FAILED"
    logger.info("=" * 60)
    logger.info(f"Data refresh {status}  (elapsed: {elapsed})")
    logger.info("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
