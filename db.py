"""
MongoDB connection for the S&P 500 prediction pipeline.
Uses MONGODB_URI from environment (set in .env or your shell).

Mirrors the connection pattern from the Financial-Dashboard-System project
so both repos share the same Atlas database.
"""

import os
from pathlib import Path

_root = Path(__file__).resolve().parent
_env_path = _root / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

DEFAULT_DB_NAME = "market_data"


def get_uri():
    """MongoDB connection URI from environment."""
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise ValueError(
            "MONGODB_URI is not set. Add it to a .env file in the project root "
            "or set the environment variable. See .env.example for format."
        )
    return uri


def get_client():
    """Return a MongoDB client."""
    import pymongo
    return pymongo.MongoClient(get_uri())


def get_database(name: str = DEFAULT_DB_NAME):
    """Return the database to use for market data."""
    return get_client()[name]


def get_collection(collection_name: str, db_name: str = DEFAULT_DB_NAME):
    """Return a collection from the market data database."""
    return get_database(db_name)[collection_name]
