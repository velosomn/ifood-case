"""Shared paths for the iFood case (used by the local notebooks)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
PRESENTATION = ROOT / "presentation"

OFFERS_JSON = DATA_RAW / "offers.json"
PROFILE_JSON = DATA_RAW / "profile.json"
TRANSACTIONS_JSON = DATA_RAW / "transactions.json"

# Saída do notebook 1 (consumida pelo notebook 2)
MODELING_TABLE_PARQUET = DATA_PROCESSED / "modeling_table.parquet"
OFFERS_PARQUET = DATA_PROCESSED / "offers.parquet"
