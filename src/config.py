"""Shared paths and Spark session factory for the iFood case."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Project layout ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
PRESENTATION = ROOT / "presentation"

OFFERS_JSON = DATA_RAW / "offers.json"
PROFILE_JSON = DATA_RAW / "profile.json"
TRANSACTIONS_JSON = DATA_RAW / "transactions.json"

# Processed outputs (single-file parquet written via pandas/pyarrow, so the
# pipeline does not depend on winutils.exe/HADOOP_HOME on Windows).
CUSTOMERS_PARQUET = DATA_PROCESSED / "customers.parquet"
OFFERS_PARQUET = DATA_PROCESSED / "offers.parquet"
MODELING_TABLE_PARQUET = DATA_PROCESSED / "modeling_table.parquet"


def get_spark(app_name: str = "ifood-case", shuffle_partitions: int = 8):
    """Create a local Spark session configured for a single-machine run.

    We pin ``PYSPARK_PYTHON`` to the current interpreter so the JVM launches the
    right Python workers on Windows, and keep the config lightweight because the
    whole dataset fits comfortably in memory.
    """
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        # NOTE: intentionally do NOT set spark.driver.memory here — on Windows it
        # forces PySpark to relaunch the JVM via spark-submit, which trips the
        # HADOOP_HOME/winutils check. In-process session avoids that entirely.
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
