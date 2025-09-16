"""
nse_high_delivery_scanner.py

This single-file deliverable contains:
  1) The full Python 3.x script implementing the High-Delivery scanner.
  2) README and requirements in the header for quick copy-paste.

Purpose: Build an automated NSE High-Delivery scanner that:
 - downloads NSE bhavcopy CSVs for past trading sessions,
 - stores cleaned EQ-series rows into a local SQLite DB,
 - computes exact 20-session average delivery quantities per symbol,
 - produces a daily snapshot for the latest successful session,
 - filters stocks with delivery >= 10,000 and delivery >= 3x 20-day average and closed green,
 - writes results to output/high_delivery_stocks_<YYYYMMDD>.csv and a scan log.

Quick start
-----------
1. Create a virtualenv and install dependencies
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Run scanner for last 30 trading sessions and store raw CSVs in ./raw
   python nse_high_delivery_scanner.py --days_back 30 --download_dir ./raw

3. Optional: send result to Telegram
   python nse_high_delivery_scanner.py --days_back 20 --telegram-token <token> --telegram-chatid <chatid>

Files produced
--------------
 - ./raw/sec_bhavdata_full_DDMMYYYY.csv  (downloaded raw files)
 - ./raw/bhavcopy.sqlite                 (SQLite DB of cleaned EQ rows)
 - ./output/high_delivery_stocks_YYYYMMDD.csv
 - ./output/scan_YYYYMMDD.log

requirements.txt
----------------
pandas>=1.5
requests>=2.28
PyYAML>=6.0
SQLAlchemy>=1.4
python-dateutil>=2.8
tqdm>=4.64
pytest>=7.0  # optional for tests

"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import requests
import yaml
from sqlalchemy import create_engine, inspect

# ----------------------
# Defaults
# ----------------------
DEFAULTS = {
    "days_back": 20,
    "max_retries_today": 5,
    "download_dir": "./raw",
    "output_dir": "./output",
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 (compatible; nse-high-delivery-scanner/1.0)",
    "timeout": 15,
    "sqlite_db": "bhavcopy.sqlite",
}

BASE_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv"

# ----------------------
# Helpers
# ----------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def setup_logging(log_path=None):
    ensure_dir(os.path.dirname(log_path) or ".")
    logger = logging.getLogger("nse_scanner")
    logger.setLevel(logging.INFO)
    # avoid duplicate handlers in repeated runs
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    if log_path:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def date_to_filename(d: datetime) -> str:
    return f"sec_bhavdata_full_{d.strftime('%d%m%Y')}.csv"


def is_weekend(d: datetime) -> bool:
    return d.weekday() >= 5

# ----------------------
# Download
# ----------------------

def download_bhavcopy_for_date(date: datetime, download_dir: str, headers: dict, timeout: int, logger=None):
    filename = date_to_filename(date)
    url = BASE_URL.format(date=date.strftime("%d%m%Y"))
    local_path = os.path.join(download_dir, filename)

    if os.path.exists(local_path):
        if logger:
            logger.info(f"Found existing file for {date.date()} -> {local_path}")
        return local_path, True

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        if logger:
            logger.warning(f"Failed to fetch {url} : {e}")
        return None, False

    if resp.status_code == 200 and resp.content:
        ensure_dir(download_dir)
        with open(local_path, "wb") as f:
            f.write(resp.content)
        if logger:
            logger.info(f"Downloaded {filename}")
        return local_path, True
    else:
        if logger:
            logger.info(f"No file for {date.date()} (HTTP {resp.status_code})")
        return None, False


def collect_last_n_trading_days(n: int, download_dir: str, headers: dict, timeout: int, max_lookback: int = 365, max_retries_today: int = 5, logger=None):
    results = []
    today = datetime.now()

    recent_date = None
    attempts = 0
    d = today
    while attempts < max_retries_today:
        if is_weekend(d):
            d = d - timedelta(days=1)
            attempts += 1
            continue
        local_path, ok = download_bhavcopy_for_date(d, download_dir, headers, timeout, logger)
        if ok:
            recent_date = d
            break
        d = d - timedelta(days=1)
        attempts += 1

    if recent_date is None:
        if logger:
            logger.error("Could not find recent trading session within max_retries_today window.")
        return results

    collected = 0
    d = recent_date
    lookback = 0
    failed_dates = []
    while collected < n and lookback < max_lookback:
        if is_weekend(d):
            d = d - timedelta(days=1)
            lookback += 1
            continue
        local_path, ok = download_bhavcopy_for_date(d, download_dir, headers, timeout, logger)
        results.append((d, local_path, ok))
        if ok:
            collected += 1
        else:
            failed_dates.append(d)
        d = d - timedelta(days=1)
        lookback += 1
    if logger:
        logger.info(f"Collected {collected} successful sessions, failed {len(failed_dates)} dates while looking back {lookback} days")
    return list(reversed(results))

# ----------------------
# Robust CSV reader (fixed)
# ----------------------

def read_and_clean_bhavcopy(csv_path: str, logger=None) -> pd.DataFrame:
    """
    Robust reader for NSE bhavcopy CSV.
    - tries multiple encodings/delimiters,
    - normalizes column names,
    - strips/uppercases SERIES values before filtering.
    Returns DataFrame of EQ rows with normalized columns.
    """
    import os

    read_attempts = [
        {"encoding": "utf-8", "sep": ","},
        {"encoding": "latin1", "sep": ","},
        {"encoding": "latin1", "sep": ";"},
        {"encoding": "latin1", "sep": "|"},
        {"encoding": "latin1", "sep": "	"},
    ]
    df = None
    last_err = None
    for opt in read_attempts:
        try:
            df = pd.read_csv(csv_path, **opt, engine="python")
            if logger:
                logger.info(f"Read {os.path.basename(csv_path)} with options {opt}, shape={df.shape}")
            break
        except Exception as e:
            last_err = e
            if logger:
                logger.debug(f"Read attempt failed for {csv_path} with {opt}: {e}")

    if df is None:
        try:
            df = pd.read_csv(csv_path, encoding="latin1", engine="python")
            if logger:
                logger.info(f"Fallback read succeeded for {os.path.basename(csv_path)} with latin1")
        except Exception as e:
            if logger:
                logger.exception(f"All read attempts failed for {csv_path}: {last_err}")
            raise

    # Normalize column names
    df.columns = [c.strip().upper() for c in df.columns]

    # Try to find/rename SERIES if variant exists
    if "SERIES" not in df.columns:
        maybe = [c for c in df.columns if "SERIES" in c]
        if maybe:
            df.rename(columns={maybe[0]: "SERIES"}, inplace=True)

    # Determine date column
    date_col = None
    for c in ["DATE1", "DATE", "TIMESTAMP"]:
        if c in df.columns:
            date_col = c
            break
    if date_col:
        df["DATE"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    else:
        df["DATE"] = pd.NaT

    # Normalize SERIES
    if "SERIES" in df.columns:
        df["SERIES"] = df["SERIES"].astype(str).str.strip().str.upper()
    else:
        df["SERIES"] = ""

    # Filter EQ
    df_eq = df[df["SERIES"] == "EQ"].copy()

    if df_eq.empty and logger:
        unique_series = df["SERIES"].astype(str).str.strip().unique().tolist()
        logger.warning(f"No EQ rows in {os.path.basename(csv_path)}. Found SERIES values: {unique_series}")

    # Numeric cleanup
    numeric_cols = [
        "PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE",
        "LAST_PRICE", "CLOSE_PRICE", "AVG_PRICE", "TTL_TRD_QNTY",
        "TURNOVER_LACS", "NO_OF_TRADES", "DELIV_QTY", "DELIV_PER",
    ]
    for c in numeric_cols:
        if c in df_eq.columns:
            df_eq[c] = df_eq[c].astype(str).str.replace(",", "", regex=False)
            df_eq[c] = pd.to_numeric(df_eq[c], errors="coerce")

    if "SYMBOL" not in df_eq.columns:
        if logger:
            logger.error(f"Missing SYMBOL column in {csv_path}")
        return pd.DataFrame()

    df_eq = df_eq[df_eq["SYMBOL"].notna() & df_eq["DATE"].notna()].copy()

    keep_cols = ["SYMBOL", "SERIES", "DATE", "PREV_CLOSE", "CLOSE_PRICE", "DELIV_QTY", "DELIV_PER"]
    present_cols = [c for c in keep_cols if c in df_eq.columns]
    df_eq = df_eq[present_cols].copy()
    df_eq["DATE"] = pd.to_datetime(df_eq["DATE"]).dt.date

    if logger:
        logger.info(f"Cleaned {os.path.basename(csv_path)} -> {len(df_eq)} EQ rows")
    return df_eq

# ----------------------
# Persistence
# ----------------------

def persist_to_sqlite(df: pd.DataFrame, db_path: str, logger=None):
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        df.to_sql("daily_bhav", conn, if_exists="append", index=False)
        # Index creation is idempotent
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON daily_bhav (SYMBOL, DATE)")
        except Exception:
            # SQLite sometimes returns error for CREATE INDEX in some contexts; ignore
            pass
    if logger:
        logger.info(f"Persisted {len(df)} rows to {db_path}")

# ----------------------
# Compute 20-day averages
# ----------------------

def compute_avg_deliv_qty_20d(engine_or_db_path: str, logger=None) -> pd.DataFrame:
    engine = create_engine(f"sqlite:///{engine_or_db_path}")
    try:
        df = pd.read_sql_table("daily_bhav", engine)
    except Exception:
        if logger:
            logger.warning("daily_bhav table not accessible when computing averages")
        return pd.DataFrame(columns=["SYMBOL", "AVG_DELIV_QTY_20D"])

    if df.empty:
        return pd.DataFrame(columns=["SYMBOL", "AVG_DELIV_QTY_20D"])

    if not pd.api.types.is_datetime64_any_dtype(df["DATE"]):
        df["DATE"] = pd.to_datetime(df["DATE"]).dt.date

    df_sorted = df.sort_values(["SYMBOL", "DATE"]).copy()

    def compute_group(g):
        g = g.copy()
        g["AVG20"] = g["DELIV_QTY"].rolling(window=20, min_periods=20).mean()
        return g

    df_roll = df_sorted.groupby("SYMBOL", group_keys=False).apply(compute_group)
    latest = df_roll.sort_values("DATE").groupby("SYMBOL", as_index=False).last()
    out = latest[["SYMBOL", "AVG20"]].rename(columns={"AVG20": "AVG_DELIV_QTY_20D"})
    out = out[out["AVG_DELIV_QTY_20D"].notna()].copy()
    if logger:
        logger.info(f"Computed AVG_DELIV_QTY_20D for {len(out)} symbols (>=20 sessions)")
    return out

# ----------------------
# Snapshot & filter
# ----------------------

def create_daily_snapshot_for_date(db_path: str, target_date: datetime.date, logger=None) -> pd.DataFrame:
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        df = pd.read_sql_table("daily_bhav", engine)
    except Exception:
        if logger:
            logger.error("daily_bhav table is missing when creating snapshot")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
    snapshot = df[df["DATE"] == target_date].copy()
    if snapshot.empty:
        if logger:
            logger.error(f"No rows for target date {target_date}")
        return pd.DataFrame()

    avg20 = compute_avg_deliv_qty_20d(db_path, logger=logger)
    merged = snapshot.merge(avg20, how="left", on="SYMBOL")

    merged["%CHANGE"] = ((merged["CLOSE_PRICE"] - merged["PREV_CLOSE"]) / merged["PREV_CLOSE"]) * 100
    merged["DELIVERY_TIMES"] = merged["DELIV_QTY"] / merged["AVG_DELIV_QTY_20D"]

    cols = ["SYMBOL", "DATE", "PREV_CLOSE", "CLOSE_PRICE", "DELIV_QTY", "DELIV_PER", "%CHANGE", "AVG_DELIV_QTY_20D", "DELIVERY_TIMES"]
    for c in cols:
        if c not in merged.columns:
            merged[c] = pd.NA
    merged = merged[cols]
    return merged


def filter_snapshot(df_snapshot: pd.DataFrame, logger=None) -> pd.DataFrame:
    if df_snapshot.empty:
        return df_snapshot

    filtered = df_snapshot.copy()
    filtered = filtered[filtered["DELIV_QTY"].fillna(0) >= 10_000]
    filtered = filtered[filtered["%CHANGE"].fillna(-9999) > 0]
    filtered = filtered[filtered["DELIVERY_TIMES"].fillna(0) >= 3]

    if logger:
        logger.info(f"Filtered snapshot: {len(df_snapshot)} -> {len(filtered)} rows")
    return filtered

# ----------------------
# Telegram helper
# ----------------------

def send_csv_to_telegram(token: str, chatid: str, csv_path: str, logger=None):
    send_url = f"https://api.telegram.org/bot{token}/sendDocument"
    try:
        with open(csv_path, "rb") as f:
            files = {"document": (os.path.basename(csv_path), f)}
            data = {"chat_id": chatid}
            resp = requests.post(send_url, data=data, files=files, timeout=30)
        if resp.status_code == 200:
            if logger:
                logger.info("Sent CSV to Telegram successfully")
            return True
        else:
            if logger:
                logger.warning(f"Telegram API returned status {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        if logger:
            logger.exception(f"Failed to send to Telegram: {e}")
        return False

# ----------------------
# CLI & main
# ----------------------

def load_config_file(path: str):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def write_tests(target_dir: str = "tests"):
    ensure_dir(target_dir)
    test_date_logic = '''
import datetime
from nse_high_delivery_scanner import is_weekend

def test_is_weekend():
    assert is_weekend(datetime.datetime(2023,9,16)) == True  # Saturday
    assert is_weekend(datetime.datetime(2023,9,15)) == False # Friday
'''
    with open(os.path.join(target_dir, "test_date_logic.py"), "w") as f:
        f.write(test_date_logic)
    print(f"Wrote pytest files to {target_dir}. Run `pytest {target_dir}` to execute them.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="NSE High-Delivery Scanner")
    parser.add_argument("--days_back", type=int, default=DEFAULTS["days_back"])
    parser.add_argument("--max_retries_today", type=int, default=DEFAULTS["max_retries_today"])
    parser.add_argument("--download_dir", type=str, default=DEFAULTS["download_dir"])
    parser.add_argument("--output_dir", type=str, default=DEFAULTS["output_dir"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--telegram-token", type=str, default=None)
    parser.add_argument("--telegram-chatid", type=str, default=None)
    parser.add_argument("--write-tests", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config_file(args.config)
    days_back = cfg.get("days_back", args.days_back)
    max_retries_today = cfg.get("max_retries_today", args.max_retries_today)
    download_dir = cfg.get("download_dir", args.download_dir)
    output_dir = cfg.get("output_dir", args.output_dir)
    sqlite_db = os.path.join(download_dir, cfg.get("sqlite_db", DEFAULTS["sqlite_db"]))

    ensure_dir(download_dir)
    ensure_dir(output_dir)

    run_date = datetime.now().strftime("%Y%m%d")
    log_path = os.path.join(output_dir, f"scan_{run_date}.log")
    logger = setup_logging(log_path)

    if args.write_tests:
        write_tests()
        return

    headers = {"User-Agent": cfg.get("user_agent", DEFAULTS["user_agent"]) }

    logger.info("Starting NSE High-Delivery Scanner")
    logger.info(f"Params: days_back={days_back}, download_dir={download_dir}")

    possible_uploaded = "/mnt/data"
    if os.path.exists(possible_uploaded):
        for fname in os.listdir(possible_uploaded):
            if fname.startswith("sec_bhavdata_full_") and fname.endswith(".csv"):
                src = os.path.join(possible_uploaded, fname)
                dst = os.path.join(download_dir, fname)
                if not os.path.exists(dst):
                    try:
                        with open(src, "rb") as fr, open(dst, "wb") as fw:
                            fw.write(fr.read())
                        logger.info(f"Copied user-uploaded bhavcopy {src} -> {dst}")
                    except Exception:
                        logger.exception(f"Failed copying {src} to {dst}")

    downloaded = collect_last_n_trading_days(days_back, download_dir, headers, DEFAULTS["timeout"], max_retries_today=max_retries_today, logger=logger)

    for d, path, ok in downloaded:
        if not ok or not path:
            logger.info(f"Skipping {d.date() if d else 'unknown'} - no file")
            continue
        try:
            cleaned = read_and_clean_bhavcopy(path, logger=logger)
            if not cleaned.empty:
                persist_to_sqlite(cleaned, sqlite_db, logger=logger)
        except Exception as e:
            logger.exception(f"Error processing {path}: {e}")

    engine = create_engine(f"sqlite:///{sqlite_db}")
    inspector = inspect(engine)
    if not inspector.has_table("daily_bhav"):
        logger.error("No daily_bhav table found in DB. No cleaned EQ rows were persisted. Exiting.")
        return

    try:
        df_all = pd.read_sql_table("daily_bhav", engine)
    except Exception as e:
        logger.exception(f"Failed to read DB: {e}")
        return

    if df_all.empty:
        logger.error("No data persisted; exiting")
        return
    df_all["DATE"] = pd.to_datetime(df_all["DATE"]).dt.date
    latest_date = df_all["DATE"].max()
    logger.info(f"Latest date in DB: {latest_date}")

    snapshot = create_daily_snapshot_for_date(sqlite_db, latest_date, logger=logger)
    if snapshot.empty:
        logger.error("Snapshot empty; exiting")
        return

    filtered = filter_snapshot(snapshot, logger=logger)

    out_csv = os.path.join(output_dir, f"high_delivery_stocks_{latest_date.strftime('%Y%m%d')}.csv")
    filtered.to_csv(out_csv, index=False)
    logger.info(f"Wrote filtered CSV to {out_csv}")

    total_processed = len(snapshot)
    total_kept = len(filtered)
    total_skipped = total_processed - total_kept

    logger.info(f"Records processed: {total_processed}, kept: {total_kept}, skipped: {total_skipped}")

    if args.telegram_token and args.telegram_chatid:
        ok = send_csv_to_telegram(args.telegram_token, args.telegram_chatid, out_csv, logger=logger)
        if not ok:
            logger.warning("Telegram upload failed")

    logger.info("Scan completed")


if __name__ == "__main__":
    main()
