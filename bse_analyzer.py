#!/usr/bin/env python3
"""
bse_high_volume_scanner.py
Automated BSE High-Volume Scanner.
Features:
 - Deduplicate database rows per (SYMBOL, DATE) before analysis,
   keeping the row with the largest TtlTradVol.
 - Ensure final output contains only one row per SYMBOL for today's snapshot.
 - Robust CSV reading + normalization
 - Stores cleaned rows into SQLite
 - Computs average volume over five most recent distinct prior dates (DATE < target_date)
 - Filters: TtlTradVol >= 10_000, %CHANGE > 0, VOLUME_TIMES >= 3
 - Outputs CSV and log; optional Telegram upload
 - Enriches final output with Market Cap from Screener.in and filters for >= ₹100 Cr.
"""
import argparse
import logging
import os
import sys
import time
import requests
import re
from datetime import datetime, timedelta, date as datetime_date
from bs4 import BeautifulSoup
import pandas as pd
#import yaml
from sqlalchemy import create_engine, inspect

# ----------------------
# Defaults
# ----------------------
DEFAULTS = {
    "days_back": 8,
    "max_retries_today": 5,
    "download_dir": "./raw",
    "output_dir": "./output",
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 (compatible; bse-high-volume-scanner/1.0)",
    "timeout": 15,
    "sqlite_db": "bhavcopy_bse.sqlite",
}

BASE_URL = "https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{date}_F_0000.CSV"

# ----------------------
# Helpers
# ----------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def setup_logging(log_path=None):
    ensure_dir(os.path.dirname(log_path) or ".")
    logger = logging.getLogger("bse_scanner")
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
    return f"BhavCopy_BSE_CM_0_0_0_{d.strftime('%Y%m%d')}_F_0000.CSV"

def is_weekend(d: datetime) -> bool:
    return d.weekday() >= 5

# ----------------------
# Download
# ----------------------
def download_bhavcopy_for_date(date: datetime, download_dir: str, headers: dict, timeout: int, logger=None):
    filename = date_to_filename(date)
    url = BASE_URL.format(date=date.strftime("%Y%m%d"))
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
# Robust CSV reader
# ----------------------
def read_and_clean_bse_bhavcopy(csv_path: str, logger=None) -> pd.DataFrame:
    """
    Robust reader for BSE bhavcopy CSV.
    - Tries multiple encodings/delimiters.
    - Normalizes column names.
    - Maps BSE columns to a standard format.
    - Returns a DataFrame with cleaned data.
    """
    import os
    read_attempts = [
        {"encoding": "utf-8", "sep": ","},
        {"encoding": "latin1", "sep": ","},
    ]
    df = None
    last_err = None
    for opt in read_attempts:
        try:
            # BSE files can have initial rows that are not data, let's try to find the header
            i = 0
            with open(csv_path, 'r', encoding=opt['encoding']) as f:
                for i, line in enumerate(f):
                    if 'TRADDT' in line.upper() or 'TRADE DATE' in line.upper():
                        break
            df = pd.read_csv(csv_path, **opt, engine="python", skiprows=i)
            if logger:
                logger.info(f"Read {os.path.basename(csv_path)} with options {opt}, shape={df.shape}")
            break
        except Exception as e:
            last_err = e
            if logger:
                logger.debug(f"Read attempt failed for {csv_path} with {opt}: {e}")

    if df is None:
        if logger:
            logger.exception(f"All read attempts failed for {csv_path}: {last_err}")
        raise

    # Normalize column names
    df.columns = [c.strip().upper() for c in df.columns]

    # Map BSE columns to our internal standard names
    # Based on the actual downloaded CSV file
    column_map = {
        "TCKRSYMB": "SYMBOL",
        "TRADDT": "DATE",
        "CLSPRIC": "CLOSE_PRICE",
        "PRVSCLSGPRIC": "PREV_CLOSE",
        "TTLTRADGVOL": "DELIV_QTY", # As per user instruction for this task
        "OPNPRIC": "OPEN_PRICE",
        "HGHPRIC": "HIGH_PRICE",
        "LWPRIC": "LOW_PRICE",
        "TTLTRFVAL": "TURNOVER_LACS"
    }
    df.rename(columns=column_map, inplace=True)

    # Date conversion
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    else:
        df["DATE"] = pd.NaT

    # Numeric cleanup
    # User wants to use TtlTradVol as delivery quantity
    numeric_cols = [
        "PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE",
        "CLOSE_PRICE", "DELIV_QTY", "TURNOVER_LACS"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(",", "", regex=False)
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "SYMBOL" not in df.columns:
        if logger:
            logger.error(f"Missing SYMBOL (from TOTALSYMB) column in {csv_path}")
        return pd.DataFrame()

    df_clean = df[df["SYMBOL"].notna() & df["DATE"].notna()].copy()

    # We don't have DELIV_PER, so we can't use it.
    keep_cols = ["SYMBOL", "DATE", "PREV_CLOSE", "CLOSE_PRICE", "DELIV_QTY"]
    present_cols = [c for c in keep_cols if c in df_clean.columns]
    df_clean = df_clean[present_cols].copy()

    # Persist DATE as date-only (python date)
    df_clean["DATE"] = pd.to_datetime(df_clean["DATE"]).dt.date

    if logger:
        logger.info(f"Cleaned {os.path.basename(csv_path)} -> {len(df_clean)} rows")
    return df_clean

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
            pass
    if logger:
        logger.info(f"Persisted {len(df)} rows to {db_path}")

# ----------------------
# DB dedupe: keep one row per (SYMBOL, DATE)
# ----------------------
def dedupe_db_by_symbol_date(db_path: str, logger=None):
    """
    Read the daily_bhav table, for each (SYMBOL, DATE) keep the row with largest DELIV_QTY.
    Overwrites the daily_bhav table with the deduplicated version.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        df = pd.read_sql_table("daily_bhav", engine)
    except Exception:
        if logger:
            logger.warning("daily_bhav table not found for deduplication")
        return
    if df.empty:
        return
    # Normalize DATE column to python date objects
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.date
    # Sort to keep the row with max DELIV_QTY for each (SYMBOL, DATE)
    if "DELIV_QTY" in df.columns:
        df_sorted = df.sort_values(["SYMBOL", "DATE", "DELIV_QTY"], ascending=[True, True, False])
    else:
        df_sorted = df.sort_values(["SYMBOL", "DATE"], ascending=[True, True])
    df_dedup = df_sorted.drop_duplicates(subset=["SYMBOL", "DATE"], keep="first").reset_index(drop=True)
    # Overwrite table
    with engine.begin() as conn:
        df_dedup.to_sql("daily_bhav", conn, if_exists="replace", index=False)
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON daily_bhav (SYMBOL, DATE)")
        except Exception:
            pass
    if logger:
        logger.info(f"Deduplicated DB: reduced to {len(df_dedup)} rows (unique SYMBOL+DATE)")

# ----------------------
# Compute 5-prior average (EXCLUDING target date)
# ----------------------
def compute_avg_deliv_qty_5_prior(db_path: str, target_date, logger=None) -> pd.DataFrame:
    """
    For each SYMBOL, calculate the mean of DELIV_QTY using the five most recent valid data points
    with DATE < target_date. Only symbols with >=5 prior distinct sessions are returned.
    Returns DataFrame: SYMBOL, AVG_DELIV_QTY_5D_PRIOR
    """
    # Normalize target_date -> python date
    if isinstance(target_date, pd.Timestamp):
        target_date = target_date.date()
    elif isinstance(target_date, datetime):
        target_date = target_date.date()
    elif isinstance(target_date, str):
        target_date = pd.to_datetime(target_date).date()
    elif isinstance(target_date, datetime_date):
        pass
    else:
        target_date = pd.to_datetime(target_date).date()
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        df = pd.read_sql_table("daily_bhav", engine)
    except Exception:
        if logger:
            logger.warning("daily_bhav table not accessible when computing 5-prior averages")
        return pd.DataFrame(columns=["SYMBOL", "AVG_DELIV_QTY_5D_PRIOR"])
    if df.empty:
        return pd.DataFrame(columns=["SYMBOL", "AVG_DELIV_QTY_5D_PRIOR"])
    # Ensure DATE is converted to python date objects for safe comparison
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.date
    # Consider only rows strictly before target_date
    df_prior = df[df["DATE"] < target_date].copy()
    if df_prior.empty:
        return pd.DataFrame(columns=["SYMBOL", "AVG_DELIV_QTY_5D_PRIOR"])
    results = []
    for sym, g in df_prior.groupby("SYMBOL"):
        g_sorted = g.sort_values("DATE", ascending=False)
        # drop duplicate calendar dates (shouldn't be needed after DB dedupe, but keep as safety)
        g_unique_dates = g_sorted.drop_duplicates(subset="DATE", keep="first")
        vals = g_unique_dates["DELIV_QTY"].dropna().astype(float).head(5).values
        if len(vals) >= 5:
            avg = float(pd.Series(vals).mean())
            results.append({"SYMBOL": sym, "AVG_DELIV_QTY_5D_PRIOR": avg})
    out = pd.DataFrame(results)
    if logger:
        logger.info(f"Computed 5-prior avg for {len(out)} symbols (>=5 prior distinct dates)")
    return out

# ----------------------
# Snapshot & filter
# ----------------------
def create_daily_snapshot_for_date(db_path: str, target_date, logger=None) -> pd.DataFrame:
    # Normalize target_date to python date
    if isinstance(target_date, pd.Timestamp):
        target_date = target_date.date()
    elif isinstance(target_date, datetime):
        target_date = target_date.date()
    elif isinstance(target_date, str):
        target_date = pd.to_datetime(target_date).date()
    elif isinstance(target_date, datetime_date):
        pass
    else:
        target_date = pd.to_datetime(target_date).date()
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        df = pd.read_sql_table("daily_bhav", engine)
    except Exception:
        if logger:
            logger.error("daily_bhav table is missing when creating snapshot")
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    # Normalize DATE column to python date (ensures consistent dtype)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.date
    # Keep only rows for target_date
    snapshot = df[df["DATE"] == target_date].copy()
    if snapshot.empty:
        if logger:
            logger.error(f"No rows for target date {target_date}")
        return pd.DataFrame()
    # Deduplicate snapshot by SYMBOL if multiple rows remain (keep row with highest DELIV_QTY)
    if "DELIV_QTY" in snapshot.columns:
        snapshot_sorted = snapshot.sort_values(["SYMBOL", "DELIV_QTY"], ascending=[True, False])
    else:
        snapshot_sorted = snapshot.sort_values("SYMBOL")
    snapshot = snapshot_sorted.drop_duplicates(subset=["SYMBOL"], keep="first").reset_index(drop=True)
    avg5 = compute_avg_deliv_qty_5_prior(db_path, target_date, logger=logger)
    merged = snapshot.merge(avg5, how="left", on="SYMBOL")
    merged["%CHANGE"] = ((merged["CLOSE_PRICE"] - merged["PREV_CLOSE"]) / merged["PREV_CLOSE"]) * 100
    merged["DELIVERY_TIMES"] = merged["DELIV_QTY"] / merged["AVG_DELIV_QTY_5D_PRIOR"]
    # DELIV_PER is not available in BSE data
    cols = ["SYMBOL", "DATE", "PREV_CLOSE", "CLOSE_PRICE", "DELIV_QTY", "%CHANGE", "AVG_DELIV_QTY_5D_PRIOR", "DELIVERY_TIMES"]
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
    # Also drop rows where AVG_DELIV_QTY_5D_PRIOR is null (i.e., <5 prior sessions)
    filtered = filtered[filtered["AVG_DELIV_QTY_5D_PRIOR"].notna()]
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
# Market Cap Enrichment (from second script)
# ----------------------
def get_market_cap_screener(stock_slug: str, logger=None):
    """
    Fetches the Market Cap for a company from Screener.in (standalone page only)
    stock_slug: the slug used in Screener URL, e.g. "JAIBALAJI"
    Returns: tuple (market_cap_string, market_cap_in_cr) or (None, None) if not found
    """
    base = "https://www.screener.in/company"
    url = f"{base}/{stock_slug}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            if logger:
                logger.warning(f"Failed to get page for {stock_slug}, status code: {resp.status_code}")
            return None, None
    except requests.RequestException as e:
        if logger:
            logger.warning(f"Request failed for {stock_slug}: {e}")
        return None, None
    soup = BeautifulSoup(resp.text, "html.parser")
    # Find all <li> tags
    li_tags = soup.find_all("li")
    for li in li_tags:
        # Look for "Market Cap" in any span or direct text
        if li.find(string=lambda text: text and "Market Cap" in text.strip()):
            # Try to find span with class "number"
            num_span = li.find("span", {"class": "number"})
            if num_span:
                cap_str = num_span.text.strip()
            else:
                # Fallback: extract text and remove "Market Cap"
                full_text = li.get_text().strip()
                cap_str = full_text.replace("Market Cap", "", 1).strip(": \t")
            # Parse numeric value from string like "â‚¹ 9,287 Cr." or "â‚¹ 87 Cr."
            cap_clean = re.sub(r'[^\d.,]', '', cap_str).replace(',', '')
            try:
                cap_value = float(cap_clean)
            except ValueError:
                if logger:
                    logger.warning(f"Could not parse market cap value: {cap_str} for {stock_slug}")
                cap_value = None
            return cap_str, cap_value
    return None, None

def enrich_with_market_cap(input_csv_path: str, logger=None) -> str:
    """
    Reads the scanner's output CSV, enriches it with market cap data from Screener.in,
    filters for Market Cap >= 100 Cr, and saves a new CSV.
    Returns the path to the new, enriched CSV file.
    """
    if not os.path.exists(input_csv_path):
        if logger:
            logger.error(f"Input CSV {input_csv_path} not found for enrichment.")
        return input_csv_path

    df = pd.read_csv(input_csv_path)

    # Initialize new columns
    df['MARKET_CAP'] = None
    df['MARKET_CAP_CR'] = None  # Numeric column for filtering
    df['SOURCE_URL'] = None
    df['TRADINGVIEW_LINK'] = None

    # Iterate through each row and fetch market cap
    for index, row in df.iterrows():
        symbol = row['SYMBOL']
        if logger:
            logger.info(f"Fetching market cap for {symbol}...")
        # Generate URLs
        screener_url = f"https://www.screener.in/company/{symbol}/"
        tradingview_url = f"https://in.tradingview.com/chart/ob4F1XSN/?symbol=BSE:{symbol}/"
        # Fetch market cap
        market_cap_str, market_cap_num = get_market_cap_screener(symbol, logger=logger)
        # Assign values
        df.at[index, 'MARKET_CAP'] = market_cap_str
        df.at[index, 'MARKET_CAP_CR'] = market_cap_num
        df.at[index, 'SOURCE_URL'] = screener_url
        df.at[index, 'TRADINGVIEW_LINK'] = tradingview_url
        # Add delay to be respectful
        time.sleep(1)

    # Filter out rows where MARKET_CAP_CR < 100 or is NaN
    df_filtered = df[df['MARKET_CAP_CR'] >= 100].copy()
    # Drop the helper column if not needed in final output
    df_filtered = df_filtered.drop(columns=['MARKET_CAP_CR'])

    # Create new filename
    base_name = os.path.splitext(input_csv_path)[0]
    output_filename = f"{base_name}_with_market_cap_filtered.csv"
    df_filtered.to_csv(output_filename, index=False)

    if logger:
        logger.info(f"Market Cap enrichment completed. Filtered results (>= ₹100 Cr) saved to {output_filename}")

    return output_filename

# ----------------------
# CLI & main
# ----------------------
def load_config_file(path: str):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}

def main(argv=None):
    parser = argparse.ArgumentParser(description="BSE High-Volume Scanner")
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
    headers = {"User-Agent": cfg.get("user_agent", DEFAULTS["user_agent"])}
    logger.info("Starting BSE High-Volume Scanner")
    logger.info(f"Params: days_back={days_back}, download_dir={download_dir}")

    possible_uploaded = "/mnt/data"
    if os.path.exists(possible_uploaded):
        for fname in os.listdir(possible_uploaded):
            if fname.startswith("BhavCopy_BSE_CM_") and fname.endswith(".CSV"):
                src = os.path.join(possible_uploaded, fname)
                dst = os.path.join(download_dir, fname)
                if not os.path.exists(dst):
                    try:
                        with open(src, "rb") as fr, open(dst, "wb") as fw:
                            fw.write(fr.read())
                        logger.info(f"Copied user-uploaded bhavcopy {src} -> {dst}")
                    except Exception:
                        logger.exception(f"Failed copying {src} to {dst}")

    # Download / collect files
    downloaded = collect_last_n_trading_days(days_back, download_dir, headers, DEFAULTS["timeout"], max_retries_today=max_retries_today, logger=logger)

    # Read & persist
    for d, path, ok in downloaded:
        if not ok or not path:
            logger.info(f"Skipping {d.date() if d else 'unknown'} - no file")
            continue
        try:
            cleaned = read_and_clean_bse_bhavcopy(path, logger=logger)
            if not cleaned.empty:
                persist_to_sqlite(cleaned, sqlite_db, logger=logger)
        except Exception as e:
            logger.exception(f"Error processing {path}: {e}")

    # Deduplicate DB on SYMBOL+DATE (keeps row with largest DELIV_QTY)
    dedupe_db_by_symbol_date(sqlite_db, logger=logger)

    # Ensure DB exists and read latest date
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
    # normalize to python date objects
    df_all["DATE"] = pd.to_datetime(df_all["DATE"], errors="coerce").dt.date
    latest_date = df_all["DATE"].max()
    logger.info(f"Latest date in DB: {latest_date}")

    # Build today's snapshot (deduped)
    snapshot = create_daily_snapshot_for_date(sqlite_db, latest_date, logger=logger)
    if snapshot.empty:
        logger.error("Snapshot empty; exiting")
        return

    # Apply initial filters (delivery, %change, delivery times)
    filtered = filter_snapshot(snapshot, logger=logger)

    # Output (only today's rows, one per symbol)
    out_csv = os.path.join(output_dir, f"bse_high_volume_stocks_{latest_date.strftime('%Y%m%d')}.csv")
    filtered.to_csv(out_csv, index=False)
    logger.info(f"Wrote filtered CSV to {out_csv}")

    total_processed = len(snapshot)
    total_kept = len(filtered)
    total_skipped = total_processed - total_kept
    logger.info(f"Records processed: {total_processed}, kept: {total_kept}, skipped: {total_skipped}")

    # --- NEW STEP: Enrich with Market Cap ---
    final_csv = enrich_with_market_cap(out_csv, logger=logger)
    # Update out_csv to point to the final, enriched file for potential Telegram upload
    out_csv = final_csv

    if args.telegram_token and args.telegram_chatid:
        ok = send_csv_to_telegram(args.telegram_token, args.telegram_chatid, out_csv, logger=logger)
        if not ok:
            logger.warning("Telegram upload failed")

    logger.info("Scan and enrichment completed")

if __name__ == "__main__":
    main()