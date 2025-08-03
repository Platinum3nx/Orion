import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import numpy as np


# --- Core Data Utilities ---

def fetch_ohlc_data(coin_id, vs_currency, days):
    """
    Fetches OHLC (Open, High, Low, Close) data from CoinGecko /ohlc endpoint.
    Granularity is automatic based on 'days'.
    - '1': Returns 30-minute data (last 1-2 days).
    - '7': Returns 4-hour data (last 3-30 days).
    - '30' or 'max': Returns 4-day data (31+ days, up to max available for free tier).
    """
    api_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency={vs_currency}&days={days}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching OHLC data: {e}")
        return None
    except json.JSONDecodeError:
        print("Error: Could not decode JSON response from OHLC API.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data fetching: {e}")
        return None


def save_dataframe_to_csv(df, filename):
    """Saves a DataFrame to a CSV file."""
    df.to_csv(filename)
    print(f"Data successfully saved to '{filename}'")


def load_dataframe_from_csv(filename):
    """Loads a DataFrame from a CSV file."""
    try:
        df = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
        print(f"Data successfully loaded from '{filename}'")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found. Please ensure the file exists in the correct directory.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading data from CSV: {e}")
        return None


# --- Pattern Detection Module: Liquidity Sweeps ---
# This section contains all logic specific to liquidity sweeps.

def find_swing_highs(df, window=5, price_tolerance_percent=0.005):
    """
    Identifies significant swing highs (peaks) in the price data.
    A high is a swing high if it's the highest in 'window' bars before and after.
    Returns a list of dictionaries: [{'timestamp': ..., 'value': ...}]
    """
    swing_points = []
    if 'high' not in df.columns:
        print("DataFrame must contain 'high' column for swing high detection.")
        return []

    for i in range(window, len(df) - window):
        current_high = df['high'].iloc[i]
        if current_high == df['high'].iloc[i - window: i + window + 1].max():
            swing_points.append({'timestamp': df.index[i], 'value': current_high})

    # Consolidate nearby levels (optional, but good for cleaner results)
    if not swing_points: return []

    # Sort by price to process highest first for consolidation
    swing_points.sort(key=lambda x: x['value'], reverse=True)

    consolidated_highs = []
    for sh in swing_points:
        is_unique = True
        for existing_sh in consolidated_highs:
            if abs(sh['value'] - existing_sh['value']) / existing_sh['value'] < price_tolerance_percent:
                is_unique = False
                break
        if is_unique:
            consolidated_highs.append(sh)

    # Sort by timestamp for consistent order
    consolidated_highs.sort(key=lambda x: x['timestamp'])
    return consolidated_highs  # Return full swing point objects


def detect_bearish_liquidity_sweeps(df, price_swing_highs_raw, wick_ratio_threshold=0.6, close_below_peak_factor=0.99):
    """
    Detects conceptual bearish liquidity sweeps.
    Looks for a candle whose high sweeps above a previous swing high,
    but then closes significantly lower, indicating rejection.

    Args:
        df (pd.DataFrame): OHLC DataFrame with 'open', 'high', 'low', 'close' columns.
        price_swing_highs_raw (list): List of identified swing high dictionaries
                                      (e.g., from find_swing_highs).
        wick_ratio_threshold (float): Minimum ratio of upper wick length to total candle range.
        close_below_peak_factor (float): Factor to determine if close is "significantly below" the swept peak.
                                         e.g., 0.99 means close must be < 99% of the swept_level.

    Returns:
        list of dict: Detected bearish liquidity sweep events.
                      Each dict contains: 'timestamp', 'type', 'swept_level', 'sweep_high', 'close_price', 'reason'.
    """
    sweeps = []
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        print("DataFrame must contain OHLC columns for liquidity sweep detection.")
        return []

    # Extract just the price values from swing_highs for easier comparison
    swing_high_prices = [sh['value'] for sh in price_swing_highs_raw]

    for i in range(1, len(df)):  # Start from the second candle
        current_candle = df.iloc[i]
        prev_candle = df.iloc[i - 1]

        swept_level = None
        # Check if current candle's high sweeps above a previous swing high
        for sh_price in swing_high_prices:
            # Ensure current candle's high is above the swing high
            # And that the previous candle's high was below or at that swing high (to confirm a "sweep")
            if current_candle['high'] > sh_price and prev_candle['high'] <= sh_price:
                swept_level = sh_price
                break  # Found a swept level

        if swept_level is not None:
            total_range = current_candle['high'] - current_candle['low']
            if total_range == 0: continue  # Avoid division by zero for flat candles

            upper_wick = current_candle['high'] - max(current_candle['open'], current_candle['close'])

            is_long_upper_wick = (upper_wick / total_range) > wick_ratio_threshold
            is_bearish_close = current_candle['close'] < current_candle['open']
            is_strong_rejection = current_candle['close'] < (swept_level * close_below_peak_factor)

            if is_long_upper_wick and is_bearish_close and is_strong_rejection:
                sweeps.append({
                    'timestamp': current_candle.name,
                    'type': 'Bearish Liquidity Sweep',
                    'swept_level': swept_level,
                    'sweep_high': current_candle['high'],
                    'close_price': current_candle['close'],
                    'reason': f"Candle wicked above previous swing high ({swept_level:.2f}) with long upper wick and closed significantly lower."
                })
    return sweeps


# --- Visualization Module ---
# This section handles plotting all detected patterns.

def plot_analysis_results(df, price_swing_high_levels, bearish_sweeps,
                          coin_id, vs_currency, days_to_fetch):
    """
    Generates a plot showing price, swing highs, and detected bearish liquidity sweeps.
    """
    plt.figure(figsize=(16, 9))

    # Plot the close price
    plt.plot(df.index, df['close'], label='Close Price', color='blue', linewidth=1.5, alpha=0.8)

    # Plot Swing Highs as horizontal lines
    if price_swing_high_levels:
        plt.axhline(y=price_swing_high_levels[0], color='gray', linestyle='--', linewidth=0.8, alpha=0.7,
                    label='Key Swing Highs')
        for level in price_swing_high_levels[1:]:
            plt.axhline(y=level, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, label='_nolegend_')

    # Mark Bearish Liquidity Sweeps
    for sweep in bearish_sweeps:
        plt.scatter(sweep['timestamp'], sweep['sweep_high'], color='red', marker='X', s=250, zorder=5,
                    label='Bearish Sweep' if sweep == bearish_sweeps[0] else "_nolegend_")
        plt.plot([sweep['timestamp'], sweep['timestamp']], [sweep['sweep_high'], sweep['close_price']], color='red',
                 linewidth=2, linestyle='-', alpha=0.8, label="_nolegend_")
        plt.axhline(y=sweep['swept_level'], color='orange', linestyle=':', linewidth=1, alpha=0.7, label='_nolegend_')

    plt.title(
        f'{coin_id.capitalize()} Price & Bearish Liquidity Sweeps ({days_to_fetch} Day Data - 30-min Granularity)',
        fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel(f'Price ({vs_currency.upper()})', fontsize=14)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), fontsize=10, loc='best')

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# --- Main Orchestration ---
def run_analysis(coin_id, vs_currency, days_to_fetch,
                 ls_swing_window, ls_price_tolerance,
                 ls_wick_threshold, ls_close_factor):
    print(
        f"--- Starting analysis for {coin_id.capitalize()} ({vs_currency.upper()}) for last {days_to_fetch} day(s) ---")

    # 1. Data Acquisition
    print("\nFetching OHLC data from CoinGecko...")
    df = fetch_ohlc_data(coin_id, vs_currency, days_to_fetch)

    if df is None or df.empty:
        print("Failed to fetch data or DataFrame is empty. Exiting.")
        return

    csv_filename = f"{coin_id}_ohlc_data_{days_to_fetch}d.csv"
    save_dataframe_to_csv(df, csv_filename)
    df = load_dataframe_from_csv(csv_filename)

    print("\nDataFrame loaded. First 5 rows:")
    print(df.head())
    print("\nDataFrame Info:")
    df.info()

    # 2. Find Swing Highs (needed for liquidity sweeps)
    print("\nIdentifying Swing Highs...")
    price_swing_highs_raw = find_swing_highs(df, window=ls_swing_window, price_tolerance_percent=ls_price_tolerance)
    price_swing_high_levels = [sh['value'] for sh in price_swing_highs_raw]

    if price_swing_high_levels:
        print(f"Identified Price Swing Highs: {price_swing_high_levels}")
    else:
        print("No significant Price Swing Highs identified. Adjust 'ls_swing_window' or 'ls_price_tolerance'.")

    # 3. Detect Bearish Liquidity Sweeps
    print("\nDetecting Bearish Liquidity Sweeps...")
    bearish_sweeps_results = detect_bearish_liquidity_sweeps(df, price_swing_highs_raw,
                                                             wick_ratio_threshold=ls_wick_threshold,
                                                             close_below_peak_factor=ls_close_factor)

    if bearish_sweeps_results:
        print(f"\nIdentified {len(bearish_sweeps_results)} Potential Bearish Liquidity Sweeps:")
        for sweep in bearish_sweeps_results:
            print(
                f"  At {sweep['timestamp'].strftime('%Y-%m-%d %H:%M')}: Swept Level={sweep['swept_level']:.2f}, Sweep High={sweep['sweep_high']:.2f}, Close={sweep['close_price']:.2f}. Reason: {sweep['reason']}")
    else:
        print("\nNo potential bearish liquidity sweeps detected with current parameters.")

    # 4. Compare Results (Currently just for Liquidity Sweeps)
    print("\n--- Summary of Detected Patterns ---")
    if bearish_sweeps_results:
        print(f"Total Bearish Liquidity Sweeps: {len(bearish_sweeps_results)}")
    else:
        print("No bearish liquidity sweeps detected.")

    # This is where you would add logic to combine or compare results from multiple indicators
    # For example:
    # if bearish_sweeps_results and bullish_rsi_divergences:
    #     print("Consider filtering sweeps that also have bullish RSI divergence for confirmation.")

    # 5. Visualize Results
    print("\nGenerating chart with detected patterns...")
    plot_analysis_results(df, price_swing_high_levels, bearish_sweeps_results,
                          coin_id, vs_currency, days_to_fetch)


if __name__ == "__main__":
    # --- Main Configuration Parameters ---
    coin_to_analyze = "bitcoin"
    currency_to_compare = "usd"
    # Set to '1' for the highest granularity (30-minute candles) available from /ohlc free tier.
    data_days_ago = "1"

    # --- Liquidity Sweep Parameters ---
    ls_swing_window = 5
    ls_price_tolerance = 0.005  # 0.5%
    ls_wick_threshold = 0.6
    ls_close_factor = 0.99

    run_analysis(
        coin_id=coin_to_analyze,
        vs_currency=currency_to_compare,
        days_to_fetch=data_days_ago,
        ls_swing_window=ls_swing_window,
        ls_price_tolerance=ls_price_tolerance,
        ls_wick_threshold=ls_wick_threshold,
        ls_close_factor=ls_close_factor
    )
