'''
This program fetches OHLC data from BTC(applicable to other cryptocurrencies, futures,
and mostly any liquid equity). It then analyzes for liquidity sweeps and conducts a 
multi-timeframe analysis to determine the likelihood of a reversal on the high 
time fram chart
'''
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
    - '1': Returns ~30-minute data.
    - '7': Returns ~4-hour data.
    - '30' or 'max': Returns ~4-day data.
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


# --- SMC Pattern Detection Module ---
# This section contains all logic specific to SMC concepts.

def find_swing_highs(df, window=5, price_tolerance_percent=0.005, is_high=True):
    """
    Identifies significant swing highs (peaks) or swing lows (troughs) in the price data.
    A point is a swing if it's the highest/lowest in 'window' bars before and after.
    Returns a list of dictionaries: [{'timestamp': ..., 'value': ...}]
    """
    swing_points = []
    series = df['high'] if is_high else df['low']
    if series.empty:
        return []

    for i in range(window, len(df) - window):
        current_val = series.iloc[i]
        is_swing = False
        if is_high and current_val == series.iloc[i - window: i + window + 1].max():
            is_swing = True
        elif not is_high and current_val == series.iloc[i - window: i + window + 1].min():
            is_swing = True

        if is_swing:
            swing_points.append({'timestamp': df.index[i], 'value': current_val})

    if not swing_points: return []

    # Consolidate nearby levels
    swing_points.sort(key=lambda x: x['value'], reverse=is_high)
    consolidated_points = []
    for sp in swing_points:
        is_unique = True
        for existing_sp in consolidated_points:
            if abs(sp['value'] - existing_sp['value']) / existing_sp['value'] < price_tolerance_percent:
                is_unique = False
                break
        if is_unique:
            consolidated_points.append(sp)

    consolidated_points.sort(key=lambda x: x['timestamp'])
    return consolidated_points


def detect_bearish_liquidity_sweeps(df, price_swing_highs_raw, wick_ratio_threshold=0.6, close_below_peak_factor=0.99):
    """
    Detects conceptual bearish liquidity sweeps.
    Looks for a candle whose high sweeps above a previous swing high,
    but then closes significantly lower, indicating rejection.

    Args:
        df (pd.DataFrame): OHLC DataFrame with 'open', 'high', 'low', 'close' columns.
        price_swing_highs_raw (list): List of identified swing high dictionaries.
        wick_ratio_threshold (float): Minimum ratio of upper wick length to total candle range.
        close_below_peak_factor (float): Factor to determine if close is "significantly below" the swept peak.

    Returns:
        list of dict: Detected bearish liquidity sweep events.
    """
    sweeps = []
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        return []

    swing_high_prices = [sh['value'] for sh in price_swing_highs_raw]

    for i in range(1, len(df)):
        current_candle = df.iloc[i]
        prev_candle = df.iloc[i - 1]

        swept_level = None
        for sh_price in swing_high_prices:
            if current_candle['high'] > sh_price and prev_candle['high'] <= sh_price:
                swept_level = sh_price
                break

        if swept_level is not None:
            total_range = current_candle['high'] - current_candle['low']
            if total_range == 0: continue

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


def detect_fair_value_gaps(df):
    """
    Identifies Fair Value Gaps (FVG) based on a three-candle pattern.
    A bullish FVG is when the high of candle 1 doesn't overlap with the low of candle 3.
    A bearish FVG is when the low of candle 1 doesn't overlap with the high of candle 3.
    Returns a list of dicts with FVG details.
    """
    fvg_bullish = []
    fvg_bearish = []

    for i in range(2, len(df)):
        candle1 = df.iloc[i - 2]
        candle2 = df.iloc[i - 1]
        candle3 = df.iloc[i]

        # Bullish FVG: a gap between the high of candle1 and the low of candle3
        if candle3['low'] > candle1['high']:
            fvg_bullish.append({
                'timestamp_start': candle1.name,
                'timestamp_end': candle3.name,
                'low_price': candle1['high'],
                'high_price': candle3['low'],
                'type': 'bullish'
            })

        # Bearish FVG: a gap between the low of candle1 and the high of candle3
        if candle3['high'] < candle1['low']:
            fvg_bearish.append({
                'timestamp_start': candle1.name,
                'timestamp_end': candle3.name,
                'low_price': candle3['high'],
                'high_price': candle1['low'],
                'type': 'bearish'
            })

    return fvg_bullish, fvg_bearish


def detect_order_blocks(df, impulse_factor=1.5):
    """
    Identifies a simplified Order Block.
    It looks for the last down-candle before a strong bullish move or the last up-candle
    before a strong bearish move.
    """
    ob_bullish = []
    ob_bearish = []

    for i in range(1, len(df)):
        current_candle = df.iloc[i]
        prev_candle = df.iloc[i - 1]

        # Check for bullish Order Block (last down-candle before a strong up-move)
        is_prev_down_candle = prev_candle['close'] < prev_candle['open']
        is_strong_up_move = (current_candle['high'] - current_candle['low']) > (
                    df['high'] - df['low']).mean() * impulse_factor

        if is_prev_down_candle and is_strong_up_move:
            ob_bullish.append({
                'timestamp_start': prev_candle.name,
                'timestamp_end': current_candle.name,
                'low_price': prev_candle['low'],
                'high_price': prev_candle['high'],
                'type': 'bullish'
            })

        # Check for bearish Order Block (last up-candle before a strong down-move)
        is_prev_up_candle = prev_candle['close'] > prev_candle['open']
        is_strong_down_move = (current_candle['high'] - current_candle['low']) > (
                    df['high'] - df['low']).mean() * impulse_factor

        if is_prev_up_candle and is_strong_down_move:
            ob_bearish.append({
                'timestamp_start': prev_candle.name,
                'timestamp_end': current_candle.name,
                'low_price': prev_candle['low'],
                'high_price': prev_candle['high'],
                'type': 'bearish'
            })

    return ob_bullish, ob_bearish


# --- Visualization Module ---

def plot_analysis_results(df_htf, df_ltf, htf_draws, ltf_reversals, ltf_continuations):
    """
    Generates a multi-panel plot showing HTF and LTF analysis.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [1, 3]}, sharex=False)

    # --- HTF Price Chart (ax1) ---
    ax1.plot(df_htf.index, df_htf['close'], label='Close Price (4hr)', color='blue', linewidth=2, alpha=0.8)

    # Plot HTF Draws on Liquidity (Swing Highs/Lows)
    if htf_draws:
        for point in htf_draws:
            color = 'red' if 'high' in point else 'green'
            ax1.axhline(y=point['value'], color=color, linestyle='--', linewidth=1, alpha=0.7, label='_nolegend_')

        # This part is for the legend, ensuring it only appears once
        ax1.axhline(y=htf_draws[0]['value'], color='gray', linestyle='--', linewidth=1, alpha=0.7,
                    label='HTF Draws on Liquidity')

    ax1.set_title('High Timeframe Analysis (4-hour)', fontsize=18)
    ax1.set_ylabel('Price', fontsize=14)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # --- LTF Price Chart (ax2) ---
    ax2.plot(df_ltf.index, df_ltf['close'], label='Close Price (30min)', color='purple', linewidth=1.5, alpha=0.8)

    # Mark LTF Reversals (Liquidity Sweeps)
    if ltf_reversals:
        # Use a dummy plot for the legend label
        ax2.scatter([], [], color='red', marker='X', s=100, label='Bearish Sweep')
        for sweep in ltf_reversals:
            ax2.scatter(sweep['timestamp'], sweep['sweep_high'], color='red', marker='X', s=100, zorder=5)

    # Mark FVG (Continuation Confluence)
    bullish_fvgs = [f for f in ltf_continuations['FVG'][0]]
    bearish_fvgs = [f for f in ltf_continuations['FVG'][1]]

    # Use dummy plots for legend labels for FVG
    if bullish_fvgs:
        ax2.axhspan(bullish_fvgs[0]['low_price'], bullish_fvgs[0]['high_price'], facecolor='green', alpha=0.2,
                    label='Bullish FVG')
        for fvg in bullish_fvgs[1:]:
            ax2.axhspan(fvg['low_price'], fvg['high_price'], facecolor='green', alpha=0.2)
    if bearish_fvgs:
        ax2.axhspan(bearish_fvgs[0]['low_price'], bearish_fvgs[0]['high_price'], facecolor='red', alpha=0.2,
                    label='Bearish FVG')
        for fvg in bearish_fvgs[1:]:
            ax2.axhspan(fvg['low_price'], bearish_fvgs[0]['high_price'], facecolor='red', alpha=0.2)

    ax2.set_title('Low Timeframe Analysis (30-minute)', fontsize=18)
    ax2.set_xlabel('Date', fontsize=14)
    ax2.set_ylabel('Price', fontsize=14)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, linestyle=':', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# --- Main Orchestration ---

def run_multi_timeframe_analysis(coin_id, vs_currency):
    """
    Orchestrates the entire multi-timeframe analysis.
    """
    # Configuration for data fetching
    htf_days = '30'  # Gets ~4-hour data
    ltf_days = '1'  # Gets ~30-minute data

    # 1. Fetch High Timeframe (HTF) Data
    print(f"--- Fetching HTF ({htf_days} days) data for {coin_id.capitalize()} ---")
    df_htf = fetch_ohlc_data(coin_id, vs_currency, htf_days)
    if df_htf is None or df_htf.empty:
        print("Failed to fetch HTF data. Exiting.")
        return

    # 2. Fetch Low Timeframe (LTF) Data
    print(f"--- Fetching LTF ({ltf_days} day) data for {coin_id.capitalize()} ---")
    df_ltf = fetch_ohlc_data(coin_id, vs_currency, ltf_days)
    if df_ltf is None or df_ltf.empty:
        print("Failed to fetch LTF data. Exiting.")
        return

    # 3. HTF Analysis: Identify Draws on Liquidity
    print("\n--- HTF Analysis: Identifying Draws on Liquidity ---")
    htf_swing_highs = find_swing_highs(df_htf, window=5, is_high=True)
    htf_swing_lows = find_swing_highs(df_htf, window=5, is_high=False)
    htf_draws_on_liquidity = htf_swing_highs + htf_swing_lows

    # 4. LTF Analysis: Detect Reversals & Continuation Confluence
    print("\n--- LTF Analysis: Detecting Patterns ---")
    # Need HTF swing highs for the LTF sweep detection logic
    ltf_reversals_sweeps = detect_bearish_liquidity_sweeps(df_ltf, htf_swing_highs, wick_ratio_threshold=0.6,
                                                           close_below_peak_factor=0.99)

    # Detect continuation confluence
    ltf_continuation_confluence = {
        'FVG': detect_fair_value_gaps(df_ltf),
        'OrderBlocks': detect_order_blocks(df_ltf),
    }

    # 5. Summarize and Plot
    print("\n--- Summary of Detected Patterns ---")
    print(f"HTF Draws on Liquidity: {len(htf_draws_on_liquidity)} levels identified.")
    print(f"LTF Bearish Liquidity Sweeps: {len(ltf_reversals_sweeps)} events.")
    print(
        f"LTF Fair Value Gaps: Bullish={len(ltf_continuation_confluence['FVG'][0])}, Bearish={len(ltf_continuation_confluence['FVG'][1])}")

    print("\nGenerating final chart...")
    plot_analysis_results(
        df_htf, df_ltf, htf_draws_on_liquidity, ltf_reversals_sweeps,
        ltf_continuation_confluence
    )


if __name__ == "__main__":
    # --- Global Configuration ---
    coin_to_analyze = "ethereum"
    currency_to_compare = "usd"

    # Run the main analysis
    run_multi_timeframe_analysis(coin_to_analyze, currency_to_compare)
