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
                'low_price': candle3['low'],
                'high_price': candle1['high'],
                'type': 'bullish'
            })

        # Bearish FVG: a gap between the low of candle1 and the high of candle3
        if candle3['high'] < candle1['low']:
            fvg_bearish.append({
                'timestamp_start': candle1.name,
                'timestamp_end': candle3.name,
                'low_price': candle1['low'],
                'high_price': candle3['high'],
                'type': 'bearish'
            })

    return fvg_bullish, fvg_bearish


def detect_order_blocks(df, volume_filter_factor=0.8, impulse_factor=1.5):
    """
    Identifies a simplified Order Block.
    It looks for the last down-candle before a strong bullish move or the last up-candle
    before a strong bearish move, using price and volume for confirmation.
    NOTE: Volume is not available in CoinGecko's /ohlc, so this function is conceptual.
    It will check for an 'impulse' move without a volume check for now.
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

        ax1.axhline(y=htf_draws[0]['value'], color=color, linestyle='--', linewidth=1, alpha=0.7,
                    label='HTF Draws on Liquidity')

    ax1.set_title('High Timeframe Analysis (4-hour)', fontsize=18)
    ax1.set_ylabel('Price', fontsize=14)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # --- LTF Price Chart (ax2) ---
    ax2.plot(df_ltf.index, df_ltf['close'], label='Close Price (30min)', color='purple', linewidth=1.5, alpha=0.8)

    # Mark LTF Reversals (Liquidity Sweeps)
    if ltf_reversals:
        for sweep in ltf_reversals:
            ax2.scatter(sweep['timestamp'], sweep['sweep_high'], color='red', marker='X', s=100, zorder=5,
                        label='Bearish Sweep' if sweep == ltf_reversals[0] else "_nolegend_")

    # Mark FVG (Continuation Confluence)
    bullish_fvgs = [f for f in ltf_continuations['FVG'] if f['type'] == 'bullish']
    bearish_fvgs = [f for f in ltf_continuations['FVG'] if f['type'] == 'bearish']

    for fvg in bullish_fvgs:
        ax2.axhspan(fvg['low_price'], fvg['high_price'], facecolor='green', alpha=0.2,
                    label='Bullish FVG' if fvg == bullish_fvgs[0] else "_nolegend_")
    for fvg in bearish_fvgs:
        ax2.axhspan(fvg['low_price'], fvg['high_price'], facecolor='red', alpha=0.2,
                    label='Bearish FVG' if fvg == bearish_fvgs[0] else "_nolegend_")

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
    if df_htf is None or df_htf.empty: return

    # 2. Fetch Low Timeframe (LTF) Data
    print(f"--- Fetching LTF ({ltf_days} day) data for {coin_id.capitalize()} ---")
    df_ltf = fetch_ohlc_data(coin_id, vs_currency, ltf_days)
    if df_ltf is None or df_ltf.empty: return

    # 3. HTF Analysis: Identify Draws on Liquidity
    # We will use the swing highs/lows as a proxy for this.
    print("\n--- HTF Analysis: Identifying Draws on Liquidity ---")
    htf_swing_highs = find_swing_highs(df_htf, window=5)
    htf_swing_lows = find_swing_highs(df_htf, window=5, is_high=False)
    htf_draws_on_liquidity = htf_swing_highs + htf_swing_lows

    # 4. LTF Analysis: Detect Reversals & Continuation Confluence
    print("\n--- LTF Analysis: Detecting Patterns ---")
    ltf_reversals = {
        'liquidity_sweeps': detect_bearish_liquidity_sweeps(df_ltf, htf_swing_highs, wick_ratio_threshold=0.6,
                                                            close_below_peak_factor=0.99)
    }

    # Detect continuation confluence
    ltf_continuation_confluence = {
        'FVG': detect_fair_value_gaps(df_ltf),
        'OrderBlocks': detect_order_blocks(df_ltf),
        # You would add 'BreakerBlocks' and other concepts here
    }

    # 5. Summarize and Plot
    print("\n--- Summary of Detected Patterns ---")
    print(f"HTF Draws on Liquidity: {len(htf_draws_on_liquidity)} levels identified.")
    print(f"LTF Bearish Liquidity Sweeps: {len(ltf_reversals['liquidity_sweeps'])} events.")
    print(
        f"LTF Fair Value Gaps: Bullish={len(ltf_continuation_confluence['FVG'][0])}, Bearish={len(ltf_continuation_confluence['FVG'][1])}")

    print("\nGenerating final chart...")
    plot_analysis_results(
        df_htf, df_ltf, htf_draws_on_liquidity, ltf_reversals['liquidity_sweeps'],
        ltf_continuation_confluence
    )


if __name__ == "__main__":
    # --- Global Configuration ---
    coin_to_analyze = "ethereum"
    currency_to_compare = "usd"

    # Run the main analysis
    run_multi_timeframe_analysis(coin_to_analyze, currency_to_compare)
