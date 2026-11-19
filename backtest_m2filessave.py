"""
backtest_m2.py  -  CLOUD BACKTEST : Method 2 (Cloud Scale-In, Google-sheet logic)   [single file, SIMULATION ONLY - no orders, no money]
=====================================================================================
Same strategy logic as the live paper-trading app (Ichimoku cloud entry, lots, profit-target / span / cloud exits, Method 1-3,
Buy-on interval, position sizing, portfolio limit) - but it runs on HISTORICAL candles instead of live data.

BACKTEST BUILDER : pick folder, exchange, FROM DATE and TO DATE, symbols, candle timeframe, capital, qty, Max lots (many) and
                   Target % (many) -> press Save -> ONE FILE is saved for every symbol x lots x target
MY BACKTESTS     : open a folder -> press 'Run backtest' -> summary table of all its files (trades, win %, P&L, return %,
                   drawdown) + CSV downloads; open any file for equity curve, signals, closed trades and the bar log.

How it works: Angel One is used ONLY to download the historical candles of [From date - cloud warm-up, To date]
(cached per symbol / timeframe / date range). The engine then walks the candles bar by bar from the From date. A position that
is still open on the To date is marked to the last close ('Open at end of data').

Run : python -m streamlit run backtest_m2.py
Login details (API key, Client ID, MPIN, TOTP secret) are typed on the first screen.
Backtest files are saved in  backtests/<ClientID>/<folder>/  next to this file.
NOTE: on Streamlit Cloud that folder is wiped when the app reboots - use  Account > Backup / Restore  (or run it on your own PC).
"""
import datetime as dt
import hashlib
import heapq
import io
import json
import logging
import os
import pickle
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
os.environ.setdefault('STREAMLIT_LOGGER_LEVEL', 'error')
import streamlit as st

logging.getLogger('streamlit').setLevel(logging.ERROR)

HERE = os.path.dirname(os.path.abspath(__file__))
STRAT_ROOT = os.path.join(HERE, 'backtests')

# timeframe -> (Angel interval, raw bar length, warm-up calendar days needed for the cloud)
def now_ist():
    """current India time as a naive Timestamp - the server (e.g. Streamlit Cloud) may run in UTC, Angel candles are IST."""
    return pd.Timestamp.now(tz='Asia/Kolkata').tz_localize(None)


def ts_ist(epoch):
    return pd.Timestamp(epoch, unit='s', tz='UTC').tz_convert('Asia/Kolkata').tz_localize(None)


TF = {
    '1D':  ('ONE_DAY',        pd.Timedelta(days=1),      260),
    '4H':  ('ONE_HOUR',       pd.Timedelta(hours=1),     160),   # 4H = 4 hourly bars, built per day
    '1H':  ('ONE_HOUR',       pd.Timedelta(hours=1),     100),
    '30m': ('THIRTY_MINUTE',  pd.Timedelta(minutes=30),  25),
    '15m': ('FIFTEEN_MINUTE', pd.Timedelta(minutes=15),  14),
    '10m': ('TEN_MINUTE',     pd.Timedelta(minutes=10),  10),
    '5m':  ('FIVE_MINUTE',    pd.Timedelta(minutes=5),   7),
    '3m':  ('THREE_MINUTE',   pd.Timedelta(minutes=3),   5),
    '1m':  ('ONE_MINUTE',     pd.Timedelta(minutes=1),   5),
}
# max days Angel returns per getCandleData request (kept a bit under the real limit)
INTERVAL_MAX_DAYS = {'ONE_DAY': 1900, 'ONE_HOUR': 380, 'THIRTY_MINUTE': 190, 'FIFTEEN_MINUTE': 190,
                     'TEN_MINUTE': 95, 'FIVE_MINUTE': 95, 'THREE_MINUTE': 55, 'ONE_MINUTE': 28}
LOTS_OPTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
TARGET_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SPAN_ONLY = {'SPAN A': 'A', 'SPAN B': 'B', 'SPAN A & B': 'A&B', 'SPAN A OR B': 'AorB'}
CLOUD_EXIT = {'Above cloud': 'CloudAbove', 'Inside cloud': 'CloudInside', 'Below cloud': 'CloudBelow'}
SPAN_ABOVE = {f'{k} (above)': v + '_up' for k, v in SPAN_ONLY.items()}      # same spans, exit when price closes ABOVE them
SPAN_DIRECTION = ['Below span', 'Above span']                     # drop-down: exit when the Close is below / above the chosen span(s)
SPAN_OPTIONS = {**SPAN_ONLY, **SPAN_ABOVE, **CLOUD_EXIT}                      # every non-% exit (stored in the file as `span`)
EXIT_OPTIONS = TARGET_OPTIONS + list(SPAN_OPTIONS)               # profit target % + span exits + cloud exits
ENTRY_OPTIONS = {'Above cloud': 'above', 'Inside cloud': 'inside', 'Below cloud': 'below'}
METHOD_OPTIONS = {'Method 1 - exit at any lot count': 1, 'Method 2 - exit only after Max lots filled': 2,
                  'Method 3 - exit below cloud, only in profit': 3}
M3_EXIT = 'Below cloud (in profit)'                              # the fixed exit of Method 3 files (no profit target / span)
AMOUNT_OPTIONS = [1000, 5000, 10000, 25000, 50000, 100000, 200000, 500000]
QTY_OPTIONS = [1, 2, 5, 10, 25, 50, 75, 100, 150, 200, 500, 1000]
# 'Buy on' = the INVESTMENT timeframe: how often a (new) lot may be bought while the price is in the entry zone.
# The timeframe of the candles / cloud itself is chosen separately (Timeframe row under it).
CANDLE_OPTIONS = {'Every candle': 'every', 'Every 30 min': 'min30', 'Every 1 hour': 'hour1', 'Every 2 hours': 'hour2',
                  'Every 4 hours': 'hour4', 'Daily': 'daily', 'Weekly': 'weekly'}
INTERVAL_MIN = {'min30': 30, 'hour1': 60, 'hour2': 120, 'hour4': 240}          # intraday investment intervals (minutes, counted from the LAST BUY)


def exit_label(x):
    if isinstance(x, tuple):                                   # combined exit: ('AND' | 'OR', [items])
        return f"{x[0]}({' + '.join(exit_label(i) for i in x[1])})"
    return f'{x}%' if isinstance(x, (int, float)) else x


def combo_spec(x):
    """('AND'|'OR', [items]) -> {'logic': 'and'|'or', 'pcts': [..], 'spans': [codes]}"""
    return {'logic': x[0].lower(), 'pcts': [i for i in x[1] if isinstance(i, (int, float))],
            'spans': [SPAN_OPTIONS[i] for i in x[1] if i in SPAN_OPTIONS]}


def size_text(s):
    if s.get('amount'):
        return f"{'equal-weight ' if s.get('sizing') == 'equal' else ''}amount/lot {s['amount']:,.0f}"
    return f"qty/lot {s['qty']:g}"


def limit_text(s):
    g = s.get('group') or {}
    return f"max {g['limit']} stocks at a time" if g.get('limit') else 'no stock limit'


def method_text(s):
    return f"method {int(s.get('method', 1))}"


def entry_text(s):
    return {'below': 'below cloud', 'inside': 'inside cloud'}.get(s.get('entry'), 'above cloud')


def exit_text(s):
    return s.get('exit_label') or f"{s['profit']}%"


def candle_text(s):
    return {v: k for k, v in CANDLE_OPTIONS.items()}.get(s.get('candle', 'every'), 'Every candle')


# =============================================================== ENGINE (M2 only, copied from m.py - same logic)
TENKAN_P, KIJUN_P, SENKOU_B_P, DISPLACEMENT = 9, 26, 52, 26


def add_ichimoku_cloud(df):
    df = df.copy()
    high, low = df['High'], df['Low']
    tenkan = (high.rolling(TENKAN_P).max() + low.rolling(TENKAN_P).min()) / 2
    kijun = (high.rolling(KIJUN_P).max() + low.rolling(KIJUN_P).min()) / 2
    df['SenkouA'] = ((tenkan + kijun) / 2).shift(DISPLACEMENT)
    df['SenkouB'] = ((high.rolling(SENKOU_B_P).max() + low.rolling(SENKOU_B_P).min()) / 2).shift(DISPLACEMENT)
    df['CloudTop'] = df[['SenkouA', 'SenkouB']].max(axis=1, skipna=False)
    df['CloudBottom'] = df[['SenkouA', 'SenkouB']].min(axis=1, skipna=False)
    df['AboveCloud'] = df['Close'] > df['CloudTop']
    return df


SPAN_RULES = {                       # exit rule for the "Span" options (price = bar Close)
    'A':    lambda p, a, b: p < a,
    'B':    lambda p, a, b: p < b,
    'A&B':  lambda p, a, b: p < a and p < b,      # below BOTH spans  (= below the cloud bottom)
    'AorB': lambda p, a, b: p < a or p < b,       # below EITHER span (= below the cloud top)
    'A_up':    lambda p, a, b: p > a,
    'B_up':    lambda p, a, b: p > b,
    'A&B_up':  lambda p, a, b: p > a and p > b,   # above BOTH spans  (= above the cloud top)
    'AorB_up': lambda p, a, b: p > a or p > b,    # above EITHER span (= above the cloud bottom)
    'CloudAbove': lambda p, a, b: p > max(a, b),  # closed ABOVE the cloud top
    'CloudBelow': lambda p, a, b: p < min(a, b),  # closed BELOW the cloud bottom
    'CloudInside': lambda p, a, b: min(a, b) <= p <= max(a, b),   # closed INSIDE the cloud
}
SPAN_TEXT = {'A': 'Span A', 'B': 'Span B', 'A&B': 'Span A & B', 'AorB': 'Span A or B',
             'CloudAbove': 'the cloud top', 'CloudBelow': 'the cloud bottom', 'CloudInside': 'the cloud'}


def span_phrase(c):
    """'A' -> 'below Span A',  'A&B_up' -> 'above Span A & B',  'CloudInside' -> 'inside the cloud' ..."""
    if c in ('CloudAbove', 'CloudBelow', 'CloudInside'):
        return {'CloudAbove': 'above', 'CloudBelow': 'below', 'CloudInside': 'inside'}[c] + ' ' + SPAN_TEXT[c]
    return ('above ' + SPAN_TEXT[c[:-3]]) if c.endswith('_up') else ('below ' + SPAN_TEXT[c])


def m2_gen(df, profit_target_pct, max_lots, start_date=None, lot_qty=1.0, trace=None, span=None, candle='every', lot_amount=None, skip=None, entry='above', method=1, state=None, combo=None):
    """GENERATOR version of Method 2. Yields ('NEXT', bar_date) before every bar and ('ASK', bar_date, fresh) when a FLAT stock wants to
    open a NEW position (fresh = the stock was NOT in the entry zone on its previous bar) - the caller answers True / False (portfolio limit). state['lots'] always shows the lots held.
    run_m2() below drives it for one stock; run_group() drives many stocks in time order.
    Method 2 (Google-sheet logic).
    span   : None = exit on Profit target %;  'A' / 'B' / 'A&B' / 'AorB' = exit when Close falls below that span
             (whole position closed; span exits go flat, no repeat exits).
    lot_amount : None = every lot buys `lot_qty` units;  number = every lot invests this AMOUNT (units = amount / lot price).
    method : 1 = profit target can exit at ANY lot count;  2 = profit target is checked ONLY once all Max lots are filled
             (span / cloud exits stay active at any lot count).
    entry  : 'above' = buy while Close is ABOVE the cloud top;  'below' = BELOW the cloud bottom;  'inside' = INSIDE the cloud.
    skip   : list of (from, to) timestamps - bars inside are ignored (strategy was paused; no buy / exit then).
    candle : 'every' = buy 1 lot on every candle above the cloud;  'min30' / 'hour1' / 'hour2' / 'hour4' = at most 1 lot per 30 min / 1 / 2 / 4
             hours (the candles / cloud can be 3m, 5m ... - the cloud is watched on every candle; the next lot is bought only after that
             many minutes have passed since the LAST BUY - exact gap, not clock slots);
             'daily' = at most 1 lot per day;
             'weekly' = at most 1 lot per week (first candle of the period that closes above the cloud).
    Returns trades DataFrame; trace (list) gets one dict per bar."""
    full = add_ichimoku_cloud(df)

    def in_zone(r):                                                  # is the ENTRY condition met on this bar?
        p_ = r['Close']
        if entry == 'above':
            return bool(r['AboveCloud'])
        if entry == 'below':
            return bool(p_ < r['CloudBottom'])
        return bool(r['CloudBottom'] <= p_ <= r['CloudTop'])         # inside the cloud

    prev_above = None                                                # entry condition on the bar BEFORE the current one (None = unknown)
    if start_date is not None:   # cloud uses full history, simulation starts flat from start_date
        _pre = full[(full['Date'] < pd.Timestamp(start_date)) & full['CloudTop'].notna() & full['CloudBottom'].notna()]
        if len(_pre):
            prev_above = in_zone(_pre.iloc[-1])
        full = full[full['Date'] >= pd.Timestamp(start_date)].reset_index(drop=True)
    rows = full.to_dict('records')
    span_hit = SPAN_RULES.get(span)
    combo_logic = (combo or {}).get('logic')                # 'and' = ALL chosen exits must be true on the same bar, 'or' = ANY one

    def pkey(d):                                             # only for 'daily' / 'weekly' (intraday intervals use last_buy_time)
        if candle == 'daily':
            return d.date()
        if candle == 'weekly':
            y, w, _ = d.isocalendar()
            return (y, w)
        return None

    if method == 3:
        exit_reason = 'Closed below the cloud while in profit (Method 3)'
    elif combo:
        _parts = [f'{p}%' for p in combo['pcts']] + [f'Close {span_phrase(c)}' for c in combo['spans']]
        exit_reason = f"Combined exit {combo_logic.upper()}: " + f" {combo_logic.upper()} ".join(_parts) + f" (Method {method})"
    else:
        exit_reason = (({'CloudAbove': 'Closed above the cloud (Method 2)', 'CloudInside': 'Closed inside the cloud (Method 2)'}.get(span) or f'Closed {span_phrase(span)} (Method 2)') if span_hit
                       else (f'Profit target {profit_target_pct}% hit after all {max_lots} lots filled (Method 2)' if method == 2
                             else f'Profit target {profit_target_pct}% hit (Method 1 - sheet logic)'))
    trades = []
    trade_id = 0
    if state is None:
        state = {}
    state['lots'] = 0
    m2_first, m2_lots, m2_avg, m2_prev_exit, m2_seg_start = True, 0, None, False, None
    m2_qty, m2_cost = 0.0, 0.0                     # units held and money spent (avg price = cost / units)
    last_buy_key = None
    last_buy_time = None                                     # time of the last BUY (intraday 'Buy on' intervals)

    for row in rows:
        if pd.isna(row['CloudTop']) or pd.isna(row['CloudBottom']):
            continue
        price, date = row['Close'], row['Date']
        if skip and any(a < date < b for a, b in skip):
            continue
        yield ('NEXT', date)
        above = in_zone(row)                                                                       # entry condition met?
        fresh = prev_above is False          # on the previous bar it was NOT in the entry zone (below / inside the cloud) -> FRESH signal
        prev_above = above
        bought, prev_lots, blocked = False, 0, False
        if m2_first:
            m2_first = False
            if above:
                if (yield ('ASK', date, fresh)):
                    m2_lots = 1
                else:
                    blocked = True                          # portfolio full -> no new position on this bar
            bought = m2_lots == 1
            if m2_lots == 1:
                last_buy_key = pkey(date)
                last_buy_time = date
        else:
            prev_lots, bought = m2_lots, False
            if candle == 'every':
                can_buy = True
            elif candle in INTERVAL_MIN:                     # exact gap since the last buy (12:03 buy -> next 12:33)
                can_buy = last_buy_time is None or (date - last_buy_time) >= pd.Timedelta(minutes=INTERVAL_MIN[candle])
            else:
                can_buy = pkey(date) != last_buy_key
            if not above or not can_buy:
                pass
            elif m2_lots == 0 and not (yield ('ASK', date, fresh)):
                blocked = True                              # flat stock, but the portfolio limit is reached
            elif m2_prev_exit:
                m2_lots, bought = 1, True
            elif m2_lots < max_lots:
                m2_lots, bought = m2_lots + 1, True
            if bought:
                last_buy_key = pkey(date)
                last_buy_time = date
        if bought:                                         # avg price changes ONLY when a lot is bought
            q = (lot_amount / price) if lot_amount else lot_qty
            if prev_lots == 0:
                m2_qty, m2_cost = q, q * price
            else:
                m2_qty, m2_cost = m2_qty + q, m2_cost + q * price
            m2_avg = (m2_cost / m2_qty) if m2_qty else price      # amount below 1 lot -> 0 units (no divide by zero)
        m2_prev_exit = False
        if m2_seg_start is None and m2_lots > 0:
            m2_seg_start = date
        if m2_lots > 0 and m2_avg:
            if method == 3:                                  # Method 3: Close BELOW the cloud AND position in profit -> exit (any lot count)
                hit = bool(price < row['CloudBottom'] and price > m2_avg)
            elif combo:
                conds = [(price - m2_avg) / m2_avg >= pc / 100.0 and (method != 2 or m2_lots >= max_lots) for pc in combo['pcts']]
                conds += [SPAN_RULES[c](price, row['SenkouA'], row['SenkouB']) for c in combo['spans']]
                hit = bool(conds) and (all(conds) if combo_logic == 'and' else any(conds))
            elif span_hit:
                hit = span_hit(price, row['SenkouA'], row['SenkouB'])
            else:
                hit = (price - m2_avg) / m2_avg >= profit_target_pct / 100.0 and (method != 2 or m2_lots >= max_lots)
            if hit:
                trade_id += 1
                trades.append({
                    'Trade ID': trade_id, 'Entry Date (1st lot)': m2_seg_start, 'Exit Date': date,
                    'Holding Days': (date - m2_seg_start).days, 'Lots Filled': m2_lots,
                    'Max Lots Allowed': max_lots, 'Avg Entry Price': round(m2_avg, 4),
                    'Exit Price': round(price, 4),
                    'Profit % (on avg entry)': round((price - m2_avg) / m2_avg * 100, 2),
                    'PnL (pts x qty)': round((price - m2_avg) * m2_qty, 2),
                    'Exit Reason': exit_reason})
                m2_prev_exit = True
                m2_seg_start = None
                m2_lots, m2_avg, m2_qty, m2_cost = 0, None, 0.0, 0.0   # exit = flat; fresh trade on the next close above cloud
        state['lots'] = m2_lots
        if trace is not None:
            trace.append({'Date': date, 'Close': price, 'CloudTop': row['CloudTop'],
                          'CloudBottom': row['CloudBottom'], 'Above': above, 'Fresh': bool(above and fresh), 'Lots': m2_lots, 'Bought': bought, 'Blocked': blocked, 'Qty': m2_qty,
                          'AvgPrice': None if not m2_avg else round(m2_avg, 4),
                          'Action': 'EXIT' if m2_prev_exit else ''})

    if m2_lots > 0 and m2_avg and not m2_prev_exit and m2_seg_start is not None and rows:
        price, date = full.iloc[-1]['Close'], full.iloc[-1]['Date']
        trade_id += 1
        trades.append({
            'Trade ID': trade_id, 'Entry Date (1st lot)': m2_seg_start, 'Exit Date': date,
            'Holding Days': (date - m2_seg_start).days, 'Lots Filled': m2_lots,
            'Max Lots Allowed': max_lots, 'Avg Entry Price': round(m2_avg, 4),
            'Exit Price': round(price, 4),
            'Profit % (on avg entry)': round((price - m2_avg) / m2_avg * 100, 2),
            'PnL (pts x qty)': round((price - m2_avg) * m2_qty, 2),
            'Exit Reason': 'Open at end of data (marked to last close)'})
    return pd.DataFrame(trades)


def run_m2(*a, **k):
    """one stock on its own (no portfolio limit) -> trades DataFrame; trace (list) gets one dict per bar."""
    g = m2_gen(*a, **k)
    try:
        msg = next(g)
        while True:
            msg = g.send(True if msg[0] == 'ASK' else None)
    except StopIteration as e:
        return e.value


def run_group(items, limit):
    """Step many stocks together, bar by bar (time order), with a portfolio limit of `limit` stocks.
    Per bar:
      1) every stock of that bar is stepped first - holders scale in or EXIT, so the freed slots are known BEFORE anybody is let in;
      2) the flat stocks that want a NEW position ('ASK') are served in PRIORITY order until the limit is reached:
           a) FRESH signals first - the stock was below / inside the cloud on its previous bar and has just come into the entry zone;
           b) then the stocks that were already waiting (in the entry zone before) - inside each group in list (rank) order.
    A stock that already holds lots keeps scaling in (no slot needed).  -> {index: trades DataFrame}"""
    heap, results = [], {}
    for i, it in enumerate(items):
        try:
            msg = next(it['g'])
            heapq.heappush(heap, (msg[1], i))
        except StopIteration as e:
            results[i] = e.value

    def _done(i, e):
        results[i] = e.value
        if items[i].get('stopped'):                                    # exited strategy: its slot is free from now on
            items[i]['state']['lots'] = 0

    while heap:
        bar = heap[0][0]
        batch = []
        while heap and heap[0][0] == bar:                              # all stocks that have a candle at this time
            batch.append(heapq.heappop(heap)[1])
        batch.sort()
        pending = []
        for i in batch:                                                # phase 1: process the bar (exits free their slots here)
            try:
                msg = items[i]['g'].send(None)
                if msg[0] == 'ASK':
                    pending.append((i, msg))                           # flat stock waiting for an answer
                else:
                    heapq.heappush(heap, (msg[1], i))
            except StopIteration as e:
                _done(i, e)
        if not pending:
            continue
        asking = {i for i, _ in pending}
        holders = [j for j, o in enumerate(items) if j not in asking and o['state']['lots'] > 0]
        pending.sort(key=lambda p: (0 if (len(p[1]) > 2 and p[1][2]) else 1, p[0]))     # fresh first, then list order
        for i, msg in pending:                                         # phase 2: hand out the free slots
            it = items[i]
            granted = len(holders) < limit
            it.setdefault('ask_log', []).append((msg[1], granted,                          # (bar, granted?, who held the slots, fresh?)
                                                 [(items[j].get('s') or {}).get('symbol', str(j)) for j in holders],
                                                 bool(len(msg) > 2 and msg[2])))
            if granted:
                holders.append(i)
            try:
                nxt = it['g'].send(granted)
                while nxt[0] == 'ASK':
                    nxt = it['g'].send(False)
                heapq.heappush(heap, (nxt[1], i))
            except StopIteration as e:
                _done(i, e)
    return results


# =============================================================== DATA FEED (Angel One SmartAPI)
SCRIP_URL = 'https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json'


def totp_now(secret):
    """6-digit TOTP code from the secret (standard RFC 6238, 30 s, SHA1) - no pyotp package needed."""
    import base64, hashlib, hmac, struct
    s = re.sub(r'[\s-]', '', str(secret)).upper()
    key = base64.b32decode(s + '=' * (-len(s) % 8), casefold=True)
    digest = hmac.new(key, struct.pack('>Q', int(time.time() // 30)), hashlib.sha1).digest()
    o = digest[-1] & 15
    return f"{(struct.unpack('>I', digest[o:o + 4])[0] & 0x7fffffff) % 10 ** 6:06d}"


def ensure_angel_packages():
    """Safety net: if the host did not install requirements.txt, install the Angel One packages once, at run time.
    site-packages can be read-only on the host, so they go into a writable folder that is added to sys.path."""
    target = os.path.join(tempfile.gettempdir(), 'angel_pkgs')
    if os.path.isdir(target) and target not in sys.path:
        sys.path.insert(0, target)
    try:
        import SmartApi, logzero, websocket                       # noqa: F401
        from SmartApi.smartWebSocketV2 import SmartWebSocketV2    # noqa: F401
        return
    except ImportError:
        pass
    import importlib, subprocess
    pk = ['smartapi-python', 'logzero', 'websocket-client', 'pycryptodome', 'pyotp']
    cmds = [[sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', '--target', target] + pk]
    uv = shutil.which('uv')
    if uv:                                                         # hosts that build the venv with uv
        cmds.append([uv, 'pip', 'install', '--quiet', '--python', sys.executable, '--target', target] + pk)
    errs = []
    for c in cmds:
        r = subprocess.run(c, capture_output=True, text=True)
        if r.returncode == 0:
            if target not in sys.path:
                sys.path.insert(0, target)
            importlib.invalidate_caches()
            return
        errs.append(' '.join(c[:5]) + ' -> ' + ((r.stderr or r.stdout or '').strip()[-300:]))
    raise RuntimeError('Could not install the Angel One packages: ' + ' || '.join(errs))


def angel_login(client_id, api_key, mpin, totp_secret):
    """Login with Client ID + MPIN + TOTP (the 6-digit code is generated from your TOTP secret)."""
    ensure_angel_packages()
    from SmartApi import SmartConnect
    smart = SmartConnect(api_key=api_key)
    res = smart.generateSession(client_id, mpin, totp_now(totp_secret))
    if not res or not res.get('status'):
        raise ValueError((res or {}).get('message', 'login failed'))
    smart._login_data = res.get('data') or {}
    return smart


@st.cache_resource(ttl=86400, show_spinner='Loading Angel One instrument list ...')
def _scrip_master():
    return requests.get(SCRIP_URL, timeout=120).json()


@st.cache_resource(ttl=86400, show_spinner=False)
def _token_index():
    idx = {}
    for r in _scrip_master():
        idx.setdefault((r['exch_seg'], r['symbol'].upper()), r)
    return idx


def _instrument_info(exchange, symbol):
    """token + exact trading symbol + exchange lot size of an instrument."""
    sym = symbol.strip().upper()
    idx = _token_index()
    for key in (sym, sym + '-EQ'):                                # exact match first
        r = idx.get((exchange, key))
        if r:
            return {'token': r['token'], 'tradingsymbol': r['symbol'], 'lotsize': max(1, int(float(r.get('lotsize') or 1)))}
    raise ValueError(f'{symbol} not found on {exchange}  (equity: SBIN-EQ / index: Nifty 50 / MCX: GOLDM05DEC25FUT)')


def _instrument_token(exchange, symbol):
    return _instrument_info(exchange, symbol)['token']




def _instrument_token_old(exchange, symbol):
    sym = symbol.strip().upper()
    hits = [r for r in _scrip_master() if r['exch_seg'] == exchange
            and r['symbol'].upper() in (sym, sym + '-EQ')]
    if not hits:
        raise ValueError(f'{symbol} not found on {exchange}  (equity: SBIN-EQ / index: Nifty 50 / MCX: GOLDM05DEC25FUT)')
    hits.sort(key=lambda r: r['symbol'].upper() != sym)          # exact match first
    return hits[0]['token']


_RATE_LOCK = threading.Lock()
_LAST_CALL = [0.0]
MIN_GAP = 0.7            # seconds between two Angel candle requests (Angel allows ~3/sec; we stay well under)
RECHECK_SECS = 45        # an already-loaded symbol is only topped up with new candles this often


def _is_rate_msg(msg):
    m = str(msg).lower()
    return 'access rate' in m or 'exceeding' in m or 'too many' in m or 'rate limit' in m


def _call_candle_api(smart, params):
    """one getCandleData call: spaced out + retried with back-off when Angel says 'exceeding access rate'."""
    last = 'candle request failed'
    for i in range(5):
        with _RATE_LOCK:
            wait = MIN_GAP - (time.time() - _LAST_CALL[0])
            if wait > 0:
                time.sleep(wait)
            _LAST_CALL[0] = time.time()
        try:
            res = smart.getCandleData(params)
        except Exception as e:
            last = str(e)
            if _is_rate_msg(last):
                time.sleep(2 * (i + 1))
                continue
            raise
        if res and res.get('status'):
            return res
        last = (res or {}).get('message', 'candle request failed')
        if _is_rate_msg(last):
            time.sleep(2 * (i + 1))
            continue
        raise ValueError(last)
    raise ValueError(f'Angel rate limit: {last}')


def _fetch_range(smart, exchange, token, interval, frm, to):
    step = pd.Timedelta(days=INTERVAL_MAX_DAYS.get(interval, 28))      # Angel per-request limits
    rows, cur = [], frm
    while cur < to:
        end = min(cur + step, to)
        res = _call_candle_api(smart, {'exchange': exchange, 'symboltoken': token, 'interval': interval,
                                       'fromdate': cur.strftime('%Y-%m-%d %H:%M'),
                                       'todate': end.strftime('%Y-%m-%d %H:%M')})
        rows += res.get('data') or []
        cur = end + pd.Timedelta(minutes=1)
    if not rows:
        return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close'])
    df = pd.DataFrame(rows, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    for c_ in ('Open', 'High', 'Low', 'Close'):
        df[c_] = pd.to_numeric(df[c_], errors='coerce')
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df['Date'] = pd.to_datetime(df['Date'])
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)
    return df[['Date', 'Open', 'High', 'Low', 'Close']]


def backtest_end(end_day):
    """last moment of the 'To date' (capped at NOW, so a candle that has not closed yet is never used)."""
    return min(pd.Timestamp(end_day).normalize() + pd.Timedelta(days=1), now_ist())


def fetch_candles(smart, exchange, symbol, tf, start_day, end_day, token=None):
    """-> raw HISTORICAL candles (Date, Open, High, Low, Close), tz-naive IST, oldest first, from (start - cloud warm-up) to the To date."""
    interval, _, warm_days = TF[tf]
    token = token or _instrument_token(exchange, symbol)
    df = _fetch_range(smart, exchange, token, interval, pd.Timestamp(start_day) - pd.Timedelta(days=warm_days), backtest_end(end_day))
    if df.empty:
        raise ValueError('no candles returned for this date range')
    return df.drop_duplicates('Date', keep='last').sort_values('Date').reset_index(drop=True)


class DataLoading(Exception):
    """candles for this symbol are still being downloaded in the background (NOT an error)."""


FAIL_RETRY_SECS = 30     # after a failed download, wait this long before trying again
LIVE_END_TTL = 600       # a range that ends today / in the future is re-downloaded after this many seconds


@st.cache_resource
def _candle_store():
    return {'lock': threading.Lock(), 'data': {}, 'pending': set(), 'fail': {}, 'smart': {},
            'login_lock': threading.Lock(), 'pool': ThreadPoolExecutor(max_workers=3, thread_name_prefix='candles')}


def _shared_smart(S, creds, stale=None):
    """ONE Angel login shared by all background workers (a worker never touches st.session_state)."""
    cid = creds['client_id']
    with S['login_lock']:
        cur = S['smart'].get(cid)
        if cur is None or (stale is not None and cur is stale):
            cur = angel_login(creds['client_id'], creds['api_key'], creds['mpin'], creds['totp'])
            S['smart'][cid] = cur
        return cur


def _with_relogin(S, creds, fn):
    smart = _shared_smart(S, creds)
    try:
        return fn(smart)
    except Exception as e:
        if _is_rate_msg(e) or 'no candles' in str(e):
            raise
        return fn(_shared_smart(S, creds, stale=smart))           # session expired? log in again once


def _bg_full(S, key, token, creds):
    exchange, symbol, tf, start_day, end_day = key
    try:
        df = _with_relogin(S, creds, lambda sm: fetch_candles(sm, exchange, symbol, tf, start_day, end_day, token))
        with S['lock']:
            S['data'][key] = {'df': df, 'checked': time.time(), 'error': None}
            S['fail'].pop(key, None)
    except Exception as ex:
        with S['lock']:
            S['fail'][key] = (time.time(), str(ex))
    finally:
        with S['lock']:
            S['pending'].discard(key)


def load_candles(creds, exchange, symbol, tf, start_day, end_day, wait=False, timeout=1800):
    """Historical candles for (symbol, timeframe, From, To) are downloaded ONCE in a BACKGROUND thread and cached.
    wait=False -> raises DataLoading until the download is ready.   wait=True -> waits for it (up to `timeout` seconds)."""
    S = _candle_store()
    key = (exchange, symbol, tf, start_day, end_day)
    token = _instrument_token(exchange, symbol)                    # (cached) - resolved here, in the script thread
    deadline = time.time() + timeout
    live_end = pd.Timestamp(end_day).date() >= now_ist().date()   # range reaches today -> data keeps growing
    while True:
        with S['lock']:
            e = S['data'].get(key)
            if e is not None and live_end and time.time() - e['checked'] > LIVE_END_TTL:
                S['data'].pop(key, None)                           # stale 'up to today' data -> download again
                e = None
            if e is not None:
                return e['df'].copy(), e['checked'], e['error']
            if key not in S['pending']:
                fail = S['fail'].get(key)
                if fail and time.time() - fail[0] < FAIL_RETRY_SECS:
                    raise ValueError(fail[1])
                S['fail'].pop(key, None)
                S['pending'].add(key)
                S['pool'].submit(_bg_full, S, key, token, creds)
        if not wait or time.time() > deadline:
            raise DataLoading('Loading candles ...')
        time.sleep(0.4)



# =============================================================== candle preparation
def closed_only(df, delta, close_t, now):
    """drop the candle that has not finished yet (its end = start+length, capped at market close)."""
    sess_end = df['Date'].dt.normalize() + pd.Timedelta(hours=close_t.hour, minutes=close_t.minute)
    end = pd.concat([df['Date'] + delta, sess_end], axis=1).min(axis=1)
    return df[end <= now].reset_index(drop=True)


def to_4h(h, close_t):
    """hourly -> 4H: per day, bars 1-4 = first 4H candle, the rest = second (e.g. NSE 09:15 + 13:15)."""
    h = h.copy()
    h['day'] = h['Date'].dt.normalize()
    h['n'] = h.groupby('day').cumcount() // 4
    out = (h.groupby(['day', 'n'])
             .agg(Date=('Date', 'first'), Open=('Open', 'first'), High=('High', 'max'),
                  Low=('Low', 'min'), Close=('Close', 'last'), cnt=('Close', 'size'), last=('Date', 'last'))
             .reset_index(drop=True))
    day_close = out['Date'].dt.normalize() + pd.Timedelta(hours=close_t.hour, minutes=close_t.minute)
    ok = (out['cnt'] == 4) | ((out['last'] + pd.Timedelta(hours=1)) >= day_close)
    return out[ok][['Date', 'Open', 'High', 'Low', 'Close']].reset_index(drop=True)


def prepare(raw, tf, close_t, now):
    _, delta, _ = TF[tf]
    df = closed_only(raw, delta, close_t, now)
    return to_4h(df, close_t) if tf == '4H' else df


# =============================================================== M2 live state
def build_tr(trace):
    tr = pd.DataFrame(trace)
    if tr.empty:
        return tr
    buy = tr['Bought'].astype(bool)
    exit_ = tr['Action'] == 'EXIT'
    tr['Signal'] = np.select([buy & exit_, exit_, buy], ['BUY+EXIT', 'EXIT', 'BUY'], 'HOLD')
    for col in ('AvgPrice', 'Qty'):                          # never bought -> column was all None (object) -> .round() crashed
        tr[col] = pd.to_numeric(tr[col], errors='coerce')
    tr['Unrealised %'] = ((tr['Close'] - tr['AvgPrice']) / tr['AvgPrice'] * 100).round(2)
    return tr


def compute_m2(df, profit_pct, max_lots, start, qty=1.0, span=None, candle='every', amount=None, skip=None, entry='above', method=1):
    trace = []
    trades = run_m2(df, profit_pct, max_lots, start_date=start, lot_qty=qty, trace=trace, span=span, candle=candle, lot_amount=amount, skip=skip, entry=entry, method=method)
    return build_tr(trace), trades


def get_smart():
    """Logged-in SmartConnect for THIS browser session (re-logs in automatically if it expired)."""
    c = st.session_state['creds']
    if 'smart' not in st.session_state:
        st.session_state['smart'] = angel_login(c['client_id'], c['api_key'], c['mpin'], c['totp'])
    return st.session_state['smart']


def relogin():
    st.session_state.pop('smart', None)
    return get_smart()


# =============================================================== strategy files (folder / file)
def safe(s):
    return re.sub(r'[^A-Za-z0-9 _.%\-]', '_', str(s)).strip() or 'x'


def client_id():
    c = st.session_state['creds']
    if c.get('demo'):                                   # Demo ID: every visitor gets a PRIVATE folder, never the owner's
        sid = st.session_state.setdefault('demo_sid', uuid.uuid4().hex[:8].upper())
        return 'DEMO-' + sid
    return c['client_id']


def user_dir():
    return os.path.join(STRAT_ROOT, safe(client_id()))


def read_json(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def list_folders():
    d = user_dir()
    return sorted(f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))) if os.path.isdir(d) else []


def delete_folder(folder):
    d = os.path.join(user_dir(), safe(folder))
    if os.path.isdir(d):
        shutil.rmtree(d)
    store = st.session_state.get('bt') or {}
    for k_ in [k for k in store if k[0] == folder]:
        store.pop(k_, None)


def list_files(folder):
    d = os.path.join(user_dir(), safe(folder))
    return sorted(f for f in os.listdir(d) if f.endswith('.json')) if os.path.isdir(d) else []


def load_strategy(folder, fname):
    with open(os.path.join(user_dir(), safe(folder), fname), encoding='utf-8') as f:
        return json.load(f)


def delete_strategy(folder, fname):
    os.remove(os.path.join(user_dir(), safe(folder), fname))
    try:
        os.remove(_result_path(folder, fname))
    except OSError:
        pass


# ---- SAVED RESULTS: every finished run is also written to  <folder>/<file>.result.pkl  so it survives logout / refresh / a new login.
# A saved result is used only while the strategy file (and, for a portfolio group, every file of the group) is unchanged.
RESULT_EXT = '.result.pkl'


def _result_path(fol, fname):
    base = fname[:-5] if fname.endswith('.json') else fname
    return os.path.join(user_dir(), safe(fol), base + RESULT_EXT)


def _file_md5(path):
    try:
        with open(path, 'rb') as fh:
            return hashlib.md5(fh.read()).hexdigest()
    except OSError:
        return ''


def _result_sig(fol, fname):
    """fingerprint of what the result depends on: the strategy file, plus all files of its group (portfolio limit)."""
    d = os.path.join(user_dir(), safe(fol))
    own = read_json(os.path.join(d, fname), {}) or {}
    g = own.get('group') or {}
    if not g.get('limit'):
        return _file_md5(os.path.join(d, fname))
    parts = []
    for f in list_files(fol):
        m = read_json(os.path.join(d, f), {}) or {}
        if (m.get('group') or {}).get('id') == g.get('id'):
            parts.append(f + ':' + _file_md5(os.path.join(d, f)))
    return hashlib.md5('|'.join(sorted(parts)).encode()).hexdigest()


def save_result(fol, fname, val):
    """persist one finished run (errors are not saved).  A failure to write never stops the backtest."""
    try:
        path = _result_path(fol, fname)
        if not (isinstance(val, dict) and 'ev' in val):
            if os.path.exists(path):
                os.remove(path)
            return
        tmp = path + '.tmp'
        with open(tmp, 'wb') as fh:
            pickle.dump({'v': 1, 'sig': _result_sig(fol, fname), 'val': val}, fh, protocol=4)
        os.replace(tmp, path)
        st.session_state.pop('_result_save_err', None)
    except Exception as e:
        st.session_state['_result_save_err'] = str(e)


def put_result(store, fol, fname, val):
    store[(fol, fname)] = val
    save_result(fol, fname, val)


def drop_result(store, fol, fname):
    store.pop((fol, fname), None)
    try:
        os.remove(_result_path(fol, fname))
    except OSError:
        pass


def hydrate_results(store, fol, fnames):
    """load saved results from disk into the session store (only for files that are not in it yet)."""
    for fn in fnames:
        if (fol, fn) in store:
            continue
        path = _result_path(fol, fn)
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'rb') as fh:
                blob = pickle.load(fh)
            if blob.get('v') == 1 and blob.get('sig') == _result_sig(fol, fn):
                store[(fol, fn)] = blob['val']
        except Exception:
            pass                                               # unreadable / old pandas version -> just run again


def create_strategies(folder, prefix, exchange, symbols, tfs, capital, sizes, close, lots_list, exits, candles, start, end, entries=('above',), methods=(1,), max_open=0, pairs=None):
    """one file per symbol x timeframe x lots x size x exit option x candle mode. An existing file with the same name is UPDATED (new dates / settings)."""
    d = os.path.join(user_dir(), safe(folder))
    os.makedirs(d, exist_ok=True)
    made, skipped = [], []
    for rank, sym in enumerate(symbols):
      for tf in tfs:
        for lots in lots_list:
          for kind, val in sizes:
            for ex_i, ex0 in enumerate(exits):
              for ent in entries:
               for mth in methods:
                if mth == 3:                                               # Method 3 has its own fixed exit -> ONE file, not one per exit
                    if ex_i != 0:
                        continue
                    ex = M3_EXIT
                elif ex0 == M3_EXIT:
                    continue
                else:
                    ex = ex0
                for cd in candles:
                    if pairs is not None and (cd, tf) not in pairs:        # this Buy-on mode was not given this timeframe
                        continue
                    cdc = CANDLE_OPTIONS[cd]
                    amt = capital / len(symbols) / lots if kind == 'eq' else val      # equal weightage: capital / symbols / max lots
                    name = (f"{prefix + ' ' if prefix else ''}{sym} {tf} {lots}lots {exit_label(ex)}"
                            + ('' if cdc == 'every' else f' {cdc}')
                            + (f' amt{val:g}' if kind == 'amt' else (' eq' if kind == 'eq' else (f' q{val:g}' if len(sizes) > 1 else '')))
                            + {'below': ' belowcloud', 'inside': ' insidecloud'}.get(ent, '')
                            + {2: ' method2', 3: ' method3'}.get(mth, ''))
                    path = os.path.join(d, safe(name) + '.json')
                    existed = os.path.exists(path)
                    is_combo = isinstance(ex, tuple)
                    is_span = (not is_combo) and ex in SPAN_OPTIONS
                    s = {'name': name, 'folder': folder, 'mode': 'backtest', 'exchange': exchange, 'symbol': sym,
                         'tf': tf, 'profit': None if (is_span or is_combo) else ex, 'span': None if is_combo else SPAN_OPTIONS.get(ex),
                         'combo': combo_spec(ex) if is_combo else None,
                         'exit_label': exit_label(ex), 'candle': cdc, 'lots': lots, 'capital': capital,
                         'qty': val if kind == 'qty' else 1.0, 'amount': amt if kind in ('amt', 'eq') else None,
                         'sizing': 'equal' if kind == 'eq' else ('amount' if kind == 'amt' else 'qty'), 'entry': ent, 'method': mth,
                         'close': close, 'start': start, 'end': end,
                         'group': ({'id': f"{start}|{end}|{tf}|{lots}|{kind}{val}|{exit_label(ex)}|{ent}|{mth}|{cdc}", 'limit': int(max_open), 'rank': rank}
                                   if max_open else None),
                         'created': now_ist().to_pydatetime().strftime('%Y-%m-%d %H:%M')}
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(s, f, indent=1)
                    (skipped if existed else made).append(name)          # skipped = existing file that was updated
    return made, skipped


# =============================================================== evaluate one saved strategy
def get_candles(s):
    """-> (raw candles, time of last data check, error text or None).  Raises DataLoading while downloading."""
    day = pd.Timestamp(s['start']).normalize().strftime('%Y-%m-%d')
    end_day = pd.Timestamp(s['end']).normalize().strftime('%Y-%m-%d')
    return load_candles(st.session_state['creds'], s['exchange'], s['symbol'], s['tf'], day, end_day,
                        wait=bool(st.session_state.get('_wait_candles')))


def first_bar_after_start(df, tf, close_t, start):
    """Date of the first bar that CLOSES after the start time (None if no such closed bar yet).
    Bars that were already closed at start never trade. (Old code compared the bar OPEN time with the
    start, so a bar that was running at start - e.g. today's daily candle - was wrongly ignored.)"""
    if df.empty:
        return None
    bar = pd.Timedelta(hours=4) if tf == '4H' else TF[tf][1]
    sess_end = df['Date'].dt.normalize() + pd.Timedelta(hours=close_t.hour, minutes=close_t.minute)
    end = pd.concat([df['Date'] + bar, sess_end], axis=1).min(axis=1)
    ok = df[end > pd.Timestamp(start)]
    return None if ok.empty else ok['Date'].iloc[0]


def _eval_prepare(s):
    """candles -> closed bars + the numbers needed to run the engine for one strategy."""
    h, m = [int(x) for x in s['close'].split(':')]
    raw, checked, data_err = get_candles(s)
    now = backtest_end(pd.Timestamp(s['end']).normalize())          # simulate up to the end of the To date (never past NOW)
    df = prepare(raw, s['tf'], dt.time(h, m), now)
    skip = None
    eff_start = first_bar_after_start(df, s['tf'], dt.time(h, m), s['start'])
    diag = {'bars': len(df), 'last_bar': None if df.empty else df['Date'].iloc[-1], 'cloud': None}
    if len(df):
        try:
            cl = add_ichimoku_cloud(df).iloc[-1]
            if not (pd.isna(cl['CloudTop']) or pd.isna(cl['CloudBottom'])):
                diag['cloud'] = {'close': float(cl['Close']), 'top': float(cl['CloudTop']), 'bottom': float(cl['CloudBottom']),
                                 'pos': 'Above cloud' if cl['Close'] > cl['CloudTop'] else ('Below cloud' if cl['Close'] < cl['CloudBottom'] else 'Inside cloud')}
        except Exception:
            pass
    return {'df': df, 'skip': skip, 'eff_start': eff_start if eff_start is not None else pd.Timestamp('2100-01-01'),
            'checked': checked, 'data_err': data_err, 'diag': diag}


def _eval_gen(s, ctx, trace, state):
    return m2_gen(ctx['df'], s.get('profit') or 0.0, int(s['lots']), start_date=ctx['eff_start'], lot_qty=s['qty'], trace=trace,
                  span=s.get('span'), candle=s.get('candle', 'every'), lot_amount=s.get('amount'), skip=ctx['skip'],
                  entry=s.get('entry', 'above'), method=int(s.get('method', 1)), state=state, combo=s.get('combo'))


def _eval_finish(s, ctx, trace, trades):
    tr = build_tr(trace)
    out = {'s': s, 'diag': ctx['diag'], 'tr': tr, 'trades': trades, 'status': 'No signal in this range', 'last': None,
           'checked': ctx['checked'], 'data_err': ctx['data_err'],
           'closed_n': 0, 'closed_pnl': 0.0, 'open_pnl': 0.0, 'invested': 0.0}
    if tr.empty:
        return out
    last = tr.iloc[-1]
    flat = last['Signal'] in ('EXIT', 'BUY+EXIT') or int(last['Lots']) == 0 or pd.isna(last['AvgPrice'])
    openrows = trades[trades['Exit Reason'].str.startswith('Open at end')] if not trades.empty else trades
    closed = trades.drop(openrows.index) if not trades.empty else trades
    out.update(last=last, flat=flat, closed_n=len(closed),
               closed_pnl=float(closed['PnL (pts x qty)'].sum()) if len(closed) else 0.0,
               open_pnl=0.0 if flat or openrows.empty else float(openrows['PnL (pts x qty)'].sum()),
               invested=0.0 if flat else float(last['Qty']) * float(last['AvgPrice']),
               status='Flat' if flat else f"Holding {int(last['Lots'])} lots")
    if flat and bool(last.get('Blocked', False)):
        out['status'] = f"Flat - portfolio full (limit {(s.get('group') or {}).get('limit', '?')})"
    return out


def evaluate(s):
    """-> dict with tr (bar log), trades and the numbers shown in the summary."""
    ctx = _eval_prepare(s)
    trace, st_ = [], {'lots': 0}
    trades = run_m2(ctx['df'], s.get('profit') or 0.0, int(s['lots']), start_date=ctx['eff_start'], lot_qty=s['qty'], trace=trace,
                    span=s.get('span'), candle=s.get('candle', 'every'), lot_amount=s.get('amount'), skip=ctx['skip'],
                    entry=s.get('entry', 'above'), method=int(s.get('method', 1)), combo=s.get('combo'))
    return _eval_finish(s, ctx, trace, trades)


def evaluate_group(strategies):
    """All strategies of ONE portfolio group (same settings, different stocks) simulated together so the
    'max open stocks' limit is applied in time order.  -> {strategy name: ev dict  or  Exception}"""
    limit = int((strategies[0].get('group') or {}).get('limit') or 0) or 10 ** 9
    out, items, ctxs = {}, [], []
    for s in strategies:
        try:
            ctxs.append(_eval_prepare(s))
        except Exception as e:
            out[s['name']] = e
            continue
        tr, stt = [], {'lots': 0}
        items.append({'s': s, 'trace': tr, 'state': stt, 'stopped': bool(s.get('stopped')), 'g': _eval_gen(s, ctxs[-1], tr, stt)})
    wait_ = next((e for e in out.values() if isinstance(e, DataLoading)), None)
    if wait_ is not None:                                          # partial data would give wrong 'max open stocks' results
        return {s['name']: wait_ for s in strategies}
    results = run_group(items, limit)
    for i, it in enumerate(items):
        out[it['s']['name']] = _eval_finish(it['s'], ctxs[i], it['trace'], results[i])
        out[it['s']['name']]['ask_log'] = it.get('ask_log', [])[-200:]
    return out


def group_members(folder, s):
    g = s.get('group') or {}
    members = []
    for f in list_files(folder):
        try:
            m = load_strategy(folder, f)
        except Exception:
            continue
        if (m.get('group') or {}).get('id') == g.get('id'):
            members.append(m)
    members.sort(key=lambda m: ((m.get('group') or {}).get('rank', 0), m['name']))
    return members


def evaluate_any(folder, s):
    """evaluate one strategy - inside its portfolio group when it has a limit."""
    g = s.get('group')
    if not (g and g.get('limit')):
        return evaluate(s)
    r = evaluate_group(group_members(folder, s))[s['name']]
    if isinstance(r, Exception):
        raise r
    return r


def evaluate_blocking(folder, s):
    """strategy page: wait (max 4 min) for the background download instead of showing 'loading'."""
    st.session_state['_wait_candles'] = True
    try:
        return evaluate_any(folder, s)
    finally:
        st.session_state['_wait_candles'] = False


def money(x):
    return f"{x:,.0f}"


# =============================================================== pages
def checkbox_dropdown(title, options, default, key, fmt=str, custom=False):
    """Drop-down with a check box beside every option + Select all. Returns the ticked options.
    custom=True adds a small "type a value + ＋" line at the bottom of the list (for quantity / amount)."""
    base = f'cb_{key}'
    extra = st.session_state.setdefault(f'{base}__extra', []) if custom else []
    opts = list(options) + [x for x in extra if x not in options]
    if custom:
        opts = sorted(opts)
    for o in opts:
        if f'{base}_{o}' not in st.session_state:
            st.session_state[f'{base}_{o}'] = o in default

    def _toggle_all():
        v = st.session_state[f'{base}__all']
        for o in opts:
            st.session_state[f'{base}_{o}'] = v

    def _add():                                                   # runs before the page re-draws
        for x in str(st.session_state.get(f'{base}__new', '')).split(','):
            try:
                v = float(x.strip())
            except ValueError:
                continue
            v = int(v) if v == int(v) else v
            if v > 0 and v not in options and v not in st.session_state[f'{base}__extra']:
                st.session_state[f'{base}__extra'].append(v)
                st.session_state[f'{base}_{v}'] = True            # new value is ticked straight away
        st.session_state[f'{base}__new'] = ''

    chosen = [o for o in opts if st.session_state[f'{base}_{o}']]
    st.session_state[f'{base}__all'] = len(chosen) == len(opts)
    with st.popover(f'{title} · {len(chosen)} selected', width='stretch'):
        st.checkbox('Select all', key=f'{base}__all', on_change=_toggle_all)
        for o in opts:
            st.checkbox(fmt(o), key=f'{base}_{o}')
        if custom:
            st.divider()
            c1, c2 = st.columns([4, 1])
            c1.text_input('Add value', key=f'{base}__new', placeholder='type a value, e.g. 15 (comma for many)',
                          label_visibility='collapsed')
            c2.button('＋', key=f'{base}__add', on_click=_add, help='Add this value to the list')
    return chosen


NIFTY50_STOCKS = ['ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE',
                  'BAJAJFINSV', 'BEL', 'BHARTIARTL', 'CIPLA', 'COALINDIA', 'DRREDDY', 'EICHERMOT', 'ETERNAL',
                  'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'INDIGO',
                  'INFY', 'ITC', 'JIOFIN', 'JSWSTEEL', 'KOTAKBANK', 'LT', 'M&M', 'MARUTI', 'NESTLEIND', 'NTPC',
                  'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN', 'SHRIRAMFIN', 'SUNPHARMA', 'TATACONSUMER',
                  'TATAMOTORS', 'TATASTEEL', 'TCS', 'TECHM', 'TITAN', 'TRENT', 'ULTRACEMCO', 'WIPRO']


@st.cache_data(ttl=86400, show_spinner='Loading instrument list ...')
def instrument_choices(exchange):
    """{'Index': [...], 'Stocks': [...], 'Commodity': [...]} for the drop-downs (built from Angel's instrument list)."""
    try:
        master = _scrip_master()
    except Exception:
        return {}
    rows = [r for r in master if r.get('exch_seg') == exchange]
    out = {}
    idx = sorted({r['symbol'] for r in rows if r.get('instrumenttype') == 'AMXIDX'})
    if idx:
        out['Index'] = idx
    if exchange == 'NSE':
        eq = {r['symbol'] for r in rows if r['symbol'].endswith('-EQ')}
        st_ = [x for x in NIFTY50_STOCKS if x + '-EQ' in eq]
        if st_:
            out['Stocks'] = st_
    if exchange == 'MCX':
        today, best = now_ist().to_pydatetime(), {}
        for r in rows:
            if r.get('instrumenttype') != 'FUTCOM':
                continue
            try:
                exp = dt.datetime.strptime(r['expiry'], '%d%b%Y')
            except Exception:
                continue
            if exp.date() >= today.date() and (r['name'] not in best or exp < best[r['name']][0]):
                best[r['name']] = (exp, r['symbol'])
        if best:
            out['Commodity'] = [v[1] for k, v in sorted(best.items())]      # nearest-expiry future of each commodity
    return out


NSE_INDEX_CSV = {                                              # niftyindices.com constituent lists (downloaded once a day)
    'NIFTY 50': 'ind_nifty50list.csv', 'NIFTY NEXT 50': 'ind_niftynext50list.csv', 'NIFTY 100': 'ind_nifty100list.csv',
    'NIFTY 200': 'ind_nifty200list.csv', 'NIFTY 500': 'ind_nifty500list.csv', 'NIFTY BANK': 'ind_niftybanklist.csv',
    'NIFTY MIDCAP 50': 'ind_niftymidcap50list.csv', 'NIFTY MIDCAP 100': 'ind_niftymidcap100list.csv',
    'NIFTY MIDCAP 150': 'ind_niftymidcap150list.csv', 'NIFTY SMALLCAP 50': 'ind_niftysmallcap50list.csv',
    'NIFTY SMALLCAP 100': 'ind_niftysmallcap100list.csv', 'NIFTY SMALLCAP 250': 'ind_niftysmallcap250list.csv',
    'NIFTY MIDSMALLCAP 400': 'ind_niftymidsmallcap400list.csv', 'NIFTY IT': 'ind_niftyitlist.csv',
    'NIFTY AUTO': 'ind_niftyautolist.csv', 'NIFTY FMCG': 'ind_niftyfmcglist.csv', 'NIFTY PHARMA': 'ind_niftypharmalist.csv',
    'NIFTY METAL': 'ind_niftymetallist.csv', 'NIFTY REALTY': 'ind_niftyrealtylist.csv', 'NIFTY ENERGY': 'ind_niftyenergylist.csv',
    'NIFTY MEDIA': 'ind_niftymedialist.csv', 'NIFTY PSU BANK': 'ind_niftypsubanklist.csv', 'NIFTY INFRA': 'ind_niftyinfralist.csv',
    'NIFTY FINANCE': 'ind_niftyfinancelist.csv', 'NIFTY PVT BANK': 'ind_nifty_privatebanklist.csv',
    'NIFTY COMMODITIES': 'ind_niftycommoditieslist.csv', 'NIFTY CONSUMPTION': 'ind_niftyconsumptionlist.csv',
}
MCX_INDEX_PREFIX = {'BULL': ('GOLD', 'SILVER'), 'METL': ('ALUMIN', 'COPPER', 'LEAD', 'NICKEL', 'ZINC'),
                    'METAL': ('ALUMIN', 'COPPER', 'LEAD', 'NICKEL', 'ZINC'), 'ENRG': ('CRUDEOIL', 'NATURALGAS', 'NATGAS'),
                    'ENERGY': ('CRUDEOIL', 'NATURALGAS', 'NATGAS')}


@st.cache_data(ttl=86400, show_spinner=False)
def _nse_index_symbols(csv_name):
    r = requests.get(f'https://www.niftyindices.com/IndexConstituent/{csv_name}', headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
    r.raise_for_status()
    return [str(x).strip() for x in pd.read_csv(io.StringIO(r.text))['Symbol'].dropna()]


MCX_SEGMENTS = {                                               # MCX segment -> commodity-name prefixes (names come from Angel's instrument list)
    'Bullion': ('GOLD', 'SILVER'),
    'Energy': ('CRUDEOIL', 'NATURALGAS', 'NATGAS'),
    'Base metals': ('ALUMIN', 'COPPER', 'LEAD', 'NICKEL', 'ZINC'),
}
MCX_NAMES = {'GOLD': 'Gold', 'GOLDM': 'Gold Mini', 'GOLDGUINEA': 'Gold Guinea', 'GOLDPETAL': 'Gold Petal',
             'SILVER': 'Silver', 'SILVERM': 'Silver Mini', 'SILVERMIC': 'Silver Micro',
             'CRUDEOIL': 'Crude Oil', 'CRUDEOILM': 'Crude Oil Mini', 'NATURALGAS': 'Natural Gas', 'NATGASMINI': 'Natural Gas Mini',
             'ALUMINIUM': 'Aluminium', 'ALUMINI': 'Aluminium Mini', 'COPPER': 'Copper', 'COPPERM': 'Copper Mini',
             'LEAD': 'Lead', 'LEADMINI': 'Lead Mini', 'NICKEL': 'Nickel', 'NICKELM': 'Nickel Mini',
             'ZINC': 'Zinc', 'ZINCMINI': 'Zinc Mini'}


def mcx_name(n):
    return MCX_NAMES.get(n, n.title())


@st.cache_data(ttl=3600, show_spinner=False)
def mcx_catalog():
    """-> {segment: {commodity name: [(expiry date, symbol), ...] sorted by expiry}}  (only running futures, from Angel's instrument list)."""
    today = now_ist().to_pydatetime().date()
    out = {}
    for r in _scrip_master():
        if r.get('exch_seg') != 'MCX' or r.get('instrumenttype') != 'FUTCOM':
            continue
        try:
            exp = dt.datetime.strptime(r['expiry'], '%d%b%Y').date()
        except Exception:
            continue
        if exp < today:
            continue
        nm = str(r['name']).upper()
        seg = next((k for k, v in MCX_SEGMENTS.items() if nm.startswith(v)), None)
        if seg:
            out.setdefault(seg, {}).setdefault(nm, []).append((exp, r['symbol']))
    return {s: {n: sorted(set(v)) for n, v in sorted(d.items())} for s, d in out.items()}


def index_members(exchange, index_name):
    """-> (list of tradable symbols that belong to that index, note). Empty list + note when unknown."""
    try:
        idx = _token_index()
        if exchange == 'NSE':
            key = re.sub(r'\s+', ' ', index_name.upper().replace('SMLCAP', 'SMALLCAP').replace('MIDSML', 'MIDSMALLCAP')
                         .replace('FIN SERVICE', 'FINANCE')).strip()
            names = None
            if key in NSE_INDEX_CSV:
                try:
                    names = _nse_index_symbols(NSE_INDEX_CSV[key])
                except Exception:
                    names = NIFTY50_STOCKS if key == 'NIFTY 50' else None      # offline fallback
            if not names:
                return [], f'{index_name}: stock list not available - use "Or type names"'
            ok = [x for x in names if ('NSE', x.upper() + '-EQ') in idx]
            return ok, '' if ok else f'{index_name}: none of its stocks found on NSE'
        if exchange == 'MCX':
            rows = [r for r in _scrip_master() if r.get('exch_seg') == 'MCX' and r.get('instrumenttype') == 'FUTCOM']
            up = index_name.upper()
            prefixes = next((v for k, v in MCX_INDEX_PREFIX.items() if k in up), None)      # unknown index -> every commodity
            today, best = now_ist().to_pydatetime(), {}
            for r in rows:
                nm = r['name'].upper()
                if prefixes and not nm.startswith(prefixes):
                    continue
                try:
                    exp = dt.datetime.strptime(r['expiry'], '%d%b%Y')
                except Exception:
                    continue
                if exp.date() >= today.date() and (r['name'] not in best or exp < best[r['name']][0]):
                    best[r['name']] = (exp, r['symbol'])
            out = [v[1] for k, v in sorted(best.items())]                  # nearest-expiry future of each commodity
            return out, '' if out else f'{index_name}: no running futures found'
    except Exception as e:
        return [], f'{index_name}: {e}'
    return [], f'{index_name}: stock list not available - use "Or type names"'


def lbl(text):                                                # same label look for EVERY field -> rows line up
    st.markdown(f"<div style='font-size:14px;line-height:1.5;padding:0 0 8px 0;opacity:.85'>{text}</div>", unsafe_allow_html=True)


def section_title(title, hint=''):
    st.markdown("<div style='display:flex;align-items:center;gap:10px;margin:2px 0 10px 0'>"
                "<span style='width:5px;height:22px;background:#ff4b4b;border-radius:2px;display:inline-block'></span>"
                f"<b style='font-size:17px'>{title}</b><span style='opacity:.6;font-size:13px'>{hint}</span></div>",
                unsafe_allow_html=True)


def typed_values(txt):
    out = []
    for x in txt.split(','):
        try:
            v = float(x.strip())
        except ValueError:
            continue
        if v > 0 and v not in out:
            out.append(v)
    return out


def page_builder():
    st.markdown('# Backtest Builder')
    with st.container(border=True):
        c1, c2 = st.columns(2)                                # ---- basic details
        with c1:
            lbl('Folder Name')
            folder = st.selectbox('Folder Name', list_folders(), index=None, accept_new_options=True,
                                  placeholder='Select or create a folder', label_visibility='collapsed')
        with c2:
            lbl('File Name')
            prefix = st.text_input('File Name', label_visibility='collapsed', placeholder='e.g. NSE morning')
        c1, c2 = st.columns(2)
        with c1:
            lbl('Exchange')
            exchange = st.selectbox('Exchange', ['NSE', 'MCX', 'NFO', 'BSE'], label_visibility='collapsed')
        with c2:
            lbl('Capital')
            capital = st.number_input('Capital', min_value=0.0, value=1000000.0, step=10000.0, label_visibility='collapsed')

        today_ = now_ist().date()                                 # ---- BACKTEST PERIOD
        c1, c2 = st.columns(2)
        with c1:
            lbl('From date (backtest starts)')
            from_d = st.date_input('From date', value=today_ - dt.timedelta(days=365), max_value=today_, format='YYYY-MM-DD',
                                   label_visibility='collapsed', key='bt_from')
        with c2:
            lbl('To date (backtest ends)')
            to_d = st.date_input('To date', value=today_, max_value=today_, format='YYYY-MM-DD',
                                 label_visibility='collapsed', key='bt_to')
        if from_d > to_d:
            st.error('From date must be on or before the To date.')
        else:
            st.caption(f'Historical candles from {from_d:%d %b %Y} to {to_d:%d %b %Y} ({(to_d - from_d).days + 1} days) will be tested. '
                       f'The cloud needs ~78 candles before the From date - they are downloaded automatically as warm-up.')

        with st.container(border=True):                       # ---- ENTRY CONDITION
            section_title('Entry condition', 'Ichimoku cloud - method, index, stocks, candle, position size')
            if exchange == 'MCX':                                 # Segment -> Commodity -> Contracts
                try:
                    cat = mcx_catalog()
                except Exception as e:
                    cat = {}
                    st.error(f'MCX instrument list not available: {e}')
                cols = st.columns(4)
                notes, members, sel_idx = [], [], []
                with cols[0]:
                    lbl(f'Segment ({len(cat)})')
                    segs = checkbox_dropdown('Select segment', list(cat), [], 'mcx_seg')
                coms = [n for s_ in segs for n in cat.get(s_, {})]
                with cols[1]:
                    lbl(f'Commodity ({len(coms)})')
                    if coms:
                        pick_com = checkbox_dropdown('Select commodity', coms, [], 'mcx_com', fmt=mcx_name)
                    else:
                        pick_com = []
                        st.popover('Choose a segment first', width='stretch', disabled=True)
                exp_of, contracts, nearest = {}, [], []
                for s_ in segs:
                    for n in pick_com:
                        for i_, (e_, sym_) in enumerate(cat.get(s_, {}).get(n, [])):
                            exp_of[sym_] = e_
                            contracts.append(sym_)
                            if i_ == 0:
                                nearest.append(sym_)                  # default tick = nearest expiry of each commodity
                with cols[2]:
                    lbl(f'Contracts ({len(contracts)})')
                    if contracts:
                        picked = checkbox_dropdown('Select contracts', contracts, nearest, 'mcx_con',
                                                   fmt=lambda x: f'{x}  ({exp_of[x]:%d %b %Y})')
                    else:
                        picked = []
                        st.popover('Choose a commodity first', width='stretch', disabled=True)
                with cols[3]:
                    lbl('Or type names (comma separated)')
                    typed_txt = st.text_input('Or type names', label_visibility='collapsed', placeholder='GOLDM05DEC25FUT',
                                              help='MCX: exact future symbol. Typed names are added to the ticked ones.')
            else:
                idx_opts = instrument_choices(exchange).get('Index', [])
                cols = st.columns(3)                                  # Index | its stocks | type names
                with cols[0]:
                    lbl(f'Index ({len(idx_opts)})')
                    sel_idx = checkbox_dropdown('Select index', idx_opts, [], f'idx_{exchange}')
                members, notes = [], []
                for ix in sel_idx:                                    # chosen index -> ITS stocks (MCX: its commodity futures)
                    m_, note_ = index_members(exchange, ix)
                    members += [x for x in m_ if x not in members]
                    if note_:
                        notes.append(note_)
                with cols[1]:
                    what = 'Commodities' if exchange == 'MCX' else 'Stocks'
                    lbl(f'{what} of the chosen index ({len(members)})')
                    if members:
                        picked = checkbox_dropdown(f'Select {what.lower()}', members, members, f'stk_{exchange}')
                    else:
                        picked = []
                        st.popover('Choose an index first', width='stretch', disabled=True)
                with cols[2]:
                    lbl('Or type names (comma separated)')
                    typed_txt = st.text_input('Or type names', label_visibility='collapsed',
                                              placeholder='Nifty 50, RELIANCE, GOLDM05DEC25FUT',
                                              help='Index: Nifty 50  |  Stock: RELIANCE (or RELIANCE-EQ)  |  MCX: exact future symbol. '
                                                   'Typed names are added to the ticked ones.')
            if notes:
                st.caption(' | '.join(notes))
            lbl('Entry method (one file each)')
            methods = [METHOD_OPTIONS[x] for x in checkbox_dropdown('Select method', list(METHOD_OPTIONS), ['Method 1 - exit at any lot count'], 'method')]
            st.caption('Method 1: profit target exits at ANY lot count.  Method 2: profit target is checked only AFTER all Max lots are filled '
                       '(span / cloud exits still work any time).  Method 3: exit when the Close is below the cloud AND the position is in profit, '
                       'at any lot count (no profit target / span exit needed).')
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                lbl('Max lots (one file each)')
                lots_list = checkbox_dropdown('Select lots', LOTS_OPTIONS, [10], 'lots')
            with c2:
                lbl('Buy on - investment timeframe (one file each)')
                candles = checkbox_dropdown('Select candle', list(CANDLE_OPTIONS), ['Every candle'], 'candle')
            with c3:
                lbl('Price vs cloud (one file each)')
                entries = [ENTRY_OPTIONS[x] for x in checkbox_dropdown('Select entry', list(ENTRY_OPTIONS), ['Above cloud'], 'entry')]
            with c4:
                lbl('Max stocks open at a time')
                max_open = st.number_input('Max stocks open at a time', min_value=0, max_value=500, value=0, step=1,
                                           label_visibility='collapsed',
                                           help='0 = no limit. Example: 50 stocks selected, limit 10 -> only the first 10 stocks that give an entry are '
                                                'taken. Another stock can enter only when one of the open stocks has fully exited. '
                                                'Stocks already open can still add lots up to Max lots.')
            # every ticked Buy-on mode gets its own timeframe picker (1m ... 1H, 4H, 1D) - one file per mode x timeframe
            TF_DEFAULT = {'Every candle': ['1D'], 'Every 30 min': ['5m'], 'Every 1 hour': ['3m'], 'Every 2 hours': ['5m'], 'Every 4 hours': ['5m'],
                          'Daily': ['1H'], 'Weekly': ['1D']}
            mode_tfs, tfs, cd_pairs = {}, [], []
            if candles:
                tcols = st.columns(len(candles))
                for col, cd in zip(tcols, candles):
                    with col:
                        lbl(f'Cloud timeframe for "{cd}" (one file each)')
                        mode_tfs[cd] = checkbox_dropdown(f'Select timeframe - {cd}', list(TF), TF_DEFAULT.get(cd, ['1D']), f'tf_{CANDLE_OPTIONS[cd]}')
                    for tf_ in mode_tfs[cd]:
                        cd_pairs.append((cd, tf_))
                        if tf_ not in tfs:
                            tfs.append(tf_)
                for cd_, tf_ in cd_pairs:                    # an interval not longer than the candle itself = same as "Every candle"
                    code_ = CANDLE_OPTIONS[cd_]
                    if code_ in INTERVAL_MIN and TF[tf_][1] >= pd.Timedelta(minutes=INTERVAL_MIN[code_]):
                        st.caption(f'ℹ️ "{cd_}" with {tf_} candles = one lot on every candle (the candle is not shorter than the interval).')
                if cd_pairs:
                    st.caption('Files will be made for (investment timeframe → cloud timeframe):  ' + '   |   '.join(f'{cd} → {", ".join(v) if v else "-"}' for cd, v in mode_tfs.items()))
            if max_open:
                st.caption(f'Portfolio limit {max_open}: at most {max_open} stocks hold a position together (separately for every '
                           f'timeframe / lots / exit / entry / method combination). A waiting stock enters when a slot frees up.')
            close_t = dt.time(23, 30) if exchange == 'MCX' else dt.time(15, 30)      # fixed by exchange (no input box)
            c = st.columns(3)
            with c[0]:
                lbl('Position - size each lot by')
                modes = checkbox_dropdown('Select position type', ['Quantity', 'Amount', 'Equal weightage'], ['Quantity'], 'mode')
            sizes, blocks = [], []
            if 'Quantity' in modes:
                blocks.append('qty')
            if 'Amount' in modes:
                blocks.append('amt')
            if 'Equal weightage' in modes:
                sizes.append(('eq', 0))
            for i, kind in enumerate(blocks):
                row = c if i == 0 else st.columns(3)
                a_, b_ = (1, 2) if i == 0 else (0, 1)
                if kind == 'qty':
                    with row[a_]:
                        lbl('Quantity per lot')
                        vals = [float(q) for q in checkbox_dropdown('Select quantity', QTY_OPTIONS, [1], 'qty', custom=True)]
                else:
                    with row[a_]:
                        lbl('Amount per lot')
                        vals = [float(q) for q in checkbox_dropdown('Select amount', AMOUNT_OPTIONS, [10000], 'amt', custom=True)]
                sizes += [(kind, v) for v in dict.fromkeys(vals)]

        with st.container(border=True):                       # ---- EXIT CONDITION
            section_title('Exit condition', 'profit target % on average entry, or Span exit')
            only3 = set(methods) == {3}
            if only3:
                st.info('Method 3 exit: the Close is BELOW the cloud and the position is in profit - at any lot count (Max lots need not be filled). '
                        'No profit target or span exit is needed.')
            targets = []
            if not only3:
                c = st.columns(3)
                with c[0]:
                    lbl('Profit target %')
                    pct_exits = checkbox_dropdown('Select target', TARGET_OPTIONS, [1, 2], 'exits_pct', fmt=exit_label)
                with c[1]:
                    lbl('Span exit')
                    span_exits = checkbox_dropdown('Select span exit', list(SPAN_ONLY), [], 'exits_span')
                span_dirs = []
                if span_exits:                                             # shown ONLY after a span is ticked
                    with c[2]:
                        lbl('Span exit when price closes')
                        span_dirs = checkbox_dropdown('Select direction', SPAN_DIRECTION, ['Below span'], 'exits_span_dir')
                span_dirs = span_dirs or ['Below span']                    # a span ticked with no direction -> Below
                span_exits = ([x for x in span_exits if 'Below span' in span_dirs]                                  # below the span(s)
                              + [f'{x} (above)' for x in span_exits if 'Above span' in span_dirs])                   # above the span(s)
                picked_exits = pct_exits + span_exits
                lbl('Combine the selected exits (optional)')
                combine = st.segmented_control('Combine', ['AND  (all selected exits must hit)', 'OR  (any one selected exit hits)'],
                                               selection_mode='single', default=None, label_visibility='collapsed', key='exit_combine3')
                if not combine or len(picked_exits) < 2:
                    targets = picked_exits                                  # nothing chosen -> one file per exit (click again to un-select)
                else:
                    targets = [('AND' if combine.startswith('AND') else 'OR', picked_exits)]      # ONE file with a combined exit
                    st.caption(f'1 file with exit  {exit_label(targets[0])}')
                if not combine:
                    st.caption('AND / OR not chosen: every selected exit makes its own file.  Click AND or OR to join them into one file '
                               '(click it again to un-select).')

                if 3 in methods:
                    st.caption('Method 3 files ignore these exits - they always exit below the cloud while in profit (one extra file).')

        start = f'{from_d:%Y-%m-%d} 00:00'                      # backtest window: From date 00:00 ...
        end = f'{to_d:%Y-%m-%d} 23:59'                          # ... To date 23:59 (capped at now)
        symbols, seen = [], set()
        for x in picked + [t.strip() for t in typed_txt.split(',') if t.strip()]:
            if x.upper() not in seen:
                seen.add(x.upper())
                symbols.append(x)
        n_other = len([m for m in methods if m != 3])
        n = len(symbols) * len(cd_pairs) * len(lots_list) * len(sizes) * len(entries) * (len(targets) * n_other + (1 if 3 in methods else 0))
        exits_to_save = targets if targets else ([M3_EXIT] if 3 in methods else [])
        st.caption(f'{len(symbols)} symbol(s) x {len(lots_list)} lots x {len(sizes)} size(s) x '
                   f'{len(targets)} exits x {len(cd_pairs)} (buy-on mode, timeframe) pair(s) x {len(entries)} entry side(s) x {len(methods)} method(s) (Method 3 = 1 fixed exit) = {n} file(s) will be saved.  Period = {start[:10]} → {end[:10]}')
        b1, b2, _ = st.columns([2, 1.6, 6])
        run_now = b1.button('▶ Save & run backtest', type='primary', help='Saves the files, runs them on the historical candles and keeps the results - '
                                                                      'they are waiting in My Backtests (also after you log in again).')
        save_only = b2.button('💾 Save only')
        if run_now or save_only:
            fname = (folder or '').strip()
            if not fname:
                st.error('Give a folder name.')
            elif from_d > to_d:
                st.error('From date must be on or before the To date.')
            elif n == 0:
                st.error('Pick (or type) at least one symbol, Buy-on mode with a timeframe, entry method, Max lots, position size, entry side, candle option and exit option.')
            elif n > 150:
                st.error('Too many files (max 150 at a time).')
            else:
                try:
                    for sym in symbols:
                        _instrument_token(exchange, sym)      # fail early if a symbol is wrong
                except Exception as e:
                    st.error(str(e))
                    return
                made, skipped = create_strategies(fname, (prefix or '').strip(), exchange, symbols, tfs, capital, sizes,
                                                  close_t.strftime('%H:%M'), lots_list, exits_to_save, candles, start, end, entries, methods, int(max_open), cd_pairs)
                if run_now and (made or skipped):
                    to_run = []
                    for nm in made + skipped:
                        fn_ = safe(nm) + '.json'
                        try:
                            to_run.append((fn_, load_strategy(fname, fn_)))
                        except Exception:
                            pass
                    store_ = st.session_state.setdefault('bt', {})
                    for fn_, _ in to_run:                           # settings may have changed -> forget any older result of these files
                        drop_result(store_, fname, fn_)
                    run_backtest(fname, to_run, label=f'Backtest \"{fname}\"')
                    ok_ = [(fn_, r_['ev']['closed_pnl'] + r_['ev']['open_pnl']) for fn_, _ in to_run
                           for r_ in [store_.get((fname, fn_), {})] if 'ev' in r_]
                    if not ok_:                                     # every file failed - stay here and show why
                        errs_ = [r_['err'] for fn_, _ in to_run for r_ in [store_.get((fname, fn_), {})] if 'err' in r_]
                        st.error(f"Backtest failed: {errs_[0] if errs_ else 'no result'}")
                        return
                    best_ = max(ok_, key=lambda t: t[1])[0]         # open the results page (graph, signals, trades) of the best file
                    st.session_state['page'] = 'strategies'
                    st.session_state['open_file'] = (fname, best_)
                    if len(to_run) > 1:
                        st.session_state['_flash'] = (f'{len(ok_)} of {len(to_run)} files ran - this is the one with the highest Total P&L. '
                                                      'Press ← Back to My Backtests to see the summary of the whole folder.')
                    st.rerun()
                if made:
                    st.success(f'Saved {len(made)} backtest file(s) in folder "{fname}". Open My Backtests and press Run backtest. (Simulation only - no orders.)')
                if skipped:
                    st.info(f'{len(skipped)} file(s) with the same name already existed - they were UPDATED with the new dates / settings.')


def save_strategy(folder, fname, s):
    with open(os.path.join(user_dir(), safe(folder), fname), 'w', encoding='utf-8') as f:
        json.dump(s, f, indent=1)


def csv_button(label, df, key):
    st.download_button(label, df.to_csv(index=False).encode('utf-8'), file_name=f'{key}.csv', mime='text/csv', key=f'dl_{key}')


def range_text(s):
    return f"{str(s['start'])[:10]} → {str(s.get('end') or '')[:10]}"


# =============================================================== running a backtest
def _candle_key(s):
    return (s['exchange'], s['symbol'], s['tf'], pd.Timestamp(s['start']).normalize().strftime('%Y-%m-%d'),
            pd.Timestamp(s['end']).normalize().strftime('%Y-%m-%d'))


def run_backtest(fol, files, label='Running backtest'):
    """files = [(fname, strategy dict)].  Downloads the historical candles (all symbols in parallel, rate-limited), runs the
    engine and keeps the result in st.session_state['bt'][(folder, file)] = {'ev': ...} or {'err': text}.
    Files that share a 'Max stocks open' limit are simulated TOGETHER (whole group)."""
    res = st.session_state.setdefault('bt', {})
    creds = st.session_state['creds']
    todo, seen = [], set()
    for f, s in files:                                             # a grouped file pulls in its whole group
        g = s.get('group') or {}
        members = group_members(fol, s) if g.get('limit') else [s]
        for m in members:
            fn = safe(m['name']) + '.json'
            if fn not in seen:
                seen.add(fn)
                todo.append((fn, m))
    for _, s in todo:                                              # start every download first (3 worker threads)
        try:
            load_candles(creds, s['exchange'], s['symbol'], s['tf'], *_candle_key(s)[3:], wait=False)
        except Exception:
            pass
    bar = st.progress(0.0, text=f'{label} ...')
    st.session_state['_wait_candles'] = True
    try:
        singles = [(fn, s) for fn, s in todo if not (s.get('group') or {}).get('limit')]
        groups = {}
        for fn, s in todo:
            g = s.get('group') or {}
            if g.get('limit'):
                groups.setdefault(g['id'], []).append((fn, s))
        total, done = len(singles) + sum(len(v) for v in groups.values()), 0
        for fn, s in singles:
            bar.progress(done / max(total, 1), text=f"{label}: {s['symbol']} {s['tf']}  ({done + 1}/{total})")
            try:
                put_result(res, fol, fn, {'ev': evaluate(s), 'ran': now_ist()})
            except Exception as e:
                put_result(res, fol, fn, {'err': str(e), 'ran': now_ist()})
            done += 1
        for gid, members in groups.items():
            members.sort(key=lambda fs: ((fs[1].get('group') or {}).get('rank', 0), fs[1]['name']))
            bar.progress(done / max(total, 1), text=f'{label}: group of {len(members)} stocks (portfolio limit) ...')
            out = evaluate_group([s for _, s in members])
            for fn, s in members:
                r = out.get(s['name'])
                put_result(res, fol, fn, {'err': str(r), 'ran': now_ist()} if isinstance(r, Exception) else {'ev': r, 'ran': now_ist()})
            done += len(members)
    finally:
        st.session_state['_wait_candles'] = False
        bar.empty()


# =============================================================== results / metrics
def closed_of(trades):
    if trades is None or trades.empty:
        return pd.DataFrame()
    return trades[~trades['Exit Reason'].str.startswith('Open at end')]


def bt_metrics(s, ev):
    """key backtest numbers of one finished run."""
    closed = closed_of(ev['trades'])
    pnl = closed['PnL (pts x qty)'] if len(closed) else pd.Series(dtype=float)
    gp, gl = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
    eq = np.concatenate([[0.0], pnl.to_numpy(dtype=float).cumsum()]) if len(pnl) else np.array([0.0])
    cap = float(s.get('capital') or 0)
    total = ev['closed_pnl'] + ev['open_pnl']
    return {'trades': len(closed), 'wins': int((pnl > 0).sum()), 'win_rate': (float((pnl > 0).mean() * 100) if len(pnl) else None),
            'realized': ev['closed_pnl'], 'open_pnl': ev['open_pnl'], 'total': total,
            'ret_pct': (total / cap * 100) if cap else None,
            'profit_factor': (gp / gl) if gl > 0 else None,
            'max_dd': float((np.maximum.accumulate(eq) - eq).max()),
            'avg_hold': float(closed['Holding Days'].mean()) if len(closed) else None,
            'best': float(pnl.max()) if len(pnl) else None, 'worst': float(pnl.min()) if len(pnl) else None,
            'open_lots': 0 if ev['last'] is None or ev['flat'] else int(ev['last']['Lots']), 'bars': ev['diag']['bars']}


def fmt(x, nd=2, suffix=''):
    return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:,.{nd}f}{suffix}'


def row_stats(s, ev, err, loading=False):
    """numbers of one file for the folder / group tables (open_n = position still open on the To date)."""
    if ev is None:
        return {'state': 'Error' if err else 'Not run', 'sub': (err or 'press Run backtest')[:70], 'open_n': 0, 'closed_n': 0,
                'realized': 0.0, 'unreal': 0.0}
    last = ev['last']
    holding = last is not None and not ev['flat']
    return {'state': 'Done', 'sub': 'No signal in this range' if last is None else ev['status'].replace('Holding', 'Open at end:'),
            'open_n': 1 if holding else 0, 'closed_n': ev['closed_n'], 'realized': ev['closed_pnl'], 'unreal': ev['open_pnl']}


def folder_summary(fol, files, evs, errs):
    rows = []
    for f, s in files:
        ev = evs.get((fol, f))
        base = {'Strategy': s['name'], 'Symbol': s['symbol'], 'TF': s['tf'], 'From': str(s['start'])[:10], 'To': str(s.get('end') or '')[:10]}
        if ev is None:
            rows.append({**base, 'Trades': None, 'Win %': None, 'Realized': None, 'Open at end': None, 'Total P&L': None,
                         'Return %': None, 'Max DD': None, 'Note': errs.get((fol, f), 'not run')[:80]})
            continue
        m = bt_metrics(s, ev)
        rows.append({**base, 'Trades': m['trades'], 'Win %': None if m['win_rate'] is None else round(m['win_rate'], 1),
                     'Realized': round(m['realized'], 2), 'Open at end': round(m['open_pnl'], 2), 'Total P&L': round(m['total'], 2),
                     'Return %': None if m['ret_pct'] is None else round(m['ret_pct'], 2), 'Max DD': round(m['max_dd'], 2), 'Note': ''})
    return pd.DataFrame(rows)


def all_trades(fol, files, evs):
    parts = []
    for f, s in files:
        ev = evs.get((fol, f))
        if ev is not None and not ev['trades'].empty:
            t = ev['trades'].copy()
            t.insert(0, 'Strategy', s['name'])
            t.insert(1, 'Symbol', s['symbol'])
            parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# =============================================================== one backtest on its own page
def page_file(fol, fname):
    if st.button('← Back to My Backtests'):
        st.session_state.pop('open_file', None)
        st.rerun()
    flash_ = st.session_state.pop('_flash', None)
    if flash_:
        st.info(flash_)
    try:
        s = load_strategy(fol, fname)
    except FileNotFoundError:
        st.session_state.pop('open_file', None)
        st.rerun()
        return
    st.markdown(f"# {fol} / {s['name']}")
    st.info(f"**{s['exchange']}** | {s['symbol']} | {s['tf']} | **{range_text(s)}** | capital {s['capital']:,.0f} | max lots {s['lots']} | "
            f"{size_text(s)} | {method_text(s)} | entry {entry_text(s)} | exit {exit_text(s)} | buy {candle_text(s).lower()} | {limit_text(s)}")
    store = st.session_state.setdefault('bt', {})
    hydrate_results(store, fol, [fname])
    if st.button('▶ Run again', key='rerun_one'):
        drop_result(store, fol, fname)
    if (fol, fname) not in store:
        with st.spinner('Downloading historical candles and running the backtest ...'):
            run_backtest(fol, [(fname, s)])
    r = store.get((fol, fname), {})
    if 'ev' not in r:
        st.error(f"Backtest failed: {r.get('err', 'unknown error')}")
        return
    ev, tr, trades = r['ev'], r['ev']['tr'], r['ev']['trades']
    m = bt_metrics(s, ev)
    st.caption(f"Backtest ran {r['ran']:%d %b %Y %H:%M:%S} IST  |  {m['bars']} closed {s['tf']} candles used (cloud warm-up included)")
    c = st.columns(5)
    c[0].metric('Total P&L', fmt(m['total']), None if m['ret_pct'] is None else f"{m['ret_pct']:+.2f}% of capital")
    c[1].metric('Realized P&L', fmt(m['realized']))
    c[2].metric('Open at end date', fmt(m['open_pnl']), f"{m['open_lots']} / {s['lots']} lots" if m['open_lots'] else 'flat')
    c[3].metric('Closed trades', m['trades'])
    c[4].metric('Win rate', fmt(m['win_rate'], 0, '%'))
    c = st.columns(5)
    c[0].metric('Profit factor', fmt(m['profit_factor']))
    c[1].metric('Max drawdown (closed trades)', fmt(m['max_dd']))
    c[2].metric('Avg holding (days)', fmt(m['avg_hold'], 1))
    c[3].metric('Best trade', fmt(m['best']))
    c[4].metric('Worst trade', fmt(m['worst']))
    if tr.empty:
        st.warning('No candle in this date range could be tested (cloud needs ~78 candles of history before the From date, '
                   'or Angel returned no data for this range).')
        return
    closed = closed_of(trades)
    key = f'{fol}_{fname}'.replace(' ', '_')
    tabs = st.tabs(['1. Equity & chart', '2. Signals', '3. Closed trades', '4. Open at end date', '5. Bar log'])
    with tabs[0]:
        if len(closed):
            eqc = closed.sort_values('Exit Date')[['Exit Date', 'PnL (pts x qty)']].copy()
            eqc['Cumulative realized P&L'] = eqc['PnL (pts x qty)'].cumsum()
            st.caption('Cumulative realized P&L (after every closed trade)')
            st.line_chart(eqc.set_index('Exit Date')[['Cumulative realized P&L']])
        pl = tr.set_index('Date')[['Close', 'CloudTop', 'CloudBottom']]
        if len(pl) > 3000:
            pl = pl.iloc[::int(np.ceil(len(pl) / 3000))]
        st.caption('Close vs Ichimoku cloud (9/26/52, shift 26)')
        st.line_chart(pl)
    with tabs[1]:
        sg = tr[tr['Signal'] != 'HOLD'][['Date', 'Signal', 'Close', 'Lots', 'AvgPrice', 'Unrealised %']]
        if sg.empty:
            st.info('No signal in this range.')
        else:
            show_df(sg.sort_values('Date', ascending=False), width='stretch', hide_index=True)
            csv_button('Download signals CSV', sg, f'signals_{key}')
    with tabs[2]:
        if closed.empty:
            st.write('No closed trades.')
        else:
            ct = closed.sort_values('Exit Date', ascending=False)
            show_df(ct, width='stretch', hide_index=True)
            csv_button('Download closed trades CSV', ct, f'closed_{key}')
    with tabs[3]:
        last = ev['last']
        if last is None or ev['flat']:
            st.write('Flat on the To date - nothing open.')
        else:
            units = float(last['Qty'])
            show_df(pd.DataFrame([{'symbol': s['symbol'], 'lots': int(last['Lots']), 'avg_entry': round(float(last['AvgPrice']), 2),
                                   'units': round(units, 4), 'invested': round(units * float(last['AvgPrice']), 2),
                                   'last_close': round(float(last['Close']), 2), 'unrealised_%': last['Unrealised %'],
                                   'unrealised_pnl': round(ev['open_pnl'], 2)}]), width='stretch', hide_index=True)
            st.caption('Marked to the last close of the date range.')
    with tabs[4]:
        bl = tr.sort_values('Date', ascending=False).head(500)
        st.caption('Latest 500 bars first (download for the full log)')
        show_df(bl, width='stretch', hide_index=True)
        csv_button('Download full bar log CSV', tr.sort_values('Date'), f'barlog_{key}')


def backup_zip():
    buf, base = io.BytesIO(), user_dir()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        if os.path.isdir(base):
            for root, _, files in os.walk(base):
                for fn in files:
                    if fn.endswith(RESULT_EXT) or fn.endswith('.tmp'):      # saved results are rebuilt by 'Run backtest'
                        continue
                    pth = os.path.join(root, fn)
                    z.write(pth, os.path.relpath(pth, base))
    return buf.getvalue()


def restore_zip(data):
    """put back the .json files of a backup zip; files that already exist are kept."""
    base, n = os.path.abspath(user_dir()), 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.endswith('.json'):
                continue
            dest = os.path.abspath(os.path.join(base, info.filename))
            if not dest.startswith(base + os.sep) or os.path.exists(dest):
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with z.open(info) as src_, open(dest, 'wb') as out_:
                shutil.copyfileobj(src_, out_)
            n += 1
    return n


def make_folder():
    name = st.session_state.get('new_folder_name', '').strip()
    if name:
        os.makedirs(os.path.join(user_dir(), safe(name)), exist_ok=True)
    st.session_state['new_folder_name'] = ''


def show_df(df, **kw):
    """st.dataframe - or, when the sidebar switch 'Light tables' is ON, a plain static table (works even when the browser cannot
    fetch the interactive table's JavaScript file)."""
    if st.session_state.get('light_tables', True):
        d = df.copy()
        if len(d) > 400:
            d = d.head(400)
        try:
            if kw.get('hide_index') and len(d.columns):
                d = d.set_index(d.columns[0])
        except Exception:
            pass
        st.table(d)
        return
    st.dataframe(df, **kw)


def pnl_txt(x, pct=None):
    col = 'green' if x > 0 else ('red' if x < 0 else 'gray')
    return f":{col}[{x:,.2f}]" + ('' if pct is None else f" ({pct:+.2f}%)")


def slot_notes(fol, files, evs):
    """Replay every ENTRY / EXIT of a folder in time order -> {(strategy name, bar time, signal): text}.
    EXIT row  : 'Slot freed ... -> X came in at 13:21'   (who took the slot afterwards)
    ENTRY row : 'Took the slot of Y (exited 13:12)'      (only when X had waited because all slots were full)"""
    groups = {}
    for f, s in files:
        g = s.get('group') or {}
        groups.setdefault(g.get('id') if g.get('limit') else None, []).append((f, s))
    notes = {}
    for gid, mem in groups.items():
        lim = int((mem[0][1].get('group') or {}).get('limit') or 0) if gid is not None else 0
        events = []                                                  # (time, order, symbol, strategy name, signal)
        for f, s in mem:
            ev = evs.get((fol, f))
            if ev is None or ev['tr'].empty:
                continue
            flat, waiting = True, False
            for _, r in ev['tr'].iterrows():
                tm, sig = r['Date'], r['Signal']
                if bool(r.get('Blocked', False)) and flat and not waiting:
                    waiting = True                                   # entry signal came but all slots were full
                    events.append((tm, 1, s['symbol'], s['name'], 'WAIT'))
                if sig == 'HOLD':
                    continue
                if sig == 'BUY' and flat:
                    flat, waiting = False, False
                    events.append((tm, 0, s['symbol'], s['name'], sig))
                elif sig in ('EXIT', 'BUY+EXIT') and not flat:
                    flat = True
                    events.append((tm, -1, s['symbol'], s['name'], sig))
        holding, freed, waiting = [], [], {}
        for tm, order, sym, name, sig in sorted(events, key=lambda e: (e[0], e[1])):
            hhmm = str(tm)[11:16]
            if order == 1:
                waiting.setdefault(sym, hhmm)
            elif order == 0:
                holding.append(sym)
                used = f' · slots {len(holding)}/{lim}' if lim else ''
                if sym in waiting and freed:
                    w = waiting.pop(sym)
                    o_sym, o_tm, o_key = freed.pop(0)
                    notes[(name, str(tm)[:16], sig)] = f'Took the slot of {o_sym} (exited {o_tm}) - waited since {w}{used}'
                    notes[o_key] = notes.get(o_key, '') + f' -> {sym} came in at {hhmm}'
                elif sym in waiting:
                    waiting.pop(sym)
                    notes[(name, str(tm)[:16], sig)] = f'Got a free slot{used}'
                else:
                    notes[(name, str(tm)[:16], sig)] = ('First fill - free slot' + used) if lim else ''
            else:
                if sym in holding:
                    holding.remove(sym)
                key = (name, str(tm)[:16], sig)
                nxt = f' · next in line: {", ".join(waiting)}' if waiting else ''
                notes[key] = (f'Slot freed ({len(holding)}/{lim} used){nxt}' if lim else 'Exited') 
                freed.append((sym, hhmm, key))
    return notes


def group_entry_view(fol, members, stats, evs, lim):
    """'View entry stocks' of one group: which stocks entered live, which exited (slot freed) and came in again, which still wait."""
    per, events = {}, []
    for f, s in members:
        ev = evs.get((fol, f))
        info = {'entries': 0, 'first': None, 'last_exit': None, 'lots': 0}
        flat, wait_ = True, False
        fresh_at = {a_[0] for a_ in ((ev or {}).get('ask_log') or []) if a_[1] and len(a_) > 3 and a_[3]}
        if ev is not None and not ev['tr'].empty:
            for _, r in ev['tr'].iterrows():
                tm, sig = r['Date'], r['Signal']
                if bool(r.get('Blocked', False)) and flat and not wait_:
                    wait_ = True
                    events.append((tm, 1, s['symbol'], 'WAIT', 0))
                if sig == 'HOLD':
                    continue
                if sig == 'BUY' and flat:
                    wait_ = False
                    flat = False
                    info['entries'] += 1
                    info['first'] = info['first'] or tm
                    events.append((tm, 0, s['symbol'], 'ENTRY - slot taken' + (' (fresh cross - priority)' if tm in fresh_at else ''), int(r['Lots'])))
                elif sig in ('EXIT', 'BUY+EXIT') and not flat:
                    flat = True
                    info['last_exit'] = tm
                    events.append((tm, -1, s['symbol'], 'EXIT - slot freed', 0))
            if ev['last'] is not None and not ev['flat']:
                info['lots'] = int(ev['last']['Lots'])
        per[f] = info
    holding = [(f, s) for f, s in members if stats[f]['open_n'] > 0]
    exited = [(f, s) for f, s in members if stats[f]['open_n'] == 0 and per[f]['entries'] > 0]
    waiting = [(f, s) for f, s in members if stats[f]['open_n'] == 0 and per[f]['entries'] == 0]
    with st.expander(f'📋 View entry stocks  ·  🟢 {len(holding)} holding  ·  ✔ {len(exited)} exited  ·  ⏳ {len(waiting)} waiting', expanded=True):
        rows = []
        for f, s in holding + exited:
            i, x = per[f], stats[f]
            rows.append({'Stock': s['symbol'],
                         'Status': (f"🟢 Holding {i['lots']} lots" if x['open_n'] else '✔ Exited - can enter again'),
                         'First entry': str(i['first'])[:16] if i['first'] is not None else '-',
                         'Last exit': str(i['last_exit'])[:16] if i['last_exit'] is not None else '-',
                         'Entries': i['entries'], 'P&L': round(x['realized'] + x['unreal'], 2)})
        if rows:
            show_df(pd.DataFrame(rows), width='stretch', hide_index=True)
        else:
            st.caption('No stock has taken an entry yet.')
        if events:                                                  # replay in time order -> who holds a slot after every event
            inn, freed, log, waitq = [], [], [], []
            for tm, order, sym, what, lots in sorted(events, key=lambda e: (e[0], e[1])):
                if order == 1:                                      # entry signal, but all slots were full
                    waitq.append(sym)
                    continue
                if order == 0:
                    if sym in waitq and freed:
                        waitq.remove(sym)
                        note = f'took the slot freed by {freed.pop(0)}'
                    else:
                        if sym in waitq:
                            waitq.remove(sym)
                        note = 'free slot (first fill)'
                    inn.append(sym)
                else:
                    if sym in inn:
                        inn.remove(sym)
                    freed.append(sym)
                    note = f'{sym} goes to the exited list - its slot is free'
                log.append({'Bar time': str(tm)[:16], 'Stock': sym, 'What': what, 'Lots after': lots,
                            'Slots used': f'{len(inn)} / {lim}', 'Holding now': ', '.join(inn) or '-', 'Note': note})
            st.caption('Entry / exit order (latest first) - who holds the slots after every entry / exit')
            show_df(pd.DataFrame(log[::-1]), width='stretch', hide_index=True)
        flat_rows = []                                              # every stock that is NOT holding: why is it not entering?
        for f, s in waiting + exited:
            ev = evs.get((fol, f))
            row = {'Stock': s['symbol'], 'Last bar': '-', 'Close': None, 'Cloud (low - high)': '-', 'Close is': '-'}
            if s.get('stopped'):
                why = 'Stopped (exited by you)'
            elif ev is None:
                why = 'Loading candles ...'
            elif ev.get('data_err') and ev['tr'].empty:
                why = 'Data problem: ' + str(ev['data_err'])[:60]
            elif ev['tr'].empty:
                why = 'No candle has closed after the start time yet'
            else:
                r = ev['tr'].iloc[-1]
                cl, top, bot = float(r['Close']), float(r['CloudTop']), float(r['CloudBottom'])
                side = 'above cloud' if cl > top else ('below cloud' if cl < bot else 'inside cloud')
                need = {'below': 'below cloud', 'inside': 'inside cloud'}.get(s.get('entry'), 'above cloud')
                row.update({'Last bar': str(r['Date'])[:16], 'Close': round(cl, 2),
                            'Cloud (low - high)': f'{bot:,.2f} - {top:,.2f}', 'Close is': side})
                if bool(r.get('Blocked', False)):
                    asks = [a for a in ev.get('ask_log', []) if a[0] == r['Date'] and not a[1]]
                    who = ', '.join(asks[-1][2]) if asks and asks[-1][2] else ''
                    why = f'Entry signal, but all {lim} slots are full' + (f' (held by: {who})' if who else '')
                elif bool(r['Above']):
                    why = 'Entry signal on the last candle (slot free) - enters on this bar'
                else:
                    why = f'No signal - entry needs the close {need}, now it is {side}'
            row['Why not holding'] = why
            flat_rows.append(row)
        if flat_rows:
            st.caption(f'⏳ Not holding ({len(flat_rows)}): waiting for a first entry or a new signal')
            show_df(pd.DataFrame(flat_rows), width='stretch', hide_index=True)


def page_view():
    of = st.session_state.get('open_file')
    if of:
        if os.path.exists(os.path.join(user_dir(), safe(of[0]), of[1])):
            page_file(of[0], of[1])
            return
        st.session_state.pop('open_file', None)
    st.markdown('# My Backtests')
    c1, c2 = st.columns(2)
    query = c1.text_input('Search', placeholder='🔍 Search folder / backtest...', label_visibility='collapsed').strip().lower()
    with c2.container(border=True):
        n1, n2 = st.columns([4, 1.3])
        n1.text_input('New folder', key='new_folder_name', placeholder='Enter new folder name', label_visibility='collapsed')
        n2.button('📁 + Create', type='primary', width='stretch', on_click=make_folder)

    folders = list_folders()
    if not folders:
        st.info('No folders yet. Create one above or in Backtest Builder.')
        return
    data = {fol: [(f, load_strategy(fol, f)) for f in list_files(fol)] for fol in folders}
    shown = [fol for fol in folders if not query or query in fol.lower() or any(query in s['name'].lower() for _, s in data[fol])]
    store = st.session_state.setdefault('bt', {})

    if st.session_state.get('_result_save_err'):
        st.warning(f"Results could not be saved to disk ({st.session_state['_result_save_err']}) - they stay only until you log out.")
    for fol in shown:
        files = data[fol]
        hydrate_results(store, fol, [f for f, _ in files])
        evs = {(fol, f): store[(fol, f)]['ev'] for f, _ in files if 'ev' in store.get((fol, f), {})}
        errs = {(fol, f): store[(fol, f)]['err'] for f, _ in files if 'err' in store.get((fol, f), {})}
        with st.expander(f'📁 {fol} · {len(files)} backtest file(s) · {len(evs)} run'):
            if not files:
                st.caption('This folder is empty. Create files in Backtest Builder.')
                with st.popover('🗑️'):
                    if st.button('✔ Yes', key=f'delfol_{fol}', type='primary'):
                        delete_folder(fol)
                        st.rerun()
                continue
            h = st.columns([2.2, 4, 1.4, 0.7])
            if h[0].button('▶ Run backtest (all files)', key=f'run_{fol}', type='primary', width='stretch'):
                run_backtest(fol, files, label=f'Backtest "{fol}"')
                st.rerun()
            with h[2].popover('Clear results', width='stretch', disabled=not evs):
                st.write('Forget the results of this folder (the files stay)')
                if st.button('✔ Yes', key=f'clr_{fol}', type='primary'):
                    for f, _ in files:
                        drop_result(store, fol, f)
                    st.rerun()
            with h[3].popover('🗑️', width='stretch'):
                st.write(f'Delete {fol} ({len(files)} files)')
                if st.button('✔ Yes', key=f'delfolx_{fol}', type='primary'):
                    delete_folder(fol)
                    st.rerun()

            stats = {f: row_stats(s, evs.get((fol, f)), errs.get((fol, f))) for f, s in files}
            if evs:
                real = sum(x['realized'] for x in stats.values())
                unreal = sum(x['unreal'] for x in stats.values())
                cap = sum(float(s.get('capital') or 0) for f, s in files if (fol, f) in evs)
                trades_n = sum(x['closed_n'] for x in stats.values())
                m = st.columns(4)
                m[0].metric('Closed trades', trades_n)
                m[1].metric('Realized P&L', f'{real:,.2f}')
                m[2].metric('Open at end date', f'{unreal:,.2f}')
                m[3].metric('Total P&L', f'{real + unreal:,.2f}', f'{(real + unreal) / cap * 100:+.2f}% of capital' if cap else None)
                summ = folder_summary(fol, files, evs, errs)
                d1, d2 = st.columns(2)
                with d1:
                    csv_button('Download summary CSV', summ, f'summary_{safe(fol)}')
                with d2:
                    at = all_trades(fol, files, evs)
                    if not at.empty:
                        csv_button('Download all trades CSV', at, f'alltrades_{safe(fol)}')
                st.caption('Summary of every file (sorted by Total P&L)')
                show_df(summ.sort_values('Total P&L', ascending=False, na_position='last'), width='stretch', hide_index=True)
            else:
                st.caption('Press **Run backtest** - historical candles of each file\'s From → To range are downloaded from Angel One and tested. '
                           'Nothing is traded; it is a simulation on past data.')
            st.divider()

            seen_groups = set()
            for f, s in files:
                g_ = s.get('group') or {}
                gid = g_.get('id') if g_.get('limit') else None
                if gid is not None:
                    if gid in seen_groups:
                        continue
                    seen_groups.add(gid)
                    members = [(ff, ss) for ff, ss in files if (ss.get('group') or {}).get('limit') and (ss.get('group') or {}).get('id') == gid]
                    members.sort(key=lambda fs: ((fs[1].get('group') or {}).get('rank', 0), fs[1]['name']))
                    s0, lim = members[0][1], g_['limit']
                    with st.container(border=True):
                        st.markdown(f"📄 **Group of {len(members)} stocks** · max {lim} open at a time · {s0['tf']} · {s0['lots']} lots · {method_text(s0)} · "
                                    f"entry {entry_text(s0)} · exit {exit_text(s0)} · {range_text(s0)}")
                        if any((fol, ff) in evs for ff, _ in members):
                            group_entry_view(fol, members, stats, evs, lim)
                            if st.toggle(f'Show the {len(members)} files of this group', key=f'grpopen_{fol}_{gid}'):
                                for ff, ss in members:
                                    _render_row(fol, ff, ss, stats, evs)
                        else:
                            st.caption('Not run yet - press Run backtest for the folder.')
                else:
                    _render_row(fol, f, s, stats, evs)


def _render_row(fol, f, s, stats, evs):
    x = stats[f]
    total = x['realized'] + x['unreal']
    pct = total / float(s['capital']) * 100 if s.get('capital') else None
    r = st.columns([3, 1.6, 2, 3.4, 0.7, 0.7])
    if r[0].button('📄 ' + s['name'], key=f'open_{fol}_{f}', type='tertiary'):
        st.session_state['open_file'] = (fol, f)
        st.rerun()
    r[0].caption(f"{s['exchange']} · {s['tf']} · {range_text(s)} · {s['lots']} lots · {method_text(s)} · entry {entry_text(s)} · exit {exit_text(s)} · "
                 f"{candle_text(s).lower()} · {size_text(s)} · {limit_text(s)} · capital {s['capital']:,.0f}")
    r[1].write(s['symbol'])
    dot = {'Done': '✅', 'Error': '🔴', 'Not run': '⚪'}[x['state']]
    r[2].write(f"{dot} {x['state']}")
    r[2].caption(x['sub'])
    if (fol, f) in evs:
        r[3].markdown(f"Trades **{x['closed_n']}** · Realized {pnl_txt(x['realized'])}  \n"
                      f"Open at end {pnl_txt(x['unreal'])} · **Total** {pnl_txt(total, pct)}")
    if r[4].button('📈', key=f'btn_{fol}_{f}', help='Open'):
        st.session_state['open_file'] = (fol, f)
        st.rerun()
    with r[5].popover('🗑️', width='stretch'):
        st.write(s['name'])
        if st.button('✔ Yes', key=f'del_{fol}_{f}', type='primary'):
            delete_strategy(fol, f)
            drop_result(st.session_state.setdefault('bt', {}), fol, f)
            st.rerun()
    st.divider()


# =============================================================== SAVED LOGIN
# The last successful login is kept in  .saved_logins.json  next to this file (NOT inside backtests/, so it is never put in the backup zip).
# Optional encryption: add  LOGIN_KEY = "<Fernet key>"  to .streamlit/secrets.toml (Streamlit Cloud: App settings > Secrets) and
# `cryptography` to requirements.txt.  Make a key with:  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Without LOGIN_KEY the file is plain JSON (owner-only permissions).
LOGIN_FILE = os.path.join(HERE, '.saved_logins.json')


def _login_fernet():
    try:
        from cryptography.fernet import Fernet
        key = st.secrets.get('LOGIN_KEY')
        return Fernet(str(key).encode()) if key else None
    except Exception:
        return None


def load_saved_login():
    """-> {'api_key','client_id','mpin','totp'} of the last login, or None."""
    try:
        with open(LOGIN_FILE, 'r', encoding='utf-8') as fh:
            raw = json.load(fh)
        if raw.get('enc'):
            f = _login_fernet()
            data = json.loads(f.decrypt(raw['data'].encode()).decode()) if f else None
        else:
            data = raw.get('data')
        if data and all(data.get(k) for k in ('api_key', 'client_id', 'mpin', 'totp')):
            return data
    except Exception:
        pass
    return None


def save_login(creds):
    try:
        f = _login_fernet()
        blob = {'enc': True, 'data': f.encrypt(json.dumps(creds).encode()).decode()} if f else {'enc': False, 'data': creds}
        tmp = LOGIN_FILE + '.tmp'
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(blob, fh)
        os.replace(tmp, LOGIN_FILE)
    except Exception:
        pass                                                   # read-only disk etc. - login still works, just not remembered


def forget_saved_login():
    try:
        os.remove(LOGIN_FILE)
    except OSError:
        pass


# ---- DEMO ID: one click logs in with the owner's details, which are kept in secrets (never in this file, never shown).
# Streamlit Cloud: App settings > Secrets   (local: .streamlit/secrets.toml):
#   [DEMO_LOGIN]
#   api_key   = "..."
#   client_id = "..."
#   mpin      = "..."
#   totp      = "..."
#   label     = "Demo ID"        # optional - the text on the button
def demo_creds():
    """-> (creds or None, problem text or '')"""
    try:
        sec = st.secrets.get('DEMO_LOGIN')
    except Exception as e:
        return None, f'secrets could not be read ({e})'
    if not sec:
        return None, 'no [DEMO_LOGIN] section found in secrets (save it, then reboot the app)'
    try:
        d = {str(k).strip().lower(): v for k, v in dict(sec).items()}
    except Exception:
        return None, '[DEMO_LOGIN] is not a section - write the header line [DEMO_LOGIN] first, then the 4 lines under it'
    miss = [k for k in ('api_key', 'client_id', 'mpin', 'totp') if not str(d.get(k) or '').strip()]
    if miss:
        return None, 'DEMO_LOGIN is missing: ' + ', '.join(miss) + ' (names must be exactly api_key, client_id, mpin, totp)'
    return {'api_key': str(d['api_key']).strip(), 'client_id': str(d['client_id']).strip().upper(),
            'mpin': str(d['mpin']).strip(), 'totp': str(d['totp']).strip(), 'demo': True,
            'label': str(d.get('label') or 'Demo ID')}, ''


def demo_login_button(key):
    """shows the 'Demo ID' button; returns the error text if the demo login failed, else None."""
    d, problem = demo_creds()
    st.divider()
    if not d:
        st.caption(f'Demo ID not available: {problem}')       # tells the owner exactly what is wrong
        return None
    st.caption('No Angel One account? Try the app with the demo ID - no keys needed.')
    if st.button(f"🎯 {d['label']}  (click to login)", key=key, width='stretch'):
        err = try_login(d, remember=False)              # never saved to .saved_logins.json
        if err is None:
            st.session_state.pop('new_login', None)
            st.rerun()
        return err
    return None


def try_login(creds, remember):
    """log in with creds; returns None on success (and saves them if `remember`), else the error text."""
    st.session_state['creds'] = creds
    try:
        get_smart()
    except Exception as e:
        st.session_state.pop('creds', None)
        st.session_state.pop('smart', None)
        return str(e)
    if remember:
        save_login(creds)
    return None


# =============================================================== app
GLOBAL_CSS = """<style>
div[data-testid='stVerticalBlock']{gap:.6rem}
/* check-box drop-downs look exactly like the other inputs (same grey, same height) */
div[data-testid='stPopover'] button{background:rgba(151,166,195,.15);border:1px solid transparent;min-height:2.5rem;border-radius:.5rem}
div[data-testid='stPopover'] button:hover{border-color:rgba(255,75,75,.6)}
</style>"""


def main():
    st.set_page_config(page_title='Cloud Backtest', layout='wide', initial_sidebar_state='expanded')
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # ---- login screen - Angel One is used ONLY to download historical candles.
    # The last successful login is remembered (see SAVED LOGIN above): after logout you get  "Continue as <ID>"  or  "Login with new ID".
    if 'creds' not in st.session_state:
        saved = load_saved_login()
        err = None
        if saved and not st.session_state.get('new_login'):
            st.title('Cloud Backtest')
            st.subheader('Welcome back')
            st.write(f"Last login: **{saved['client_id']}**")
            c1, c2 = st.columns(2)
            if c1.button(f"Continue as {saved['client_id']}", type='primary', width='stretch'):
                err = try_login(saved, remember=False)
                if err is None:
                    st.rerun()
            if c2.button('Login with new ID', width='stretch'):
                st.session_state['new_login'] = True
                st.rerun()
            if err:
                st.error(f'Login failed: {err}')
            if st.button('Forget saved login', type='tertiary'):
                forget_saved_login()
                st.rerun()
            derr = demo_login_button('demo_login_saved')
            if derr:
                st.error(f'Demo login failed: {derr}')
            return
        st.title('Cloud Backtest')
        st.subheader('Enter your Angel One details')
        st.caption('Only used to download historical candles. No order is ever placed.')
        api_key = st.text_input('API key', type='password')
        client_id = st.text_input('Client ID')
        mpin = st.text_input('MPIN', type='password')
        totp = st.text_input('TOTP secret key', type='password',
                             help='The long text shown when you enable TOTP (not the 6-digit code)')
        remember = st.checkbox('Remember this login', value=True,
                               help='Saves these details on the machine running this app, so next time you only press "Continue".')
        c1, c2 = st.columns([1, 4])
        if c1.button('Login') and api_key and client_id and mpin and totp:
            err = try_login({'api_key': api_key.strip(), 'client_id': client_id.strip().upper(),
                             'mpin': mpin.strip(), 'totp': totp.strip()}, remember=remember)
            if err is None:
                st.session_state.pop('new_login', None)
                st.rerun()
        if saved and c2.button('← Back to saved login'):
            st.session_state.pop('new_login', None)
            st.rerun()
        if err:
            st.error(f'Login failed: {err}')
        derr = demo_login_button('demo_login_form')
        if derr:
            st.error(f'Demo login failed: {derr}')
        return

    page = st.session_state.setdefault('page', 'strategies')      # ---- navigation
    with st.sidebar:
        st.markdown('### Cloud Backtest')
        st.toggle('Light tables', value=True, key='light_tables',
                  help='ON = plain tables (works even when the browser shows "TypeError: Failed to fetch dynamically imported module"). '
                       'OFF = interactive tables (sortable, scrollable).')
        if st.button('📁 My Backtests', type='primary' if page == 'strategies' else 'secondary', width='stretch'):
            st.session_state['page'] = 'strategies'
            st.session_state.pop('open_file', None)
            st.rerun()
        if st.button('🛠️ Backtest Builder', type='primary' if page == 'builder' else 'secondary', width='stretch'):
            st.session_state['page'] = 'builder'
            st.rerun()

    if page == 'builder':
        page_builder()
    else:
        page_view()

    st.write('')
    with st.expander('⚙️ Account'):
        st.write(f"Client ID: {client_id()}" if not st.session_state['creds'].get('demo') else f"Logged in with the **{st.session_state['creds']['label']}** (your files are private to this browser session: {client_id()})")
        st.caption('On Streamlit Cloud the backtest files are wiped when the app reboots - download a backup and restore it later.')
        st.download_button('⬇️ Backup my backtests (zip)', backup_zip(), file_name=f"backtests_{client_id()}.zip", mime='application/zip')
        up_ = st.file_uploader('Restore from backup zip', type='zip', key='restore_zip')
        if up_ is not None and st.button('Restore files'):
            st.success(f'{restore_zip(up_.getvalue())} file(s) restored')
            st.rerun()
        if st.button('Logout / change details'):
            for k_ in ('creds', 'smart', 'bt', 'new_login'):
                st.session_state.pop(k_, None)
            st.rerun()


if __name__ == '__main__':
    main()
