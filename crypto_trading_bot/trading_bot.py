#!/usr/bin/env python3
# trading_bot.py - Мульти-аккаунтный торговый бот
# Версия 37.0 - Исправленная синхронизация + email уведомления

import os, sys, time, asyncio, json, requests, hmac, hashlib, numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import warnings, logging, smtplib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import aiohttp
import re
import pandas as pd
import traceback

warnings.filterwarnings('ignore')

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import talib

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN = True
except:
    IMBLEARN = False

# ==================== НАСТРОЙКИ ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/var/log/tradingbot-multi.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "trading_data")
ACCOUNTS_DIR = os.path.join(BASE_DIR, "accounts")
EXCEL_DIR = os.path.join(DATA_DIR, "exports")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ACCOUNTS_DIR, exist_ok=True)
os.makedirs(EXCEL_DIR, exist_ok=True)

load_dotenv(os.path.join(BASE_DIR, '.env'))

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
EMAIL_FROM = os.getenv('EMAIL_FROM', '')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
EMAIL_TO = "oxykbuff11@yandex.com"

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "AVAXUSDT", "DOTUSDT", "DOGEUSDT", "TRXUSDT", "LINKUSDT", "POLUSDT",
    "LTCUSDT", "ATOMUSDT", "UNIUSDT", "AAVEUSDT", "ALGOUSDT", "XLMUSDT",
    "VETUSDT", "FILUSDT", "ETCUSDT", "NEARUSDT", "APTUSDT", "ARUSDT",
    "OPUSDT", "ROSEUSDT", "BCHUSDT", "INJUSDT", "SEIUSDT", "SUIUSDT",
    "RUNEUSDT", "ZROUSDT", "1INCHUSDT", "CROUSDT", "HFTUSDT", "MNTUSDT"
]

# Параметры монет
MIN_ORDER_SIZES = {}
ORDER_STEPS = {}
PRICE_PRECISION = {}
TICK_SIZE = {}

for s in SYMBOLS:
    if s == "BTCUSDT":
        MIN_ORDER_SIZES[s] = 0.001
        ORDER_STEPS[s] = 0.001
        PRICE_PRECISION[s] = 1
        TICK_SIZE[s] = 0.01
    elif s in ["BNBUSDT", "ETHUSDT", "BCHUSDT"]:
        MIN_ORDER_SIZES[s] = 0.01
        ORDER_STEPS[s] = 0.01
        PRICE_PRECISION[s] = 2
        TICK_SIZE[s] = 0.01
    elif s in ["SOLUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "ATOMUSDT", "INJUSDT"]:
        MIN_ORDER_SIZES[s] = 0.1
        ORDER_STEPS[s] = 0.1
        PRICE_PRECISION[s] = 2
        TICK_SIZE[s] = 0.01
    elif s in ["POLUSDT", "ALGOUSDT", "NEARUSDT", "APTUSDT", "ZROUSDT", "SEIUSDT", "XLMUSDT"]:
        MIN_ORDER_SIZES[s] = 1.0
        ORDER_STEPS[s] = 1.0
        PRICE_PRECISION[s] = 4
        TICK_SIZE[s] = 0.0001
    elif s == "ADAUSDT":
        MIN_ORDER_SIZES[s] = 1.0
        ORDER_STEPS[s] = 1.0
        PRICE_PRECISION[s] = 5
        TICK_SIZE[s] = 0.000001
    elif s in ["XRPUSDT", "TRXUSDT", "DOGEUSDT", "SUIUSDT"]:
        MIN_ORDER_SIZES[s] = 10.0
        ORDER_STEPS[s] = 10.0
        PRICE_PRECISION[s] = 5
        TICK_SIZE[s] = 0.000001
    else:
        MIN_ORDER_SIZES[s] = 1.0
        ORDER_STEPS[s] = 1.0
        PRICE_PRECISION[s] = 4
        TICK_SIZE[s] = 0.0001

EXCHANGE_COMMISSION = 0.00055
POSITION_IDX = 0

REAL_TRADING_CRITERIA = {
    'min_trades_per_coin': 30,
    'min_winrate_per_coin': 55.0,
    'min_profit_factor': 1.2,
    'risk_percent_real': 0.3,
}

TRADING_MODE = {
    'cycle_sleep': 1800,
    'risk_percent_demo': 0.5,
    'risk_percent_real': 0.3,
    'max_trades_per_cycle': 10,
    'base_signal_threshold': 6,
    'ml_enabled': True,
    'ml_decision_threshold': 0.35,
    'ai_enabled': True,
    'ai_confidence_threshold': 0.65,
    'min_stop_pct': 0.02,
    'max_stop_pct': 0.08,
    'base_stop_multiplier': 1.5,
    'atr_period': 14,
    'volatility_lookback': 20,
    'trend_interval': '240',
    'swing_interval': '60',
    'entry_interval': '30',
    'price_tolerance_pct': 0.005,
    'ml_min_closed_trades': 15,
    'demo_trading_enabled': True,
    'real_trading_enabled': False,
}

COIN_STATS_FILE = os.path.join(DATA_DIR, "coin_stats.json")

def send_email(subject, body, attachment=None):
    """Отправка email уведомления"""
    if not EMAIL_FROM or not EMAIL_PASSWORD:
        return False
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        if attachment and os.path.exists(attachment):
            with open(attachment, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(attachment)}"')
                msg.attach(part)
        
        with smtplib.SMTP_SSL("smtp.yandex.ru", 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info(f"📧 Email отправлен: {subject}")
        return True
    except Exception as e:
        logger.error(f"Email error: {e}")
        return False


# ==================== КЛАСС ДЛЯ АНАЛИЗА МОНЕТ ====================
class CoinAnalyzer:
    def __init__(self):
        self.stats_file = COIN_STATS_FILE
        self.coin_stats = self.load_stats()
        self.processed_trades = set()

    def load_stats(self):
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {}

    def save_stats(self):
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.coin_stats, f, indent=2)
        except:
            pass

    def update_from_trade(self, trade):
        if trade.get('exit') is None:
            return
        
        trade_id = trade.get('id')
        if trade_id:
            if trade_id in self.processed_trades:
                return
            self.processed_trades.add(trade_id)
        
        symbol = trade['symbol']
        net = trade.get('net', 0)
        
        if symbol not in self.coin_stats:
            self.coin_stats[symbol] = {
                'total_trades': 0, 'wins': 0, 'losses': 0,
                'total_profit': 0.0, 'total_loss': 0.0,
                'winrate': 0.0, 'profit_factor': 0.0,
                'qualified': False, 'qualified_at': None, 'last_update': None
            }
        
        stats = self.coin_stats[symbol]
        stats['total_trades'] += 1
        
        if net > 0:
            stats['wins'] += 1
            stats['total_profit'] += net
        else:
            stats['losses'] += 1
            stats['total_loss'] += abs(net)
        
        if stats['total_trades'] > 0:
            stats['winrate'] = (stats['wins'] / stats['total_trades']) * 100
        
        if stats['total_loss'] > 0:
            stats['profit_factor'] = stats['total_profit'] / stats['total_loss']
        else:
            stats['profit_factor'] = stats['total_profit'] if stats['total_profit'] > 0 else 0
        
        was_qualified = stats.get('qualified', False)
        stats['qualified'] = (
            stats['total_trades'] >= REAL_TRADING_CRITERIA['min_trades_per_coin'] and
            stats['winrate'] >= REAL_TRADING_CRITERIA['min_winrate_per_coin'] and
            stats['profit_factor'] >= REAL_TRADING_CRITERIA['min_profit_factor']
        )
        
        if not was_qualified and stats['qualified']:
            stats['qualified_at'] = datetime.now().isoformat()
            logger.info(f"🎉 МОНЕТА {symbol} ПРОШЛА КВАЛИФИКАЦИЮ!")
            body = f"""🎉 НОВАЯ МОНЕТА ГОТОВА К РЕАЛЬНОЙ ТОРГОВЛЕ!

Монета: {symbol}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 СТАТИСТИКА:
Всего сделок: {stats['total_trades']}
Винрейт: {stats['winrate']:.1f}%
Profit Factor: {stats['profit_factor']:.2f}
✅ Критерии выполнены!"""
            send_email(f"✅ {symbol} ГОТОВА К РЕАЛЬНОЙ ТОРГОВЛЕ", body)
        
        stats['last_update'] = datetime.now().isoformat()
        self.save_stats()

    def get_qualified_coins(self):
        return [s for s, st in self.coin_stats.items() if st.get('qualified', False)]

    def is_qualified(self, symbol):
        return self.coin_stats.get(symbol, {}).get('qualified', False)

    def generate_excel_report(self, all_trades):
        if not all_trades:
            return None
        
        trades_data = []
        for t in all_trades:
            trades_data.append({
                'time': t.get('time', '')[:19], 'account': t.get('account', ''),
                'symbol': t.get('symbol', ''), 'side': t.get('side', ''),
                'entry': t.get('entry', 0), 'exit': t.get('exit', 0) if t.get('exit') else '',
                'size': t.get('size', 0), 'leverage': t.get('lev', 1),
                'pnl': t.get('net', 0) if t.get('exit') else '',
                'close_reason': t.get('reason', 'открыта'),
                'ml_prob': t.get('ml_prob', 0), 'ai_score': t.get('ai_score', 0),
                'ai_reason': t.get('ai_reason', ''),
                'trend': t.get('trend', ''), 'rsi': t.get('rsi', 0),
                'adx': t.get('adx', 0), 'williams': t.get('williams', 0),
                'pattern_type': t.get('pattern_type', ''), 'market_phase': t.get('market_phase', '')
            })
        
        df_trades = pd.DataFrame(trades_data)
        filename = os.path.join(EXCEL_DIR, f"trades_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df_trades.to_excel(writer, sheet_name='Сделки', index=False)
        
        return filename


# ==================== ТЕХНИЧЕСКИЙ АНАЛИЗ ====================
class TechnicalAnalyzer:
    @staticmethod
    def rsi(prices, p=14):
        if len(prices) < p+1: return 50
        r = talib.RSI(np.array(prices), timeperiod=p)
        return float(r[-1]) if not np.isnan(r[-1]) else 50

    @staticmethod
    def ema(prices, p=20):
        if len(prices) < p: return prices[-1]
        e = talib.EMA(np.array(prices), timeperiod=p)
        return float(e[-1]) if not np.isnan(e[-1]) else prices[-1]

    @staticmethod
    def macd(prices):
        if len(prices) < 26: return 0,0,0
        m,s,h = talib.MACD(np.array(prices))
        return float(m[-1]), float(s[-1]), float(h[-1])

    @staticmethod
    def atr(highs, lows, closes, p=14):
        if len(closes) < p: return 0
        a = talib.ATR(np.array(highs), np.array(lows), np.array(closes), timeperiod=p)
        return float(a[-1]) if not np.isnan(a[-1]) else 0

    @staticmethod
    def adx(highs, lows, closes, p=14):
        if len(closes) < p+1:
            return 25
        adx = talib.ADX(np.array(highs), np.array(lows), np.array(closes), timeperiod=p)
        return float(adx[-1]) if not np.isnan(adx[-1]) else 25

    @staticmethod
    def williams_r(highs, lows, closes, p=14):
        if len(closes) < p+1:
            return -50
        wr = talib.WILLR(np.array(highs), np.array(lows), np.array(closes), timeperiod=p)
        return float(wr[-1]) if not np.isnan(wr[-1]) else -50

    @staticmethod
    def donchian_channel(highs, lows, period=20):
        if len(highs) < period:
            return None, None, None
        upper = max(highs[-period:])
        lower = min(lows[-period:])
        middle = (upper + lower) / 2
        return upper, middle, lower

    @staticmethod
    def find_patterns(highs, lows, closes):
        patterns = {'double_top': False, 'double_bottom': False, 'signal': 0}
        if len(highs) < 30:
            return patterns
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]
        peaks = []
        for i in range(5, len(recent_highs)-5):
            if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i+1]:
                peaks.append((i, recent_highs[i]))
        if len(peaks) >= 2:
            if abs(peaks[-1][1] - peaks[-2][1]) / peaks[-1][1] < 0.02:
                patterns['double_top'] = True
                patterns['signal'] = -1
        troughs = []
        for i in range(5, len(recent_lows)-5):
            if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i+1]:
                troughs.append((i, recent_lows[i]))
        if len(troughs) >= 2:
            if abs(troughs[-1][1] - troughs[-2][1]) / troughs[-1][1] < 0.02:
                patterns['double_bottom'] = True
                patterns['signal'] = 1
        return patterns

    @staticmethod
    def dynamic_stop(closes, highs, lows):
        price = closes[-1]
        atr_val = TechnicalAnalyzer.atr(highs, lows, closes, 14)
        atr_stop = (atr_val / price) * 1.2
        recent = closes[-20:]
        vol = np.std(recent) / np.mean(recent) if np.mean(recent) > 0 else 0.02
        adj = 1 + (vol - 0.02) * 0.6
        adj = max(0.5, min(1.5, adj))
        stop = atr_stop * adj
        return max(0.02, min(0.08, stop))


# ==================== BYBIT API КЛИЕНТ ====================
class BybitClient:
    def __init__(self, api_key, secret_key, demo=False, name=None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.demo = demo
        self.name = name
        self.base = "https://api-demo.bybit.com" if demo else "https://api.bybit.com"
        self.session = requests.Session()

    def _sign(self, params=None, post=False):
        ts = str(int(time.time() * 1000))
        recv_window = "5000"
        if post and params:
            body = json.dumps(params)
            param_str = ts + self.api_key + recv_window + body
        elif params:
            query = "&".join([f"{k}={v}" for k, v in params.items()])
            param_str = ts + self.api_key + recv_window + query
        else:
            param_str = ts + self.api_key + recv_window
        signature = hmac.new(bytes(self.secret_key, "utf-8"), bytes(param_str, "utf-8"), hashlib.sha256).hexdigest()
        return ts, recv_window, signature

    def _req(self, method, endpoint, params=None):
        try:
            url = self.base + endpoint
            ts, rw, sig = self._sign(params, method == "POST")
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": ts,
                "X-BAPI-RECV-WINDOW": rw,
                "X-BAPI-SIGN": sig,
                "Content-Type": "application/json"
            }
            if method == "GET":
                r = self.session.get(url, params=params, headers=headers, timeout=30)
            else:
                r = self.session.post(url, json=params, headers=headers, timeout=30)
            return r.json()
        except Exception as e:
            return {"retCode": 500, "retMsg": str(e)}

    def check(self):
        r = self._req("GET", "/v5/market/time")
        return r.get('retCode') == 0

    def balance(self):
        try:
            r = self._req("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"})
            if r.get('retCode') == 0:
                for account in r.get('result', {}).get('list', []):
                    for coin in account.get('coin', []):
                        if coin.get('coin') == 'USDT':
                            val = coin.get('availableToWithdraw')
                            if val is None or val == '':
                                val = coin.get('walletBalance', '0')
                            if val and val != '':
                                return float(val)
            return 0.0
        except:
            return 0.0

    def positions(self, symbol=None):
        p = {"category": "linear", "settleCoin": "USDT", "limit": 200}
        if symbol:
            p["symbol"] = symbol
        r = self._req("GET", "/v5/position/list", p)
        res = []
        for pos in r.get('result', {}).get('list', []):
            sz = float(pos.get('size', 0))
            if sz > 0:
                res.append({
                    'symbol': pos.get('symbol'),
                    'side': pos.get('side'),
                    'size': sz,
                    'avgPrice': float(pos.get('avgPrice', 0)),
                    'leverage': float(pos.get('leverage', 1)),
                })
        return res

    def price(self, symbol):
        r = self._req("GET", "/v5/market/tickers", {"category": "linear", "symbol": symbol})
        if r.get('retCode') == 0:
            return float(r.get('result', {}).get('list', [{}])[0].get('lastPrice', 0))
        return 0.0

    def klines(self, symbol, interval="60", limit=150):
        r = self._req("GET", "/v5/market/kline", {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit})
        data = []
        for k in reversed(r.get('result', {}).get('list', [])):
            data.append({
                'timestamp': int(k[0]),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
            })
        return data

    def format_qty(self, symbol, qty):
        mn = MIN_ORDER_SIZES.get(symbol, 0.001)
        st = ORDER_STEPS.get(symbol, 0.001)
        qty = max(mn, qty)
        qty = round(qty / st) * st
        if st >= 1: return str(int(qty))
        if st >= 0.1: return f"{qty:.1f}"
        if st >= 0.01: return f"{qty:.2f}"
        return f"{qty:.6f}"

    def round_price(self, symbol, price):
        tick = TICK_SIZE.get(symbol, 0.0001)
        return round(price / tick) * tick

    def order(self, symbol, side, qty, sl=None, tp=None, lev=2):
        qty_str = self.format_qty(symbol, qty)
        p = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": qty_str,
            "timeInForce": "GTC",
            "positionIdx": POSITION_IDX,
            "marketUnit": "quoteCoin"
        }
        if sl:
            p["stopLoss"] = f"{self.round_price(symbol, sl):.{PRICE_PRECISION.get(symbol, 2)}f}"
        if tp:
            p["takeProfit"] = f"{self.round_price(symbol, tp):.{PRICE_PRECISION.get(symbol, 2)}f}"
        r = self._req("POST", "/v5/order/create", p)
        if r.get('retCode') == 0:
            order_id = r.get('result', {}).get('orderId')
            return {"ok": True, "id": order_id}
        return {"ok": False, "err": r.get('retMsg')}

    def get_closed_trade(self, symbol, entry_price=None, size=None, hours_back=48):
        """Получение данных о закрытой сделке - проверенная версия"""
        end_time = int(time.time() * 1000)
        start_time = end_time - (hours_back * 60 * 60 * 1000)
        params = {"category": "linear", "startTime": start_time, "endTime": end_time, "limit": 100}
        if symbol:
            params["symbol"] = symbol
        r = self._req("GET", "/v5/position/closed-pnl", params)
        if r.get('retCode') != 0:
            return None
        
        trades = r.get('result', {}).get('list', [])
        if not trades:
            return None
        
        # Если есть цена входа - ищем по ней
        if entry_price:
            for trade in trades:
                if trade.get('symbol') == symbol:
                    trade_entry = float(trade.get('avgEntryPrice', 0))
                    if trade_entry > 0 and abs(trade_entry - entry_price) / entry_price < 0.03:
                        return {
                            'exit': float(trade.get('avgExitPrice', 0)),
                            'pnl': float(trade.get('closedPnl', 0)),
                            'entry': trade_entry,
                            'time': trade.get('updatedTime', 0)
                        }
        
        # Если не нашли по цене, берем последнюю сделку по символу
        for trade in reversed(trades):
            if trade.get('symbol') == symbol:
                return {
                    'exit': float(trade.get('avgExitPrice', 0)),
                    'pnl': float(trade.get('closedPnl', 0)),
                    'entry': float(trade.get('avgEntryPrice', 0)),
                    'time': trade.get('updatedTime', 0)
                }
        
        return None

    def get_all_closed_trades(self, hours_back=168):
        end_time = int(time.time() * 1000)
        start_time = end_time - (hours_back * 60 * 60 * 1000)
        params = {"category": "linear", "startTime": start_time, "endTime": end_time, "limit": 100}
        r = self._req("GET", "/v5/position/closed-pnl", params)
        if r.get('retCode') != 0:
            return []
        trades = []
        for trade in r.get('result', {}).get('list', []):
            trades.append({
                'symbol': trade.get('symbol'),
                'exit': float(trade.get('avgExitPrice', 0)),
                'pnl': float(trade.get('closedPnl', 0)),
                'entry': float(trade.get('avgEntryPrice', 0)),
                'size': abs(float(trade.get('size', 0))),
                'time': trade.get('updatedTime', 0)
            })
        return trades

    def close(self, symbol):
        pos = self.positions(symbol)
        if not pos:
            return {"ok": False, "err": "No position"}
        p = pos[0]
        side = "Buy" if p['side'] == "Sell" else "Sell"
        qty_str = self.format_qty(symbol, p['size'])
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": qty_str,
            "reduceOnly": True,
            "positionIdx": POSITION_IDX
        }
        r = self._req("POST", "/v5/order/create", params)
        return {"ok": r.get('retCode') == 0, "err": r.get('retMsg')}

    def leverage(self, symbol, lev):
        r = self._req("POST", "/v5/position/set-leverage", {
            "category": "linear", "symbol": symbol,
            "buyLeverage": str(lev), "sellLeverage": str(lev), "positionIdx": POSITION_IDX
        })
        return r.get('retCode') == 0

    def get_order_fill(self, order_id):
        r = self._req("GET", "/v5/order/realtime", {"category": "linear", "orderId": order_id})
        if r.get('retCode') == 0:
            orders = r.get('result', {}).get('list', [])
            if orders:
                return {
                    'avg_price': float(orders[0].get('avgPrice', 0)),
                    'executed_qty': float(orders[0].get('cumExecQty', 0))
                }
        return None


# ==================== УПРАВЛЕНИЕ АККАУНТАМИ ====================
@dataclass
class Account:
    name: str
    acc_type: str
    api_key: str = ''
    secret: str = ''
    active: bool = True
    balance: float = 0.0
    positions: Dict = field(default_factory=dict)
    history: List = field(default_factory=list)

    def display(self):
        return {'main': '👑 MAIN', 'dasha': '👩 DASHA', 'demo': '🧪 DEMO'}.get(self.acc_type, self.acc_type.upper())

    def is_demo(self):
        return self.acc_type == 'demo'


class AccountManager:
    def __init__(self):
        self.accounts: Dict[str, Account] = {}
        self.clients: Dict[str, BybitClient] = {}
        self.current = None

    def load(self):
        if not os.path.exists(ACCOUNTS_DIR):
            os.makedirs(ACCOUNTS_DIR)
            return False
        for f in os.listdir(ACCOUNTS_DIR):
            if not f.endswith('.env'): continue
            cfg = {}
            with open(os.path.join(ACCOUNTS_DIR, f), 'r') as fp:
                for line in fp:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        cfg[k.strip()] = v.strip()
            name = f.replace('.env', '')
            t = cfg.get('ACCOUNT_TYPE', 'demo').lower()
            key = cfg.get('BYBIT_API_KEY', '')
            sec = cfg.get('BYBIT_SECRET_KEY', '')
            if key and sec:
                self.accounts[name] = Account(name, t, key, sec)
        return len(self.accounts) > 0

    def init_clients(self):
        for n, a in self.accounts.items():
            c = BybitClient(a.api_key, a.secret, a.is_demo(), n)
            if c.check():
                self.clients[n] = c
                a.balance = c.balance()
                logger.info(f"✅ {n} баланс: ${a.balance:.2f}")
            else:
                a.active = False
                logger.warning(f"❌ {n} неактивен (ошибка API)")

    def get_active(self):
        return [n for n, a in self.accounts.items() if a.active]

    def get_client(self):
        if self.current and self.current in self.clients:
            return self.clients[self.current]
        return None

    def get_account(self):
        if self.current and self.current in self.accounts:
            return self.accounts[self.current]
        return None

    def switch(self, name):
        if name in self.accounts and name in self.clients:
            self.current = name
            return True
        return False

    def update_balances(self):
        for n, c in self.clients.items():
            self.accounts[n].balance = c.balance()


# ==================== ML СИСТЕМА ====================
class MLSystem:
    def __init__(self):
        self.classifier = None
        self.scaler = RobustScaler()
        self.trained = False
        self.metrics = {'f1': 0, 'threshold': TRADING_MODE['ml_decision_threshold']}

    def extract_features(self, market, direction='BUY'):
        try:
            closes = market['closes']
            highs = market['highs']
            lows = market['lows']
            price = closes[-1]

            features = []
            atr = TechnicalAnalyzer.atr(highs, lows, closes, 14)
            features.append(atr / price if price > 0 else 0)
            rsi = TechnicalAnalyzer.rsi(closes)
            features.append(rsi / 100)
            _, _, hist = TechnicalAnalyzer.macd(closes)
            features.append(hist / price if price > 0 else 0)
            adx = TechnicalAnalyzer.adx(highs, lows, closes, 14)
            features.append(adx / 100)
            williams = TechnicalAnalyzer.williams_r(highs, lows, closes, 14)
            features.append(williams / -100)
            features.append(1 if direction == 'BUY' else 0)
            return np.array(features, dtype=np.float32)
        except:
            return None

    def train(self, history, market_data):
        X, y = [], []
        for trade in history:
            if trade.get('exit') is None: continue
            symbol = trade['symbol']
            if symbol not in market_data: continue
            features = self.extract_features(market_data[symbol], trade['side'])
            if features is None: continue
            X.append(features)
            y.append(1 if trade.get('net', 0) > 0 else 0)

        if len(X) < TRADING_MODE['ml_min_closed_trades']:
            return False

        X = np.array(X)
        y = np.array(y)
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        if IMBLEARN:
            try:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            except: pass

        self.classifier = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced')
        self.classifier.fit(X_train, y_train)

        y_pred = self.classifier.predict(X_test)
        self.metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)

        self.trained = True
        logger.info(f"ML обучен: F1={self.metrics['f1']:.3f}")
        return True

    def predict(self, market, direction='BUY'):
        if not self.trained or self.classifier is None:
            return 0.5
        features = self.extract_features(market, direction)
        if features is None:
            return 0.5
        features_scaled = self.scaler.transform([features])
        proba = self.classifier.predict_proba(features_scaled)[0]
        return proba[1] if len(proba) == 2 else proba[0]


# ==================== AI СИСТЕМА ====================
class AISystem:
    def __init__(self):
        self.enabled = TRADING_MODE['ai_enabled'] and bool(DEEPSEEK_API_KEY)
        self.cache = {}
        self.cache_file = os.path.join(DATA_DIR, "ai_cache.json")
        self.load_cache()

    def load_cache(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
        except:
            pass

    def save_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except:
            pass

    def parse_ai_response(self, response_text):
        try:
            logger.info(f"📝 AI RAW RESPONSE: {response_text[:500]}")
            cleaned_text = re.sub(r'```json\s*', '', response_text)
            cleaned_text = re.sub(r'```\s*', '', cleaned_text)
            json_match = re.search(r'\{[^{}]*"score"\s*:\s*[\d.]+[^{}]*\}', cleaned_text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                score = result.get('score') or result.get('confidence_score')
                if score is not None:
                    confidence_score = max(0.0, min(1.0, float(score)))
                    reason = result.get('reason', '')
                    logger.info(f"✅ AI уверенность: {confidence_score:.2f}")
                    return confidence_score, reason
                return 0.5, "Нет оценки"
            number_match = re.search(r'(\d+\.?\d*)', response_text)
            if number_match:
                num_value = float(number_match.group(1))
                if num_value > 1:
                    confidence_score = num_value / 100.0
                else:
                    confidence_score = num_value
                return max(0.0, min(1.0, confidence_score)), "Извлечено из текста"
            return 0.5, "Не удалось распарсить"
        except Exception as e:
            logger.error(f"Ошибка парсинга AI: {e}")
            return 0.5, f"Ошибка: {e}"

    async def confirm(self, signal, account_balance):
        if not self.enabled:
            return True, 0.5, "AI отключён"

        key = f"{signal['symbol']}_{signal['action']}_{signal['price']:.4f}"
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - datetime.fromisoformat(entry['time']) < timedelta(hours=6):
                return entry['confirm'], entry['score'], entry.get('reason', '')

        prompt = f"""Оцени торговый сигнал для криптовалюты от 0 до 1 как профессиональный трейдер.

Монета: {signal['symbol']}
Действие: {signal['action']}
Цена входа: ${signal['price']:.6f}
Стоп-лосс: ${signal['stop']:.6f} ({signal['stop_pct']*100:.1f}%)
Тейк-профит: ${signal['tp']:.6f} ({signal['tp_pct']*100:.1f}%)
Соотношение риск/прибыль: {signal['rr']:.1f}:1
Уверенность сигнала: {signal['confidence']}/10
Риск на сделку: {signal['risk_percent']:.2f}% от баланса
Баланс аккаунта: ${account_balance:.2f}

Технические индикаторы:
- Тренд (12H): {signal.get('trend', 'NEUTRAL')}
- Свинг (4H): {signal.get('swing', 'NEUTRAL')}
- RSI (1H): {signal.get('rsi', 50):.1f}
- ADX: {signal.get('adx', 25):.0f}
- Williams %R: {signal.get('williams', -50):.0f}
- EMA50: ${signal.get('ema50', 0):.4f}
- EMA200: ${signal.get('ema200', 0):.4f}
- Фаза рынка: {signal.get('market_phase', 'NEUTRAL')}
- Тип входа: {signal.get('pattern_type', 'Стандартный')}

Ответь ТОЛЬКО JSON формате:
{{"score": 0.0-1.0, "reason": "развернутый анализ"}}"""

        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 500
                    }
                    async with session.post(
                        "https://api.deepseek.com/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=30
                    ) as resp:
                        data = await resp.json()
                        text = data['choices'][0]['message']['content']
                        score, reason = self.parse_ai_response(text)
                        confirm = score >= TRADING_MODE['ai_confidence_threshold']
                        self.cache[key] = {
                            'confirm': confirm,
                            'score': score,
                            'reason': reason,
                            'time': datetime.now().isoformat()
                        }
                        self.save_cache()
                        logger.info(f"🤖 AI: {signal['symbol']} {signal['action']} -> {score:.2f}")
                        return confirm, score, reason
            except asyncio.TimeoutError:
                logger.warning(f"AI таймаут, попытка {attempt+1}/3")
                await asyncio.sleep(1)
            except Exception as e:
                logger.warning(f"AI ошибка: {e}, попытка {attempt+1}/3")
                await asyncio.sleep(1)
        
        return True, 0.6, "AI временно недоступен"


# ==================== ТОРГОВАЯ СТРАТЕГИЯ ====================
class Strategy:
    def __init__(self, client, bot, name):
        self.client = client
        self.bot = bot
        self.name = name

    def get_market(self, symbol):
        data = self.client.klines(symbol, '30', 150)
        if len(data) < 80:
            return None
        return {
            'symbol': symbol,
            'closes': [d['close'] for d in data],
            'highs': [d['high'] for d in data],
            'lows': [d['low'] for d in data],
            'volumes': [d['volume'] for d in data],
            'price': data[-1]['close'],
        }

    def check_position(self, symbol):
        return any(p['symbol'] == symbol for p in self.client.positions())

    def trend(self, symbol):
        data = self.client.klines(symbol, '240', 100)
        if len(data) < 20:
            return 'NEUTRAL'
        closes = [d['close'] for d in data]
        ema20 = TechnicalAnalyzer.ema(closes, 20)
        ema50 = TechnicalAnalyzer.ema(closes, 50)
        p = closes[-1]
        if p > ema20 > ema50: return 'BULLISH'
        if p < ema20 < ema50: return 'BEARISH'
        return 'NEUTRAL'

    def swing(self, symbol):
        data = self.client.klines(symbol, '60', 50)
        if len(data) < 15:
            return 'NEUTRAL', 0
        closes = [d['close'] for d in data]
        rsi = TechnicalAnalyzer.rsi(closes)
        _, _, hist = TechnicalAnalyzer.macd(closes)
        if hist > 0 and rsi > 50: return 'BULLISH', rsi
        if hist < 0 and rsi < 50: return 'BEARISH', rsi
        return 'NEUTRAL', rsi

    def find_entry_signal(self, symbol, acc, coin_analyzer):
        if self.check_position(symbol):
            return None
        
        for t in acc.history:
            if t.get('symbol') == symbol and t.get('exit') is None:
                return None

        data_1h = self.client.klines(symbol, '30', 100)
        if len(data_1h) < 50:
            return None

        closes = [d['close'] for d in data_1h]
        highs = [d['high'] for d in data_1h]
        lows = [d['low'] for d in data_1h]
        volumes = [d['volume'] for d in data_1h]
        price = self.client.price(symbol)
        
        if price <= 0:
            return None

        trend = self.trend(symbol)
        swing, rsi4 = self.swing(symbol)
        rsi = TechnicalAnalyzer.rsi(closes)
        macd, sig, hist = TechnicalAnalyzer.macd(closes)
        adx = TechnicalAnalyzer.adx(highs, lows, closes, 14)
        williams = TechnicalAnalyzer.williams_r(highs, lows, closes, 14)
        ema50 = TechnicalAnalyzer.ema(closes, 50)
        ema200 = TechnicalAnalyzer.ema(closes, 200)
        
        donchian_high, donchian_mid, donchian_low = TechnicalAnalyzer.donchian_channel(highs, lows, 20)
        
        patterns = TechnicalAnalyzer.find_patterns(highs, lows, closes)
        
        market_phase = 'NEUTRAL'
        if ema50 > ema200 and adx > 25:
            market_phase = 'TREND_UP'
        elif ema50 < ema200 and adx > 25:
            market_phase = 'TREND_DOWN'
        elif adx < 20:
            market_phase = 'CONSOLIDATION'

        action = None
        conf = 0
        entry_type = ""

        if donchian_high and price > donchian_high and market_phase in ['TREND_UP', 'CONSOLIDATION']:
            if volumes[-1] > np.mean(volumes[-10:]) * 1.2:
                action = 'BUY'
                conf = 8
                entry_type = "Пробитие H-20"

        elif market_phase in ['TREND_UP', 'CONSOLIDATION'] and abs(price - ema50) / price < 0.02:
            if williams < -80 or rsi < 35:
                action = 'BUY'
                conf = 7
                entry_type = "Откат к MA-50"

        if donchian_low and price < donchian_low and market_phase in ['TREND_DOWN', 'CONSOLIDATION']:
            if volumes[-1] > np.mean(volumes[-10:]) * 1.2:
                action = 'SELL'
                conf = 8
                entry_type = "Пробитие H-20 (шорт)"

        elif market_phase in ['TREND_DOWN', 'CONSOLIDATION'] and abs(price - ema50) / price < 0.02:
            if williams > -20 or rsi > 65:
                action = 'SELL'
                conf = 7
                entry_type = "Откат к MA-50 (шорт)"

        if not action:
            if patterns.get('double_bottom'):
                action = 'BUY'
                conf = 7
                entry_type = "Фигура 'Двойное дно'"
            elif patterns.get('double_top'):
                action = 'SELL'
                conf = 7
                entry_type = "Фигура 'Двойная вершина'"

        if not action:
            return None

        stop_pct = TechnicalAnalyzer.dynamic_stop(closes, highs, lows)
        if action == 'BUY':
            stop = price * (1 - stop_pct)
            tp = price * (1 + stop_pct * 2)
        else:
            stop = price * (1 + stop_pct)
            tp = price * (1 - stop_pct * 2)
        
        rr = 3.0

        return {
            'symbol': symbol, 'action': action, 'price': price,
            'stop': stop, 'tp': tp, 'confidence': conf,
            'rr': rr, 'stop_pct': stop_pct, 'tp_pct': stop_pct * 2,
            'trend': trend, 'swing': swing, 'rsi': rsi,
            'adx': adx, 'williams': williams,
            'ema50': ema50, 'ema200': ema200,
            'market_phase': market_phase,
            'pattern_type': entry_type,
            'is_qualified': coin_analyzer.is_qualified(symbol)
        }

    def position_size(self, symbol, price, stop_pct, action, acc, is_real_trade=False):
        bal = acc.balance
        if bal <= 0:
            return MIN_ORDER_SIZES.get(symbol, 0.1), 0, 0, 1
        
        if is_real_trade:
            risk_percent = REAL_TRADING_CRITERIA['risk_percent_real']
        elif acc.is_demo():
            risk_percent = TRADING_MODE['risk_percent_demo']
        else:
            risk_percent = 0.3
        
        risk_amount_usdt = bal * (risk_percent / 100)
        position_with_leverage = risk_amount_usdt / stop_pct
        
        target_margin_percent = 35.0
        leverage = target_margin_percent / (stop_pct * 100)
        leverage = max(2, min(12, round(leverage)))
        
        margin = position_with_leverage / leverage
        size = position_with_leverage / price
        
        mn = MIN_ORDER_SIZES.get(symbol, 0.1)
        size = max(mn, size)
        st = ORDER_STEPS.get(symbol, 0.1)
        size = round(size / st) * st
        
        actual_position_with_leverage = size * price
        actual_leverage = actual_position_with_leverage / margin if margin > 0 else leverage
        actual_risk_amount = actual_position_with_leverage * stop_pct
        actual_risk_percent = (actual_risk_amount / bal) * 100
        
        return size, actual_risk_percent, actual_risk_amount, actual_leverage


# ==================== ОСНОВНОЙ БОТ ====================
class TradingBot:
    def __init__(self):
        self.acc_mgr = AccountManager()
        self.strategies = {}
        self.ml = MLSystem()
        self.ai = AISystem()
        self.coin_analyzer = CoinAnalyzer()
        self.shutdown = False
        self.init()

    def init(self):
        logger.info("🚀 Инициализация...")
        if not self.acc_mgr.load():
            logger.error("❌ Нет аккаунтов")
            return
        self.acc_mgr.init_clients()
        if not self.acc_mgr.clients:
            logger.error("❌ Нет активных клиентов")
            return
        
        self.sync_all_positions()
        
        for n, c in self.acc_mgr.clients.items():
            self.strategies[n] = Strategy(c, self, n)
        
        self.load_history()
        
        logger.info("✅ Бот готов")
        logger.info(f"📊 Стратегия: Тренд + Фигуры + Канал Дончана")
        logger.info(f"🎯 Риск для демо: {TRADING_MODE['risk_percent_demo']}% от баланса")
        logger.info(f"🎯 Риск для реальных: {REAL_TRADING_CRITERIA['risk_percent_real']}% от баланса")
        self.print_status()

    def sync_all_positions(self):
        """Синхронизация позиций с биржей - исправленная версия"""
        for n, a in self.acc_mgr.accounts.items():
            if not a.active:
                continue

            logger.info(f"🔄 Синхронизация для {n}...")
            c = self.acc_mgr.clients[n]

            exchange_positions = c.positions()
            exchange_symbols = {p['symbol'] for p in exchange_positions}
            
            a.positions.clear()
            for pos in exchange_positions:
                sym = pos['symbol']
                a.positions[sym] = {
                    'side': pos['side'],
                    'entry': float(pos['avgPrice']),
                    'size': float(pos['size']),
                    'lev': float(pos['leverage']),
                    'sl': None,
                    'tp': None,
                    'order_id': None
                }
            
            updated = 0
            for trade in a.history:
                if trade.get('exit') is not None:
                    continue
                
                sym = trade.get('symbol')
                if sym in exchange_symbols:
                    continue
                
                entry_price = trade.get('entry')
                if entry_price:
                    closed = c.get_closed_trade(sym, entry_price=float(entry_price), hours_back=168)
                    
                    if closed:
                        trade['exit'] = float(closed['exit'])
                        try:
                            time_val = int(closed['time']) if isinstance(closed['time'], str) else closed['time']
                            trade['exit_time'] = datetime.fromtimestamp(time_val / 1000).isoformat()
                        except:
                            trade['exit_time'] = datetime.now().isoformat()
                        trade['reason'] = "TAKE_PROFIT" if closed['pnl'] > 0 else "STOP_LOSS"
                        trade['net'] = float(closed['pnl'])
                        
                        entry = float(trade['entry'])
                        size = float(trade['size'])
                        side = trade['side']
                        exit_price = float(closed['exit'])
                        
                        if side == "BUY":
                            gross = size * (exit_price - entry)
                        else:
                            gross = size * (entry - exit_price)
                        comm = size * entry * EXCHANGE_COMMISSION + size * exit_price * EXCHANGE_COMMISSION
                        trade['gross'] = gross
                        trade['commission'] = comm
                        
                        logger.info(f"   ✅ {sym}: закрыта | P&L: ${closed['pnl']:+.2f}")
                        updated += 1
            
            if updated > 0:
                logger.info(f"✅ {n}: синхронизировано {updated} закрытых сделок")
                self.save_history(n)
            
            logger.info(f"📊 {n}: открыто {len(a.positions)} позиций")

    def load_history(self):
        self.coin_analyzer.processed_trades = set()
        
        for n, a in self.acc_mgr.accounts.items():
            f = os.path.join(DATA_DIR, f"history_{n}.json")
            if os.path.exists(f):
                with open(f, 'r') as fp:
                    a.history = json.load(fp)
                logger.info(f"📂 История {n}: {len(a.history)} записей")
                
                closed_count = 0
                for trade in a.history:
                    if trade.get('exit'):
                        self.coin_analyzer.update_from_trade(trade)
                        closed_count += 1
                logger.info(f"   Обновлено статистики: {closed_count} закрытых сделок")

    def save_history(self, n):
        a = self.acc_mgr.accounts[n]
        f = os.path.join(DATA_DIR, f"history_{n}.json")
        with open(f, 'w') as fp:
            json.dump(a.history, fp, indent=2, default=str)

    def print_status(self):
        logger.info("=" * 60)
        logger.info("📊 СТАТУС ОБУЧЕНИЯ ПО МОНЕТАМ:")
        
        qualified = self.coin_analyzer.get_qualified_coins()
        logger.info(f"   Квалифицировано монет: {len(qualified)}/{len(SYMBOLS)}")
        
        for symbol, stats in sorted(self.coin_analyzer.coin_stats.items(), 
                                    key=lambda x: x[1]['total_trades'], reverse=True)[:15]:
            status = "✅" if stats.get('qualified') else "⏳"
            needed = max(0, REAL_TRADING_CRITERIA['min_trades_per_coin'] - stats['total_trades'])
            logger.info(f"   {status} {symbol:10} | Сделок: {stats['total_trades']:3} | Винрейт: {stats['winrate']:5.1f}% | Нужно еще: {needed}")
        
        logger.info("=" * 60)

    async def daily_report(self):
        while not self.shutdown:
            now = datetime.now()
            target = now.replace(hour=12, minute=0, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            await asyncio.sleep((target - now).total_seconds())
            
            all_trades = []
            for a in self.acc_mgr.accounts.values():
                all_trades.extend(a.history)
            
            excel_file = self.coin_analyzer.generate_excel_report(all_trades)
            
            lines = ["=" * 60]
            lines.append(f"ОТЧЁТ ПО ТОРГОВЛЕ - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            lines.append("=" * 60)
            lines.append("")
            
            for n, a in self.acc_mgr.accounts.items():
                if not a.active:
                    continue
                closed = [t for t in a.history if t.get('exit')]
                wins = sum(1 for t in closed if t.get('net', 0) > 0)
                profit = sum(t.get('net', 0) for t in closed)
                lines.append(f"📊 {a.display()} ({n}): ${a.balance:.2f}")
                lines.append(f"   Сделок: {len(closed)}, Винрейт: {(wins/len(closed)*100 if closed else 0):.1f}%, Прибыль: ${profit:+.2f}")
                lines.append("")
            
            lines.append(f"💰 Суммарный баланс: ${sum(a.balance for a in self.acc_mgr.accounts.values() if a.active):.2f}")
            lines.append("")
            
            qualified = self.coin_analyzer.get_qualified_coins()
            lines.append(f"🎯 МОНЕТЫ, ГОТОВЫЕ К РЕАЛЬНОЙ ТОРГОВЛЕ: {len(qualified)}")
            for sym in qualified[:10]:
                stats = self.coin_analyzer.coin_stats[sym]
                lines.append(f"   ✅ {sym}: {stats['total_trades']} сделок, винрейт {stats['winrate']:.1f}%")
            
            body = "\n".join(lines)
            send_email(f"Торговый отчёт {datetime.now().strftime('%Y-%m-%d')}", body, excel_file)
            logger.info("✅ Ежедневный отчет отправлен")

    async def cycle_loop(self):
        cycle_count = 0
        asyncio.create_task(self.daily_report())
        
        while not self.shutdown:
            try:
                cycle_count += 1
                self.acc_mgr.update_balances()
                
                for n, s in self.strategies.items():
                    a = self.acc_mgr.accounts[n]
                    if not a.active:
                        continue
                    
                    if not a.is_demo():
                        logger.info(f"⏸️ {a.display()} ({n}) - реальный аккаунт, торговля отключена до обучения ML")
                        continue
                    
                    self.acc_mgr.switch(n)
                    logger.info(f"🔄 Цикл {n} (баланс: ${a.balance:.2f}) | Позиций: {len(a.positions)}")
                    
                    await self.check_closed_positions(n)
                    await self.monitor_sltp(n)
                    await self.signals(n)
                
                if cycle_count % 10 == 0:
                    self.print_status()
                
                await asyncio.sleep(TRADING_MODE['cycle_sleep'])
            except Exception as e:
                logger.error(f"Ошибка цикла: {e}")
                await asyncio.sleep(60)

    async def check_closed_positions(self, name):
        s = self.strategies[name]
        a = self.acc_mgr.accounts[name]
        
        exchange_positions = s.client.positions()
        exchange_symbols = {p['symbol'] for p in exchange_positions}
        
        for sym in list(a.positions.keys()):
            if sym not in exchange_symbols:
                logger.info(f"🔔 {name} {sym} закрыта на бирже")
                entry_price = a.positions[sym]['entry']
                closed = s.client.get_closed_trade(sym, entry_price=entry_price, hours_back=48)
                if closed:
                    await self.close_position_with_data(name, sym, closed)
                else:
                    price = s.client.price(sym)
                    if price > 0:
                        await self.close_position(name, sym, "CLOSED_ON_EXCHANGE", price)

    async def monitor_sltp(self, name):
        s = self.strategies[name]
        a = self.acc_mgr.accounts[name]
        
        for sym, pos_data in list(a.positions.items()):
            price = s.client.price(sym)
            if price == 0:
                continue
            entry = pos_data['entry']
            sl = pos_data.get('sl', 0)
            tp = pos_data.get('tp', 0)
            side = pos_data['side']
            tol = entry * 0.005
            reason = None
            if side == "Buy":
                if tp and price >= tp - tol:
                    reason = "TAKE_PROFIT"
                elif sl and price <= sl + tol:
                    reason = "STOP_LOSS"
            else:
                if tp and price <= tp + tol:
                    reason = "TAKE_PROFIT"
                elif sl and price >= sl - tol:
                    reason = "STOP_LOSS"
            if reason:
                logger.info(f"🔔 {name} {sym} {reason}")
                await self.close_position(name, sym, reason, price)

    async def signals(self, name):
        s = self.strategies[name]
        a = self.acc_mgr.accounts[name]
        
        signals = []
        for sym in SYMBOLS:
            if sym in a.positions:
                continue
            
            sig = s.find_entry_signal(sym, a, self.coin_analyzer)
            if not sig:
                continue
            
            size, risk_percent, risk_amount, lev = s.position_size(
                sym, sig['price'], sig['stop_pct'], sig['action'], a, False
            )
            
            sig['size'] = size
            sig['risk_percent'] = risk_percent
            sig['risk_amount_usdt'] = risk_amount
            sig['lev'] = lev
            signals.append(sig)
        
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        for sig in signals[:TRADING_MODE['max_trades_per_cycle']]:
            await self.open_trade(name, sig)

    async def open_trade(self, name, sig):
        s = self.strategies[name]
        a = self.acc_mgr.accounts[name]
        sym = sig['symbol']

        if s.check_position(sym):
            return

        market = s.get_market(sym)
        ml_prob = self.ml.predict(market, sig['action']) if market else 0.5
        if ml_prob < TRADING_MODE['ml_decision_threshold']:
            logger.info(f"ML отклонил {sym}: {ml_prob:.2f}")
            return

        confirm, ai_score, ai_reason = await self.ai.confirm(sig, a.balance)
        if not confirm:
            logger.info(f"AI отклонил {sym}: {ai_score:.2f} - {ai_reason[:100]}")
            return

        s.client.leverage(sym, sig['lev'])
        side = "Buy" if sig['action'] == "BUY" else "Sell"
        res = s.client.order(sym, side, sig['size'], sig['stop'], sig['tp'], sig['lev'])

        if res.get('ok'):
            order_id = res.get('id')
            
            await asyncio.sleep(3)
            
            real_price = None
            real_qty = None
            
            if order_id:
                order_fill = s.client.get_order_fill(order_id)
                if order_fill and order_fill['avg_price'] > 0:
                    real_price = order_fill['avg_price']
                    real_qty = order_fill['executed_qty']
                    logger.info(f"✅ {name}: {sym} {sig['action']} открыта | Цена: {real_price:.6f}")
            
            if real_price is None:
                real_price = sig['price']
                real_qty = sig['size']
            
            stop_pct = sig['stop_pct']
            tp_pct = sig['tp_pct']
            stop = real_price * (1 - stop_pct) if sig['action'] == "BUY" else real_price * (1 + stop_pct)
            tp = real_price * (1 + tp_pct) if sig['action'] == "BUY" else real_price * (1 - tp_pct)
            
            position_value = real_qty * real_price
            
            a.positions[sym] = {
                'side': side,
                'entry': real_price,
                'size': real_qty,
                'lev': sig['lev'],
                'sl': stop,
                'tp': tp,
                'order_id': order_id
            }

            trade_id = f"{name}_{sym}_{int(time.time())}_{real_price:.8f}"
            trade = {
                'id': trade_id,
                'account': name,
                'time': datetime.now().isoformat(),
                'symbol': sym,
                'side': sig['action'],
                'entry': real_price,
                'size': real_qty,
                'lev': sig['lev'],
                'sl': stop,
                'tp': tp,
                'risk_percent': sig['risk_percent'],
                'risk_amount_usdt': sig['risk_amount_usdt'],
                'ml_prob': ml_prob,
                'ai_score': ai_score,
                'ai_reason': ai_reason,
                'trend': sig.get('trend', ''),
                'rsi': sig.get('rsi', 0),
                'adx': sig.get('adx', 0),
                'williams': sig.get('williams', 0),
                'pattern_type': sig.get('pattern_type', ''),
                'market_phase': sig.get('market_phase', ''),
                'exit': None,
            }
            a.history.append(trade)
            self.save_history(name)
            
            side_emoji = "🟢" if sig['action'] == "BUY" else "🔴"
            body = f"""
{side_emoji} СДЕЛКА ОТКРЫТА
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Аккаунт: {a.display()}
Монета: {sym}
Направление: {sig['action']}
Тип входа: {sig.get('pattern_type', 'Стандартный')}

📊 ПАРАМЕТРЫ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Цена входа: ${real_price:.6f}
Размер: {real_qty:.6f} {sym.replace('USDT', '')}
Стоимость позиции: ${position_value:.2f}
Плечо: {sig['lev']:.0f}x
Стоп-лосс: ${stop:.6f} ({stop_pct*100:.1f}%)
Тейк-профит: ${tp:.6f} ({tp_pct*100:.1f}%)
RR: {sig['rr']:.1f}:1

📈 ТЕХНИЧЕСКИЙ АНАЛИЗ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Тренд: {sig.get('trend', 'NEUTRAL')} | ADX: {sig.get('adx', 0):.0f}
RSI: {sig.get('rsi', 50):.1f} | Williams %R: {sig.get('williams', 0):.0f}
Фаза рынка: {sig.get('market_phase', 'NEUTRAL')}

💰 РИСК:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Риск: {sig['risk_percent']:.2f}% от баланса
Сумма риска: ${sig['risk_amount_usdt']:.2f}
Баланс: ${a.balance:.2f}

🤖 AI ОЦЕНКА:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Уверенность: {sig['confidence']}/10 | ML: {ml_prob:.2f}
AI скор: {ai_score:.2f}
{ai_reason}
"""
            send_email(f"{side_emoji} ОТКРЫТА {a.display()} {sym} {sig['action']}", body)
        else:
            logger.error(f"❌ {name} {sym} ошибка: {res.get('err')}")

    async def close_position(self, name, sym, reason, price, existing_trade=None):
        s = self.strategies[name]
        a = self.acc_mgr.accounts[name]
        
        if existing_trade is None:
            for t in a.history:
                if t.get('symbol') == sym and t.get('exit') is None:
                    existing_trade = t
                    break
        
        if existing_trade is None:
            logger.warning(f"Не найдена сделка {sym} в истории")
            return
        
        entry = existing_trade['entry']
        size = existing_trade['size']
        side = existing_trade['side']
        
        if side == "BUY":
            gross = size * (price - entry)
        else:
            gross = size * (entry - price)
        comm = size * entry * EXCHANGE_COMMISSION + size * price * EXCHANGE_COMMISSION
        net = gross - comm
        
        existing_trade['exit'] = price
        existing_trade['exit_time'] = datetime.now().isoformat()
        existing_trade['reason'] = reason
        existing_trade['net'] = net
        existing_trade['gross'] = gross
        existing_trade['commission'] = comm
        
        self.save_history(name)
        self.coin_analyzer.update_from_trade(existing_trade)
        
        if sym in a.positions:
            del a.positions[sym]
        
        profit_emoji = "📈" if net > 0 else "📉"
        body = f"""
{profit_emoji} СДЕЛКА ЗАКРЫТА
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Аккаунт: {a.display()}
Монета: {sym}
Причина: {reason}

📊 РЕЗУЛЬТАТ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Цена входа: ${entry:.6f}
Цена выхода: ${price:.6f}
Размер: {size:.6f}
P&L: ${net:+.2f}
Комиссия: ${comm:.4f}
"""
        send_email(f"{profit_emoji} ЗАКРЫТА {a.display()} {sym} {reason}", body)
        logger.info(f"✅ {name} {sym} закрыта | {reason} | P&L: ${net:+.2f}")

    async def close_position_with_data(self, name, sym, closed_data, existing_trade=None):
        s = self.strategies[name]
        a = self.acc_mgr.accounts[name]
        
        if existing_trade is None:
            for t in a.history:
                if t.get('symbol') == sym and t.get('exit') is None:
                    existing_trade = t
                    break
        
        if existing_trade is None:
            logger.warning(f"Не найдена сделка {sym} в истории")
            return
        
        entry = existing_trade['entry']
        size = existing_trade['size']
        side = existing_trade['side']
        exit_price = closed_data['exit']
        pnl = closed_data['pnl']
        
        if side == "BUY":
            gross = size * (exit_price - entry)
        else:
            gross = size * (entry - exit_price)
        comm = size * entry * EXCHANGE_COMMISSION + size * exit_price * EXCHANGE_COMMISSION
        
        existing_trade['exit'] = exit_price
        try:
            time_val = int(closed_data['time']) if isinstance(closed_data['time'], str) else closed_data['time']
            existing_trade['exit_time'] = datetime.fromtimestamp(time_val / 1000).isoformat()
        except:
            existing_trade['exit_time'] = datetime.now().isoformat()
        existing_trade['reason'] = "TAKE_PROFIT" if pnl > 0 else "STOP_LOSS"
        existing_trade['net'] = pnl
        existing_trade['gross'] = gross
        existing_trade['commission'] = comm
        
        self.save_history(name)
        self.coin_analyzer.update_from_trade(existing_trade)
        
        if sym in a.positions:
            del a.positions[sym]
        
        profit_emoji = "📈" if pnl > 0 else "📉"
        body = f"""
{profit_emoji} СДЕЛКА ЗАКРЫТА
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Аккаунт: {a.display()}
Монета: {sym}
Причина: {existing_trade['reason']}

📊 РЕЗУЛЬТАТ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Цена входа: ${entry:.6f}
Цена выхода: ${exit_price:.6f}
Размер: {size:.6f}
P&L: ${pnl:+.2f}
Комиссия: ${comm:.4f}
"""
        send_email(f"{profit_emoji} ЗАКРЫТА {a.display()} {sym} {existing_trade['reason']}", body)
        logger.info(f"✅ {name} {sym} закрыта | {existing_trade['reason']} | P&L: ${pnl:+.2f}")


async def main():
    bot = TradingBot()
    await bot.cycle_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Бот остановлен")
