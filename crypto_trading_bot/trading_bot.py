#!/usr/bin/env python3
# trading_bot.py - Мульти-аккаунтный торговый бот
# Версия 51.0 - Исправлено плечо, AI порог 0.6, полные email

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
from enum import Enum

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
    'ai_confidence_threshold': 0.6,  # ИСПРАВЛЕНО: 0.6
    'min_stop_pct': 0.015,
    'max_stop_pct': 0.05,
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

# ==================== ФУНКЦИЯ ОТПРАВКИ EMAIL (УВЕЛИЧЕНА ДО 4000 СИМВОЛОВ) ====================
def send_email(subject, body, attachment=None):
    """Отправка email уведомления с поддержкой до 4000 символов"""
    if not EMAIL_FROM or not EMAIL_PASSWORD:
        return False
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        msg['Subject'] = subject
        
        # Увеличиваем лимит для текста
        if len(body) > 4000:
            body = body[:3997] + "..."
        
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

# ==================== BYBIT API КЛИЕНТ (С ИСПРАВЛЕННЫМ ПЛЕЧОМ) ====================
class BybitClient:
    def __init__(self, api_key, secret_key, demo=False, name=None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.demo = demo
        self.name = name
        self.base = "https://api-demo.bybit.com" if demo else "https://api.bybit.com"
        self.session = requests.Session()
        self._leverage_cache = {}

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
                    'stopLoss': pos.get('stopLoss'),
                    'takeProfit': pos.get('takeProfit')
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

    def set_leverage(self, symbol, lev):
        """Установка кредитного плеча - ИСПРАВЛЕНО"""
        try:
            lev = int(round(lev))
            lev = max(1, min(20, lev))
            
            # Проверяем кэш
            cache_key = f"{symbol}_{lev}"
            if cache_key in self._leverage_cache:
                if time.time() - self._leverage_cache[cache_key] < 300:
                    return True
            
            r = self._req("POST", "/v5/position/set-leverage", {
                "category": "linear",
                "symbol": symbol,
                "buyLeverage": str(lev),
                "sellLeverage": str(lev),
                "positionIdx": POSITION_IDX
            })
            if r.get('retCode') == 0:
                self._leverage_cache[cache_key] = time.time()
                logger.info(f"⚙️ {self.name}: плечо {symbol} установлено на {lev}x")
                return True
            else:
                logger.warning(f"⚠️ {self.name}: ошибка установки плеча {symbol}: {r.get('retMsg')}")
                return False
        except Exception as e:
            logger.error(f"Ошибка set_leverage: {e}")
            return False

    def order(self, symbol, side, qty, sl=None, tp=None, lev=2):
        qty_str = self.format_qty(symbol, qty)
        
        # КРИТИЧЕСКИ ВАЖНО: Устанавливаем плечо ПЕРЕД открытием позиции
        self.set_leverage(symbol, lev)
        
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

    def set_stop_loss_take_profit(self, symbol, side, sl, tp, position_idx=0):
        params = {
            "category": "linear",
            "symbol": symbol,
            "positionIdx": position_idx
        }
        if side == "Buy":
            if sl:
                params["stopLoss"] = f"{self.round_price(symbol, sl):.{PRICE_PRECISION.get(symbol, 2)}f}"
            if tp:
                params["takeProfit"] = f"{self.round_price(symbol, tp):.{PRICE_PRECISION.get(symbol, 2)}f}"
        else:
            if sl:
                params["stopLoss"] = f"{self.round_price(symbol, sl):.{PRICE_PRECISION.get(symbol, 2)}f}"
            if tp:
                params["takeProfit"] = f"{self.round_price(symbol, tp):.{PRICE_PRECISION.get(symbol, 2)}f}"
        
        r = self._req("POST", "/v5/position/set-trading-stop", params)
        if r.get('retCode') == 0:
            return {"ok": True}
        else:
            return {"ok": False, "err": r.get('retMsg')}

    def get_closed_trade(self, symbol, entry_price=None, size=None, hours_back=48):
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
        return self.set_leverage(symbol, lev)

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


# ==================== КЛАСС ДЛЯ АНАЛИЗА МОНЕТ ====================
class CoinAnalyzer:
    def __init__(self):
        self.stats_file = COIN_STATS_FILE
        self.coin_stats = self.load_stats()
        self.processed_trades = set()
        self.processed_keys = set()

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
        if trade_id and trade_id in self.processed_trades:
            return
        
        exit_time = trade.get('exit_time', '')
        symbol = trade.get('symbol', '')
        entry = trade.get('entry', 0)
        unique_key = f"{symbol}_{entry}_{exit_time[:19]}"
        
        if unique_key in self.processed_keys:
            return
        
        if trade_id:
            self.processed_trades.add(trade_id)
        self.processed_keys.add(unique_key)
        
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
                'strategy': t.get('strategy', ''), 'market_regime': t.get('market_regime', ''),
            })
        
        df_trades = pd.DataFrame(trades_data)
        filename = os.path.join(EXCEL_DIR, f"trades_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df_trades.to_excel(writer, sheet_name='Сделки', index=False)
        
        return filename


class MarketRegime(Enum):
    STRONG_TREND_UP = "STRONG_TREND_UP"
    WEAK_TREND_UP = "WEAK_TREND_UP"
    STRONG_TREND_DOWN = "STRONG_TREND_DOWN"
    WEAK_TREND_DOWN = "WEAK_TREND_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"


# ==================== БАЗОВАЯ СТРАТЕГИЯ ====================
class BaseStrategy:
    def __init__(self, name: str, weight: float, min_confidence: float = 0.6):
        self.name = name
        self.weight = weight
        self.min_confidence = min_confidence
        self.total_signals = 0
        self.win_signals = 0
    
    def get_winrate(self) -> float:
        if self.total_signals == 0:
            return 0.5
        return self.win_signals / self.total_signals
    
    def update_performance(self, was_winner: bool):
        self.total_signals += 1
        if was_winner:
            self.win_signals += 1
    
    def analyze(self, data: Dict) -> Optional[Dict]:
        raise NotImplementedError
    
    def get_signal_strength(self, data: Dict) -> float:
        raise NotImplementedError


# ==================== СТРАТЕГИЯ 1: MA CROSSOVER ====================
class MACrossoverStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("MA_CROSSOVER", weight=1.0, min_confidence=0.55)
        self.fast_period = 9
        self.slow_period = 21
    
    def analyze(self, data: Dict) -> Optional[Dict]:
        closes = np.array(data['closes'])
        if len(closes) < self.slow_period + 10:
            return None
        
        fast_ma = talib.SMA(closes, timeperiod=self.fast_period)
        slow_ma = talib.SMA(closes, timeperiod=self.slow_period)
        
        curr_fast = fast_ma[-1]
        curr_slow = slow_ma[-1]
        prev_fast = fast_ma[-2]
        prev_slow = slow_ma[-2]
        price = closes[-1]
        
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            confidence = 0.7 + (abs(curr_fast - curr_slow) / price) * 10
            confidence = min(0.95, confidence)
            return {
                'action': 'BUY',
                'confidence': confidence,
                'reason': f'Золотой крест MA{self.fast_period}/MA{self.slow_period}',
                'strategy': self.name
            }
        
        if prev_fast >= prev_slow and curr_fast < curr_slow:
            confidence = 0.7 + (abs(curr_fast - curr_slow) / price) * 10
            confidence = min(0.95, confidence)
            return {
                'action': 'SELL',
                'confidence': confidence,
                'reason': f'Мертвый крест MA{self.fast_period}/MA{self.slow_period}',
                'strategy': self.name
            }
        
        return None
    
    def get_signal_strength(self, data: Dict) -> float:
        return 0.5


# ==================== СТРАТЕГИЯ 2: ADX + DI ====================
class ADXStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("ADX_DI", weight=1.2, min_confidence=0.6)
        self.period = 14
    
    def analyze(self, data: Dict) -> Optional[Dict]:
        highs = np.array(data['highs'])
        lows = np.array(data['lows'])
        closes = np.array(data['closes'])
        
        if len(closes) < self.period + 10:
            return None
        
        adx = talib.ADX(highs, lows, closes, timeperiod=self.period)
        plus_di = talib.PLUS_DI(highs, lows, closes, timeperiod=self.period)
        minus_di = talib.MINUS_DI(highs, lows, closes, timeperiod=self.period)
        
        curr_adx = adx[-1]
        curr_plus = plus_di[-1]
        curr_minus = minus_di[-1]
        
        if curr_adx > 25:
            if curr_plus > curr_minus and curr_plus > 25:
                confidence = 0.6 + (curr_adx - 25) / 100
                confidence = min(0.9, confidence)
                return {
                    'action': 'BUY',
                    'confidence': confidence,
                    'reason': f'ADX={curr_adx:.1f}, +DI > -DI',
                    'strategy': self.name
                }
            elif curr_minus > curr_plus and curr_minus > 25:
                confidence = 0.6 + (curr_adx - 25) / 100
                confidence = min(0.9, confidence)
                return {
                    'action': 'SELL',
                    'confidence': confidence,
                    'reason': f'ADX={curr_adx:.1f}, -DI > +DI',
                    'strategy': self.name
                }
        
        return None
    
    def get_signal_strength(self, data: Dict) -> float:
        return 0.5


# ==================== СТРАТЕГИЯ 3: RSI DIVERGENCE ====================
class RSIDivergenceStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("RSI_DIVERGENCE", weight=1.3, min_confidence=0.65)
        self.rsi_period = 14
        self.lookback = 20
    
    def analyze(self, data: Dict) -> Optional[Dict]:
        closes = np.array(data['closes'])
        
        if len(closes) < self.lookback + 10:
            return None
        
        rsi = talib.RSI(closes, timeperiod=self.rsi_period)
        
        price_lows = self._find_swing_lows(closes[-self.lookback:])
        rsi_lows = self._find_swing_lows(rsi[-self.lookback:])
        price_highs = self._find_swing_highs(closes[-self.lookback:])
        rsi_highs = self._find_swing_highs(rsi[-self.lookback:])
        
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            price_low1, price_low2 = price_lows[-2], price_lows[-1]
            rsi_low1, rsi_low2 = rsi_lows[-2], rsi_lows[-1]
            
            if price_low2 < price_low1 and rsi_low2 > rsi_low1:
                confidence = 0.75
                if rsi_low2 < 30:
                    confidence += 0.1
                return {
                    'action': 'BUY',
                    'confidence': min(0.95, confidence),
                    'reason': f'Бычья дивергенция RSI',
                    'strategy': self.name
                }
        
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            price_high1, price_high2 = price_highs[-2], price_highs[-1]
            rsi_high1, rsi_high2 = rsi_highs[-2], rsi_highs[-1]
            
            if price_high2 > price_high1 and rsi_high2 < rsi_high1:
                confidence = 0.75
                if rsi_high2 > 70:
                    confidence += 0.1
                return {
                    'action': 'SELL',
                    'confidence': min(0.95, confidence),
                    'reason': f'Медвежья дивергенция RSI',
                    'strategy': self.name
                }
        
        return None
    
    def _find_swing_lows(self, arr):
        lows = []
        for i in range(2, len(arr) - 2):
            if arr[i] < arr[i-1] and arr[i] < arr[i-2] and arr[i] < arr[i+1] and arr[i] < arr[i+2]:
                lows.append(arr[i])
        return lows
    
    def _find_swing_highs(self, arr):
        highs = []
        for i in range(2, len(arr) - 2):
            if arr[i] > arr[i-1] and arr[i] > arr[i-2] and arr[i] > arr[i+1] and arr[i] > arr[i+2]:
                highs.append(arr[i])
        return highs
    
    def get_signal_strength(self, data: Dict) -> float:
        return 0.5


# ==================== СТРАТЕГИЯ 4: BOLLINGER BANDS ====================
class BollingerBandsStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("BOLLINGER_REVERSAL", weight=1.1, min_confidence=0.6)
        self.period = 20
        self.std_dev = 2
    
    def analyze(self, data: Dict) -> Optional[Dict]:
        closes = np.array(data['closes'])
        
        if len(closes) < self.period + 5:
            return None
        
        upper, middle, lower = talib.BBANDS(
            closes, timeperiod=self.period, nbdevup=self.std_dev, 
            nbdevdn=self.std_dev, matype=0
        )
        
        curr_upper = upper[-1]
        curr_lower = lower[-1]
        price = closes[-1]
        prev_price = closes[-2]
        
        if price <= curr_lower * 1.002 and prev_price > curr_lower:
            confidence = 0.65
            return {
                'action': 'BUY',
                'confidence': min(0.9, confidence),
                'reason': f'Отбой от нижней полосы BB',
                'strategy': self.name
            }
        
        if price >= curr_upper * 0.998 and prev_price < curr_upper:
            confidence = 0.65
            return {
                'action': 'SELL',
                'confidence': min(0.9, confidence),
                'reason': f'Отбой от верхней полосы BB',
                'strategy': self.name
            }
        
        return None
    
    def get_signal_strength(self, data: Dict) -> float:
        return 0.5


# ==================== СТРАТЕГИЯ 5: VOLUME BREAKOUT ====================
class VolumeBreakoutStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("VOLUME_BREAKOUT", weight=1.4, min_confidence=0.7)
        self.volume_ma_period = 20
        self.volume_multiplier = 1.5
    
    def analyze(self, data: Dict) -> Optional[Dict]:
        closes = np.array(data['closes'])
        volumes = np.array(data['volumes'])
        
        if len(volumes) < self.volume_ma_period + 5:
            return None
        
        volume_ma = talib.SMA(volumes, timeperiod=self.volume_ma_period)
        curr_volume = volumes[-1]
        avg_volume = volume_ma[-1]
        
        if curr_volume < avg_volume * self.volume_multiplier:
            return None
        
        highs = np.array(data['highs'])
        lows = np.array(data['lows'])
        
        resistance = np.max(highs[-20:-5]) if len(highs) > 20 else highs[-1]
        support = np.min(lows[-20:-5]) if len(lows) > 20 else lows[-1]
        price = closes[-1]
        
        if price > resistance and curr_volume > avg_volume * 1.5:
            confidence = 0.7
            volume_ratio = curr_volume / avg_volume
            confidence += min(0.2, (volume_ratio - 1.5) / 5)
            return {
                'action': 'BUY',
                'confidence': min(0.95, confidence),
                'reason': f'Пробой сопротивления с объемом',
                'strategy': self.name
            }
        
        if price < support and curr_volume > avg_volume * 1.5:
            confidence = 0.7
            volume_ratio = curr_volume / avg_volume
            confidence += min(0.2, (volume_ratio - 1.5) / 5)
            return {
                'action': 'SELL',
                'confidence': min(0.95, confidence),
                'reason': f'Пробой поддержки с объемом',
                'strategy': self.name
            }
        
        return None
    
    def get_signal_strength(self, data: Dict) -> float:
        return 0.5


# ==================== СТРАТЕГИЯ 6: CANDLE PATTERNS ====================
class CandlePatternStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("CANDLE_PATTERNS", weight=1.0, min_confidence=0.55)
    
    def analyze(self, data: Dict) -> Optional[Dict]:
        opens = np.array(data['opens'])
        highs = np.array(data['highs'])
        lows = np.array(data['lows'])
        closes = np.array(data['closes'])
        
        if len(closes) < 10:
            return None
        
        engulfing = talib.CDLENGULFING(opens, highs, lows, closes)
        if engulfing[-1] == 100:
            return {'action': 'BUY', 'confidence': 0.75, 'reason': 'Бычье поглощение', 'strategy': self.name}
        if engulfing[-1] == -100:
            return {'action': 'SELL', 'confidence': 0.75, 'reason': 'Медвежье поглощение', 'strategy': self.name}
        
        hammer = talib.CDLHAMMER(opens, highs, lows, closes)
        if hammer[-1] != 0:
            return {'action': 'BUY', 'confidence': 0.65, 'reason': 'Молот', 'strategy': self.name}
        
        hanging_man = talib.CDLHANGINGMAN(opens, highs, lows, closes)
        if hanging_man[-1] != 0:
            return {'action': 'SELL', 'confidence': 0.65, 'reason': 'Повешенный', 'strategy': self.name}
        
        return None
    
    def get_signal_strength(self, data: Dict) -> float:
        return 0.5


# ==================== СТРАТЕГИЯ 7: MACD ====================
class MACDStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("MACD_CROSSOVER", weight=1.1, min_confidence=0.55)
        self.fast = 12
        self.slow = 26
        self.signal = 9
    
    def analyze(self, data: Dict) -> Optional[Dict]:
        closes = np.array(data['closes'])
        
        if len(closes) < self.slow + self.signal + 5:
            return None
        
        macd, signal, hist = talib.MACD(
            closes, fastperiod=self.fast, slowperiod=self.slow, signalperiod=self.signal
        )
        
        curr_macd = macd[-1]
        curr_signal = signal[-1]
        prev_macd = macd[-2]
        prev_signal = signal[-2]
        
        if prev_macd <= prev_signal and curr_macd > curr_signal:
            confidence = 0.65
            return {
                'action': 'BUY',
                'confidence': min(0.9, confidence),
                'reason': f'MACD пересек сигнальную линию вверх',
                'strategy': self.name
            }
        
        if prev_macd >= prev_signal and curr_macd < curr_signal:
            confidence = 0.65
            return {
                'action': 'SELL',
                'confidence': min(0.9, confidence),
                'reason': f'MACD пересек сигнальную линию вниз',
                'strategy': self.name
            }
        
        return None
    
    def get_signal_strength(self, data: Dict) -> float:
        return 0.5


# ==================== СТРАТЕГИЯ 8: MEAN REVERSION ====================
class MeanReversionStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("MEAN_REVERSION", weight=0.9, min_confidence=0.55)
        self.period = 20
        self.entry_zscore = 2.0
    
    def analyze(self, data: Dict) -> Optional[Dict]:
        closes = np.array(data['closes'])
        
        if len(closes) < self.period + 5:
            return None
        
        mean = np.mean(closes[-self.period:])
        std = np.std(closes[-self.period:])
        
        if std == 0:
            return None
        
        zscore = (closes[-1] - mean) / std
        
        if zscore < -self.entry_zscore:
            confidence = 0.6
            if zscore < -2.5:
                confidence += 0.1
            return {
                'action': 'BUY',
                'confidence': min(0.85, confidence),
                'reason': f'Z-score = {zscore:.2f}',
                'strategy': self.name
            }
        
        if zscore > self.entry_zscore:
            confidence = 0.6
            if zscore > 2.5:
                confidence += 0.1
            return {
                'action': 'SELL',
                'confidence': min(0.85, confidence),
                'reason': f'Z-score = {zscore:.2f}',
                'strategy': self.name
            }
        
        return None
    
    def get_signal_strength(self, data: Dict) -> float:
        return 0.5


# ==================== МЕНЕДЖЕР СТРАТЕГИЙ ====================
class StrategyManager:
    def __init__(self):
        self.strategies: List[BaseStrategy] = []
        self.market_regime = MarketRegime.RANGING
        self._init_strategies()
    
    def _init_strategies(self):
        self.strategies = [
            MACrossoverStrategy(),
            ADXStrategy(),
            RSIDivergenceStrategy(),
            BollingerBandsStrategy(),
            VolumeBreakoutStrategy(),
            CandlePatternStrategy(),
            MACDStrategy(),
            MeanReversionStrategy()
        ]
        logger.info(f"✅ Инициализировано {len(self.strategies)} стратегий")
    
    def detect_market_regime(self, data: Dict) -> MarketRegime:
        closes = np.array(data['closes'])
        if len(closes) < 50:
            return MarketRegime.RANGING
        
        returns = np.diff(np.log(closes))
        volatility = np.std(returns) * np.sqrt(365)
        
        if volatility > 0.8:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.3:
            return MarketRegime.LOW_VOLATILITY
        
        return MarketRegime.RANGING
    
    def get_best_signal(self, data: Dict, symbol: str) -> Optional[Dict]:
        self.market_regime = self.detect_market_regime(data)
        
        best_signal = None
        best_score = 0
        
        for strategy in self.strategies:
            signal = strategy.analyze(data)
            if signal and signal['confidence'] >= strategy.min_confidence:
                score = signal['confidence'] * strategy.weight
                
                if score > best_score:
                    best_score = score
                    best_signal = signal
                    best_signal['market_regime'] = self.market_regime.value
        
        if best_signal:
            logger.info(f"🎯 {symbol}: {best_signal['strategy']} (уверенность: {best_signal['confidence']:.2f})")
        
        return best_signal


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
            atr = talib.ATR(np.array(highs), np.array(lows), np.array(closes), timeperiod=14)
            features.append(atr[-1] / price if price > 0 else 0)
            rsi = talib.RSI(np.array(closes), timeperiod=14)
            features.append(rsi[-1] / 100)
            macd, signal, hist = talib.MACD(np.array(closes))
            features.append(hist[-1] / price if price > 0 else 0)
            adx = talib.ADX(np.array(highs), np.array(lows), np.array(closes), timeperiod=14)
            features.append(adx[-1] / 100)
            williams = talib.WILLR(np.array(highs), np.array(lows), np.array(closes), timeperiod=14)
            features.append(williams[-1] / -100)
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


# ==================== AI СИСТЕМА (ПОРОГ 0.6) ====================
class AISystem:
    def __init__(self):
        self.enabled = TRADING_MODE['ai_enabled'] and bool(DEEPSEEK_API_KEY)
        self.cache = {}
        self.cache_file = os.path.join(DATA_DIR, "ai_cache.json")
        self.load_cache()
        if self.enabled:
            logger.info("✅ AI система инициализирована (DeepSeek)")
            logger.info(f"🎯 AI порог уверенности: {TRADING_MODE['ai_confidence_threshold']}")

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
                    recommended_stop_pct = result.get('recommended_stop_pct')
                    recommended_tp_pct = result.get('recommended_tp_pct')
                    
                    return {
                        'score': confidence_score,
                        'reason': reason,
                        'recommended_stop_pct': recommended_stop_pct,
                        'recommended_tp_pct': recommended_tp_pct
                    }
            
            number_match = re.search(r'(\d+\.?\d*)', response_text)
            if number_match:
                num_value = float(number_match.group(1))
                confidence_score = num_value / 100.0 if num_value > 1 else num_value
                return {'score': max(0.0, min(1.0, confidence_score)), 'reason': 'Извлечено из текста'}
            
            return {'score': 0.5, 'reason': 'Не удалось распарсить'}
            
        except Exception as e:
            logger.error(f"Ошибка парсинга AI: {e}")
            return {'score': 0.5, 'reason': f"Ошибка: {e}"}
    
    def get_cache_key(self, signal):
        symbol = signal.get('symbol', '')
        action = signal.get('action', '')
        price = signal.get('price', 0)
        price_bucket = round(price, 2) if price > 1 else round(price, 4)
        timestamp = int(time.time() // 600)
        return f"{symbol}_{action}_{price_bucket}_{timestamp}"

    async def confirm(self, signal, account_balance):
        if not self.enabled:
            return True, 0.5, "AI отключён", None, None

        if account_balance <= 0.01:
            return True, 0.5, "Баланс ниже минимума", None, None

        cache_key = self.get_cache_key(signal)
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            cache_time = datetime.fromisoformat(entry['timestamp'])
            if datetime.now() - cache_time < timedelta(minutes=10):
                logger.info(f"🔄 Используем кэш AI для {signal['symbol']}")
                return entry['confirm'], entry['score'], entry.get('reason', ''), entry.get('stop_pct'), entry.get('tp_pct')

        base_stop_pct = signal.get('stop_pct', 0.03)
        base_tp_pct = signal.get('tp_pct', 0.06)

        prompt = f"""Оцени торговый сигнал.

Монета: {signal['symbol']}
Действие: {signal['action']}
Текущая цена: ${signal['price']:.6f}
Стратегия: {signal.get('strategy', 'UNKNOWN')}
Режим рынка: {signal.get('market_regime', 'UNKNOWN')}

Баланс счета: ${account_balance:.2f}

Ответь ТОЛЬКО JSON формате:
{{"score": 0.0-1.0, "reason": "анализ"}}"""

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
                        "max_tokens": 1000
                    }
                    async with session.post(
                        "https://api.deepseek.com/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=30
                    ) as resp:
                        data = await resp.json()
                        text = data['choices'][0]['message']['content']
                        result = self.parse_ai_response(text)

                        score = result['score']
                        reason = result['reason']

                        # AI порог 0.6 (ИСПРАВЛЕНО)
                        confirm = score >= TRADING_MODE['ai_confidence_threshold']

                        self.cache[cache_key] = {
                            'confirm': confirm,
                            'score': score,
                            'reason': reason,
                            'stop_pct': base_stop_pct,
                            'tp_pct': base_tp_pct,
                            'timestamp': datetime.now().isoformat()
                        }
                        self.save_cache()

                        logger.info(f"🤖 AI: {signal['symbol']} {signal['action']} -> score={score:.2f}, confirm={confirm}")
                        return confirm, score, reason, base_stop_pct, base_tp_pct

            except asyncio.TimeoutError:
                logger.warning(f"AI таймаут, попытка {attempt+1}/3")
                await asyncio.sleep(1)
            except Exception as e:
                logger.warning(f"AI ошибка: {e}, попытка {attempt+1}/3")
                await asyncio.sleep(1)

        return False, 0.0, "AI временно недоступен", base_stop_pct, base_tp_pct


# ==================== ТОРГОВАЯ СТРАТЕГИЯ ====================
class Strategy:
    def __init__(self, client, bot, name):
        self.client = client
        self.bot = bot
        self.name = name
        self.strategy_manager = StrategyManager()

    def get_market_data(self, symbol):
        data_30m = self.client.klines(symbol, '30', 150)
        if len(data_30m) < 80:
            return None
        
        return {
            'symbol': symbol,
            'opens': [d['open'] for d in data_30m],
            'closes': [d['close'] for d in data_30m],
            'highs': [d['high'] for d in data_30m],
            'lows': [d['low'] for d in data_30m],
            'volumes': [d['volume'] for d in data_30m],
            'price': data_30m[-1]['close'],
        }

    def check_position(self, symbol):
        return any(p['symbol'] == symbol for p in self.client.positions())

    def calculate_dynamic_stop(self, market_data, action):
        closes = np.array(market_data['closes'])
        highs = np.array(market_data['highs'])
        lows = np.array(market_data['lows'])
        
        atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
        price = closes[-1]
        atr_pct = atr / price
        
        stop_pct = atr_pct * 1.5
        stop_pct = max(TRADING_MODE['min_stop_pct'], min(TRADING_MODE['max_stop_pct'], stop_pct))
        
        return stop_pct

    def find_entry_signal(self, symbol, acc, coin_analyzer):
        if self.check_position(symbol):
            return None
        
        for t in acc.history:
            if t.get('symbol') == symbol and t.get('exit') is None:
                return None
        
        market_data = self.get_market_data(symbol)
        if not market_data:
            return None
        
        best_signal = self.strategy_manager.get_best_signal(market_data, symbol)
        
        if not best_signal:
            return None
        
        stop_pct = self.calculate_dynamic_stop(market_data, best_signal['action'])
        price = market_data['price']
        
        if best_signal['action'] == 'BUY':
            stop = price * (1 - stop_pct)
            tp = price * (1 + stop_pct * 2)
        else:
            stop = price * (1 + stop_pct)
            tp = price * (1 - stop_pct * 2)
        
        best_signal.update({
            'symbol': symbol,
            'price': price,
            'stop': stop,
            'tp': tp,
            'stop_pct': stop_pct,
            'tp_pct': stop_pct * 2,
            'rr': 2.0,
            'is_qualified': coin_analyzer.is_qualified(symbol)
        })
        
        return best_signal

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
        logger.info(f"🎯 Риск для демо: {TRADING_MODE['risk_percent_demo']}% от баланса")
        logger.info(f"🤖 AI порог: {TRADING_MODE['ai_confidence_threshold']}")
        self.print_status()

    def sync_all_positions(self):
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
        self.coin_analyzer.processed_keys = set()
        
        for n, a in self.acc_mgr.accounts.items():
            f = os.path.join(DATA_DIR, f"history_{n}.json")
            if os.path.exists(f):
                with open(f, 'r') as fp:
                    a.history = json.load(fp)
                logger.info(f"📂 История {n}: {len(a.history)} записей")
                
                closed_count = 0
                for trade in a.history:
                    if trade.get('exit'):
                        closed_count += 1
                        self.coin_analyzer.update_from_trade(trade)
                logger.info(f"   Закрытых сделок: {closed_count}")

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
                        logger.info(f"⏸️ {a.display()} ({n}) - реальный аккаунт, торговля отключена")
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
                traceback.print_exc()
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
                closed = s.client.get_closed_trade(sym, entry_price=entry_price, hours_back=168)
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
        
        signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        for sig in signals[:TRADING_MODE['max_trades_per_cycle']]:
            await self.open_trade(name, sig)

    async def open_trade(self, name, sig):
        s = self.strategies[name]
        a = self.acc_mgr.accounts[name]
        sym = sig['symbol']

        if s.check_position(sym):
            return

        # ML проверка
        market = s.get_market_data(sym)
        ml_prob = self.ml.predict(market, sig['action']) if market else 0.5
        if ml_prob < TRADING_MODE['ml_decision_threshold']:
            logger.info(f"ML отклонил {sym}: {ml_prob:.2f}")
            return

        # AI проверка (порог 0.6)
        confirm, ai_score, ai_reason, ai_stop_pct, ai_tp_pct = await self.ai.confirm(sig, a.balance)

        if not confirm:
            logger.info(f"AI отклонил {sym}: {ai_score:.2f} - {ai_reason[:100]}")
            return

        # Открываем рыночный ордер (плечо устанавливается внутри order)
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
                    logger.info(f"✅ {name}: {sym} {sig['action']} открыта | Цена: {real_price:.6f} | Плечо: {sig['lev']}x")
            
            if real_price is None:
                real_price = sig['price']
                real_qty = sig['size']
            
            # Корректируем SL/TP под реальную цену
            if sig['action'] == "BUY":
                real_stop = real_price * (1 - sig['stop_pct'])
                real_tp = real_price * (1 + sig['tp_pct'])
            else:
                real_stop = real_price * (1 + sig['stop_pct'])
                real_tp = real_price * (1 - sig['tp_pct'])
            
            position_value = real_qty * real_price
            
            a.positions[sym] = {
                'side': side,
                'entry': real_price,
                'size': real_qty,
                'lev': sig['lev'],
                'sl': real_stop,
                'tp': real_tp,
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
                'sl': real_stop,
                'tp': real_tp,
                'risk_percent': sig['risk_percent'],
                'risk_amount_usdt': sig['risk_amount_usdt'],
                'ml_prob': ml_prob,
                'ai_score': ai_score,
                'ai_reason': ai_reason[:300],
                'strategy': sig.get('strategy', ''),
                'market_regime': sig.get('market_regime', ''),
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
Стратегия: {sig.get('strategy', 'N/A')}
Режим рынка: {sig.get('market_regime', 'N/A')}

📊 ПАРАМЕТРЫ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Цена входа: ${real_price:.6f}
Размер: {real_qty:.6f} {sym.replace('USDT', '')}
Стоимость позиции: ${position_value:.2f}
Плечо: {sig['lev']:.0f}x
Стоп-лосс: ${real_stop:.6f} ({sig['stop_pct']*100:.1f}%)
Тейк-профит: ${real_tp:.6f} ({sig['tp_pct']*100:.1f}%)
RR: {sig['rr']:.1f}:1

💰 РИСК:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Риск: {sig['risk_percent']:.2f}% от баланса
Сумма риска: ${sig['risk_amount_usdt']:.2f}
Баланс: ${a.balance:.2f}

🤖 AI ОЦЕНКА: {ai_score:.2f}
{ai_reason[:200]}
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
Стратегия: {existing_trade.get('strategy', 'N/A')}

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
Стратегия: {existing_trade.get('strategy', 'N/A')}

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


# ==================== ЗАПУСК ====================
async def main():
    bot = TradingBot()
    
    # Обработка остановки
    def signal_handler():
        logger.info("🛑 Получен сигнал остановки...")
        bot.shutdown = True
    
    import signal
    signal.signal(signal.SIGINT, lambda s, f: signal_handler())
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler())
    
    await bot.cycle_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        traceback.print_exc()
