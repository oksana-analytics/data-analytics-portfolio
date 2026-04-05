# Crypto Trading Bot v50.0 — 12 стратегий с AI-оптимизацией

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Bybit](https://img.shields.io/badge/Bybit-API-blue.svg)](https://www.bybit.com/)
[![AI](https://img.shields.io/badge/DeepSeek-AI-green.svg)](https://deepseek.com/)
[![ML](https://img.shields.io/badge/ML-RandomForest-green.svg)](https://scikit-learn.org/)

## 🎯 О чём этот проект?

Это мульти-аккаунтный торговый бот для криптовалютной биржи Bybit, который объединяет **12 торговых стратегий**, **машинное обучение** и **AI-верификацию** для автоматической торговли. Бот адаптируется под режим рынка и оптимизирует параметры сделок через DeepSeek AI.

### Что умеет бот?

1. **Анализировать 36 криптовалют** — BTC, ETH, SOL, BNB и другие
2. **Применять 12 стратегий** — от MA Crossover до Ichimoku
3. **Адаптироваться к рынку** — определяет 8 режимов и выбирает нужные стратегии
4. **Фильтровать сигналы** — через ML (Random Forest) и AI (DeepSeek)
5. **Управлять рисками** — динамический стоп-лосс, риск 0.5% на сделку
6. **Отправлять уведомления** — email на каждую сделку

---

## 📁 Архитектура бота

### Компонент 1: 12 торговых стратегий

| Тип | Стратегии |
|-----|-----------|
| **Трендовые** | MA Crossover, ADX+DI, MACD, Ichimoku |
| **Контртрендовые** | Bollinger Bands, Mean Reversion, RSI Divergence |
| **Пробойные** | Volume Breakout, Double Pattern |
| **Паттерновые** | Candle Patterns, Support/Resistance, Fibonacci |

### Компонент 2: Адаптивный выбор (StrategyManager)

| Функция | Что делает |
|---------|------------|
| Определение режима | Анализирует ADX, волатильность, положение цены |
| Активация стратегий | Включает только релевантные под текущий режим |
| Взвешенная оценка | Учитывает confidence, вес и винрейт стратегии |

### Компонент 3: AI оптимизация (DeepSeek)

| Функция | Что делает |
|---------|------------|
| Верификация сигнала | Оценивает сделку от 0 до 1 |
| Оптимизация SL/TP | Рекомендует лучшие уровни стоп-лосса и тейк-профита |
| Анализ рынка | Учитывает технические индикаторы и баланс |

### Компонент 4: Риск-менеджмент

| Параметр | Значение |
|----------|----------|
| Стоп-лосс | 2-8% (динамический через ATR) |
| Тейк-профит | Стоп × 2 (RR 1:2) |
| Риск на сделку (демо) | 0.5% от баланса |
| Риск на сделку (реал) | 0.3% от баланса |
| Плечо | 2x — 12x (адаптивное) |

---

## 📊 Режимы рынка и стратегии

| Режим рынка | Активные стратегии |
|-------------|---------------------|
| Strong Trend Up | MA Crossover, ADX+DI, Ichimoku, Volume Breakout |
| Strong Trend Down | MA Crossover, ADX+DI, Ichimoku, Volume Breakout |
| Weak Trend Up | MA Crossover, MACD, Ichimoku |
| Weak Trend Down | MA Crossover, MACD, Ichimoku |
| Ranging (боковик) | Bollinger Bands, Mean Reversion, RSI Divergence, S/R Bounce |
| High Volatility | Volume Breakout, Double Pattern, Candle Patterns |
| Low Volatility | Bollinger Bands, Fibonacci, S/R Bounce |
| Breakout (пробой) | Volume Breakout, Double Pattern, MA Crossover |

---

## 🔍 Что мы сделали шаг за шагом?

### 1. Создали 12 классов стратегий

Каждая стратегия наследуется от `BaseStrategy` и реализует методы:
- `analyze(data)` — поиск сигнала
- `get_signal_strength(data)` — оценка силы сигнала

```python
class MACrossoverStrategy(BaseStrategy):
    def analyze(self, data):
        fast_ma = talib.SMA(closes, timeperiod=9)
        slow_ma = talib.SMA(closes, timeperiod=21)
        
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return {'action': 'BUY', 'confidence': 0.7}
