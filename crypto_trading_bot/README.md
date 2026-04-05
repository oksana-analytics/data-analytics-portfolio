# Crypto Trading Bot v50.0 — 12 стратегий с AI-оптимизацией

## О чём этот проект

Мульти-аккаунтный торговый бот для криптовалютной биржи Bybit с адаптивным выбором стратегий под режим рынка и AI-оптимизацией уровней стоп-лосса и тейк-профита.

## Что умеет бот

- Анализировать 36 криптовалютных пар
- Применять 12 торговых стратегий
- Определять 8 режимов рынка
- Фильтровать сигналы через ML (Random Forest)
- Оптимизировать SL/TP через DeepSeek AI
- Отправлять email-уведомления о сделках

## 12 торговых стратегий

| Тип | Стратегии |
|-----|-----------|
| Трендовые | MA Crossover, ADX+DI, MACD, Ichimoku |
| Контртрендовые | Bollinger Bands, Mean Reversion, RSI Divergence |
| Пробойные | Volume Breakout, Double Pattern |
| Паттерновые | Candle Patterns, Support/Resistance, Fibonacci |

## 8 режимов рынка

| Режим рынка | Активные стратегии |
|-------------|---------------------|
| Strong Trend Up | MA Crossover, ADX+DI, Ichimoku, Volume Breakout |
| Strong Trend Down | MA Crossover, ADX+DI, Ichimoku, Volume Breakout |
| Weak Trend Up | MA Crossover, MACD, Ichimoku |
| Weak Trend Down | MA Crossover, MACD, Ichimoku |
| Ranging | Bollinger Bands, Mean Reversion, RSI Divergence, S/R Bounce |
| High Volatility | Volume Breakout, Double Pattern, Candle Patterns |
| Low Volatility | Bollinger Bands, Fibonacci, S/R Bounce |
| Breakout | Volume Breakout, Double Pattern, MA Crossover |

## Архитектура

**1. Технический анализ**
- 4H (тренд): EMA 20/50, ADX
- 1H (свинг): RSI, MACD, Williams %R
- 30M (вход): Канал Дончана, паттерны, объем

**2. ML фильтр (Random Forest)**
- Признаки: ATR, RSI, MACD, ADX, Williams, направление
- Порог: ≥ 0.35

**3. AI оптимизация (DeepSeek)**
- Верификация сигнала (score 0-1)
- Рекомендация stop_pct и tp_pct

**4. Риск-менеджмент**
- Динамический стоп на основе ATR (2-8%)
- RR = 1:2
- Риск: 0.5% (демо) / 0.3% (реал)

## Ключевые метрики

| Метрика | Значение |
|---------|----------|
| Количество стратегий | 12 |
| Режимов рынка | 8 |
| Монет в сканировании | 36 |
| Винрейт (демо) | 65-70% |
| AI порог | 0.5 |
| ML порог | 0.35 |

## Технологический стек

| Компонент | Технология |
|-----------|------------|
| Язык | Python 3.9+ |
| Стратегии | TA-Lib |
| ML | Scikit-learn (Random Forest, SMOTE) |
| AI | DeepSeek API |
| API | Bybit REST API |
| Асинхронность | asyncio, aiohttp |

## Структура проекта

```text
crypto_trading_bot/
├── trading_bot.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
├── accounts/
│   ├── demo.env
│   └── main.env
└── trading_data/



Быстрый старт
Установка зависимостей:

bash
pip install python-dotenv requests aiohttp pandas numpy scikit-learn TA-Lib openpyxl imbalanced-learn
Создайте .env:

env
DEEPSEEK_API_KEY=your_key
EMAIL_FROM=your_email@yandex.com
EMAIL_PASSWORD=your_password
Создайте accounts/demo.env:

env
ACCOUNT_TYPE=demo
BYBIT_API_KEY=your_demo_api_key
BYBIT_SECRET_KEY=your_demo_secret_key
    ├── history_*.json
    ├── coin_stats.json
    └── exports/
