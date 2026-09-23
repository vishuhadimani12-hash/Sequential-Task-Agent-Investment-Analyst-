# 📊 AI Investment Analyst Agent

A production-ready, end-to-end **Sequential Task AI Agent** that autonomously performs investment analysis using **Python**, **Streamlit**, and **Groq AI**.

---

## ✨ Features

| Step | Agent | Description |
|------|-------|-------------|
| 1 | Data Agent | Fetches prices, income statement, balance sheet, cash flow from Yahoo Finance |
| 2 | Validation Agent | Scores data completeness, cleans NaN values, flags issues |
| 3 | KPI Agent | Extracts 15+ key performance indicators |
| 4 | Ratio Agent | Calculates liquidity, profitability, valuation, leverage & efficiency ratios |
| 5 | Benchmark Agent | Compares target company against competitor peers |
| 6 | Sentiment Agent | Classifies news headlines as Positive/Neutral/Negative using Groq AI |
| 7 | Forecast Agent | Computes SMA, RSI, MACD, Bollinger Bands + 30-day ML regression forecast |
| 8 | Risk Agent | Scores financial risk 0–100 across 8 signal categories |
| 9 | AI Recommendation | Generates structured investment thesis via Groq (`openai/gpt-oss-120b`) |
| 10 | Report Agent | Exports a professional multi-page PDF report |

---

## 🛠️ Tech Stack

- **Python 3.11+**
- **Streamlit** — Interactive dashboard
- **Groq SDK** — AI inference (`openai/gpt-oss-120b`)
- **yfinance** — Market data
- **Plotly** — Interactive charts (candlestick, radar, gauge, forecast, bubble)
- **pandas / numpy** — Data processing
- **FPDF2** — PDF report generation
- **NewsAPI / Yahoo Finance news** — News sentiment source

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd investment_analyst_agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
GROQ_API_KEY=your_groq_api_key_here
NEWS_API_KEY=your_newsapi_key_here   # optional
```

---

## 🔑 API Setup

### Groq API
1. Visit [https://console.groq.com](https://console.groq.com)
2. Create a free account
3. Navigate to **API Keys** and generate a key
4. Paste the key into your `.env` file

### NewsAPI (optional)
1. Visit [https://newsapi.org](https://newsapi.org)
2. Sign up for a free developer account
3. Copy your API key into `.env`
4. If omitted, the app automatically uses Yahoo Finance news

---

## ▶️ Running the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## 📁 Project Architecture

```
investment_analyst_agent/
├── app.py                  # Main Streamlit dashboard (orchestrator)
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── data_agent.py       # Step 1 — Yahoo Finance data collection
│   ├── validation_agent.py # Step 2 — Data quality validation
│   ├── kpi_agent.py        # Step 3 — KPI extraction
│   ├── ratio_agent.py      # Step 4 — Financial ratio calculation
│   ├── benchmark_agent.py  # Step 5 — Competitor benchmarking
│   ├── sentiment_agent.py  # Step 6 — News sentiment (Groq AI)
│   ├── forecast_agent.py   # Step 7 — Technical indicators & forecast
│   ├── risk_agent.py       # Step 8 — Risk scoring
│   └── report_agent.py     # Step 10 — PDF report generation
│
├── utils/
│   ├── groq_client.py      # Reusable Groq API wrapper
│   ├── charts.py           # Plotly chart factory
│   └── helpers.py          # Shared utility functions
│
├── assets/
│   └── style.css           # Custom dark-theme Streamlit CSS
│
└── reports/                # Generated PDF reports (gitignored)
```

---

## 📊 Dashboard Sections

1. **Company Overview** — Sector, industry, 52-week range, business description
2. **Stock Price Chart** — Interactive candlestick with volume
3. **Financial KPIs** — Metric cards for all key indicators
4. **Financial Ratios** — Grouped tables + radar chart
5. **Competitor Benchmark** — Tabbed bar charts + full ranking table + bubble chart
6. **News Sentiment** — Pie chart + article list with AI sentiment badges
7. **Technical Analysis** — Price forecast + MACD charts
8. **Risk Meter** — Gauge chart with detected risk factors
9. **AI Recommendation** — Structured Groq-generated investment thesis
10. **Export** — PDF download + CSV download

---

## 📸 Screenshots

> _Add screenshots here after running the app_

---

## ⚠️ Disclaimer

This tool is for **educational and informational purposes only**. It does not constitute financial advice. Always consult a qualified financial professional before making investment decisions. Past performance is not indicative of future results.

---

## 📝 License

MIT License — see `LICENSE` for details.
