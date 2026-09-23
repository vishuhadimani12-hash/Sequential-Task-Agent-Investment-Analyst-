"""
AI Investment Analyst Agent — Main Streamlit Application
Sequential 10-step pipeline driven by Groq AI.
"""

import os
import sys
import time
import math
import logging
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & path setup
# ---------------------------------------------------------------------------
load_dotenv()
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# Page configuration — MUST be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Investment Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inject custom CSS + force sidebar open via JS
# ---------------------------------------------------------------------------
css_path = BASE_DIR / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Force sidebar expanded — works by setting the internal Streamlit sidebar state
st.markdown(
    """
    <script>
    // Expand sidebar on load if it is collapsed
    const tryExpand = () => {
        const btn = window.parent.document.querySelector(
            '[data-testid="stSidebarCollapseButton"] button, '
            + '[data-testid="stSidebarNavToggleButton"] button'
        );
        if (btn) {
            const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
            if (sidebar && sidebar.getAttribute('aria-expanded') === 'false') {
                btn.click();
            }
        } else {
            setTimeout(tryExpand, 200);
        }
    };
    document.addEventListener('DOMContentLoaded', tryExpand);
    setTimeout(tryExpand, 400);
    </script>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Local imports (after path is set)
# ---------------------------------------------------------------------------
from utils.helpers import (
    period_to_dates, fmt_large, fmt_pct, to_float, now_label, risk_label, risk_color
)
from utils.charts import (
    candlestick_chart, revenue_trend_chart, ratio_radar_chart,
    competitor_bar_chart, sentiment_pie, risk_gauge,
    forecast_chart, market_cap_bubble, macd_chart,
)
from agents.data_agent       import DataAgent
from agents.validation_agent import ValidationAgent
from agents.kpi_agent        import KPIAgent
from agents.ratio_agent      import RatioAgent
from agents.benchmark_agent  import BenchmarkAgent
from agents.sentiment_agent  import SentimentAgent
from agents.forecast_agent   import ForecastAgent
from agents.risk_agent       import RiskAgent
from agents.report_agent     import ReportAgent
from utils.groq_client       import generate_investment_analysis

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "analysis_done":    False,
        "raw_data":         None,
        "validation":       None,
        "kpis":             None,
        "fmt_kpis":         None,
        "trend_data":       None,
        "ratios":           None,
        "ratio_groups":     None,
        "radar_labels":     None,
        "radar_values":     None,
        "benchmark_df":     None,
        "sentiment":        None,
        "forecast":         None,
        "risk":             None,
        "ai_rec":           None,
        "company_name":     "",
        "current_ticker":   "",
        "pdf_bytes":        None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
          <div class="sidebar-logo">
            <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
              <!-- Background circle -->
              <circle cx="24" cy="24" r="24" fill="#1e3a5f"/>
              <!-- Bar chart bars -->
              <rect x="8"  y="28" width="7" height="12" rx="2" fill="#60a5fa"/>
              <rect x="17" y="20" width="7" height="20" rx="2" fill="#3b82f6"/>
              <rect x="26" y="14" width="7" height="26" rx="2" fill="#2563eb"/>
              <!-- Trend line -->
              <polyline points="11,27 21,19 30,13 40,9"
                        stroke="#34d399" stroke-width="2.5"
                        stroke-linecap="round" stroke-linejoin="round"/>
              <!-- Trend dot -->
              <circle cx="40" cy="9" r="3" fill="#34d399"/>
            </svg>
          </div>
          <div class="sidebar-brand-text">
            <span class="sidebar-title">Investment Analyst</span>
            <span class="sidebar-sub">AI-Powered · Sequential Agent</span>
          </div>
        </div>

        <nav class="sidebar-nav">
          <a href="#company-overview"   class="nav-item">🏢 Company Overview</a>
          <a href="#stock-price"        class="nav-item">💹 Stock Price</a>
          <a href="#financial-kpis"     class="nav-item">📊 Financial KPIs</a>
          <a href="#financial-ratios"   class="nav-item">📐 Ratios</a>
          <a href="#competitor"         class="nav-item">🏆 Competitors</a>
          <a href="#sentiment"          class="nav-item">📰 News Sentiment</a>
          <a href="#forecast"           class="nav-item">📈 Forecast</a>
          <a href="#risk"               class="nav-item">🚨 Risk Analysis</a>
          <a href="#ai-recommendation"  class="nav-item">🤖 AI Recommendation</a>
          <a href="#export"             class="nav-item">📄 Export Report</a>
        </nav>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    ticker_input = st.text_input(
        "Company Ticker Symbol",
        value="AAPL",
        placeholder="e.g. AAPL, TSLA, MSFT",
        help="Enter a valid Yahoo Finance ticker symbol.",
    ).upper().strip()

    competitor_raw = st.text_input(
        "Competitor Tickers (comma-separated)",
        value="MSFT, GOOGL, AMZN",
        placeholder="e.g. MSFT, GOOGL",
    )
    competitors = [c.strip().upper() for c in competitor_raw.split(",") if c.strip()]

    period = st.selectbox(
        "Analysis Period",
        ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y"],
        index=3,
    )

    st.divider()
    run_btn    = st.button("🚀 Run Sequential Analysis", use_container_width=True)
    report_btn = st.button("🤖 Generate AI Report",      use_container_width=True)
    export_btn = st.button("📄 Export PDF",              use_container_width=True)

    st.divider()
    st.markdown(
        "<small style='color:#475569'>Data: Yahoo Finance · News: NewsAPI / YF<br>"
        "AI Model: openai/gpt-oss-120b (Groq)</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def step_badge(number: int, label: str, status: str = "active") -> None:
    icons = {"active": "⚙️", "success": "✅", "error": "❌", "warning": "⚠️"}
    css_cls = {"active": "", "success": "success", "error": "error", "warning": "warning"}
    icon = icons.get(status, "⚙️")
    cls = css_cls.get(status, "")
    st.markdown(
        f'<div class="step-badge {cls}">{icon} Step {number}: {label}</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def render_kpi_cards(fmt_kpis: dict, cols: int = 4) -> None:
    keys_to_show = [
        "Revenue", "Net Income", "EBITDA", "Gross Profit",
        "Market Cap", "EPS", "PE Ratio", "ROE",
        "ROA", "Operating Margin", "Net Profit Margin", "Free Cash Flow",
    ]
    items = [(k, fmt_kpis.get(k, "N/A")) for k in keys_to_show if k in fmt_kpis]
    rows = [items[i:i+cols] for i in range(0, len(items), cols)]
    for row in rows:
        columns = st.columns(len(row))
        for col, (label, value) in zip(columns, row):
            col.markdown(
                f'<div class="kpi-card"><div class="label">{label}</div>'
                f'<div class="value">{value}</div></div>',
                unsafe_allow_html=True,
            )


def render_news_table(articles: list) -> None:
    if not articles:
        st.info("No news articles available.")
        return
    for a in articles[:15]:
        title    = a.get("title", "No title")
        source   = a.get("source", {}).get("name", "") if isinstance(a.get("source"), dict) else a.get("source", "")
        url      = a.get("url", "#")
        sent     = a.get("sentiment", "Neutral")
        conf     = a.get("confidence", 0.5)
        st.markdown(
            f'<div class="news-row">'
            f'<div class="news-badge badge-{sent}">{sent}<br>{conf:.0%}</div>'
            f'<div class="headline"><a href="{url}" target="_blank" style="color:#93c5fd;text-decoration:none">{title}</a>'
            f'<div class="source">{source}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_ai_recommendation(ai: dict) -> None:
    sections = [
        ("investment_summary",      "📋 Investment Summary"),
        ("strengths",               "💪 Strengths"),
        ("weaknesses",              "⚠️ Weaknesses"),
        ("key_risks",               "🚨 Key Risks"),
        ("growth_opportunities",    "🌱 Growth Opportunities"),
        ("short_term_outlook",      "📅 Short-Term Outlook"),
        ("long_term_outlook",       "🔭 Long-Term Outlook"),
        ("portfolio_recommendation","💼 Portfolio Recommendation"),
    ]
    for key, label in sections:
        val = ai.get(key)
        if not val:
            continue
        if isinstance(val, list):
            items_html = "".join(f"<li>{item}</li>" for item in val)
            body = f"<ul>{items_html}</ul>"
        else:
            body = f"<p>{val}</p>"
        st.markdown(
            f'<div class="ai-block"><h4>{label}</h4>{body}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main sequential analysis pipeline
# ---------------------------------------------------------------------------
def run_analysis(ticker: str, competitors: list[str], period: str) -> None:
    start_date, end_date = period_to_dates(period)

    progress_bar = st.progress(0, text="Starting analysis…")
    status_box   = st.empty()

    total_steps  = 10

    def update(step: int, label: str):
        pct = int(step / total_steps * 100)
        progress_bar.progress(pct, text=f"Step {step}/{total_steps} — {label}")
        status_box.markdown(
            f'<div class="step-badge">⚙️ {label}…</div>', unsafe_allow_html=True
        )

    # -----------------------------------------------------------------------
    # Step 1 — Data Gathering
    # -----------------------------------------------------------------------
    update(1, "Collecting financial data")
    with st.expander("📥 Step 1 — Data Gathering", expanded=False):
        step_badge(1, "Data Gathering", "active")
        sub_prog = st.empty()
        try:
            agent = DataAgent(ticker, start_date, end_date)

            def data_cb(s, t, lbl):
                pct = int(s / t * 100) if t else 0
                sub_prog.progress(pct, text=lbl)

            raw = agent.run(progress_callback=data_cb)
            company_name = raw.get("info", {}).get("shortName", ticker)
            st.session_state.raw_data     = raw
            st.session_state.company_name = company_name

            if raw.get("errors"):
                for e in raw["errors"]:
                    st.warning(f"⚠️ {e}")

            info = raw.get("info", {})
            col1, col2, col3 = st.columns(3)
            col1.metric("Price History rows", len(raw.get("history", pd.DataFrame())))
            col2.metric("Income Stmt cols",   len(raw.get("income_stmt", pd.DataFrame()).columns))
            col3.metric("Balance Sheet rows",  len(raw.get("balance_sheet", pd.DataFrame())))
            step_badge(1, "Data Gathering", "success")
        except Exception as exc:
            st.error(f"❌ Data collection failed: {exc}")
            step_badge(1, "Data Gathering", "error")
            st.stop()

    # -----------------------------------------------------------------------
    # Step 2 — Validation
    # -----------------------------------------------------------------------
    update(2, "Validating data quality")
    with st.expander("✅ Step 2 — Data Validation", expanded=False):
        step_badge(2, "Data Validation", "active")
        try:
            val_agent  = ValidationAgent(st.session_state.raw_data)
            val_report = val_agent.run()
            st.session_state.validation = val_report

            score = val_report["validation_score"]
            st.metric("Validation Score", f"{score}%", help="% of expected fields present")
            st.write(f"**Quality**: {val_agent.get_score_label()}")

            if val_report["missing_info_fields"]:
                st.warning("Missing fields: " + ", ".join(val_report["missing_info_fields"][:8]))
            if val_report["issues"]:
                for iss in val_report["issues"]:
                    st.warning(iss)

            step_badge(2, "Data Validation", "success")
        except Exception as exc:
            st.error(f"❌ Validation failed: {exc}")

    # -----------------------------------------------------------------------
    # Step 3 — KPI Extraction
    # -----------------------------------------------------------------------
    update(3, "Extracting KPIs")
    cleaned = st.session_state.validation["cleaned_data"]
    with st.expander("📊 Step 3 — KPI Extraction", expanded=False):
        step_badge(3, "KPI Extraction", "active")
        try:
            kpi_agent = KPIAgent(cleaned)
            kpis      = kpi_agent.run()
            fmt_kpis  = kpi_agent.formatted_kpis()
            trend_d   = kpi_agent.trend_data()

            st.session_state.kpis      = kpis
            st.session_state.fmt_kpis  = fmt_kpis
            st.session_state.trend_data = trend_d

            render_kpi_cards(fmt_kpis)
            step_badge(3, "KPI Extraction", "success")
        except Exception as exc:
            st.error(f"❌ KPI extraction failed: {exc}")

    # -----------------------------------------------------------------------
    # Step 4 — Financial Ratios
    # -----------------------------------------------------------------------
    update(4, "Calculating financial ratios")
    with st.expander("📐 Step 4 — Financial Ratios", expanded=False):
        step_badge(4, "Financial Ratios", "active")
        try:
            ratio_agent  = RatioAgent(st.session_state.kpis, cleaned["info"])
            ratios       = ratio_agent.run()
            groups       = ratio_agent.grouped_table()
            r_labels, r_vals = ratio_agent.radar_data()

            st.session_state.ratios       = ratios
            st.session_state.ratio_groups = groups
            st.session_state.radar_labels = r_labels
            st.session_state.radar_values = r_vals

            for group_name, group_data in groups.items():
                if not group_data:
                    continue
                st.write(f"**{group_name}**")
                df_g = pd.DataFrame(
                    [{"Ratio": k, "Value": v} for k, v in group_data.items()]
                )
                st.dataframe(df_g, use_container_width=True, hide_index=True)

            step_badge(4, "Financial Ratios", "success")
        except Exception as exc:
            st.error(f"❌ Ratio calculation failed: {exc}")

    # -----------------------------------------------------------------------
    # Step 5 — Competitor Benchmark
    # -----------------------------------------------------------------------
    update(5, "Benchmarking competitors")
    with st.expander("🏆 Step 5 — Competitor Benchmark", expanded=False):
        step_badge(5, "Competitor Benchmark", "active")
        try:
            bench_sub = st.empty()
            bench_agent = BenchmarkAgent(ticker, competitors)

            def bench_cb(s, t, lbl):
                pct = int(s / t * 100) if t else 0
                bench_sub.progress(pct, text=lbl)

            bdf = bench_agent.run(progress_callback=bench_cb)
            st.session_state.benchmark_df = bdf

            st.dataframe(bdf, use_container_width=True, hide_index=True)
            step_badge(5, "Competitor Benchmark", "success")
        except Exception as exc:
            st.warning(f"⚠️ Benchmark failed: {exc}")
            st.session_state.benchmark_df = pd.DataFrame()

    # -----------------------------------------------------------------------
    # Step 6 — News Sentiment
    # -----------------------------------------------------------------------
    update(6, "Analysing news sentiment")
    with st.expander("📰 Step 6 — News Sentiment", expanded=False):
        step_badge(6, "News Sentiment", "active")
        try:
            sent_agent = SentimentAgent(ticker, st.session_state.company_name)
            sentiment  = sent_agent.run()
            st.session_state.sentiment = sentiment

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall", sentiment["overall_sentiment"])
            col2.metric("Positive", sentiment["positive"])
            col3.metric("Neutral",  sentiment["neutral"])
            col4.metric("Negative", sentiment["negative"])

            if sentiment.get("note"):
                st.info(sentiment["note"])

            render_news_table(sentiment.get("articles", []))
            step_badge(6, "News Sentiment", "success")
        except Exception as exc:
            st.warning(f"⚠️ Sentiment analysis failed: {exc}")
            st.session_state.sentiment = {
                "overall_sentiment": "Neutral",
                "sentiment_score": 0.5,
                "positive": 0, "neutral": 0, "negative": 0,
                "articles": [],
            }

    # -----------------------------------------------------------------------
    # Step 7 — Trend Forecast
    # -----------------------------------------------------------------------
    update(7, "Computing technical indicators & forecast")
    history = cleaned.get("history", pd.DataFrame())
    with st.expander("📈 Step 7 — Trend Forecast", expanded=False):
        step_badge(7, "Trend Forecast", "active")
        try:
            fc_agent = ForecastAgent(history)
            forecast = fc_agent.run()
            st.session_state.forecast = forecast

            if "error" in forecast:
                st.warning(forecast["error"])
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Trend",            forecast["trend"])
                col2.metric("Bullish Prob.",    f"{forecast['bullish_prob']}%")
                col3.metric("Current RSI",      str(forecast.get("current_rsi", "N/A")))
            step_badge(7, "Trend Forecast", "success")
        except Exception as exc:
            st.warning(f"⚠️ Forecast failed: {exc}")
            st.session_state.forecast = {}

    # -----------------------------------------------------------------------
    # Step 8 — Risk Detection
    # -----------------------------------------------------------------------
    update(8, "Detecting risks")
    with st.expander("🚨 Step 8 — Risk Detection", expanded=False):
        step_badge(8, "Risk Detection", "active")
        try:
            risk_agent = RiskAgent(
                st.session_state.kpis,
                st.session_state.ratios or {},
                st.session_state.sentiment or {},
                st.session_state.forecast or {},
            )
            risk = risk_agent.run()
            st.session_state.risk = risk

            col1, col2 = st.columns(2)
            col1.metric("Risk Score", f"{risk['score']:.1f} / 100")
            col2.metric("Category",   risk["label"])
            for r in risk.get("detected_risks", []):
                st.markdown(f"- {r}")
            step_badge(8, "Risk Detection", "success")
        except Exception as exc:
            st.warning(f"⚠️ Risk detection failed: {exc}")
            st.session_state.risk = {"score": 50, "label": "Moderate",
                                     "color": "#f59e0b", "detected_risks": []}

    # -----------------------------------------------------------------------
    # Step 9 — AI Recommendation
    # -----------------------------------------------------------------------
    update(9, "Generating AI investment recommendation")
    with st.expander("🤖 Step 9 — AI Recommendation", expanded=False):
        step_badge(9, "AI Recommendation", "active")
        try:
            context = {
                "company":    st.session_state.company_name,
                "ticker":     ticker,
                "kpis":       {k: v for k, v in (st.session_state.kpis or {}).items()},
                "ratios":     st.session_state.ratios or {},
                "sentiment": {
                    "overall":  st.session_state.sentiment.get("overall_sentiment", ""),
                    "score":    st.session_state.sentiment.get("sentiment_score", 0.5),
                },
                "risk": {
                    "score": st.session_state.risk.get("score", 50),
                    "label": st.session_state.risk.get("label", ""),
                    "flags": st.session_state.risk.get("detected_risks", []),
                },
                "forecast": {
                    "trend":        st.session_state.forecast.get("trend", ""),
                    "bullish_prob": st.session_state.forecast.get("bullish_prob", 50),
                    "rsi":          st.session_state.forecast.get("current_rsi", ""),
                },
            }
            ai_rec = generate_investment_analysis(context)
            st.session_state.ai_rec = ai_rec
            render_ai_recommendation(ai_rec)
            step_badge(9, "AI Recommendation", "success")
        except Exception as exc:
            st.warning(f"⚠️ AI recommendation failed: {exc}")
            st.session_state.ai_rec = {
                "investment_summary": f"AI analysis unavailable: {exc}",
            }

    # -----------------------------------------------------------------------
    # Step 10 — Report ready
    # -----------------------------------------------------------------------
    update(10, "Finalising report")
    st.session_state.analysis_done   = True
    st.session_state.current_ticker  = ticker
    progress_bar.progress(100, text="✅ Analysis complete!")
    status_box.empty()
    st.success(f"✅ Sequential analysis for **{ticker}** completed — scroll down to view the full dashboard.")


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------
def generate_pdf() -> bytes | None:
    if not st.session_state.analysis_done:
        st.warning("Run the analysis first before exporting PDF.")
        return None
    try:
        report = ReportAgent(
            ticker        = st.session_state.current_ticker,
            company_name  = st.session_state.company_name,
            kpis          = st.session_state.kpis or {},
            ratios        = st.session_state.ratios or {},
            benchmark_df  = st.session_state.benchmark_df,
            sentiment     = st.session_state.sentiment or {},
            risk          = st.session_state.risk or {},
            forecast      = st.session_state.forecast or {},
            ai_recommendation = st.session_state.ai_rec or {},
            fmt_kpis      = st.session_state.fmt_kpis or {},
        )
        pdf_bytes = report.generate()
        st.session_state.pdf_bytes = pdf_bytes
        return pdf_bytes
    except Exception as exc:
        st.error(f"PDF generation failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Trigger actions from sidebar buttons
# ---------------------------------------------------------------------------
if run_btn:
    if not ticker_input:
        st.sidebar.error("Please enter a ticker symbol.")
    else:
        run_analysis(ticker_input, competitors, period)

if report_btn:
    if not st.session_state.analysis_done:
        st.sidebar.warning("Run the analysis first.")
    else:
        with st.spinner("Generating AI investment report…"):
            ctx = {
                "company":   st.session_state.company_name,
                "ticker":    st.session_state.current_ticker,
                "kpis":      st.session_state.kpis or {},
                "ratios":    st.session_state.ratios or {},
                "sentiment": st.session_state.sentiment or {},
                "risk":      st.session_state.risk or {},
                "forecast":  st.session_state.forecast or {},
            }
            ai_rec = generate_investment_analysis(ctx)
            st.session_state.ai_rec = ai_rec
        st.sidebar.success("AI report generated!")

if export_btn:
    pdf = generate_pdf()
    if pdf:
        fname = f"{st.session_state.current_ticker}_investment_report_{now_label().replace(':', '-').replace(' ', '_')}.pdf"
        st.sidebar.download_button(
            label     = "⬇️ Download PDF Report",
            data      = pdf,
            file_name = fname,
            mime      = "application/pdf",
        )
        st.sidebar.success("PDF ready — click Download above.")


# ---------------------------------------------------------------------------
# ===== MAIN DASHBOARD — rendered after analysis =====
# ---------------------------------------------------------------------------
if not st.session_state.analysis_done:
    # Welcome screen
    st.markdown(
        """
        <div style="text-align:center; padding: 80px 0 40px;">
          <h1 style="font-size:2.5rem; font-weight:800; color:#60a5fa;">📊 AI Investment Analyst</h1>
          <p style="font-size:1.1rem; color:#94a3b8; max-width:600px; margin:16px auto;">
            A sequential AI agent that autonomously collects, validates, analyses,
            and summarises financial data for any publicly traded company.
          </p>
          <br>
          <p style="color:#64748b">← Enter a ticker in the sidebar and click <strong>Run Sequential Analysis</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature cards
    features = [
        ("📥", "Data Collection",    "Prices, income, balance sheet & cash flow from Yahoo Finance"),
        ("✅", "Validation",         "Automatic data quality scoring and cleansing"),
        ("📊", "KPI Extraction",     "12+ key performance indicators"),
        ("📐", "Financial Ratios",   "Liquidity, profitability, valuation & leverage"),
        ("🏆", "Benchmark",          "Side-by-side competitor comparison"),
        ("📰", "Sentiment",          "AI-powered news sentiment classification"),
        ("📈", "Forecast",           "SMA, RSI, MACD, Bollinger Bands + 30-day ML forecast"),
        ("🚨", "Risk Analysis",      "Multi-factor risk scoring 0-100"),
        ("🤖", "AI Recommendation",  "Groq-powered investment thesis generation"),
        ("📄", "PDF Report",         "Professional downloadable investment report"),
    ]
    cols = st.columns(5)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 5]:
            st.markdown(
                f'<div class="kpi-card" style="height:120px;text-align:left;">'
                f'<div style="font-size:1.4rem">{icon}</div>'
                f'<div style="font-weight:700;color:#93c5fd;font-size:0.85rem">{title}</div>'
                f'<div style="font-size:0.72rem;color:#64748b;margin-top:4px">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.stop()


# ---------------------------------------------------------------------------
# Dashboard header
# ---------------------------------------------------------------------------
ticker_display = st.session_state.current_ticker
company_display = st.session_state.company_name

st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:16px;padding:12px 0 20px;">
      <div style="font-size:2rem">📊</div>
      <div>
        <h1 style="margin:0;font-size:1.8rem;font-weight:800;color:#f1f5f9">{company_display}</h1>
        <span style="color:#64748b;font-size:0.9rem">{ticker_display} · Analysis generated {now_label()}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Section 1: Company Overview
# ---------------------------------------------------------------------------
section_header("🏢 Company Overview")
info = (st.session_state.validation or {}).get("cleaned_data", {}).get("info", {})
if info:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sector",       info.get("sector", "N/A"))
    col2.metric("Industry",     info.get("industry", "N/A"))
    col3.metric("52W High",     f"${to_float(info.get('fiftyTwoWeekHigh')):.2f}")
    col4.metric("52W Low",      f"${to_float(info.get('fiftyTwoWeekLow')):.2f}")

    with st.expander("Full Company Description"):
        st.write(info.get("longBusinessSummary", "Not available."))

# ---------------------------------------------------------------------------
# Section 2: Stock Price Chart
# ---------------------------------------------------------------------------
section_header("💹 Stock Price History")
history = (st.session_state.validation or {}).get("cleaned_data", {}).get("history", pd.DataFrame())
if not history.empty:
    st.plotly_chart(
        candlestick_chart(history, ticker_display),
        use_container_width=True,
    )
else:
    st.info("No price history available.")

# ---------------------------------------------------------------------------
# Section 3: Financial KPIs
# ---------------------------------------------------------------------------
section_header("📊 Financial KPIs")
if st.session_state.fmt_kpis:
    render_kpi_cards(st.session_state.fmt_kpis)

    td = st.session_state.trend_data or {}
    if td.get("years"):
        st.plotly_chart(
            revenue_trend_chart(td["years"], td["revenue"], td["net_income"]),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Section 4: Financial Ratios
# ---------------------------------------------------------------------------
section_header("📐 Financial Ratios")
if st.session_state.ratio_groups:
    col_a, col_b = st.columns(2)
    groups = st.session_state.ratio_groups
    group_items = list(groups.items())
    with col_a:
        for name, data in group_items[:3]:
            if data:
                st.write(f"**{name}**")
                df_r = pd.DataFrame([{"Ratio": k, "Value": v} for k, v in data.items()])
                st.dataframe(df_r, use_container_width=True, hide_index=True)
    with col_b:
        for name, data in group_items[3:]:
            if data:
                st.write(f"**{name}**")
                df_r = pd.DataFrame([{"Ratio": k, "Value": v} for k, v in data.items()])
                st.dataframe(df_r, use_container_width=True, hide_index=True)

    if st.session_state.radar_labels:
        st.plotly_chart(
            ratio_radar_chart(
                st.session_state.radar_labels,
                st.session_state.radar_values,
                ticker_display,
            ),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Section 5: Competitor Benchmark
# ---------------------------------------------------------------------------
section_header("🏆 Competitor Benchmark")
bdf = st.session_state.benchmark_df
if bdf is not None and not bdf.empty:
    tab1, tab2, tab3 = st.tabs(["📊 PE Ratio", "💰 Market Cap", "📈 ROE"])
    with tab1:
        m_df = bdf[["Ticker", "PE"]].dropna() if "PE" in bdf.columns else pd.DataFrame()
        if not m_df.empty:
            m_df = m_df.rename(columns={"PE": "PE"})
            m_df["PE"] = pd.to_numeric(m_df["PE"], errors="coerce")
            st.plotly_chart(competitor_bar_chart(m_df.dropna(), "PE"), use_container_width=True)
    with tab2:
        mc_df = bdf[["Ticker", "Market Cap"]].dropna() if "Market Cap" in bdf.columns else pd.DataFrame()
        if not mc_df.empty:
            mc_df["Market Cap"] = pd.to_numeric(mc_df["Market Cap"], errors="coerce")
            st.plotly_chart(competitor_bar_chart(mc_df.dropna(), "Market Cap"), use_container_width=True)
    with tab3:
        roe_df = bdf[["Ticker", "ROE"]].dropna() if "ROE" in bdf.columns else pd.DataFrame()
        if not roe_df.empty:
            roe_df["ROE"] = pd.to_numeric(roe_df["ROE"], errors="coerce")
            st.plotly_chart(competitor_bar_chart(roe_df.dropna(), "ROE"), use_container_width=True)

    # Market-cap bubble
    bubble_df_cols = {"Ticker": "Ticker", "Market Cap": "MarketCap", "PE": "PE", "Revenue": "Revenue"}
    bubble_cols = [c for c in bubble_df_cols if c in bdf.columns]
    if len(bubble_cols) >= 3:
        bub = bdf[bubble_cols].rename(columns=bubble_df_cols)
        for col in ["MarketCap", "PE", "Revenue"]:
            if col in bub.columns:
                bub[col] = pd.to_numeric(bub[col], errors="coerce")
        st.plotly_chart(market_cap_bubble(bub.dropna(subset=["MarketCap", "PE"])), use_container_width=True)

    st.write("**Full Ranking Table**")
    st.dataframe(bdf, use_container_width=True, hide_index=True)
else:
    st.info("No benchmark data available.")

# ---------------------------------------------------------------------------
# Section 6: News Sentiment
# ---------------------------------------------------------------------------
section_header("📰 Market News Sentiment")
sentiment = st.session_state.sentiment or {}
if sentiment:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(
            sentiment_pie(
                sentiment.get("positive", 0),
                sentiment.get("neutral", 0),
                sentiment.get("negative", 0),
            ),
            use_container_width=True,
        )
    with col2:
        st.metric("Overall Sentiment", sentiment.get("overall_sentiment", "N/A"))
        score = sentiment.get("sentiment_score", 0.5)
        # st.progress expects int 0-100 in Streamlit ≥1.35
        st.progress(int(score * 100), text=f"Sentiment Score: {score:.0%}")
        render_news_table(sentiment.get("articles", []))

# ---------------------------------------------------------------------------
# Section 7: Technical Indicators & Forecast
# ---------------------------------------------------------------------------
section_header("📈 Technical Analysis & 30-Day Forecast")
forecast = st.session_state.forecast or {}


def _clean_series(lst):
    """Replace NaN/None with None so Plotly renders gaps cleanly."""
    import math
    if not lst:
        return lst
    return [None if (v is None or (isinstance(v, float) and math.isnan(v))) else v
            for v in lst]


if forecast and "error" not in forecast:
    tab_fc, tab_macd = st.tabs(["📊 Price Forecast", "📉 MACD"])
    with tab_fc:
        st.plotly_chart(
            forecast_chart(
                hist_dates   = forecast.get("dates", []),
                hist_prices  = _clean_series(forecast.get("prices", [])),
                fc_dates     = forecast.get("forecast_dates", []),
                fc_prices    = _clean_series(forecast.get("forecast_prices", [])),
                sma20        = _clean_series(forecast.get("sma20")),
                sma50        = _clean_series(forecast.get("sma50")),
                bb_upper     = _clean_series(forecast.get("bb_upper")),
                bb_lower     = _clean_series(forecast.get("bb_lower")),
                ticker       = ticker_display,
            ),
            use_container_width=True,
        )
        col1, col2, col3 = st.columns(3)
        # Strip emoji from trend for safe display in metric labels
        trend_str = forecast.get("trend", "N/A")
        col1.metric("Trend",         trend_str)
        col2.metric("Bullish Prob.", f"{forecast.get('bullish_prob', 0)}%")
        col3.metric("RSI",           str(forecast.get("current_rsi", "N/A")))

    with tab_macd:
        macd_data = forecast.get("macd_line")
        # Check the list is non-empty and has at least one real (non-NaN) value
        has_macd = bool(macd_data) and any(
            v is not None and not (isinstance(v, float) and math.isnan(v))
            for v in macd_data
        )
        if has_macd:
            st.plotly_chart(
                macd_chart(
                    forecast["dates"],
                    _clean_series(forecast["macd_line"]),
                    _clean_series(forecast["signal_line"]),
                    _clean_series(forecast["histogram"]),
                ),
                use_container_width=True,
            )
        else:
            st.info("MACD data not available for the selected period.")
elif forecast.get("error"):
    st.warning(forecast["error"])

# ---------------------------------------------------------------------------
# Section 8: Risk Meter
# ---------------------------------------------------------------------------
section_header("🚨 Risk Analysis")
risk = st.session_state.risk or {}
if risk:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(risk_gauge(risk.get("score", 50)), use_container_width=True)
    with col2:
        label = risk.get("label", "N/A")
        color_cls = {"Low": "risk-low", "Moderate": "risk-moderate", "High": "risk-high"}.get(label, "")
        st.markdown(
            f'<div style="margin-top:30px"><span class="section-title">Category: </span>'
            f'<span class="{color_cls}" style="font-size:1.3rem">{label}</span></div>',
            unsafe_allow_html=True,
        )
        st.write("**Detected Risk Factors:**")
        for rf in risk.get("detected_risks", ["None detected."]):
            st.markdown(f"- {rf}")

# ---------------------------------------------------------------------------
# Section 9: AI Recommendation
# ---------------------------------------------------------------------------
section_header("🤖 AI Investment Recommendation")
ai_rec = st.session_state.ai_rec or {}
if ai_rec:
    render_ai_recommendation(ai_rec)
else:
    st.info("Run the analysis or click 'Generate AI Report' to receive an AI investment recommendation.")

# ---------------------------------------------------------------------------
# Section 10: Download Report
# ---------------------------------------------------------------------------
section_header("📄 Export & Download")
col1, col2 = st.columns(2)
with col1:
    if st.button("📄 Generate & Export PDF Report", use_container_width=True):
        with st.spinner("Building PDF report…"):
            pdf = generate_pdf()
        if pdf:
            fname = f"{ticker_display}_report_{now_label().replace(':', '-').replace(' ', '_')}.pdf"
            st.download_button(
                label     = "⬇️ Download PDF",
                data      = pdf,
                file_name = fname,
                mime      = "application/pdf",
                use_container_width=True,
            )
with col2:
    if st.session_state.kpis:
        kpi_df = pd.DataFrame(
            list(st.session_state.fmt_kpis.items()), columns=["KPI", "Value"]
        )
        csv = kpi_df.to_csv(index=False).encode()
        st.download_button(
            label     = "⬇️ Download KPIs as CSV",
            data      = csv,
            file_name = f"{ticker_display}_kpis.csv",
            mime      = "text/csv",
            use_container_width=True,
        )

st.markdown(
    "<br><hr style='border-color:rgba(100,116,139,0.2)'>"
    "<p style='text-align:center;color:#475569;font-size:0.75rem'>"
    "AI Investment Analyst · For educational purposes only · Not financial advice"
    "</p>",
    unsafe_allow_html=True,
)
