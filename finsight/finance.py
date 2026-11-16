"""Auditable, deterministic financial analytics. Monetary figures are USD millions."""
from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "financials.csv"
METRICS = ("revenue", "gross_profit", "operating_income", "net_income")
METRIC_LABELS = {
    "revenue": "Revenue", "gross_profit": "Gross profit",
    "operating_income": "Operating income", "net_income": "Net income",
    "gross_margin": "Gross margin", "operating_margin": "Operating margin",
    "net_margin": "Net margin", "revenue_growth": "Revenue growth (YoY)",
}
RATIO_NUMERATORS = {
    "gross_margin": "gross_profit", "operating_margin": "operating_income",
    "net_margin": "net_income",
}


@dataclass(frozen=True)
class Observation:
    company: str
    ticker: str
    year: int
    year_end: str
    metric: str
    value: float
    unit: str
    source_url: str
    source_label: str

    @property
    def citation(self) -> str:
        return f"{self.company} FY{self.year}: {self.source_label} ({self.source_url})"


class FinancialStore:
    def __init__(self, csv_path: str | Path = DATA_PATH):
        self.frame = pd.read_csv(csv_path, dtype={"ticker": "string"})
        self._validate()

    def _validate(self):
        required = {"company", "ticker", "fiscal_year", "fiscal_year_end", "metric",
                    "value_usd_millions", "source_url", "source_label"}
        if not required.issubset(self.frame.columns):
            raise ValueError(f"Missing columns: {sorted(required - set(self.frame.columns))}")
        if self.frame[list(required)].isna().any().any():
            raise ValueError("Financial dataset contains missing values")
        if self.frame.duplicated(["ticker", "fiscal_year", "metric"]).any():
            raise ValueError("Duplicate company/year/metric observations")
        if not self.frame.metric.isin(METRICS).all():
            raise ValueError("Unexpected metric")
        if not self.frame.source_url.str.startswith("https://").all():
            raise ValueError("Each observation must have a source HTTPS URL")

    @property
    def tickers(self) -> list[str]:
        return sorted(self.frame.ticker.unique().tolist())

    def years(self, ticker: str) -> list[int]:
        return sorted(self.frame.loc[self.frame.ticker == ticker.upper(), "fiscal_year"].unique().tolist())

    def get(self, ticker: str, year: int, metric: str) -> Observation | None:
        ticker = ticker.upper().strip()
        if metric in RATIO_NUMERATORS:
            numerator = self.get(ticker, year, RATIO_NUMERATORS[metric])
            denominator = self.get(ticker, year, "revenue")
            if not numerator or not denominator or denominator.value == 0:
                return None
            return Observation(numerator.company, ticker, year, numerator.year_end, metric,
                               100.0 * numerator.value / denominator.value, "%",
                               numerator.source_url, numerator.source_label)
        if metric == "revenue_growth":
            current = self.get(ticker, year, "revenue")
            previous = self.get(ticker, year - 1, "revenue")
            if not current or not previous or previous.value == 0:
                return None
            return Observation(current.company, ticker, year, current.year_end, metric,
                               100.0 * (current.value / previous.value - 1), "%",
                               current.source_url, current.source_label)
        rows = self.frame[(self.frame.ticker == ticker) &
                          (self.frame.fiscal_year == int(year)) &
                          (self.frame.metric == metric)]
        if rows.empty:
            return None
        row = rows.iloc[0]
        return Observation(str(row.company), ticker, int(year), str(row.fiscal_year_end),
                           metric, float(row.value_usd_millions), "USD million",
                           str(row.source_url), str(row.source_label))

    def compare(self, tickers: list[str], years: list[int], metric: str) -> list[Observation]:
        if metric not in METRIC_LABELS:
            raise ValueError("Unsupported metric")
        result = []
        for ticker in dict.fromkeys(t.upper() for t in tickers):
            for year in sorted(set(int(y) for y in years)):
                item = self.get(ticker, year, metric)
                if item:
                    result.append(item)
        return result


def display_value(obs: Observation) -> str:
    return f"{obs.value:,.2f}%" if obs.unit == "%" else f"${obs.value:,.0f}m"


def detect_metric(question: str) -> str | None:
    q = question.lower()
    phrases = [
        (r"\b(gross margin|marge brute)\b", "gross_margin"),
        (r"\b(operating margin|marge op[eé]rationnelle)\b", "operating_margin"),
        (r"\b(net margin|marge nette)\b", "net_margin"),
        (r"\b(growth|croissance|yoy|year.over.year|progression)\b", "revenue_growth"),
        (r"\b(gross profit|b[eé]n[eé]fice brut)\b", "gross_profit"),
        (r"\b(operating income|operating profit|r[eé]sultat op[eé]rationnel)\b", "operating_income"),
        (r"\b(net income|net profit|b[eé]n[eé]fice net|r[eé]sultat net)\b", "net_income"),
        (r"\b(revenue|revenues|sales|turnover|chiffre d.affaires|ventes)\b", "revenue"),
    ]
    for pattern, metric in phrases:
        if re.search(pattern, q):
            return metric
    return None


def parse_financial_query(question: str, store: FinancialStore) -> tuple[list[str], list[int], str] | None:
    metric = detect_metric(question)
    if not metric:
        return None
    q = question.lower()
    tickers = []
    for ticker in store.tickers:
        name = str(store.frame.loc[store.frame.ticker == ticker, "company"].iloc[0])
        if re.search(rf"\b{re.escape(ticker.lower())}\b", q) or name.lower() in q:
            tickers.append(ticker)
    if not tickers:
        return None  # Never silently select a company for free-form questions
    years = [int(y) for y in re.findall(r"\b(?:20\d{2})\b", q)]
    if not years:
        years = [max(store.years(t)) for t in tickers]
    elif metric == "revenue_growth":
        years = [max(years)] if len(years) > 1 else years
    return tickers, years, metric


def answer_financial_query(question: str, store: FinancialStore) -> str | None:
    parsed = parse_financial_query(question, store)
    if not parsed:
        return None
    tickers, years, metric = parsed
    items = store.compare(tickers, years, metric)
    if not items:
        return "No matching fiscal-year observations are available in the verified dataset."
    heading = METRIC_LABELS[metric]
    lines = [f"**{heading}** (fiscal years, not calendar years):"]
    for obs in items:
        lines.append(f"- **{obs.company} ({obs.ticker}), FY{obs.year}** "
                     f"(ended {obs.year_end}): {display_value(obs)} "
                     f"— [official filing]({obs.source_url})")
    if metric == "revenue_growth":
        lines.append("Growth = (current FY revenue / preceding FY revenue − 1) × 100.")
    elif metric in RATIO_NUMERATORS:
        lines.append("Margin = corresponding profit / revenue × 100.")
    lines.append("USD figures are in **millions**. Companies may have different fiscal year-ends.")
    return "\n".join(lines)
