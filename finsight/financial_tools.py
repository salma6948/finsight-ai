"""Auditable calculations and cautious extraction from reported financial passages."""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from finsight.documents import Chunk


@dataclass(frozen=True)
class FinancialFact:
    company: str
    year: int
    metric: str
    value: Decimal
    unit: str
    citation: str
    original: str


def parse_amount(value: str, unit: str) -> Decimal:
    """Normalize thousands separators and magnitude to millions; never guess currency."""
    cleaned = re.sub(r'[\s,]', '', value)
    try:
        amount = Decimal(cleaned)
    except InvalidOperation as exc:
        raise ValueError('Invalid financial amount') from exc
    unit = unit.lower().strip()
    multipliers = {'million': Decimal(1), 'millions': Decimal(1), 'm': Decimal(1),
                   'billion': Decimal(1000), 'billions': Decimal(1000), 'bn': Decimal(1000)}
    if unit not in multipliers:
        raise ValueError('Unknown magnitude; cannot safely normalize amount')
    return amount * multipliers[unit]


def yoy_growth(current: Decimal, previous: Decimal) -> Decimal:
    if previous == 0:
        raise ValueError('Previous fiscal year amount cannot be zero')
    return (current / previous - Decimal(1)) * Decimal(100)


def profit_margin(profit: Decimal, revenue: Decimal) -> Decimal:
    if revenue == 0:
        raise ValueError('Revenue cannot be zero')
    return profit / revenue * Decimal(100)


def extract_reported_revenue(chunks: list[Chunk], company: str, year: int) -> list[FinancialFact]:
    """Conservative extraction of labeled revenue totals, with currency/year in same passage.

    Does not guess table column order. Ambiguous table rows remain evidence for the LLM,
    not verified structured numeric facts.
    """
    results = []
    for chunk in chunks:
        if str(year) not in chunk.text:
            continue
        text = re.sub(r'\s+', ' ', chunk.text)
        # E.g. '€80.8bn ... REVENUE FOR 2025' / 'Revenue for 2025: EUR 80.8 billion'.
        patterns = [
            rf'(?P<currency>€|EUR|\$|USD)\s*(?P<amount>\d[\d ,.]*\d|\d+)\s*(?P<scale>bn|billion|million|m)\b[^.\n]{{0,80}}?\b(?:revenue|sales|chiffre d.affaires)\s+(?:for\s+)?{year}\b',
            rf'\b(?:revenue|sales|chiffre d.affaires)\s+(?:for\s+)?{year}\b\s*[:=]?\s*(?P<currency>€|EUR|\$|USD)\s*(?P<amount>\d[\d ,.]*\d|\d+)\s*(?P<scale>bn|billion|million|m)\b'
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.I):
                raw = match.group(0)
                # Slide layouts often place the NEXT KPI immediately after 'REVENUE FOR 2025'.
                # Do not label that next amount as revenue when it is followed by another KPI.
                if re.match(r'(?i)^(revenue|sales|chiffre)', raw) and re.search(r'\b(PROFIT|EBITDA|EARNINGS|INCOME)\b', text[match.end():match.end() + 55], re.I):
                    continue
                currency = 'EUR' if match['currency'] in ('€', 'EUR') else 'USD'
                try:
                    value = parse_amount(match['amount'], match['scale'])
                except ValueError:
                    continue
                results.append(FinancialFact(company, year, 'revenue', value,
                                             f'{currency} million', chunk.cite(), raw))
    # Deduplicate a number appearing in multiple overlapping chunks.
    unique = {}
    for fact in results:
        unique[(fact.value, fact.unit, fact.citation)] = fact
    return list(unique.values())


def summarize_extracted_facts(chunks: list[Chunk], question: str) -> str:
    """Use extraction only when company/year explicitly appear in the question."""
    year_match = re.search(r'\b(20\d{2})\b', question)
    if not year_match or not re.search(r'\b(revenue|sales|chiffre d.affaires)\b', question, re.I):
        return ''
    year = int(year_match[1])
    candidates = sorted({c.document.rsplit('.pdf', 1)[0].split('_')[0] for c in chunks})
    selected = next((name for name in candidates if name and name.lower() in question.lower()), None)
    if not selected:
        return ''
    facts = extract_reported_revenue(chunks, selected, year)
    # Conflicting extracted values are not certified.
    distinct = {(f.value, f.unit) for f in facts}
    if len(distinct) != 1:
        return ''
    fact = facts[0]
    return (f'**Document-extracted revenue (verify against original PDF):** '
            f'{fact.company}, FY{year}: {fact.value:,.2f} {fact.unit} '
            f'[{fact.citation}]. Extracted phrase: "{fact.original[:160]}".')
