# The ETF Fee Paradox

The typical new US ETF is getting more expensive even as the fee paid by the
average invested dollar stays near record lows. This repository rebuilds that
distribution from raw SEC filings and supports the accompanying research note.

**Read the essay:** https://plvch.github.io/indexes_cost/

## Main findings

Using the fiscal-2025 census of 3,954 US '40 Act ETFs:

- The asset-weighted net expense ratio is **0.13%**; the median fund charges
  **0.50%**.
- Funds below 0.10% are **8% of fund count** and hold **73% of assets**.
- Funds above 0.50% are **55% of fund count** and hold about **5.5% of assets**.
- The median 2025 launch charges **0.69%**, and **85%** of 2025 launches are
  actively managed.
- Organic 2020-2025 launches hold **4.3% of assets** and produce **15.4% of
  implied fee revenue**.

## Repository

- `post/` contains the standalone published essay.
- `analysis/build_snapshot.py` builds the current ETF fee and AUM panel.
- `analysis/build_history.py` builds fee, launch, closure, and active-share
  histories.
- `analysis/build_sponsors.py` compares recent launches with sponsors'
  incumbent books.
- `analysis/out/` contains the resulting panels and summary CSVs used to check
  the published figures.
- `scripts/fetch_history.sh` downloads and filters the required SEC quarterly
  files.

Raw SEC downloads are excluded from Git because they total roughly 485 MB.

## Reproduce

Requirements: `curl`, `unzip`, `awk`, Python, and
[uv](https://docs.astral.sh/uv/).

The SEC asks automated clients to send an identifying user agent:

```sh
export SEC_USER_AGENT="Your Name your.email@example.com"
./scripts/fetch_history.sh
```

Then run the analyses in order:

```sh
uv run --with pandas --with numpy --with matplotlib python analysis/build_snapshot.py
uv run --with pandas --with numpy --with matplotlib python analysis/build_history.py
uv run --with pandas --with numpy --with matplotlib python analysis/build_sponsors.py
```

The scripts write their results to `analysis/out/`. The essay's charts are
hand-coded HTML/CSS and inline SVG, with values taken from those outputs.

## Data and method

The analysis joins SEC series IDs across:

- [Form N-CEN data sets](https://www.sec.gov/data-research/sec-markets-data/form-n-cen-data-sets)
  for ETF status, index status, first filings, and fiscal-year average assets.
- [Mutual Fund Prospectus Risk/Return Summary data sets](https://www.sec.gov/data-research/sec-markets-data/mutual-fund-prospectus-riskreturn-summary-data-sets)
  for prospectus expense ratios.

Key choices:

- The universe is US '40 Act ETFs. UIT ETFs such as SPY, QQQ, and DIA, plus
  '33 Act commodity and crypto trusts, are outside the fund-level panel.
- A fund's investor fee is its net expense ratio where a waiver is reported;
  otherwise it is the gross expense ratio.
- The series-level fee is the median across values in the latest filing, which
  avoids choosing an arbitrary share class.
- Counts use trailing-four-quarter N-CEN windows because the census is annual
  and filings are seasonal.
- Organic launches use first N-CEN filings from 2020 onward and exclude funds
  arriving with more than $500 million of first-year assets, a proxy for
  conversions, adopted funds, and late first filings.

Important limitation: Vanguard ETF series share classes are blended with their
mutual-fund classes in the available series-level asset figures. The essay
states this and the other material caveats in full.
