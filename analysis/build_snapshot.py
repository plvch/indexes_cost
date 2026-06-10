"""First-pass snapshot of the US '40 Act ETF universe: AUM and net-fee distributions.

Inputs (data/raw/): N-CEN FUND_REPORTED_INFO + REGISTRANT for the trailing four
quarters (2025Q2-2026Q1 = one full annual census), RR num.tsv for the trailing
nine quarters (2024Q1-2026Q1; funds refresh prospectuses annually, the extra
quarters cover late filers -- only the latest filing per series is used).
Outputs (analysis/out/): etf_snapshot.csv, summary CSVs, report.html.
"""

import base64
import io
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

NCEN_QS = ["2025q2", "2025q3", "2025q4", "2026q1"]  # chronological
RR_QS = ["2024q1_rr1", "2024q2_rr1", "2024q3_rr1", "2024q4_rr1",
         "2025q1_rr1", "2025q2_rr1", "2025q3_rr1", "2025q4_rr1", "2026q1_rr1"]
FEE_TAGS = {
    "NetExpensesOverAssets": "net_er",
    "ExpensesOverAssets": "gross_er",
    "ManagementFeesOverAssets": "mgmt_fee",
}

# ---------------------------------------------------------------- N-CEN panel
frames = []
for q in NCEN_QS:
    matches = sorted(RAW.glob(f"{q}_ncen*"))
    if not matches:
        raise FileNotFoundError(f"Missing N-CEN data for {q} under {RAW}")
    d = matches[0]
    f = pd.read_csv(
        d / "FUND_REPORTED_INFO.tsv", sep="\t", low_memory=False,
        usecols=["ACCESSION_NUMBER", "SERIES_ID", "FUND_NAME", "IS_ETF", "IS_ETMF",
                 "IS_INDEX", "IS_FIRST_FILING", "MONTHLY_AVG_NET_ASSETS"],
    )
    reg = pd.read_csv(d / "REGISTRANT.tsv", sep="\t", low_memory=False,
                      usecols=["ACCESSION_NUMBER", "REGISTRANT_NAME"])
    f = f.merge(reg, on="ACCESSION_NUMBER", how="left")
    f["quarter"] = q
    frames.append(f)

ncen = pd.concat(frames)
ncen = ncen[ncen["IS_ETF"] == "Y"].dropna(subset=["SERIES_ID"])
# keep the latest filing per series; within a quarter, the highest accession
# number is the latest (amended) filing
ncen = (ncen.sort_values(["quarter", "ACCESSION_NUMBER"])
        .drop_duplicates("SERIES_ID", keep="last"))
ncen["is_active"] = ncen["IS_INDEX"] != "Y"
ncen["aum"] = pd.to_numeric(ncen["MONTHLY_AVG_NET_ASSETS"], errors="coerce")
ncen.loc[ncen["aum"] <= 0, "aum"] = np.nan

# ------------------------------------------------------------------- RR fees
fee_rows = []
for q in RR_QS:
    n = pd.read_csv(
        RAW / q / "num.tsv", sep="\t", low_memory=False,
        usecols=["adsh", "tag", "ddate", "series", "otherdims", "value"],
        dtype={"ddate": str},
    )
    n = n[n["tag"].isin(FEE_TAGS) & n["series"].notna()]
    n["value"] = pd.to_numeric(n["value"], errors="coerce")
    n = n.dropna(subset=["value"])
    n["class_id"] = n["otherdims"].str.extract(r"Class=(C\d+)", flags=re.I)[0].fillna("")
    fee_rows.append(n)

fees = pd.concat(fee_rows)
# keep only the latest filing per series/tag, then take the median across that
# filing's values: multi-class funds (e.g. Vanguard) tag one value per class,
# often without a Class dimension, so collapsing to a single row picks an
# arbitrary class -- the median is the robust series-level figure
fees["filing"] = fees["ddate"].fillna("") + "_" + fees["adsh"]
latest = fees.groupby(["series", "tag"])["filing"].transform("max")
per_series = (fees[fees["filing"] == latest]
              .groupby(["series", "tag"])["value"].median()
              .unstack().rename(columns=FEE_TAGS))

panel = ncen.merge(per_series, left_on="SERIES_ID", right_index=True, how="left")
# NetExpensesOverAssets is only tagged when a fund has a fee-waiver line;
# no-waiver funds report only ExpensesOverAssets (gross == net for them)
panel["has_waiver"] = panel["net_er"].notna() & panel["gross_er"].notna()
panel["waiver_gap"] = (panel["gross_er"] - panel["net_er"]).where(panel["has_waiver"], 0.0)
panel["net_er"] = panel["net_er"].fillna(panel["gross_er"])
# fee sanity: drop net ER > 5% as data errors (count them first)
n_fee_outliers = int((panel["net_er"] > 0.05).sum())
panel.loc[panel["net_er"] > 0.05, ["net_er", "gross_er", "mgmt_fee", "waiver_gap"]] = np.nan

BRANDS = [  # registrant (legal trust) -> sponsor brand
    (r"VANGUARD", "Vanguard"), (r"ISHARES|BLACKROCK", "BlackRock (iShares)"),
    (r"SPDR|SSGA|SELECT SECTOR", "State Street (SPDR)"), (r"INVESCO", "Invesco"),
    (r"SCHWAB", "Schwab"), (r"FIDELITY", "Fidelity"), (r"FIRST TRUST", "First Trust"),
    (r"JPMORGAN|J\.P\. MORGAN", "JPMorgan"), (r"DIMENSIONAL", "Dimensional"),
    (r"PROSHARES", "ProShares"), (r"DIREXION", "Direxion"),
    (r"WISDOMTREE", "WisdomTree"), (r"VANECK", "VanEck"),
    (r"GLOBAL X", "Global X"), (r"PACER", "Pacer"), (r"CAPITAL GROUP", "Capital Group"),
    (r"AVANTIS|AMERICAN CENTURY", "American Century (Avantis)"),
    (r"GRANITESHARES", "GraniteShares"), (r"YIELDMAX|TIDAL", "Tidal (YieldMax)"),
]


def brand(reg):
    if pd.isna(reg):
        return "(unknown)"
    for pat, b in BRANDS:
        if re.search(pat, reg, re.I):
            return b
    return reg.title()


panel["sponsor"] = panel["REGISTRANT_NAME"].map(brand)

cols = ["SERIES_ID", "FUND_NAME", "REGISTRANT_NAME", "sponsor", "quarter",
        "is_active", "IS_FIRST_FILING", "aum", "net_er", "gross_er", "mgmt_fee",
        "waiver_gap", "has_waiver"]
panel[cols].to_csv(OUT / "etf_snapshot.csv", index=False)

# ------------------------------------------------------------------- checks
both = panel.dropna(subset=["aum", "net_er"])
aw_er = float(np.average(both["net_er"], weights=both["aum"]))
checks = {
    "ETF series in census (TTM)": len(panel),
    "  active / passive": f"{int(panel['is_active'].sum())} / {int((~panel['is_active']).sum())}",
    "AUM coverage": f"{panel['aum'].notna().mean():.1%}",
    "fee join coverage": f"{panel['net_er'].notna().mean():.1%}",
    "fee outliers dropped (net ER >5%)": n_fee_outliers,
    "funds with fee waiver line": f"{panel['has_waiver'].mean():.1%}",
    "total AUM ($T, fiscal-yr avg)": f"{panel['aum'].sum() / 1e12:.2f}",
    "asset-weighted net ER": f"{aw_er:.4%}  (Morningstar benchmark ~0.15-0.16%)",
    "equal-weighted net ER": f"{panel['net_er'].mean():.4%}",
    "median net ER": f"{panel['net_er'].median():.4%}",
}

# --------------------------------------------------------------- aggregates
BUCKETS = [0, 0.001, 0.002, 0.005, 0.01, np.inf]
BUCKET_LABELS = ["<0.10%", "0.10-0.20%", "0.20-0.50%", "0.50-1.00%", ">1.00%"]
both = panel.dropna(subset=["aum", "net_er"]).copy()
both["bucket"] = pd.cut(both["net_er"], BUCKETS, labels=BUCKET_LABELS, right=False)
both["revenue"] = both["aum"] * both["net_er"]
bucket_tbl = both.groupby("bucket", observed=True).agg(
    n_funds=("SERIES_ID", "count"), aum=("aum", "sum"), revenue=("revenue", "sum"))
bucket_tbl["count_share"] = bucket_tbl["n_funds"] / bucket_tbl["n_funds"].sum()
bucket_tbl["aum_share"] = bucket_tbl["aum"] / bucket_tbl["aum"].sum()
bucket_tbl["revenue_share"] = bucket_tbl["revenue"] / bucket_tbl["revenue"].sum()
bucket_tbl.to_csv(OUT / "fee_buckets.csv")

ap = panel.groupby("is_active").agg(
    n=("SERIES_ID", "count"), aum=("aum", "sum"),
    ew_er=("net_er", "mean"), med_er=("net_er", "median"))
ap["aw_er"] = [
    np.average(g.dropna(subset=["aum", "net_er"])["net_er"],
               weights=g.dropna(subset=["aum", "net_er"])["aum"])
    for _, g in panel.groupby("is_active")]
ap.index = ["passive", "active"]
ap.to_csv(OUT / "active_passive.csv")

aum_sorted = both.sort_values("aum", ascending=False)
total_aum = aum_sorted["aum"].sum()
conc = {
    "top 10 funds": aum_sorted["aum"].head(10).sum() / total_aum,
    "top 1% of funds": aum_sorted["aum"].head(max(1, len(aum_sorted) // 100)).sum() / total_aum,
    "top 10% of funds": aum_sorted["aum"].head(len(aum_sorted) // 10).sum() / total_aum,
    "funds under $100M (count share)": (both["aum"] < 1e8).mean(),
    "funds under $100M (AUM share)": both.loc[both["aum"] < 1e8, "aum"].sum() / total_aum,
    "funds under $50M (count share)": (both["aum"] < 5e7).mean(),
}

fam = (panel.dropna(subset=["aum"]).groupby("sponsor")
       .agg(n=("SERIES_ID", "count"), aum=("aum", "sum"), ew_er=("net_er", "mean"))
       .sort_values("aum", ascending=False))
fam.to_csv(OUT / "families.csv")

top15 = aum_sorted.head(15)[["FUND_NAME", "REGISTRANT_NAME", "aum", "net_er", "is_active"]]

newf = panel[panel["IS_FIRST_FILING"] == "Y"]
oldf = panel[panel["IS_FIRST_FILING"] != "Y"]
vintage = pd.DataFrame({
    "n": [len(oldf), len(newf)],
    "active share": [oldf["is_active"].mean(), newf["is_active"].mean()],
    "median net ER": [oldf["net_er"].median(), newf["net_er"].median()],
    "median AUM ($M)": [oldf["aum"].median() / 1e6, newf["aum"].median() / 1e6],
}, index=["incumbents", "first N-CEN filing (new funds)"])
vintage.to_csv(OUT / "new_vs_incumbent.csv")

# ------------------------------------------------------------------- charts
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
ACT, PAS = "#d1495b", "#3a6ea5"


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# 1. AUM histogram (log scale), active vs passive
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.logspace(5, 12, 60)
ax.hist(both.loc[~both["is_active"], "aum"], bins=bins, alpha=0.7, label="passive", color=PAS)
ax.hist(both.loc[both["is_active"], "aum"], bins=bins, alpha=0.7, label="active", color=ACT)
ax.set_xscale("log")
ax.axvline(1e8, color="gray", ls="--", lw=1)
ax.text(1.1e8, ax.get_ylim()[1] * 0.9, "$100M\n(~breakeven)", fontsize=8, color="gray")
ax.set_xlabel("fund AUM, $ (log scale)")
ax.set_ylabel("number of ETFs")
ax.legend()
img_aum = b64(fig)

# 2. Lorenz curve
fig, ax = plt.subplots(figsize=(5, 4.5))
v = np.sort(both["aum"].values)
cum = np.insert(np.cumsum(v) / v.sum(), 0, 0)
x = np.linspace(0, 1, len(cum))
ax.plot(x, cum, color=PAS)
ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
gini = 1 - 2 * np.trapezoid(cum, x)
ax.set_xlabel("cumulative share of funds (smallest first)")
ax.set_ylabel("cumulative share of AUM")
ax.set_title(f"AUM concentration - Gini {gini:.2f}")
img_lorenz = b64(fig)

# 3. Fee histogram, active vs passive
fig, ax = plt.subplots(figsize=(8, 4))
fbins = np.arange(0, 0.0205, 0.0005)
ax.hist(panel.loc[~panel["is_active"], "net_er"].dropna(), bins=fbins, alpha=0.7,
        label="passive", color=PAS)
ax.hist(panel.loc[panel["is_active"], "net_er"].dropna(), bins=fbins, alpha=0.7,
        label="active", color=ACT)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
ax.axvline(aw_er, color="black", ls="--", lw=1)
ax.text(aw_er + 0.0002, ax.get_ylim()[1] * 0.9,
        f"asset-weighted avg {aw_er:.2%}", fontsize=8)
ax.set_xlabel("net expense ratio")
ax.set_ylabel("number of ETFs")
ax.legend()
img_fees = b64(fig)

# 4. Centerpiece: count vs AUM vs revenue share by fee bucket
fig, ax = plt.subplots(figsize=(8, 4))
xpos = np.arange(len(bucket_tbl))
w = 0.27
ax.bar(xpos - w, bucket_tbl["count_share"], w, label="share of fund count", color="#888")
ax.bar(xpos, bucket_tbl["aum_share"], w, label="share of AUM", color=PAS)
ax.bar(xpos + w, bucket_tbl["revenue_share"], w, label="share of fee revenue", color=ACT)
ax.set_xticks(xpos, bucket_tbl.index)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.set_xlabel("net expense ratio bucket")
ax.legend()
img_buckets = b64(fig)

# 5. Median fee by AUM decile
fig, ax = plt.subplots(figsize=(8, 3.5))
both["decile"] = pd.qcut(both["aum"], 10, labels=False) + 1
dec = both.groupby("decile")["net_er"].median()
ax.bar(dec.index, dec.values, color=PAS)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
ax.set_xticks(range(1, 11))
ax.set_xlabel("AUM decile (1 = smallest funds, 10 = largest)")
ax.set_ylabel("median net ER")
img_decile = b64(fig)

# --------------------------------------------------------------------- html
def tbl(df, fmt=None):
    return df.to_html(border=0, float_format=(fmt or "{:,.2f}").format,
                      classes="t", escape=True)


top15_fmt = top15.copy()
top15_fmt["aum"] = (top15_fmt["aum"] / 1e9).map("{:,.0f}".format) + " B"
top15_fmt["net_er"] = top15_fmt["net_er"].map(lambda v: f"{v:.2%}" if pd.notna(v) else "n/a")
top15_fmt["is_active"] = top15_fmt["is_active"].map({True: "active", False: "passive"})

bucket_fmt = bucket_tbl.copy()
for c in ["count_share", "aum_share", "revenue_share"]:
    bucket_fmt[c] = bucket_fmt[c].map("{:.1%}".format)
bucket_fmt["aum"] = (bucket_fmt["aum"] / 1e9).map("{:,.0f} B".format)
bucket_fmt["revenue"] = (bucket_fmt["revenue"] / 1e9).map("{:,.2f} B".format)

ap_fmt = ap.copy()
for c in ["ew_er", "med_er", "aw_er"]:
    ap_fmt[c] = ap_fmt[c].map("{:.2%}".format)
ap_fmt["aum"] = (ap_fmt["aum"] / 1e12).map("{:,.2f} T".format)

vint_fmt = vintage.copy()
vint_fmt["active share"] = vint_fmt["active share"].map("{:.0%}".format)
vint_fmt["median net ER"] = vint_fmt["median net ER"].map("{:.2%}".format)
vint_fmt["median AUM ($M)"] = vint_fmt["median AUM ($M)"].map("{:,.0f}".format)

fam_fmt = fam.head(15).copy()
fam_fmt["aum"] = (fam_fmt["aum"] / 1e9).map("{:,.0f} B".format)
fam_fmt["ew_er"] = fam_fmt["ew_er"].map(lambda v: f"{v:.2%}" if pd.notna(v) else "n/a")

checks_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in checks.items())
conc_html = "".join(f"<tr><td>{k}</td><td>{v:.1%}</td></tr>" for k, v in conc.items())
rev_total = both["revenue"].sum()

html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>US ETF universe snapshot - AUM and fees</title>
<style>
 body{{font:15px/1.55 -apple-system,Segoe UI,sans-serif;max-width:880px;margin:2em auto;
      padding:0 1em;color:#222}}
 h1{{font-size:1.5em}} h2{{font-size:1.15em;margin-top:2em;border-bottom:1px solid #ddd;
      padding-bottom:.2em}}
 table.t{{border-collapse:collapse;font-size:13px;margin:.8em 0}}
 .t td,.t th{{padding:3px 10px;text-align:right;border-bottom:1px solid #eee}}
 .t th{{text-align:right;background:#f7f7f7}} .t td:first-child,.t th:first-child{{text-align:left}}
 img{{max-width:100%}} .note{{color:#666;font-size:13px}}
 .big{{font-size:1.05em;background:#f4f7fb;border-left:3px solid #3a6ea5;padding:.7em 1em;
      margin:1em 0}}
</style></head><body>
<h1>US '40 Act ETF universe: AUM &amp; fee snapshot</h1>
<p class="note">Built from raw SEC filings: N-CEN census (trailing 4 quarters, 2025Q2-2026Q1)
joined with prospectus Risk/Return XBRL fee tables (latest filing per series,
2024Q1-2026Q1) on SEC series ID. Net fee = NetExpensesOverAssets where a waiver line
exists, else ExpensesOverAssets.
AUM = fiscal-year average net assets as reported on N-CEN. Excludes non-'40 Act ETPs
(crypto/commodity trusts, ETNs). Generated 2026-06-09.</p>

<div class="big"><b>{len(panel):,} ETFs</b> (active {panel['is_active'].mean():.0%} of count) ·
<b>${panel['aum'].sum() / 1e12:,.2f}T</b> AUM ·
asset-weighted net fee <b>{aw_er:.2%}</b> vs equal-weighted <b>{panel['net_er'].mean():.2%}</b> ·
implied annual fee revenue <b>${rev_total / 1e9:,.1f}B</b></div>

<h2>Validation &amp; coverage checks</h2>
<table class="t">{checks_html}</table>

<h2>AUM distribution</h2>
<img src="data:image/png;base64,{img_aum}">
<img src="data:image/png;base64,{img_lorenz}">
<table class="t"><tr><th>concentration</th><th>value</th></tr>{conc_html}</table>

<h2>Net fee distribution</h2>
<img src="data:image/png;base64,{img_fees}">
<h2>The centerpiece: who holds the funds vs who pays the fees</h2>
<img src="data:image/png;base64,{img_buckets}">
{tbl(bucket_fmt)}

<h2>Active vs passive</h2>
{tbl(ap_fmt)}

<h2>Fee vs fund size</h2>
<img src="data:image/png;base64,{img_decile}">

<h2>New funds vs incumbents</h2>
<p class="note">"New" = fund's first-ever N-CEN filing fell in the trailing census
(roughly: launched within the last ~2 years).</p>
{tbl(vint_fmt)}

<h2>Top 15 ETFs by AUM</h2>
{tbl(top15_fmt)}
<h2>Top 15 sponsors by AUM</h2>
{tbl(fam_fmt)}

<h2>Caveats</h2>
<ul class="note">
<li>AUM is the fiscal-year <i>average</i> from N-CEN, not point-in-time; totals lag a
rising market vs year-end figures.</li>
<li>Universe is '40 Act ETFs by SEC series; headline "ETF counts" in the press include
~300-400 exchange-traded products (crypto/commodity trusts, ETNs) absent here.</li>
<li>ETFs organized as unit investment trusts file N-CEN without fund-level detail and are
<b>excluded</b>: SPY (~$660B, 0.0945%), QQQ (~$380B, 0.20%), DIA (~$40B, 0.16%) -
roughly $1.1T of additional AUM, all cheap and passive, so including them would push the
asset-weighted fee and concentration slightly further in the directions shown.</li>
<li>Vanguard-style series include mutual-fund share classes; their AUM (series-level) and
fee (median across classes) blend the ETF class with the others.</li>
<li>Fees joined for the share of funds shown in coverage checks; newest launches may not
have a structured RR filing yet.</li>
<li>{n_fee_outliers} funds with reported net ER &gt; 5% treated as tagging errors and
excluded from fee stats.</li>
<li>Multi-class series use the median fee across the latest filing's class values
(class attribution is impossible when filers omit the Class dimension).</li>
</ul>
</body></html>"""

(OUT / "report.html").write_text(html)

print("--- checks ---")
for k, v in checks.items():
    print(f"{k}: {v}")
print("\n--- concentration ---")
for k, v in conc.items():
    print(f"{k}: {v:.1%}")
print("\n--- fee buckets ---")
print(bucket_tbl[["n_funds", "count_share", "aum_share", "revenue_share"]].round(3))
print("\n--- active/passive ---")
print(ap.round(4))
print("\n--- new vs incumbents ---")
print(vintage.round(4))
print(f"\ntotal implied fee revenue: ${rev_total / 1e9:,.1f}B/yr")
print(f"wrote {OUT / 'report.html'} and 5 CSVs")
