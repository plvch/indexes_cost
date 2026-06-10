"""History panel: ETF counts, launches, AUM and fees over time, from raw SEC filings.

N-CEN 2018Q3-2026Q1 (flags, AUM, launches) + RR fee tables 2010Q4-2026Q1.
Outputs: analysis/out/etf_fund_year.csv, time-series CSVs, history.html.
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
FEE_TAGS = {"NetExpensesOverAssets": "net_er", "ExpensesOverAssets": "gross_er"}

# ------------------------------------------------------------- N-CEN history
frames = []
for d in sorted(RAW.glob("*_ncen*")):
    q = d.name[:6]
    f = pd.read_csv(d / "FUND_REPORTED_INFO.tsv", sep="\t", low_memory=False,
                    usecols=lambda c: c in {"ACCESSION_NUMBER", "SERIES_ID",
                                            "FUND_NAME", "IS_ETF",
                                            "IS_INDEX", "IS_FIRST_FILING",
                                            "IS_MULTI_INVERSE_INDEX",
                                            "MONTHLY_AVG_NET_ASSETS"})
    f["quarter"] = q
    frames.append(f)
ncen = pd.concat(frames).dropna(subset=["SERIES_ID"])
ncen = ncen[ncen["IS_ETF"] == "Y"]
ncen["year"] = ncen["quarter"].str[:4].astype(int)
ncen["qnum"] = ncen["quarter"].str[5].astype(int)
ncen["aum"] = pd.to_numeric(ncen["MONTHLY_AVG_NET_ASSETS"], errors="coerce")
ncen["is_active"] = ncen["IS_INDEX"] != "Y"
ncen["is_levinv"] = ncen["IS_MULTI_INVERSE_INDEX"] == "Y"
# canonical row per series-quarter: amended/duplicate filings exist within a
# quarter (~1.1k series-quarter dupes); keep the highest accession = latest filing
ncen = (ncen.sort_values(["quarter", "ACCESSION_NUMBER"])
        .drop_duplicates(["SERIES_ID", "quarter"], keep="last"))

etf_series = set(ncen["SERIES_ID"])

# trailing-4-quarter census count, evaluated at each quarter
quarters = sorted(ncen["quarter"].unique())
census = []
for i in range(3, len(quarters)):
    win = quarters[i - 3:i + 1]
    w = ncen[ncen["quarter"].isin(win)].drop_duplicates("SERIES_ID", keep="last")
    act_aum = w.loc[w["is_active"], "aum"].sum()
    census.append({"quarter": quarters[i], "n_etf": len(w),
                   "n_active": int(w["is_active"].sum()),
                   "n_levinv": int(w["is_levinv"].sum()),
                   "aum_T": w["aum"].sum() / 1e12,
                   "aum_act_B": act_aum / 1e9,
                   "act_aum_share": act_aum / w["aum"].sum(),
                   "act_cnt_share": w["is_active"].mean()})
census = pd.DataFrame(census)
census.to_csv(OUT / "census_by_quarter.csv", index=False)
census[["quarter", "aum_act_B", "act_aum_share", "act_cnt_share"]].to_csv(
    OUT / "aum_split.csv", index=False)

# launches: first-ever N-CEN filing of a series. A first filing with a large
# first-fiscal-year AUM is a conversion / adopted fund / late-filing trust, not
# an organic launch (same rule as build_sponsors.py)
first = ncen.sort_values("quarter").drop_duplicates("SERIES_ID", keep="first")
first["launch_year"] = first["year"]
first["is_transfer"] = first["aum"] > 5e8
# 2018-2019 filings include the pre-existing stock; clean launch years start 2020
launches = first[(first["launch_year"] >= 2020) & ~first["is_transfer"]]
launch_mix = launches.groupby("launch_year").agg(
    n=("SERIES_ID", "count"), active_share=("is_active", "mean"),
    levinv_share=("is_levinv", "mean"))
launch_mix.to_csv(OUT / "launch_mix.csv")

# closures: series whose last filing is >5 quarters before the latest quarter
last = ncen.sort_values("quarter").drop_duplicates("SERIES_ID", keep="last")
cutoff = quarters[-6]
closed = last[last["quarter"] < cutoff]
closures = closed.groupby("year")["SERIES_ID"].count()
closures.rename("n_closed").to_csv(OUT / "closures_by_year.csv")

# --------------------------------------------------------------- RR history
fee_rows = []
for d in sorted(RAW.glob("*_rr1")):
    n = pd.read_csv(d / "num.tsv", sep="\t", low_memory=False,
                    usecols=["adsh", "tag", "ddate", "series", "value"],
                    dtype={"ddate": str})
    n = n[n["tag"].isin(FEE_TAGS) & n["series"].notna()]
    n["value"] = pd.to_numeric(n["value"], errors="coerce")
    n = n.dropna(subset=["value"])
    fee_rows.append(n)
fees = pd.concat(fee_rows)
fees["year"] = fees["ddate"].str[:4].astype(float)
fees["filing"] = fees["ddate"].fillna("") + "_" + fees["adsh"]
# per series-year-tag: median across the latest filing's class values
latest = fees.groupby(["series", "year", "tag"])["filing"].transform("max")
fy = (fees[fees["filing"] == latest]
      .groupby(["series", "year", "tag"])["value"].median()
      .unstack().rename(columns=FEE_TAGS).reset_index())
fy["fee"] = fy["net_er"].fillna(fy["gross_er"])
fy.loc[fy["fee"] > 0.05, "fee"] = np.nan
fy = fy.dropna(subset=["fee"])
fy["is_etf_ever"] = fy["series"].isin(etf_series)

# ETF-only fund-year panel with AUM + flags joined (2018+).
# A series can file in two quarters of one calendar year -> dedupe to the
# latest filing per series-year, else the join expands fund-year rows
flags = (ncen.sort_values(["quarter", "ACCESSION_NUMBER"])
         .drop_duplicates(["SERIES_ID", "year"], keep="last")
         .set_index(["SERIES_ID", "year"])[["aum", "is_active", "is_levinv"]])
assert flags.index.is_unique
panel = (fy[fy["is_etf_ever"]]
         .join(flags, on=["series", "year"]))
panel.to_csv(OUT / "etf_fund_year.csv", index=False)

# EW vs AW fee by year (ETF-ever set; AW only where AUM known, 2018+)
def yearly(g):
    aw = np.nan
    h = g.dropna(subset=["aum"])
    if h["aum"].sum() > 0:
        aw = np.average(h["fee"], weights=h["aum"])
    return pd.Series({"n": len(g), "ew": g["fee"].mean(),
                      "median": g["fee"].median(), "aw": aw})

trend = panel.groupby("year").apply(yearly, include_groups=False)
trend = trend[(trend.index >= 2011) & (trend.index <= 2026)]
trend.to_csv(OUT / "fee_trend.csv")

# launch-vintage fee: first fee observation per series, by its first-obs year
fy_etf = fy[fy["is_etf_ever"]].sort_values("year")
vint = fy_etf.drop_duplicates("series", keep="first")
vint = vint[(vint["year"] >= 2011) & (vint["year"] <= 2025)]
vint = vint.join(first.set_index("SERIES_ID")["is_active"], on="series")
vintage = vint.groupby("year").agg(n=("series", "count"), med_fee=("fee", "median"))
vintage["med_fee_active"] = vint[vint["is_active"] == True].groupby("year")["fee"].median()
vintage["med_fee_passive"] = vint[vint["is_active"] == False].groupby("year")["fee"].median()
vintage.to_csv(OUT / "vintage_fees.csv")

# ------------------------------------------------------------- zombie pairs
PAIRS = [("EEM/IEMG", "iShares MSCI Emerging Markets ETF", "iShares Core MSCI Emerging Markets ETF"),
         ("EFA/IEFA", "iShares MSCI EAFE ETF", "iShares Core MSCI EAFE ETF")]
zrows = []
name_to_sid = (ncen.drop_duplicates("SERIES_ID", keep="last")
               .set_index("FUND_NAME")["SERIES_ID"].to_dict())
for label, zn, tn in PAIRS:
    for leg, nm in [("zombie", zn), ("twin", tn)]:
        sid = name_to_sid.get(nm)
        if sid is None:
            continue
        a = ncen[ncen["SERIES_ID"] == sid].groupby("year")["aum"].last()
        fz = panel[panel["series"] == sid].groupby("year")["fee"].last()
        for y in a.index:
            zrows.append({"pair": label, "leg": leg, "fund": nm, "year": y,
                          "aum_B": a[y] / 1e9, "fee": fz.get(y, np.nan)})
zombie = pd.DataFrame(zrows)
zombie.to_csv(OUT / "zombie_pairs.csv", index=False)

# ------------------------------------------------------------------- charts
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
ACT, PAS = "#d1495b", "#3a6ea5"


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# 1. census count + AUM
fig, ax = plt.subplots(figsize=(8, 4))
x = range(len(census))
ax.plot(x, census["n_etf"], color=PAS, label="ETF count (census)")
ax.plot(x, census["n_active"], color=ACT, label="of which active")
ax.set_xticks(list(x)[::4], census["quarter"][::4], rotation=45)
ax.set_ylabel("number of ETFs")
ax2 = ax.twinx()
ax2.plot(x, census["aum_T"], color="gray", ls=":", label="AUM $T (right)")
ax2.set_ylabel("AUM, $T")
ax.legend(loc="upper left"); ax2.legend(loc="lower right")
img_census = b64(fig)

# 2. EW vs AW fee
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(trend.index, trend["ew"], color=ACT, label="equal-weighted")
ax.plot(trend.index, trend["median"], color=ACT, ls="--", label="median")
ax.plot(trend.index, trend["aw"], color=PAS, label="asset-weighted (2018+)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
ax.set_ylabel("net expense ratio"); ax.legend()
img_trend = b64(fig)

# 3. vintage fees
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(vintage.index, vintage["med_fee"], color="black", label="all launches")
ax.plot(vintage.index, vintage["med_fee_active"], color=ACT, label="active")
ax.plot(vintage.index, vintage["med_fee_passive"], color=PAS, label="passive")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
ax.set_ylabel("median fee of new funds"); ax.set_xlabel("first-fee-filing vintage")
ax.legend()
img_vintage = b64(fig)

# 4. launch mix
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(launch_mix.index, launch_mix["n"], color="#bbb", label="launches (first N-CEN)")
ax.set_ylabel("new ETF series")
ax2 = ax.twinx()
ax2.plot(launch_mix.index, launch_mix["active_share"], color=ACT, marker="o",
         label="active share")
ax2.plot(launch_mix.index, launch_mix["levinv_share"], color=PAS, marker="s",
         label="leveraged/inverse share")
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax2.set_ylim(0, 1)
ax.legend(loc="upper left"); ax2.legend(loc="center left")
img_mix = b64(fig)

# 5. zombie pairs
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)
for ax, (label, _, _) in zip(axes, PAIRS):
    zp = zombie[zombie["pair"] == label]
    for leg, color in [("zombie", ACT), ("twin", PAS)]:
        d = zp[zp["leg"] == leg]
        ax.plot(d["year"], d["aum_B"], color=color, marker="o", label=leg)
        for _, r in d.iterrows():
            if not np.isnan(r["fee"]) and r["year"] in (d["year"].min(), d["year"].max()):
                ax.annotate(f"{r['fee']:.2%}", (r["year"], r["aum_B"]), fontsize=7,
                            textcoords="offset points", xytext=(0, 6), color=color)
    ax.set_title(label); ax.set_ylabel("AUM, $B"); ax.legend(fontsize=8)
img_zombie = b64(fig)

# --------------------------------------------------------------------- html
def tbl(df, fmt="{:,.2f}"):
    return df.to_html(border=0, float_format=fmt.format, classes="t")


html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>US ETF universe - history from raw SEC filings</title>
<style>
 body{{font:15px/1.55 -apple-system,Segoe UI,sans-serif;max-width:880px;margin:2em auto;
      padding:0 1em;color:#222}}
 h2{{font-size:1.15em;margin-top:2em;border-bottom:1px solid #ddd;padding-bottom:.2em}}
 table.t{{border-collapse:collapse;font-size:13px;margin:.8em 0}}
 .t td,.t th{{padding:3px 10px;text-align:right;border-bottom:1px solid #eee}}
 .t th{{background:#f7f7f7}} .t td:first-child{{text-align:left}}
 img{{max-width:100%}} .note{{color:#666;font-size:13px}}
</style></head><body>
<h1>US '40 Act ETF universe: history</h1>
<p class="note">N-CEN census 2018Q3-2026Q1 (counts, AUM, flags) + Risk/Return XBRL fee
tables 2010Q4-2026Q1. Counts use trailing-4-quarter windows. Launch years from first
N-CEN filing (clean from 2020). Pre-2018 fee history covers series that survived to
file an N-CEN (survivorship caveat). Generated 2026-06-09.</p>
<h2>Universe growth (census)</h2><img src="data:image/png;base64,{img_census}">
<h2>Equal- vs asset-weighted fee</h2><img src="data:image/png;base64,{img_trend}">
{tbl(trend.tail(8))}
<h2>Launch-vintage median fee</h2><img src="data:image/png;base64,{img_vintage}">
<h2>Launch mix</h2><img src="data:image/png;base64,{img_mix}">
{tbl(launch_mix, "{:.2f}")}
<h2>Zombie pairs (fee labels at curve ends)</h2><img src="data:image/png;base64,{img_zombie}">
</body></html>"""
(OUT / "history.html").write_text(html)

print("--- census (last 6 quarters) ---")
print(census.tail(6).to_string(index=False))
print("\n--- fee trend ---")
print(trend.round(4).to_string())
print("\n--- vintage fees ---")
print(vintage.round(4).to_string())
print("\n--- launch mix ---")
print(launch_mix.round(3).to_string())
print("\n--- zombie pairs (first/last year) ---")
print(zombie.groupby(["pair", "leg"]).agg(aum_first=("aum_B", "first"),
      aum_last=("aum_B", "last"), fee_last=("fee", "last")).round(2).to_string())
print(f"\nclosures by last-filing year:\n{closures.to_string()}")
print(f"\nwrote {OUT / 'history.html'}")
