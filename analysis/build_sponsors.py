"""Sponsor 'flight to fee' analysis: do the giants launch above their own book?

For each top sponsor: launches per year (N-CEN first filing, clean from 2020),
median fee of those launches (first RR fee observation), active share, and the
comparison to the sponsor's incumbent book (asset-weighted fee of the current
snapshot). Plus: what share of the sponsor's current AUM vs fee revenue comes
from post-2020 launches.

Inputs: data/raw/ N-CEN + RR quarters, analysis/out/etf_snapshot.csv (run
build_snapshot.py first).
Outputs: out/sponsor_launches.csv (sponsor x vintage), out/sponsor_flight.csv
(one row per sponsor), printed QA block.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent / "out"
FEE_TAGS = {"NetExpensesOverAssets": "net_er", "ExpensesOverAssets": "gross_er"}

BRANDS = [  # keep in sync with build_snapshot.py
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


# ------------------------------------------------------------- N-CEN history
frames = []
for d in sorted(RAW.glob("*_ncen*")):
    f = pd.read_csv(d / "FUND_REPORTED_INFO.tsv", sep="\t", low_memory=False,
                    usecols=lambda c: c in {"ACCESSION_NUMBER", "SERIES_ID",
                                            "FUND_NAME", "IS_ETF", "IS_INDEX",
                                            "IS_MULTI_INVERSE_INDEX",
                                            "MONTHLY_AVG_NET_ASSETS"})
    reg = pd.read_csv(d / "REGISTRANT.tsv", sep="\t", low_memory=False,
                      usecols=["ACCESSION_NUMBER", "REGISTRANT_NAME"])
    f = f.merge(reg, on="ACCESSION_NUMBER", how="left")
    f["quarter"] = d.name[:6]
    frames.append(f)
ncen = pd.concat(frames).dropna(subset=["SERIES_ID"])
ncen = ncen[ncen["IS_ETF"] == "Y"]
# canonical row per series-quarter: keep the latest accession (amended filings)
ncen = (ncen.sort_values(["quarter", "ACCESSION_NUMBER"])
        .drop_duplicates(["SERIES_ID", "quarter"], keep="last"))
ncen["year"] = ncen["quarter"].str[:4].astype(int)
ncen["is_active"] = ncen["IS_INDEX"] != "Y"
ncen["is_levinv"] = ncen["IS_MULTI_INVERSE_INDEX"] == "Y"

# launches: first-ever N-CEN filing of a series; clean launch years start 2020
# (2018-2019 filings absorb the pre-existing stock)
first = ncen.sort_values("quarter").drop_duplicates("SERIES_ID", keep="first")
first["sponsor"] = first["REGISTRANT_NAME"].map(brand)
first = first.rename(columns={"year": "launch_year"})
# a "launch" with a large first-fiscal-year AUM is not organic new product: it is a
# mutual-fund conversion (Dimensional, JPMorgan JREG, Fidelity Enhanced), an adopted
# fund, or a pre-existing fund whose trust filed its first N-CEN late (PIMCO MINT,
# Global X QYLD). Flag and exclude from launch stats; treat as incumbent.
first["aum0"] = pd.to_numeric(first["MONTHLY_AVG_NET_ASSETS"], errors="coerce")
first["is_transfer"] = first["aum0"] > 5e8
launches_all = first[first["launch_year"].between(2020, 2025)].copy()
launches = launches_all[~launches_all["is_transfer"]].copy()
n_transfers = int(launches_all["is_transfer"].sum())

# --------------------------------------------------------------- launch fees
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
latest = fees.groupby(["series", "year", "tag"])["filing"].transform("max")
fy = (fees[fees["filing"] == latest]
      .groupby(["series", "year", "tag"])["value"].median()
      .unstack().rename(columns=FEE_TAGS).reset_index())
fy["fee"] = fy["net_er"].fillna(fy["gross_er"])  # net only exists with a waiver line
fy.loc[fy["fee"] > 0.05, "fee"] = np.nan
fy = fy.dropna(subset=["fee"])

# launch fee = first fee observation per series, within [launch_year - 2,
# launch_year + 2]: the initial prospectus precedes the launch and the first
# N-CEN can lag it by over a year; fees older than that mean the fund
# pre-existed its first census filing
first_fee = fy.sort_values("year").drop_duplicates("series", keep="first")
launches = launches.merge(first_fee[["series", "year", "fee"]],
                          left_on="SERIES_ID", right_on="series", how="left")
stale = ((launches["year"] > launches["launch_year"] + 2)
         | (launches["year"] < launches["launch_year"] - 2))
launches.loc[stale, "fee"] = np.nan
fee_cov = launches["fee"].notna().mean()

# ------------------------------------------------- current book (snapshot)
snap = pd.read_csv(OUT / "etf_snapshot.csv")
snap = snap.dropna(subset=["aum"])
snap["revenue"] = snap["aum"] * snap["net_er"]
snap = snap.merge(first.set_index("SERIES_ID")[["launch_year", "is_transfer"]],
                  left_on="SERIES_ID", right_index=True, how="left")
# same window as the launch stats (2026 first-filers excluded on both sides)
snap["is_new"] = (snap["launch_year"].between(2020, 2025)
                  & ~snap["is_transfer"].fillna(False))

TOP = (snap.groupby("sponsor")["aum"].sum().sort_values(ascending=False)
       .head(10).index.tolist())

# ------------------------------------------------------------------ outputs
sl = (launches[launches["sponsor"].isin(TOP)]
      .groupby(["sponsor", "launch_year"])
      .agg(n=("SERIES_ID", "count"), med_launch_fee=("fee", "median"),
           active_share=("is_active", "mean"), levinv_share=("is_levinv", "mean"))
      .reset_index())
sl.to_csv(OUT / "sponsor_launches.csv", index=False)


def sponsor_row(s):
    book = snap[snap["sponsor"] == s]
    b = book.dropna(subset=["net_er"])
    aw = np.average(b["net_er"], weights=b["aum"]) if len(b) else np.nan
    inc = book[~book["is_new"].fillna(False)].dropna(subset=["net_er"])
    aw_inc = np.average(inc["net_er"], weights=inc["aum"]) if len(inc) else np.nan
    la = launches[launches["sponsor"] == s]
    new = book[book["is_new"].fillna(False)]
    return {
        "sponsor": s,
        "book_aum_B": book["aum"].sum() / 1e9,
        "book_aw_fee": aw,
        "incumbent_aw_fee": aw_inc,
        "n_launches_2020_25": len(la),
        "med_launch_fee": la["fee"].median(),
        "launch_active_share": la["is_active"].mean(),
        "launch_fee_multiple": la["fee"].median() / aw_inc if aw_inc else np.nan,
        "new_aum_share": new["aum"].sum() / book["aum"].sum(),
        "new_rev_share": new["revenue"].sum() / book["revenue"].sum(),
    }


flight = pd.DataFrame([sponsor_row(s) for s in TOP])
book = snap
b = book.dropna(subset=["net_er"])
mkt = {
    "sponsor": "ALL '40 Act ETFs",
    "book_aum_B": book["aum"].sum() / 1e9,
    "book_aw_fee": np.average(b["net_er"], weights=b["aum"]),
    "incumbent_aw_fee": np.nan,
    "n_launches_2020_25": len(launches),
    "med_launch_fee": launches["fee"].median(),
    "launch_active_share": launches["is_active"].mean(),
    "launch_fee_multiple": np.nan,
    "new_aum_share": book.loc[book["is_new"].fillna(False), "aum"].sum() / book["aum"].sum(),
    "new_rev_share": book.loc[book["is_new"].fillna(False), "revenue"].sum() / book["revenue"].sum(),
}
flight = pd.concat([flight, pd.DataFrame([mkt])], ignore_index=True)
flight.to_csv(OUT / "sponsor_flight.csv", index=False)

pd.DataFrame([{
    "n_organic_launches_2020_25": len(launches),
    "n_transfers_excluded": n_transfers,
    "launch_fee_coverage": fee_cov,
}]).to_csv(OUT / "sponsor_qa.csv", index=False)

# ----------------------------------------------------------------------- QA
print(f"organic launches 2020-2025: {len(launches):,} "
      f"(excluded {n_transfers} transfers/conversions/late-filers with first-year AUM > $500M)"
      f" | launch-fee coverage {fee_cov:.1%}")
print(f"top-10 sponsors by current AUM: {TOP}")
print("\n--- sponsor flight table ---")
fmt = flight.copy()
for c in ["book_aw_fee", "incumbent_aw_fee", "med_launch_fee"]:
    fmt[c] = fmt[c].map(lambda v: f"{v:.2%}" if pd.notna(v) else "-")
for c in ["launch_active_share", "new_aum_share", "new_rev_share"]:
    fmt[c] = fmt[c].map(lambda v: f"{v:.0%}" if pd.notna(v) else "-")
fmt["launch_fee_multiple"] = flight["launch_fee_multiple"].map(
    lambda v: f"{v:.1f}x" if pd.notna(v) else "-")
fmt["book_aum_B"] = fmt["book_aum_B"].map("{:,.0f}".format)
print(fmt.to_string(index=False))
print("\n--- launches per sponsor-year (top 10) ---")
print(sl.round(4).to_string(index=False))
