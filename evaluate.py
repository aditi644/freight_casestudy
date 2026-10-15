"""Eval harness. Run: python evaluate.py   (exit code 1 if anything fails)

Four independent checks:
  A. Number check    - recompute metrics with a slow, plain-Python loop and compare to the pipeline;
                       and compare against the 3 rows given in sample_output_format_v2.csv.
  B. Labeled set     - verdicts I labeled by hand from reading the 10 notes (labeled_set.csv),
                       written WITHOUT looking at the code's output rules.
  C. Fabrication audit - every 'justified' row: cited note exists, scope matches, window overlaps the
                       week, note describes a cost rise (re-checked by hand-coded facts, not notes.py).
  D. Adversarial tests - perturb the notes and confirm the system refuses to invent a reason.
"""
import sys
from datetime import date, timedelta

import pandas as pd

from run import build_output
from pipeline import load_shipments

SHIP = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/shipment_records.csv"
NOTES = sys.argv[2] if len(sys.argv) > 2 else "/mnt/user-data/uploads/context_notes.csv"
SAMPLE = sys.argv[3] if len(sys.argv) > 3 else "/mnt/user-data/uploads/sample_output_format_v2.csv"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


out = build_output(SHIP, NOTES)
notes = pd.read_csv(NOTES)

# ---- A. numbers ---------------------------------------------------------------------------------
d = load_shipments(SHIP)


def slow_metrics(route, week):
    """Plain loops, no groupby/rolling -- deliberately different code path from pipeline.py."""
    def cpt(r, wk):
        rows = [x for x in d[(d.route == r) & (d.week_of == wk)].itertuples()]
        return sum(x.freight_cost_inr for x in rows) / sum(x.quantity_tonnes * x.distance_km for x in rows)
    wk = pd.Timestamp(week)
    prior = [cpt(route, wk - timedelta(weeks=k)) for k in range(1, 9)
             if len(d[(d.route == route) & (d.week_of == wk - timedelta(weeks=k))])]
    rtype = d[d.route == route].route_type.iloc[0]
    others = [r for r in d[d.route_type == rtype].route.unique() if r != route]
    peer = sum(cpt(r, wk) for r in others) / len(others)
    own = cpt(route, wk)
    return own, (own / (sum(prior) / len(prior)) - 1) * 100, (own / peer - 1) * 100


bad = 0
for r in out.itertuples():
    own, op, pp = slow_metrics(r.route, r.week_of)
    got_op = float(r.vs_own_history.split("%")[0]); got_pp = float(r.vs_similar_routes.split("%")[0])
    if abs(own - r.cost_per_tonne_km) > 0.005 or abs(op - got_op) > 0.05 or abs(pp - got_pp) > 0.05:
        bad += 1
check("A1 slow recomputation matches pipeline on every output row", bad == 0, f"({len(out)} rows, {bad} mismatches)")

# The provided sample has an unquoted comma inside one `reason` cell, so strict CSV parsing fails;
# split on the first 7 commas only (the 8th field, `reason`, is free text).
with open(SAMPLE, encoding="utf-8") as fh:
    _lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
_cols = _lines[0].split(",")
sample = pd.DataFrame([ln.split(",", 7) for ln in _lines[1:]], columns=_cols)
sample["cost_per_tonne_km"] = sample["cost_per_tonne_km"].astype(float)
m = sample.merge(out, on=["route", "week_of"], suffixes=("_s", "_o"))
ok = len(m) == len(sample) and all(
    (m[c + "_s"] == m[c + "_o"]).all() for c in
    ["cost_per_tonne_km", "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id"]
    if c != "matched_note_id") and (m["matched_note_id_s"].fillna("") == m["matched_note_id_o"].fillna("")).all()
check("A2 the 3 sample-file rows reproduced exactly (numbers, flag, note id)", ok, f"({len(m)}/{len(sample)} found)")
check("A3 output columns/order match sample_output_format_v2.csv", list(out.columns) == list(sample.columns))

# ---- B. hand-labeled verdicts -------------------------------------------------------------------
lab = pd.read_csv("labeled_set.csv", dtype=str).fillna("")
m = lab.merge(out.fillna(""), on=["route", "week_of"], how="left", suffixes=("_lab", ""))
missing = m["flagged"].eq("").sum()
wrong = m[(m["flagged"] != m["expected_flagged"]) | (m["matched_note_id"] != m["expected_note"])]
check("B1 labeled set: verdict and note id correct on every labeled row", len(wrong) == 0 and missing == 0,
      f"({len(lab) - len(wrong)}/{len(lab)} correct, {missing} missing)")
if len(wrong):
    print(wrong[["route", "week_of", "expected_flagged", "flagged", "expected_note", "matched_note_id"]])

# ---- C. fabrication audit (hand-coded note facts, independent of notes.py) -----------------------
FACTS = {  # note_id: (route or None=all, first day, last day) -- ONLY the notes that report a cost rise
    "N001": ("Chennai-Bangalore", date(2025, 2, 24), date(2025, 3, 8)),
    "N002": ("Ahmedabad-Mumbai", date(2025, 1, 20), date(2025, 1, 26)),
    "N003": (None, date(2025, 5, 5), date(2025, 6, 29)),
}
bad = []
for r in out[out.flagged != "Yes"].itertuples():
    f = FACTS.get(r.matched_note_id)
    wk = pd.Timestamp(r.week_of).date()
    days = 0 if f is None else (min(f[2], wk + timedelta(days=6)) - max(f[1], wk)).days + 1
    if f is None or (f[0] not in (None, r.route)) or days < 4:
        bad.append((r.route, r.week_of, r.matched_note_id))
check("C1 every justified row cites a real note for the right route and time", not bad, f"(bad: {bad})")
check("C2 every unexplained row has an empty matched_note_id",
      (out[out.flagged == "Yes"].matched_note_id.fillna("") == "").all())
check("C3 no justified row cites a 'no impact' note (N004-N010)",
      not out[out.flagged != "Yes"].matched_note_id.isin([f"N{i:03d}" for i in range(4, 11)]).any())

# ---- D. adversarial perturbations ---------------------------------------------------------------
def justified(df):
    return set(map(tuple, df[df.flagged != "Yes"][["route", "week_of", "matched_note_id"]].values))


base = justified(out)

wrong_route = notes.copy()  # D1: attach the flood note to the wrong route
wrong_route.loc[wrong_route.note_id == "N001", "applies_to"] = "Mumbai-Pune"
wrong_route.loc[wrong_route.note_id == "N001", "note"] = wrong_route.loc[
    wrong_route.note_id == "N001", "note"].str.replace("Chennai-Bangalore", "Mumbai-Pune")
j = justified(build_output(SHIP, NOTES, wrong_route))
check("D1 flood note moved to another route: Chennai-Bangalore weeks stop being justified",
      not any(x[0] == "Chennai-Bangalore" and x[2] == "N001" for x in j))

shifted = notes.copy()  # D2: all note dates a year late -> windows no longer overlap
shifted["date"] = (pd.to_datetime(shifted["date"]) + pd.DateOffset(years=1)).dt.strftime("%Y-%m-%d")
check("D2 all note dates shifted a year: no row justified", not justified(build_output(SHIP, NOTES, shifted)))

check("D3 empty note set: no row justified, none cited",
      not justified(build_output(SHIP, NOTES, notes.iloc[0:0])))

negated = notes.copy()  # D4: rewrite the 3 supporting notes to say 'no effect' -> nothing may be justified
for nid in ("N001", "N002", "N003"):
    negated.loc[negated.note_id == nid, "note"] += " Costs were not significantly affected."
check("D4 supporting notes reworded to 'not significantly affected': none justified",
      not justified(build_output(SHIP, NOTES, negated)))

only_neg = notes[notes.note_id.isin([f"N{i:03d}" for i in range(4, 11)])]
check("D5 only the 7 'no impact' notes available: none justified", not justified(build_output(SHIP, NOTES, only_neg)))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
