"""Usage: python run.py [shipments.csv] [notes.csv] [out.csv]  ->  writes output in sample_output_format_v2.csv layout."""
import sys

import pandas as pd

from notes import NoteIndex
from pipeline import add_flags, load_shipments, weekly_metrics

COLUMNS = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes",
           "flagged", "matched_note_id", "reason"]


def build_output(shipments_path, notes_path, notes_override=None):
    w = add_flags(weekly_metrics(load_shipments(shipments_path)))
    index = NoteIndex(notes_override if notes_override is not None else pd.read_csv(notes_path))
    rows = []
    for r in w[w["candidate"]].sort_values(["week_of", "route"]).itertuples():
        note_id, reason = index.verify(r.route, r.week_of, r.own_pct, r.own_leg, r.peer_leg)
        rows.append({
            "route": r.route,
            "week_of": r.week_of.strftime("%Y-%m-%d"),
            "cost_per_tonne_km": round(r.cpt, 2),
            "vs_own_history": f"{r.own_pct:+.1f}% vs this route's past average",
            "vs_similar_routes": f"{r.peer_pct:+.1f}% vs similar-length routes this week",
            "flagged": "No (justified)" if note_id else "Yes",
            "matched_note_id": note_id,
            "reason": reason,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


if __name__ == "__main__":
    ship = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/shipment_records.csv"
    notes = sys.argv[2] if len(sys.argv) > 2 else "/mnt/user-data/uploads/context_notes.csv"
    out = sys.argv[3] if len(sys.argv) > 3 else "output.csv"
    df = build_output(ship, notes)
    df.to_csv(out, index=False)
    print(f"{len(df)} rows -> {out}  ({(df.flagged == 'Yes').sum()} unexplained, "
          f"{(df.flagged != 'Yes').sum()} justified)")
