"""Stretch goal: ask about the findings in plain English. Answers ONLY from output.csv (the system's own
results) -- if nothing matches it says so instead of guessing.

  python ask.py "why did Chennai-Bangalore get pricier in March?"
"""
import re
import sys

import pandas as pd

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]


def answer(question, out_path="output.csv"):
    out = pd.read_csv(out_path, dtype=str).fillna("")
    q = question.lower().replace("_", "-")
    routes = [r for r in out.route.unique() if r.lower() in q or r.lower().replace("-", " ") in q]
    if not routes:   # match on just one city name (e.g. "Pune")
        routes = [r for r in out.route.unique() if any(c.lower() in q for c in r.split("-"))]
    months = [i + 1 for i, m in enumerate(MONTHS) if m in q or m[:3] + " " in q + " "]
    year = re.search(r"\b(20\d\d)\b", q)
    df = out.copy()
    if routes:
        df = df[df.route.isin(routes)]
    if months:
        df = df[pd.to_datetime(df.week_of).dt.month.isin(months)]
    if year:
        df = df[df.week_of.str.startswith(year.group(1))]
    if df.empty:
        return ("None of the weeks I flagged match that question. That means either costs were not "
                "rising abnormally then, or the rise was below my flag threshold -- I have no finding to explain.")
    lines = []
    for r in df.itertuples():
        verdict = (f"explained by note {r.matched_note_id}" if r.matched_note_id else "UNEXPLAINED (no supporting note)")
        lines.append(f"- {r.route}, week of {r.week_of}: {r.cost_per_tonne_km} INR/tonne-km, "
                     f"{r.vs_own_history.split(' vs')[0]} vs its own history, {r.vs_similar_routes.split(' vs')[0]} "
                     f"vs similar routes -> {verdict}.\n    {r.reason}")
    return f"I found {len(df)} flagged week(s):\n" + "\n".join(lines)


if __name__ == "__main__":
    print(answer(" ".join(sys.argv[1:]) or input("Question: ")))
