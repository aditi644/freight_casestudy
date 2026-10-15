"""Note retrieval + verification. Retrieve first, then verify -- a note justifies a flag only if ALL hold:

  1. scope  : note applies to this route (or says "All Routes")
  2. window : the note's effective window covers >= 4 days of the flagged Mon-Sun week
  3. effect : the note actually describes a cost INCREASE (notes saying "not affected",
              "returned to normal", "no rate change", ... never justify)
  4. magnitude (if the note states one): the observed rise is not far above it
  5. leg    : an "All Routes" note hits the route's peers equally, so it can explain a rise vs the
              route's own history but never a gap vs similar routes.

No LLM is used: retrieval is TF-IDF cosine (scikit-learn); verdicts and wording are rules/templates,
so output is identical on every run and cannot invent a reason.
"""
import re
from datetime import timedelta

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MIN_OVERLAP_DAYS = 4
OPEN_ENDED_WEEKS = 8          # level-shift notes: after 8 weeks the 8-week baseline has absorbed them
MAGNITUDE_TOLERANCE_PP = 3.0  # observed rise may exceed a note's stated upper bound by this much
CLOSEST_NOTE_DAYS = 21

NO_EFFECT = [
    r"not significantly affected", r"not part of this dataset", r"no major disruptions",
    r"no significant disruptions", r"returned to normal", r"remained normal",
    r"without a rate change", r"improved road", r"remained stable",
]
RAISES_COST = [r"surcharge", r"pushing up", r"higher trip costs", r"prices rose"]
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def monday(ts):
    return ts - timedelta(days=ts.weekday())


def _parse_md(text, year):
    m = re.match(r"([A-Za-z]{3})\w*\s+(\d{1,2})", text.strip())
    return pd.Timestamp(year=year, month=MONTHS[m.group(1).lower()], day=int(m.group(2)))


def structure_note(row):
    """Turn a raw note row into scope / window / effect / magnitude (deterministic rules)."""
    text, d = row["note"], pd.Timestamp(row["date"])
    low = text.lower()
    rng = re.search(r"from ([A-Za-z]+ \d{1,2}) to ([A-Za-z]+ \d{1,2})", text)
    if rng:                                                    # explicit window in the text
        start, end = _parse_md(rng.group(1), d.year), _parse_md(rng.group(2), d.year)
    elif "starting this week" in low:                          # open-ended level shift
        start = monday(d)
        end = start + timedelta(weeks=OPEN_ENDED_WEEKS) - timedelta(days=1)
    else:                                                      # point-in-time note: its own week
        start = monday(d)
        end = start + timedelta(days=6)
    no_effect = any(re.search(p, low) for p in NO_EFFECT)
    raises = any(re.search(p, low) for p in RAISES_COST) and not no_effect
    pct = re.search(r"(\d+)-(\d+)%", text)
    return {
        "note_id": row["note_id"], "date": d, "text": text,
        "scope": row["applies_to"].strip(), "start": start, "end": end,
        "raises_cost": raises, "max_pct": float(pct.group(2)) if pct else None,
    }


class NoteIndex:
    def __init__(self, notes_df):
        self.notes = [structure_note(r) for _, r in notes_df.iterrows()]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.mat = (self.vec.fit_transform([n["text"] + " " + n["scope"] for n in self.notes])
                    if self.notes else None)   # no notes -> nothing can be justified

    def retrieve(self, route):
        """All notes ranked by similarity to the route + 'cost rise' vocabulary (ties -> note_id)."""
        if not self.notes:
            return []
        q = f"{route.replace('-', ' ')} {route} cost increase surcharge diesel toll flood disruption"
        sims = cosine_similarity(self.vec.transform([q]), self.mat)[0]
        order = sorted(range(len(self.notes)), key=lambda i: (-round(sims[i], 9), self.notes[i]["note_id"]))
        return [self.notes[i] for i in order]

    @staticmethod
    def _in_scope(n, route):
        return n["scope"].lower() == "all routes" or n["scope"].lower() == route.lower()

    @staticmethod
    def _overlap_days(n, week_of):
        a, b = max(n["start"], week_of), min(n["end"], week_of + timedelta(days=6))
        return max((b - a).days + 1, 0)

    def verify(self, route, week_of, own_pct, own_leg, peer_leg):
        """Return (matched_note_id, reason). matched_note_id == '' means unexplained."""
        ranked = self.retrieve(route)
        rejections = []
        for n in ranked:
            if not self._in_scope(n, route) or self._overlap_days(n, week_of) < MIN_OVERLAP_DAYS:
                continue
            if not n["raises_cost"]:
                rejections.append(f"{n['note_id']} ({n['date'].date()}) covers this week but reports no cost increase")
                continue
            all_routes = n["scope"].lower() == "all routes"
            if all_routes and peer_leg and not own_leg:
                rejections.append(f"{n['note_id']} is network-wide, so it cannot explain a gap vs similar routes "
                                  f"(they are affected equally)")
                continue
            if n["max_pct"] is not None and own_pct > n["max_pct"] + MAGNITUDE_TOLERANCE_PP:
                rejections.append(f"{n['note_id']} accounts for about {n['max_pct']:.0f}% but the observed rise is "
                                  f"{own_pct:+.1f}%, so most of it stays unexplained")
                continue
            return n["note_id"], (
                f"Matches note {n['note_id']} dated {n['date'].date()} "
                f"(applies to {n['scope']}; effective {n['start'].date()} to {n['end'].date()}): "
                f"\"{n['text']}\" The cost rise falls inside that window and has a clear explanation.")
        reason = "No matching note found that explains a cost rise for this route and week. "
        if rejections:
            reason += "Checked and rejected: " + "; ".join(rejections) + ". "
        else:
            near = self._closest(route, week_of)
            if near:
                reason += (f"The closest note ({near['note_id']}, {near['date'].date()}) says: \"{near['text']}\" "
                           f"-- it does not describe a reason for a cost rise on this route. ")
        return "", reason + "Cost rise looks unexplained and worth a human review."

    def _closest(self, route, week_of):
        c = [n for n in self.notes if self._in_scope(n, route)
             and abs((n["date"] - week_of).days) <= CLOSEST_NOTE_DAYS]
        return min(c, key=lambda n: (abs((n["date"] - week_of).days), n["note_id"])) if c else None
