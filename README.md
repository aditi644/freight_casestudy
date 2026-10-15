# FreightTiger - Shipping Cost Watcher

Spots routes whose cost per tonne-km is rising abnormally and says, in plain English and **only from the
provided notes**, whether there is a real reason. If no note supports a reason, the answer is "unexplained".

```
pip install pandas scikit-learn
python run.py shipment_records.csv context_notes.csv output.csv   # -> output.csv (21 rows)
python evaluate.py        # 12 checks, exit code 1 on failure
./repro_check.sh          # 3 fresh runs, byte-for-byte diff
python ask.py "why did Chennai-Bangalore get pricier in March?"   # stretch goal
```

## Files
| File | Role |
|---|---|
| `pipeline.py` | grouping, Monday-based weeks, cost per tonne-km, the two graded baselines, flag rule |
| `notes.py` | note structuring, TF-IDF retrieval, verification rules, reason text |
| `run.py` | glue; writes the CSV in the `sample_output_format_v2.csv` layout |
| `evaluate.py`, `labeled_set.csv` | eval harness (see "How I know it is right") |
| `repro_check.sh`, `repro/` | 3-run reproducibility check and its hashes |
| `ask.py` | plain-English Q&A over the system's own findings |
| `token_cost_log.md` | token / cost log for one full run |

## How it works
1. **Weekly metric.** `route = origin-destination`; `week_of` = Monday of the Mon-Sun week of `shipment_date`;
   `cost_per_tonne_km = sum(freight_cost) / sum(quantity x distance)` per route-week (a ratio of sums, as defined).
   `route_type` is taken as given (verified: each route has exactly one type). Data is clean: 7 routes,
   104 weeks each, no nulls/duplicates.
2. **Baselines (exact brief definitions).**
   *vs own history* = mean of the route's previous 8 weeks, strictly before the current week (no look-ahead).
   For a route's first 7 weeks fewer than 8 prior weeks exist, so **all available prior weeks are used - no padding**;
   its very first week has no history and cannot be flagged.
   *vs similar routes* = simple mean of the other routes' cost per tonne-km for the same `route_type` and week
   (self excluded). Short and Long types have only **one** peer route each, so those peer numbers are noisy.
3. **Flag rule (my choice - the brief leaves it open).** A route-week is a candidate when cost is rising vs its own
   history (> 0%) **and** either (a) it is >= +8% above its own trailing average, or (b) it is >= +20% above similar routes.
   Why: normal week-to-week noise is about 3-4% (std of the own-history % on stable routes), so +8% is roughly 2 sigma;
   leg (b) is needed because a slow creep (Mumbai-Pune, below) is absorbed by a trailing average and would never trip (a).
   Thresholds are constants at the top of `pipeline.py`.
4. **Retrieval + verification ("RAG without a generator").** For each candidate, all notes are ranked by TF-IDF
   similarity (scikit-learn; with only 10 notes I rank all of them rather than cut to top-k, because missing a note
   is worse than a few extra comparisons). A note then **justifies** a flag only if it passes *all* of:
   - applies to this route (or "All Routes");
   - its effective window covers >= 4 days of the flagged week (windows come from the note: explicit dates such as
     "Feb 24 to Mar 8"; "festival week" = that week; "starting this week" = 8 weeks, after which the 8-week baseline has
     absorbed the shift);
   - it actually describes a cost **increase** (notes that say "not significantly affected", "returned to normal",
     "no rate change", "no major disruptions", "improved roads", "not part of this dataset" never justify);
   - if it states a size (N003: 5-7%), the observed rise is not more than 3 points above the top of it;
   - a network-wide note cannot explain a gap vs *similar routes* (peers are hit equally).
   Otherwise the verdict is `Yes` (unexplained, `matched_note_id` blank) and the reason says which notes were
   checked and why they were rejected.
5. **Reasons are templates that quote the note verbatim**, so nothing can be invented. No LLM is used (see cost log).

## Results (21 flagged route-weeks: 3 justified, 18 unexplained)
- **Justified:** Ahmedabad-Mumbai 2025-01-20 (N002 festival surcharge); Chennai-Bangalore 2025-02-24 and 2025-03-03 (N001 flood).
- **Deliberately NOT justified (the traps):** Chennai-Bangalore 03-10 and 03-17 (costs still high, but the flood window
  ended Mar 8 and N009 says conditions normal); Mumbai-Delhi 05-12 (diesel note N003 explains ~5-7%, observed +11.6%);
  Mumbai-Pune 2025-10-27 (N010 says no rate change); Mumbai-Pune 2025-09-15 (N006 says stable demand).
- **Main finding for a human:** Mumbai-Pune has crept up since mid-2025 (about +35-39% vs its Short-route peer by December)
  with no supporting note - 12 of the 18 unexplained rows.

## How I know it is right (and why I trusted it enough to submit)
`python evaluate.py` runs 12 checks, all passing:
- **A. Numbers.** A slow plain-Python recomputation (different code path) matches the pipeline on all 21 rows,
  and the 3 rows in `sample_output_format_v2.csv` are reproduced exactly (cost, both percentages, flag, note id).
- **B. Labeled set.** 12 route-weeks I labeled by hand from the notes (`labeled_set.csv`, with my reasoning): 12/12 correct.
  This set is small and I chose the hard cases on purpose; it is a sanity net, not a statistical accuracy estimate.
- **C. Fabrication audit.** Every justified row's cited note is re-checked against hand-coded note facts: right route,
  window overlaps the week, note reports a rise. No "no impact" note is ever cited; unexplained rows never cite one.
- **D. Adversarial tests.** Move the flood note to another route / shift all note dates a year / remove all notes /
  reword supporting notes to "not affected" / keep only the "no impact" notes: in every case nothing is falsely justified.
  (Writing D3 exposed a real crash on an empty note set; fixed.)

## Reproducibility
`./repro_check.sh` runs three fresh processes with different `PYTHONHASHSEED`s and diffs the whole file:
all three have SHA-256 `c745a676...9894a40`, byte-identical - flags, numbers, cited notes **and reason text**.
Nothing is random: no sampling and no LLM, so there is no temperature to set.

## Cost
0 LLM calls, 0 input / 0 output tokens, Rs 0, 1.2 s wall time (`token_cost_log.md`).

## Known limits / trade-offs (honest list)
- Flag thresholds (8% / 20%) are judgment calls; the brief gives none. Smaller diesel-sized rises (5-7%) that stay
  under 8% on their own are not flagged at all.
- Note windows and effects are extracted by rules tuned to these 10 notes; unseen phrasing in new notes would need
  new patterns. With a larger notes set I would add a vector DB with real embeddings (not possible here: model
  downloads are blocked) and an LLM to extract note fields at temperature 0, cached per note, with the same
  hard verification rules on top so the LLM can never decide a verdict alone.
- Peer groups for Short/Long contain one other route, so "vs similar routes" is a one-route comparison.
- The provided `sample_output_format_v2.csv` has an unquoted comma in one `reason` cell; my output quotes such cells properly.
