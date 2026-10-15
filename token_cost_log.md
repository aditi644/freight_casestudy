# Token / cost log -- one full run over the entire shipment file (2,940 rows)

| Metric | Value |
|---|---|
| LLM calls | **0** |
| Total input tokens | **0** |
| Total output tokens | **0** |
| Cost at published rates | **Rs 0 / $0** (no paid API; TF-IDF retrieval via scikit-learn, runs locally) |
| Wall time | 1.2 s measured on CPU |
| Randomness | none (no sampling, no LLM), so no temperature to set |

**Why no LLM:** only 10 short notes exist and the brief penalises any reason that is not grounded in them.
Verdicts are rules over structured note fields and the `reason` text is a template that quotes the note
verbatim, so it cannot invent a cause. This is also why "explanation wording may vary" never applies here:
all three runs are byte-identical.

**If an LLM were added later (estimate, NOT measured):** one call per flagged row only (21 rows), to rephrase the
reason, given only the matched note text and the numbers: roughly 21 calls x ~350 input tokens = ~7.4k input
and 21 x ~90 = ~1.9k output tokens in total. At temperature 0 with the verdict, note id and numbers
fixed outside the model, the LLM could not change the verdict. With a free-tier / open-source model that costs Rs 0.
