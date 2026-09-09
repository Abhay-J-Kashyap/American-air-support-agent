# Golden set: sampling and labelling

<!-- Written on Day 2, before any agent code runs. See DECISIONS.md #1. -->

## Sampling frame
200 examples in two strata, kept separate in all reporting.

- **Stratum A (n=100)** — uniform random from the held-out evaluation window.
  This is the *only* stratum from which population-level metrics may be quoted.
- **Stratum B (n=100)** — stratified across intents plus deliberately selected
  hard cases (sarcasm, multi-intent, non-English, ambiguous escalation).
  For coverage of rare classes. **Not** a population estimate.

## Labels per example
| Field | Notes |
|---|---|
| `intent` | one of the curated taxonomy, or `other` |
| `action` | `auto` or `escalate` |
| `reason` | reason enum, required when `action=escalate` |
| `reference_reply` | the brand's actual historical reply |
| `reference_is_good` | was that reply actually any good? |

`reference_is_good` exists because the references are themselves noisy — a
large share are "DM us" boilerplate. Without this field, cases where the agent
beats the reference are indistinguishable from errors.

## Agreement
50 examples double-labelled for intra-rater reliability. Kappa reported in the
report; it is the ceiling on measurable accuracy.
