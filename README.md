# @AmericanAir support agent

An AI support agent for a single brand (`@AmericanAir`) built on the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. It classifies an incoming customer message into a data-derived intent,
drafts a reply grounded in how the brand has historically resolved similar
issues, and decides whether to auto-handle or escalate — with a stated reason.

The evaluation is the point. See `report/` for what the headline number does
and does not mean.

## Reproduce the headline results (no API key, no spend)

```bash
make setup
make eval
```

`make eval` runs entirely from the committed response cache and finishes in
under 15 minutes on a laptop. It needs no API keys and costs nothing. The
`reproduce` job in CI runs exactly this on every push, so the claim cannot rot.

To regenerate every prediction against the live APIs instead — hours, and
subject to free-tier rate limits — use `make eval-live`.

## Full pipeline

```bash
cp .env.example .env      # fill in free-tier keys, see below
make check-providers      # confirm every model is live BEFORE a long run
make data                 # stream-filter the raw CSV to the brand slice
make intents              # cluster and propose the intent taxonomy
make index                # build the retrieval index (training window only)
make golden               # launch the labelling CLI
make eval-live            # regenerate predictions and judgements
make report               # render metrics, tables, figures
```

`make help` lists every target.

## Providers

Four roles, four model slots, all on no-credit-card free tiers. Roles are
resolved from `config/config.yaml`; swapping a provider is a config edit, never
a refactor.

| Role | Model | Provider |
|---|---|---|
| classifier | Llama 3.1 8B | Groq |
| drafter | Llama 3.3 70B | Groq |
| judge A | GPT-OSS 120B | Cerebras |
| judge B | Mistral Small | Mistral |

Judges deliberately sit on different providers *and* different model families
from the drafter — that separation is what makes the self-preference bias
analysis meaningful, and a CI test asserts it so a convenient config edit
cannot quietly destroy the experiment.

OpenRouter is wired in as a fallback for every role but is never primary: its
free variants allow only 50 requests per day on an unfunded account, and failed
calls still burn quota.

Sign up (all free, no card): [Groq](https://console.groq.com),
[Cerebras](https://cloud.cerebras.ai), [Mistral](https://console.mistral.ai),
[OpenRouter](https://openrouter.ai/keys).

### Rate limits are the real constraint

Published free-tier limits go stale fast and vary by account. `make
check-providers` pings every configured model and prints the limits the API
actually reports in its response headers. Run it before every long sweep.

Note that tokens-per-minute usually binds before requests-per-minute: Groq's
free tier allows 30 requests/minute but only 6,000 tokens/minute, so a
~1,350-token drafting call caps you near 4 requests/minute in practice. The
client budgets both dimensions.

## Disk

Nothing here needs much space. The raw CSV is ~500 MB; `make data`
stream-filters it in chunks and deletes it, leaving an ~8 MB brand slice.
Embeddings are float16 (~12 MB). The response cache is text — a full run is
single-digit MB, gzipped for committing by `make freeze-cache`. Total committed
artifacts stay under 40 MB, no Git LFS required.

## Development

```bash
make test         # no network, no API keys, fake clock
make lint fmt typecheck
```

CI runs lint, format check, mypy and the test suite on Python 3.10 and 3.12,
plus the cached-reproduction job. Tests marked `live` hit real APIs and are
excluded from CI — CI must never depend on a free tier being up.

## Documents

- `DECISIONS.md` — non-obvious choices and why.
- `SAMPLING.md` — how the golden set was sampled and labelled.
- `CREDITS.md` — dataset, models and borrowed code.
- `report/` — problem framing, results vs baselines, failure analysis, and the
  mandatory "what is misleading about my headline number" section.
