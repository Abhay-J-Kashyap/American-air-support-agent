# Decision log

Non-obvious choices and why. Written as we go, not reconstructed at the end.

---

**1. Label the golden set before the agent exists.**
If the system is built first, labels drift toward what it happens to do, and
the evaluation silently measures agreement-with-itself. Labelling is scheduled
on Day 2 morning, before any drafting code runs. Cost: cannot use model output
to speed up labelling. Worth it.

**2. Brand = @AmericanAir, not @AppleSupport or @SpotifyCares.**
Apple and Amazon are larger but their replies are overwhelmingly "DM us"
boilerplate, so a model that learns to say "DM us" scores well and resolves
nothing. Airline support has real escalation tiers (medical needs,
unaccompanied minors, disability accommodation, DOT complaint threats) which
makes the escalate/auto decision load-bearing rather than a confidence
threshold.

**3. Intents discovered, then hand-curated -- not chosen a priori.**
Embed first-turn inbound messages, k-means with a silhouette sweep over
k=8..14, LLM proposes a label per cluster, then a human edit pass down to
~11 intents plus `other`. Time-boxed to two hours; no HDBSCAN/UMAP tuning.
The cluster-to-taxonomy audit trail ships as an artifact.

**4. Time-based split, not random.**
Airline support volume is event-correlated: one storm produces thousands of
near-identical tweets in hours. A random split puts near-duplicates on both
sides of the boundary, so retrieval finds an almost-exact match and scores
inflate for reasons that will not generalise. Index on the earlier 70%,
evaluate on the later 30%.

**5. Explicit near-duplicate screen between index and golden set.**
Belt and braces on #4. Drop golden examples with cosine > 0.95 against any
indexed message; report how many were dropped and report scores with and
without the screen. That delta is a headline finding, not a footnote.

**6. Context overflow raises instead of truncating.**
The Cerebras free tier caps context at 8K. A silently truncated judge prompt
still returns a well-formed score, which corrupts every downstream metric
without erroring. `assert_fits` fails loudly at the call boundary.

**7. Token counts are estimated, then reconciled against reported usage.**
Three providers, three tokenizers. Shipping three vocabularies to improve a
budgeting heuristic is not worth it. We estimate at ~3.6 chars/token
(conservative for hashtag- and emoji-dense tweets), then charge the difference
back to the bucket from the API's `usage` block so drift does not compound.

**8. Judges sit on different providers AND different families from the drafter.**
Drafter is Llama on Groq; judges are GPT-OSS on Cerebras and Mistral Small on
Mistral. This is what makes the self-preference bias measurement meaningful,
and it is asserted in CI so a convenient config edit cannot quietly destroy
the experiment.

**9. OpenRouter is configured but never primary.**
Free variants are 20 RPM and 50 requests per *day* on an unfunded account, and
failed calls still burn quota. Wired in advance so a provider dying at 11pm is
a config line, not a new integration written while tired.

**10. Append-only JSONL cache rather than sqlite.**
A killed run keeps everything already written, resumption is free, and there is
no partial-write recovery path to get wrong. A torn final line is skipped on
reload and the call simply re-runs. Committed gzipped so a grader reproduces
every number with no API key and no spend.

**11. Cache records store the model id the API echoed back, not the config alias.**
Free-tier model names get retired without notice and aliases like
`-latest` move. "Which model produced this number" has to be answerable from
the artifact.

**12. fastembed/ONNX for embeddings, not sentence-transformers.**
`sentence-transformers` pulls PyTorch: ~2.5GB installed, and it filled the
disk on the first install attempt. fastembed runs the same class of MiniLM/BGE
models on ONNX Runtime in ~50MB and produces vectors of equivalent quality for
retrieval over 30-word tweets. Embeddings are stored float16 (~12MB for 15k
messages) rather than float32. `pip install -e ".[torch]"` restores the old
backend if the swap ever needs auditing.

**13. Free-tier model catalogs are a moving target -- verify before every long run, not just at project start.**
Three weeks (in project time) after picking `llama-3.1-8b-instant` and
`llama-3.3-70b-versatile` on Groq, both were decommissioned (2026-08-16) and
`check-providers` returned 404 on every one of them. Groq's free flagship
consolidated around `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, and a preview
`qwen/qwen3.6-27b`. This is the entire reason `check-providers` exists as a
standing target rather than a one-time setup step -- a report that claims
reproducibility from a free-tier stack has to treat provider catalogs as
unstable, not fixed at design time.

**14. Cerebras disabled for this account -- a live 402 outranks aggregator blogs.**
`gpt-oss-120b`, the only model in Cerebras's public production catalog,
returns 402 Payment Required on this account. Several third-party trackers
describe a perpetual, no-card 1M-tokens/day free tier; Cerebras's own current
pricing structure describes a one-time $5 trial credit plus a paid Developer
tier, which matches the observed 402 far better. `providers.cerebras.enabled`
is set to `false` rather than left half-working. Re-enable only after
confirming billing status directly in the Cerebras console -- never take an
aggregator's "free tier" claim over what the provider's own API just said.

**15. judge_a lost provider independence from the drafter; judge_b is what preserves the experiment.**
The original design put judges on different providers AND different model
families from the drafter (#8). With Cerebras disabled (#14), Groq is the
only remaining free provider with enough throughput for our volume, so both
the drafter (`qwen/qwen3.6-27b`) and judge_a (`openai/gpt-oss-120b`) now sit
on Groq -- family-independent, not provider-independent. judge_b (Mistral
Small) is the one judge with full independence and is what the
self-preference analysis leans on for a clean reading. Disclosed in the
report as a possible shared-infrastructure confound for judge_a (e.g.
Groq's TruePoint Numerics precision reduction affecting both calls
identically). CI (`test_at_least_one_judge_is_fully_independent_of_the_drafter`)
guards that this last clean signal can't be silently lost to a future config edit.
