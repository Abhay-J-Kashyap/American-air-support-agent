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

**16. Ingestion is two passes; thread connectivity can't be known in one streaming pass.**
Whether a customer's opening tweet belongs to the brand slice depends on
where its thread eventually leads, which isn't knowable until the whole
reply graph exists. Pass 1 builds a lightweight tweet_id -> parent-id index
over the full file (a few numeric columns, not text -- tens of MB, not
hundreds). Pass 2 re-reads the file and keeps only rows in the
brand-connected set found in pass 1. This only applies to `make data`, which
isn't on the 15-minute reproduction path (`make eval` runs from cache) --
it's fine for this step to take longer.

**17. Multi-part reply merge reparents onto the FIRST part's parent, keeps the SECOND part's id.**
When a brand reply is split "...(1/2)" then a same-author follow-up, the
merged row has to keep the id anything downstream actually points at (the
last part posted), while its own parent pointer skips over the dropped
first part to whatever the first part originally replied to. Getting this
backwards silently breaks the thread graph rather than erroring, so it's
covered by `test_multipart_reply_merges_and_skips_the_dropped_parent` with
an explicit check on `in_response_to_tweet_id`, not just on the merged text.

**18. A confirmation-code regex without a digit requirement redacts its own label.**
First version of the booking-reference regex matched a trigger word,
optionally "number", optionally a linking word, then `[A-Z0-9]{5,8}` for the
code. Under `re.IGNORECASE`, the literal word "number" is itself six
letters -- it satisfies that character class as well as a real code does,
so `"confirmation number is AB12CD"` redacted "number" and left the actual
code untouched. Fixed with a lookahead requiring at least one digit in the
captured span. Caught by `test_confirmation_code_redacted`, which is why it
asserts on the literal code string disappearing rather than just checking
that `[REF]` appears somewhere in the output -- a weaker assertion would
have passed against the bug.

**19. Three ingestion bugs that only surface on real data, found by adversarial probing rather than by the test suite.**
The fixture-based tests all passed while these were live, because clean
fixtures don't contain the pathologies a 3M-row scrape does. Found by
deliberately constructing the messy cases:
* **Cyclic parent references** (`a -> b -> a`, or a self-reply) sent
  `reconstruct_threads` into an infinite loop. `make data` would hang with
  no output and no error. Fixed with a `seen` guard; cycles collapse to
  `min(cycle)` so thread_ids are independent of row order.
* **Duplicate tweet_ids** made `set_index("tweet_id")` non-unique, so
  `.loc[parent_id]` returned a DataFrame instead of a Series and the
  multipart merge died with pandas' "truth value of a Series is ambiguous"
  -- an opaque error far from its cause. Now deduplicated in pass 2
  (`keep="first"`), with an explicit precondition check in the merge that
  names the real problem.
* **Unparseable timestamps** became `NaT`, and since `NaT <= cutoff` is
  False, those rows landed in the EVAL window unannounced -- silently
  polluting the held-out set, the worst outcome available in this pipeline.
  Now dropped in pass 2 and counted.
Both drop counts are surfaced in the `make data` output rather than being
silently swallowed: if the real file turns out to have thousands of either,
that is a finding about the dataset worth reporting, not a detail to hide.

**20. Timestamps parsed with an explicit format; ISO-8601 test fixtures were unrepresentative.**
twcs.csv uses Twitter's native format ("Tue Oct 31 22:10:47 +0000 2017"),
not ISO-8601. Every fixture in the test suite used ISO-8601 and passed
anyway, because `errors="coerce"` quietly fell back to dateutil -- so the
tests were green while parsing ~88k values individually and emitting a
UserWarning on every real run. Now parsed with an explicit format (3x
faster, no warning), with a per-row fallback so an oddly-formatted row is
recovered rather than dropped by the undated filter. Two tests now pin the
real format. Same lesson as #19: fixtures written from assumption rather
than from data hide the bugs they were meant to catch.

**21. Multi-part reply merging retained but effectively dead for this brand -- deliberately not "fixed".**
Only 10 of 36,764 AmericanAir replies (0.03%, 4 conversations) carry an
`(N/M)` marker. Investigated whether to loosen the pattern to bare `N/M`
and the answer is an emphatic no: 129 replies contain bare `\d+/\d+`, and
inspection shows they are all false positives -- "we're here 24/7" and
dates like "10/23". A looser regex would have merged unrelated tweets and
corrupted the thread graph. The strict parenthesised pattern is correct
*because* it is strict. The merge code stays (it costs nothing, is tested,
and would matter for a brand like Delta that does split replies), but for
AmericanAir the honest expected value is ~0 merges and that is reported as
such rather than tuned until the number looks impressive. 140->280 char
expansion in late 2017, mid-dataset, is the likely reason splitting is rare.
