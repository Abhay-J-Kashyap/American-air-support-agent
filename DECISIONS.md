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

**22. Intents clustered from first-turn messages only, in the training window only.**
Intent is a property of what the customer first asked. Clustering every
inbound turn would over-represent long argumentative threads and fill the
taxonomy with conversational states -- "still waiting", "any update?" --
rather than intents. Restricted to the training window so the taxonomy is
not derived from held-out data (#4). The LLM never sees the corpus and never
chooses k: it only names groups the geometry already found, keeping the
taxonomy grounded in message distribution rather than in a model's priors
about what airline complaints should look like.

**23. Silhouette is computed on a 5k subsample; inertia is recorded but cannot select k.**
Silhouette is O(n^2) and will hang or OOM on 15k+ points, so it is always
subsampled -- acceptable because it is a heuristic for choosing k, not a
reported metric. Inertia falls monotonically with k by construction so it
can never pick k on its own, but it is logged because the shape of the curve
is what a human reads when overruling the silhouette pick, and the brief
expects a human edit pass rather than blind automation.

**24. `groupby().first()` silently fabricated rows; replaced with `drop_duplicates`.**
pandas' `groupby().first()` returns the first NON-NULL value for each column
INDEPENDENTLY. For a thread whose opening message had a null text, it
emitted a Frankenstein row: tweet_id and created_at from the real first
message, text silently borrowed from a later one -- a fabricated training
example that would look completely normal downstream. Found by adversarial
probing, not by the passing test suite. Same failure class as #19 and #20:
green tests over fixtures that lack the pathology.

**25. Embedding sits behind a Protocol so tests never download a model.**
`Embedder` is a Protocol with a deterministic `StubEmbedder` in the test
suite, so all 88 tests run with no ONNX download. Same principle as never
letting CI depend on a live free-tier API (#9): a test suite that needs a
network fetch is a test suite that fails for reasons unrelated to the code.
Embeddings are cached to disk keyed by (model, dtype, content hash) so a
crashed or re-parameterised run does not re-embed 15k messages.

**26. Embedding uses all CPU cores; fastembed's default is single-core.**
`TextEmbedding.embed()` accepts a per-call `parallel` kwarg (confirmed via
`inspect.signature()` against the installed package, not docs -- docs have
drifted from behavior twice already in this project, #13 and #20).
Default `None` means single-core inference, which pegs one core for
minutes on a batch of thousands of short tweets while the rest sit idle.
Now called with `parallel=0` (use every available core), matching
fastembed's own documented recommendation for large-dataset offline
encoding, which this always is.

**27. Embedding cache writes are now atomic; np.save silently corrupted the temp-file rename once already.**
`np.save` is not atomic, so a run killed mid-write can leave a truncated
`.npy` at the real cache path -- a future run then crashes on `np.load`
with a confusing numpy error nowhere near its actual cause. Fixed with a
temp-file-then-rename pattern. First attempt at this fix was itself broken:
`np.save` silently appends `.npy` to any filename that doesn't already end
with it, so naming the temp file `*.npy.tmp` made numpy actually write
`*.npy.tmp.npy`, and the rename then failed with a hard-to-parse
FileNotFoundError. Caught by directly testing the temp-file behavior
against a real numpy call rather than trusting the first version once it
merely looked correct. The cache also now self-heals: a pre-existing
corrupt file (from before this fix, or any other partial write) is treated
as a miss and re-embedded rather than crashing the whole run.

**28. Progress reporting is a correctness concern, not a nicety.**
The first version wrapped model download, model load, and encoding of 15k
messages under one static "Embedding..." spinner. When a real run appeared
to stall, that made "is this hung or just slow?" literally unanswerable
without a debugger -- and because the embedding cache only writes after the
full batch completes, waiting longer accrued no partial progress either.
Now: model load and encoding report separately, encoding shows an
incremental bar with an ETA (fastembed's `.embed()` returns a generator, so
per-item progress is free), and throughput is printed on completion so the
next run's cost is predictable rather than guessed.

**29. Embedding parallelism is configurable, defaulting to all cores.**
`--parallel 0` (all cores) by default, `-1` for single-core, `N` for an
explicit count. Not hard-coded, because data-parallel encoding spawns
worker PROCESSES and Windows uses spawn rather than fork: each worker
re-imports the module and loads its own ~67MB copy of the ONNX model. For
small batches that startup cost can exceed the compute it saves, so the
single-core fallback is a real escape hatch. Recommended practice is to
measure throughput on `--sample 500` before committing to a full run.

**30. JSON parsing rewritten for reasoning models; 21.6% of the corpus was silently unlabelled.**
The first real run left two clusters (1,325 and 1,926 messages, 21.6% of
the sample) as `unlabelled_cluster_N`. Cause: the free-tier models are
reasoning models (gpt-oss, qwen) that emit analysis before their answer
despite explicit instructions, and the original parser took the FIRST
`{...}` span via `find("{")`/`rfind("}")` -- which spans from the first
brace to the last and yields garbage when reasoning text contains braces.
Rewritten to strip `<think>` and harmony analysis channels first, then
brace-count (quote-aware, so braces inside string values don't break depth
tracking) and try candidates LAST-first, since the real answer follows the
analysis. A failed parse now retries once with a blunter format
instruction, and if that also fails the RAW RESPONSE is retained on the
cluster and written to cluster_audit.json -- the original code discarded it,
making the failure undiagnosable without a full re-run.

**31. Duplicate labels, parse failures, and edge-of-range k are now surfaced as warnings.**
The first run produced two clusters both labelled `positive_feedback`
(31.3% of messages combined) and chose k=8, the minimum of the swept 8-14
range -- both signals that the taxonomy needs work, and both easy to miss
in a table. The CLI now explicitly warns on duplicate labels, names the
clusters that failed to parse, and flags when the chosen k sits at the edge
of the swept range (which means the true optimum may lie outside it).
Warnings, not errors: the human edit pass decides, the tool just refuses to
let the problem pass unnoticed.

**32. Reasoning models returned EMPTY content, not malformed content -- the real cause of the unlabelled clusters.**
On the first two real runs, 21.6% then 51% of the intent corpus came back
as `unlabelled_cluster_N`. Inspecting the cached raw responses showed the
model returned `''` -- empty string, HTTP 200, full completion_tokens
count. gpt-oss is a REASONING model: it spends max_tokens on internal
reasoning before emitting any answer, and `classifier.max_tokens` was 200,
so the budget was exhausted mid-reasoning and no content was ever produced.
A public HF project hit the identical signature with max_tokens=20. Fixed
three ways: max_tokens raised (1200 classifier / 2000 judge_a),
`reasoning_effort` set per role (low for classification, none for drafting
where chain-of-thought buys nothing, medium for judging where it is the
point), and `reasoning_format: hidden` -- which Groq requires alongside
JSON mode anyway. The client now also raises a NAMED error on empty
content rather than passing "" downstream to fail as a mystifying parse
error far from its cause. Note the first diagnosis was wrong: I assumed
reasoning-text braces confused the parser and rewrote the parser
accordingly. That rewrite is retained (#30) because it is independently
correct, but it was not the bug. Reading the cached raw response was what
actually identified it -- which is precisely why raw responses are now
retained on failure.

**33. Silhouette cannot select k for this corpus; k is a human choice and the report must say so.**
Swept k=4..16: silhouette peaked at 0.042 and fell monotonically. Against
Rousseeuw's scale (>0.70 strong, 0.50-0.70 reasonable, 0.25-0.50 weak,
<0.25 no substantial structure) 0.042 means the data has essentially NO
cluster structure -- short-text embeddings form a semantic continuum, not
discrete blobs. Because the curve decreases monotonically, argmax
necessarily returns the smallest k swept, so the "chosen" k=4 was an
artifact of `--k-min`, not a finding. k-means still returns clusters, but
their boundaries are imposed rather than discovered. The CLI now detects
both conditions and says so explicitly, and `--k` allows setting k
deliberately. This goes in the report's "what is misleading about my
headline number" section: any per-intent metric inherits the arbitrariness
of these boundaries, and a taxonomy presented as data-derived is only
partly so.

**34. The actionable intents are genuinely rare, so clustering alone cannot produce a usable taxonomy.**
Keyword incidence over 35,249 inbound training-window messages: refund
1.6%, booking-change 1.4%, check-in 1.8%, seat 5.4%, loyalty 2.9%,
accessibility 0.4%. These are not hidden by poor embeddings -- they are a
long tail. k=10 over 15,000 messages yields ~1,500-message clusters, so a
1.6% intent (~240 messages) is arithmetically impossible to isolate;
surfacing it would need k~50+, and silhouette (0.031) says there is no
structure to find at any k. The taxonomy is therefore HYBRID: dominant
themes from clustering, long-tail intents added by hand, with `source:
cluster|keyword` recorded per intent so the report never blurs the two
kinds of evidence. This is the human edit pass the brief asks for, not a
workaround for a failed method.

**35. What this channel is actually for reframes "good" for this brand.**
~30% of first-turn messages are praise or pre-flight excitement needing no
resolution at all, and another ~48% are undirected complaints about delays
and service. Actual service REQUESTS are the minority. Consequences: (a) a
system that handles praise perfectly and refunds badly will post an
excellent headline accuracy while being operationally worthless, so
accuracy is the wrong headline metric for this brand; (b) `positive_no_action`
being counted as "successfully handled" is the single largest inflator of
any aggregate score, and is called out in the misleading-number section;
(c) per-intent metrics matter far more than any average here.

**36. Escalation for safety-critical intents must be a deterministic rule, not a confidence threshold.**
`accessibility_medical` (wheelchair, medical need, disability accommodation)
is the rarest intent at 0.4% -- roughly 140 messages in the entire training
window. No learned classifier trained on this distribution will have usable
recall on it, and its errors are the most costly in the taxonomy. This is
empirical support for what was originally a design preference (#8 era):
`risk_tier: always_escalate` is enforced by rule regardless of model
confidence, and recall on this class is reported SEPARATELY because it is
invisible in a macro average.

**37. The curated taxonomy is a separate committed file from the discovery proposal.**
`discover-intents` rewrites `artifacts/intents/taxonomy.json` on every run.
The taxonomy that the classifier, retriever and escalation policy build
against lives in `config/taxonomy.yaml` and is never written by the tool.
Without this split, re-running discovery to try a different k would
silently clobber the hand-curated taxonomy everything downstream depends
on. A regression test asserts the separation holds.

**38. First keyword prevalence figures were wrong three ways; corrected before they reached the taxonomy.**
The initial long-tail shares were computed by a throwaway command with
three defects, all mine: (a) DENOMINATOR -- it filtered on `inbound` and
`in_training_window` but not first-turn, so it measured 35,249 all-turn
messages while cluster shares measured 15,000 first-turn ones, making the
two sets non-comparable side by side; (b) SINGLE KEYWORDS ON MERGED
INTENTS -- `seat_or_upgrade` got "seat" only, `refund_or_compensation` got
"refund" only, ignoring voucher/credit, so every merged intent was
understated; (c) an arithmetic slip put `flight_disruption` at 30% when its
three constituent clusters sum to 35.1%. Re-derived using union patterns
against `extract_first_inbound(training_only=True)`, so the denominator now
matches the clustering exactly. Refund moved 1.6% -> 3.1%, booking_change
2.9% -> 4.8%, seat_or_upgrade 5.4% -> 8.1%. Cluster shares now sum to
99.9%, confirming the merges neither double-count nor drop a cluster --
and that check is now a test.

**39. checkin_boarding's 8.3% is flagged as untrustworthy inside the artifact itself.**
Its pattern included `\bgate\b`, and "gate" saturates delay complaints
("sat at the gate for an hour") which are flight_disruption, not check-in
problems. The true share is probably nearer the 1.8% that `check.?in`
alone yields. Rather than silently substituting a guess, the figure is
retained with an explicit SUSPECT warning in `config/taxonomy.yaml` and a
test asserting that warning survives, so the caveat travels with the
artifact instead of living only in a conversation. The golden set will
settle it.

**40. Cluster 6 (`delay_and_service_complaint`, 8.6%) assigned to flight_disruption -- a documented coin-flip.**
It genuinely straddles flight_disruption and service_complaint; assigning
it to the latter would move 8.6pp between two of the five largest intents.
Kept with flight_disruption on the reasoning that the disruption is the
trigger and the service failure is its consequence. Recorded in the
taxonomy note as a judgement call to revisit if golden-set labelling
disagrees, because a merge decision of this size silently changes every
per-intent metric downstream.

**41. Consolidation audit found the curated taxonomy was orphaned and the CLI contradicted it.**
A full-state audit after fourteen incremental patches found three
integration defects that no test caught, because each file was individually
correct: (a) `taxonomy.py` was imported by nothing but its own tests -- dead
code; (b) `TAXONOMY_PATH` was a hardcoded relative path, the only
configuration not in config.yaml, silently breaking if run from another
directory; (c) `discover-intents` ended by telling the user to hand-edit
`taxonomy.json` -- the file it overwrites on every run -- directly
contradicting decision #37 and the separation that decision exists to
protect. Fixed by routing the path through config, adding a
`taxonomy-show` command (so the curated file is reachable and validated
rather than orphaned), and correcting the message. Lesson: unit tests
verify components, not that components are wired to each other. A
periodic whole-project audit is a distinct activity from running the suite.
