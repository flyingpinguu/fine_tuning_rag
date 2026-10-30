# Evaluation and decision report: baseline RAG versus normal LoRA

This report implements project sections 5 and 6. It evaluates only the baseline and the original adapter trained at learning rate **0.0002**. The 100× and 500× experiments are excluded.

## Decision

Deterministic checks alone cannot establish semantic answer quality. Retain both raw answer paths for review; a quality recommendation requires semantic judgments. Rerun with --judge-reviews to include the saved assistant review.

## Experimental scope and fairness

There are **14 held-out questions: 10 supported and 4 unsupported**, 6 Markdown sources, 16 paragraph chunks and 16 training examples. Retrieval is executed once per case. Both paths receive the same accepted passages, identical serialized prompt and generation settings. The baseline temporarily disables the adapter on the same frozen base model; the hybrid enables it. Dropout is off for both during evaluation. Expected sources, required terms, forbidden claims and judge references are never placed in generation prompts.

The timed rerun reproduced all 28 answers from the original normal-rate run exactly. Each generated answer was repeated three times and the script checked token-for-token equality across repeats. Latency repetitions are timing samples, not independent quality samples. No retraining or hyperparameter changes were performed for this evaluation.

Training and evaluation IDs and normalized questions are separate, but policies and scenarios overlap. For example, standard shipping is a training paraphrase, the annual-refund scenario already appears in training, and diagnostic-log restrictions appear in training. This is an in-domain exercise, not a sealed test of unseen policies. The evaluator's prior access to outputs and labels is disclosed below.

## Configuration and provenance

| Component | Configuration |
| --- | --- |
| Base model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Base revision | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Retrieval | Qdrant local in-memory; cosine similarity; top_k=3; threshold=0.3; title + paragraph embedded; source ID/title/position/text preserved |
| Chunking | Blank-line paragraphs; Markdown heading retained as title; no overlap; rebuilt once per run |
| Prompt | Native chat template; shared system instruction; [source-id] passage text + question; assistant generation prefix |
| Generation | Greedy (do_sample=False); maximum 80 new tokens; float32; mps; seed=42; model context-window check, no prompt truncation |
| Evaluation machine | Apple M4 Pro; 24 GiB unified memory; local macOS |
| Adapter | `/Users/christinck/Documents/ml_bootcamp/optional-project/fine_tuning_rag/artifacts/support-lora-adapter` |
| LoRA | targets=['q_proj', 'v_proj']; rank=8; alpha=16; dropout=0.05 |
| Training | epochs=8; learning_rate=0.0002; batch_size=1; 128 updates; AdamW; betas=(0.9, 0.999); epsilon=1e-8; weight_decay=0.01; no scheduler or gradient clipping; fixed example order |
| Training tokens | Maximum 256 prompt+answer tokens; answer-only loss; prompt labels -100; no silent truncation; float32 |
| Parameters | 460,800 trainable of 134,975,808 including adapter (0.341%) |
| Training duration | 7.747 s; optimization + saving/summary work, excludes downloading, loading and tokenization |
| Training loss | 1.5471 → 0.5060; diagnostic only, no held-out quality claim |
| Training SHA-256 | `a842ca7e98e2f0e37877f0c6d8d60b62479b217e1e0338ef0187b9fd6e65de07` |
| Evaluation SHA-256 | `96a9338946b68411d54131e1568e91a844a2d4652cad9a77cb4b1e44a67169f0` |
| Timed run | `artifacts/comparison-20261029-171844-841895/results.json`; UTC 2026-10-29T17:18:44.841895+00:00 |
| Library versions observed during evaluation | torch=2.14.1; transformers=5.20.2; peft=0.21.2; sentence-transformers=6.2.0; qdrant-client=1.20.0 |

The trace stores every accepted and pre-threshold top-3 passage, similarity score, exact prompt, raw answer, token count, per-repeat timing and knowledge-base file hashes. The raw answer files remain unchanged. Judgments are bound to the trace and individual answer hashes.

## Explicit deterministic rubric (no key required)

- **Recall@k:** fraction of labeled relevant source IDs appearing in the top-k paragraphs, macro-averaged over the 10 supported cases. All cases have one expected source; this reduces to source hit rate. MRR is the reciprocal rank of the first expected-source paragraph, averaged over those same 10 cases.
- **Precision@3:** fraction of the three retrieved paragraphs whose source ID is an expected source. This is source-label precision, not passage-level semantic relevance; useful style context can count as non-relevant under this label.
- **Required-term coverage:** case-insensitive whole-phrase matching, with word boundaries. Macro averages case coverage; micro divides all matched terms by all required terms. Only supported cases are included; empty labels do not earn free points. Synonyms and meaning are not inferred.
- **Citation validity:** at least one [source-id] citation and every cited ID belongs to accepted retrieved evidence. Citation-based faithfulness additionally requires cited IDs to belong to expected_sources and none of the exact forbidden phrases. This is a formatting/label proxy, not semantic entailment or claim-level citation coverage.
- **Style proxy:** mean of three checks: ≤80 whitespace words, first sentence ≤30 words, and no whole-word maybe/probably/perhaps. Reported over the 12 generated outputs. This deliberately simple proxy does not detect contradictions, wrong actions or instruction echoing.
- **Answer/abstain accuracy:** supported cases need an answer; unsupported cases need the exact canonical refusal or no-evidence fallback. A partly cautious statement followed by invented policy is not counted as abstention. An incorrect substantive answer can still pass this decision metric.
- **Projected citation guard:** answered if citations exist and all IDs were retrieved, rejected otherwise; clean refusals remain abstained. This is an offline projection of the project's citation requirement. It does not replace or rescore raw outputs and does not prove correctness.
- **Cap hit:** generated length equals the 80-token limit; this flags potential truncation, not certain truncation. Token counts include model-generated special tokens; text omits them.

## Shared retrieval quality and threshold evidence

| Metric | Shared result |
| --- | --- |
| recall_at_1 | 1.000 |
| recall_at_3 | 1.000 |
| accepted_recall_at_3 | 1.000 |
| precision_at_3 | 0.667 |
| reciprocal_rank | 1.000 |
| gate_accuracy | 85.7% |
| unsupported_rejection_rate | 50.0% |

The 0.30 gate accepts all 10 supported questions and 2 unsupported questions, and rejects the other 2 unsupported questions: TP=10, FP=2, TN=2, FN=0; gate precision=83.3%, supported recall=100%, unsupported specificity=50%, balanced accuracy=75%. The two false positives concern bank chargebacks and late-shipping compensation: topical similarity is not evidence that the requested fact exists.

| Case | Supported? | Top cosine score | First relevant rank | 0.30 gate |
| --- | --- | --- | --- | --- |
| eval-refund-ticket | yes | 0.7433 | 1 | accept |
| eval-duplicate-payment | yes | 0.5246 | 1 | accept |
| eval-authenticator | yes | 0.6071 | 1 | accept |
| eval-passport | yes | 0.6059 | 1 | accept |
| eval-packed-address | yes | 0.8412 | 1 | accept |
| eval-standard-shipping | yes | 0.6538 | 1 | accept |
| eval-annual-refund | yes | 0.7052 | 1 | accept |
| eval-pause | yes | 0.8908 | 1 | accept |
| eval-outage-updates | yes | 0.7079 | 1 | accept |
| eval-log-secrets | yes | 0.6541 | 1 | accept |
| eval-bank-chargeback | no | 0.6621 | n/a | accept |
| eval-shipping-compensation | no | 0.5486 | n/a | accept |
| eval-cafeteria | no | 0.1321 | n/a | reject |
| eval-bicycle | no | 0.2078 | n/a | reject |

| Threshold | Supported accepted /10 | Unsupported accepted /4 | Gate accuracy | Balanced accuracy |
| --- | --- | --- | --- | --- |
| 0.00 | 10 | 4 | 71.4% | 50.0% |
| 0.20 | 10 | 3 | 78.6% | 62.5% |
| 0.30 | 10 | 2 | 85.7% | 75.0% |
| 0.40 | 10 | 2 | 85.7% | 75.0% |
| 0.50 | 10 | 2 | 85.7% | 75.0% |
| 0.55 | 9 | 1 | 85.7% | 82.5% |
| 0.60 | 9 | 1 | 85.7% | 82.5% |
| 0.65 | 7 | 1 | 71.4% | 72.5% |
| 0.67 | 5 | 0 | 64.3% | 75.0% |
| 0.70 | 5 | 0 | 64.3% | 75.0% |
| 0.75 | 2 | 0 | 42.9% | 60.0% |
| 0.85 | 1 | 0 | 35.7% | 55.0% |
| 0.90 | 0 | 0 | 28.6% | 50.0% |

The weakest supported top score is 0.5246; the strongest unsupported score is 0.6621. Therefore no single similarity cutoff perfectly separates answerable from unanswerable cases. Raising the threshold enough to block both near-domain unsupported cases also rejects legitimate questions. The 0.30 threshold was inherited, not selected using a separate calibration set. This sweep is post-hoc benchmark diagnosis; do not tune on it and claim fresh held-out performance. Keep it provisional and calibrate with new development cases, including near-domain unknown facts.

## Aggregate automatic comparison

| Metric (denominator) | Baseline RAG | Normal LoRA |
| --- | --- | --- |
| Required-term macro coverage (10 supported) | 60.0% | 65.0% |
| Required-term micro coverage (all supported labels) | 61.1% | 61.1% |
| Full required-term coverage (count /10) | 5 | 5 |
| Valid citation present (count /10 supported) | 0 | 5 |
| Valid citation present (count /12 generated) | 0 | 7 |
| Citation-based faithfulness proxy (count /10 supported) | 0 | 5 |
| Citation-based faithfulness proxy including fallbacks (count /14) | 2 | 7 |
| Exact forbidden-phrase violations (count /14) | 0 | 0 |
| Style proxy average (12 generated) | 97.2% | 100.0% |
| All three style checks pass (count /12 generated) | 11 | 12 |
| Raw answer/abstain accuracy (14 cases) | 85.7% | 85.7% |
| Supported cases answered (count /10) | 10 | 10 |
| Unsupported cases cleanly abstained (count /4) | 2 | 2 |
| Near-domain unsupported generated refusals (count /2) | 0 | 0 |
| Projected citation-guard rejections (count /14) | 12 | 5 |
| Unsupported answers passing citation guard (count /4) | 0 | 2 |
| Generation token-cap hits (count /12) | 3 | 0 |
| Mean words (12 generated) | 40.4 | 24.5 |
| Mean generated tokens (12 generated) | 49.6 | 33.3 |

All observed citation occurrences name retrieved sources. Baseline has no citations, so occurrence precision is undefined rather than 0% or 100%; LoRA's occurrence precision does not establish that its claims follow from those sources. Neither system hits the supplied exact forbidden phrases, although both contradict explicit prohibitions. Thus an exact forbidden-string test is particularly weak here.

## Assistant judge: separate semantic assessment

Not included. Deterministic evaluation was completed without a model judge or API key.

## Latency and resource costs

Warm-up each path; alternate system order by case; synchronize MPS around timers; median generation time across repeats. Answer latency = shared retrieval + prompt preparation + median generation; startup excluded.

Three sequential repetitions per generated answer, one warmed model process on Apple MPS, one-token warm-up on each path, alternating system-first order by case. The baseline uses the same PEFT-wrapped model with its adapter disabled. Measurements include Python overhead and decoding, but exclude startup, report writing and network calls. Shared retrieval and tokenization are measured once per case and attributed to both paths; answer latency is this common work plus median generation time, an estimate rather than two separately timed end-to-end requests. No time-to-first-token measurement was made. Token-cap hits and shorter answers affect latency.

| Latency / throughput metric | Baseline | LoRA |
| --- | --- | --- |
| Mean answer latency, all 14 including fallbacks (s) | 0.2410 | 0.2092 |
| Mean answer latency, 12 generated (s) | 0.2805 | 0.2434 |
| Median answer latency, 12 generated (s) | 0.2652 | 0.2322 |
| p95 answer latency, 12 generated (s; linear interpolation) | 0.4290 | 0.3981 |
| Mean generation-only median time (s) | 0.2738 | 0.2367 |
| Aggregate generated tokens / generation second | 181.1 | 140.8 |

Measured startup with cached model files: index build **0.697 s**, base+adapter load **0.490 s**, before warm-up. These are warm-cache observations, not cold-download benchmarks.
Adapter tensor file size: **1.773 MiB**. Float32 trainable tensor payload is approximately 1.758 MiB, but training also needs the frozen model, activations, gradients and optimizer state. The small trainable percentage is not total memory or compute savings. Peak memory, energy, full cold start, deployment concurrency and financial cost were not measured. No paid model API is used by these scripts; this report uses the current assistant conversation as judge.

Operationally the hybrid adds adapter storage, training-data/version tracking, validation and retraining work. Faster total responses here can reflect fewer generated tokens; compare throughput and output length before claiming an inference speedup.

## Failure and fragile-success analysis

1. **Correct evidence, wrong polarity:** workshop refunds and authentication tokens both have the needed rule in the first passage. Both systems nevertheless begin with yes. LoRA's refund answer even quotes the prohibition after saying yes. This is a reasoning/context-following failure after successful retrieval, not missing knowledge. The term proxy can reward a copied prohibition inside an incorrect answer.
2. **Citation-shaped hallucinations:** LoRA cites account-security while saying password recovery is not recommended, and cites subscriptions while replacing pause frequency/duration with cancellation advice. Both pass the source-ID check but fail semantic correctness and grounding. Claim-level review or an answerability/entailment check is needed beyond source membership.
3. **Unsupported near-domain questions:** bank chargeback and late-shipping compensation retrieve high-similarity policies but the requested facts are absent. Both generators invent details instead of cleanly abstaining. LoRA's incorrect answers contain valid source IDs, so a citation guard can let them through; the uncited baseline would be blocked. This prevents interpreting more accepted answers as safer answers.
4. **Some genuine improvements and important regressions:** LoRA fixes the standard-shipping interval, annual-refund decision and outage-update answer. Baseline is better on passport handling and subscription pauses. Shipping and annual refunds overlap strongly with training scenarios, so these successes do not prove generalization. Outage content is grounded but uncited; standard shipping is the clearest fully acceptable hybrid answer.
5. **Prompt and passage interaction:** top-3 retrieval includes related but inapplicable rules. The baseline transfers internal finance review time to an external bank deadline; LoRA mixes shipping timing with workshops. The shared design is fair for comparing adapters, but does not isolate whether fewer/more focused passages would improve either path. That needs a new controlled experiment.

## Per-case evidence and judgments

| Case | Supported? | Baseline term coverage | LoRA term coverage | Baseline C/F /2 | LoRA C/F /2 | Preferred |
| --- | --- | --- | --- | --- | --- | --- |
| eval-refund-ticket | yes | 0.00 | 1.00 | not judged | not judged | not judged |
| eval-duplicate-payment | yes | 1.00 | 1.00 | not judged | not judged | not judged |
| eval-authenticator | yes | 0.50 | 1.00 | not judged | not judged | not judged |
| eval-passport | yes | 0.50 | 0.00 | not judged | not judged | not judged |
| eval-packed-address | yes | 0.00 | 0.50 | not judged | not judged | not judged |
| eval-standard-shipping | yes | 0.00 | 1.00 | not judged | not judged | not judged |
| eval-annual-refund | yes | 1.00 | 0.50 | not judged | not judged | not judged |
| eval-pause | yes | 1.00 | 0.00 | not judged | not judged | not judged |
| eval-outage-updates | yes | 1.00 | 1.00 | not judged | not judged | not judged |
| eval-log-secrets | yes | 1.00 | 0.50 | not judged | not judged | not judged |
| eval-bank-chargeback | no | n/a | n/a | not judged | not judged | not judged |
| eval-shipping-compensation | no | n/a | n/a | not judged | not judged | not judged |
| eval-cafeteria | no | n/a | n/a | not judged | not judged | not judged |
| eval-bicycle | no | n/a | n/a | not judged | not judged | not judged |

### eval-refund-ticket

**Question:** A digital workshop starts in 12 hours. Can the customer still receive a refund?

**Label:** supported; expected sources: ['refunds']; required terms: ['24 hours', 'outside']; forbidden phrases: ['eligible inside the final 24 hours'].

**Accepted evidence supplied identically to both systems:**

- [refunds] paragraph 1; cosine=0.7433: Digital workshop tickets are refundable until 24 hours before the workshop starts. Requests received inside the final 24 hours are outside the refund window.
- [refunds] paragraph 0; cosine=0.4754: Standard purchases are refundable within 30 calendar days when the customer provides the order ID. Approved refunds return to the original payment method.
- [refunds] paragraph 2; cosine=0.4065: For a duplicate payment, collect the invoice number and both payment references. Finance reviews a complete duplicate-payment request within three business days.

**Baseline RAG — raw output:**

```text
Yes, the customer can still receive a refund. The provided context states that the digital workshop starts in 12 hours, and the customer provides the order ID and payment references within 30 calendar days. This means that the customer can receive a refund within the 12-hour timeframe.
```

Term coverage=0.0; citations=[]; citation valid=False; faithfulness proxy=False; style proxy=1.00; answer/abstain correct=True; projected guard=rejected; tokens=61; cap hit=False; answer latency=0.3523 s.

**Normal LoRA — raw output:**

```text
Yes, the customer can still receive a refund. The digital workshop tickets are refundable until 24 hours before the workshop starts. Requests received inside the final 24 hours are outside the refund window.
```

Term coverage=1.0; citations=[]; citation valid=False; faithfulness proxy=False; style proxy=1.00; answer/abstain correct=True; projected guard=rejected; tokens=44; cap hit=False; answer latency=0.3085 s.

### eval-duplicate-payment

**Question:** What must support collect for a duplicate payment investigation?

**Label:** supported; expected sources: ['refunds']; required terms: ['invoice number', 'payment references']; forbidden phrases: [].

**Accepted evidence supplied identically to both systems:**

- [refunds] paragraph 2; cosine=0.5246: For a duplicate payment, collect the invoice number and both payment references. Finance reviews a complete duplicate-payment request within three business days.
- [refunds] paragraph 0; cosine=0.3592: Standard purchases are refundable within 30 calendar days when the customer provides the order ID. Approved refunds return to the original payment method.
- [refunds] paragraph 1; cosine=0.3209: Digital workshop tickets are refundable until 24 hours before the workshop starts. Requests received inside the final 24 hours are outside the refund window.

**Baseline RAG — raw output:**

```text
To support collect for a duplicate payment investigation, the following must be present:

1. A valid invoice number: The invoice number must be provided to the customer.
2. Both payment references: The payment references must be provided to the customer.
3. A complete duplicate-payment request within three business days: The customer must provide the order ID and both payment references within the specified timeframe.
```

Term coverage=1.0; citations=[]; citation valid=False; faithfulness proxy=False; style proxy=1.00; answer/abstain correct=True; projected guard=rejected; tokens=80; cap hit=True; answer latency=0.4234 s.

**Normal LoRA — raw output:**

```text
Collect the invoice number and both payment references to ensure a complete refund. [refunds]
```

Term coverage=1.0; citations=['refunds']; citation valid=True; faithfulness proxy=True; style proxy=1.00; answer/abstain correct=True; projected guard=answered; tokens=20; cap hit=False; answer latency=0.1555 s.

### eval-authenticator

**Question:** A customer lost their authenticator. What information and process should support use?

**Label:** supported; expected sources: ['account-security']; required terms: ['registered email', 'secure upload portal']; forbidden phrases: ['identity verification may use ordinary chat'].

**Accepted evidence supplied identically to both systems:**

- [account-security] paragraph 1; cosine=0.6071: For account recovery, ask for the registered email address and send a secure recovery link. If a customer lost access to an authenticator, identity verification must use the secure upload portal, never ordinary chat.
- [account-security] paragraph 2; cosine=0.3983: For a suspicious login, recommend resetting the password, signing out other sessions, and enabling multi-factor authentication.
- [response-style] paragraph 0; cosine=0.3621: Start with the direct answer. Follow with the next action the customer or support colleague should take.

**Baseline RAG — raw output:**

```text
The customer should use the secure upload portal to recover their authenticator.
```

Term coverage=0.5; citations=[]; citation valid=False; faithfulness proxy=False; style proxy=1.00; answer/abstain correct=True; projected guard=rejected; tokens=15; cap hit=False; answer latency=0.1090 s.

**Normal LoRA — raw output:**

```text
Ask for the registered email address and send a secure recovery link. If a customer lost access to an authenticator, use the secure upload portal to reset the password and enable multi-factor authentication. Follow the instructions to ensure the authenticator is properly secured. [account-security]
```

Term coverage=1.0; citations=['account-security']; citation valid=True; faithfulness proxy=True; style proxy=1.00; answer/abstain correct=True; projected guard=answered; tokens=57; cap hit=False; answer latency=0.3924 s.

### eval-passport

**Question:** Should a customer paste a passport scan into chat to recover an account?

**Label:** supported; expected sources: ['account-security']; required terms: ['never', 'secure upload portal']; forbidden phrases: ['passport scans are safe in chat'].

**Accepted evidence supplied identically to both systems:**

- [account-security] paragraph 1; cosine=0.6059: For account recovery, ask for the registered email address and send a secure recovery link. If a customer lost access to an authenticator, identity verification must use the secure upload portal, never ordinary chat.
- [account-security] paragraph 2; cosine=0.3627: For a suspicious login, recommend resetting the password, signing out other sessions, and enabling multi-factor authentication.
- [incidents] paragraph 1; cosine=0.3593: Diagnostic logs may include request IDs and timestamps, but must not include passwords, authentication tokens, one-time passcodes, or full payment-card details.

**Baseline RAG — raw output:**

```text
No, a customer should not paste a passport scan into chat to recover an account. This action is not recommended as it can lead to a security breach and unauthorized access to the customer's account. Instead, the customer should use the secure upload portal, secure password reset, and multi-factor authentication to recover their account.
```

Term coverage=0.5; citations=[]; citation valid=False; faithfulness proxy=False; style proxy=1.00; answer/abstain correct=True; projected guard=rejected; tokens=65; cap hit=False; answer latency=0.3616 s.

**Normal LoRA — raw output:**

```text
Password recovery is not recommended. [account-security]
```

Term coverage=0.0; citations=['account-security']; citation valid=True; faithfulness proxy=True; style proxy=1.00; answer/abstain correct=True; projected guard=answered; tokens=12; cap hit=False; answer latency=0.1123 s.

### eval-packed-address

**Question:** Can support change a delivery address after the warehouse packed the order?

**Label:** supported; expected sources: ['shipping']; required terms: ['cannot', 'carrier']; forbidden phrases: ['support can change the address after packing'].

**Accepted evidence supplied identically to both systems:**

- [shipping] paragraph 1; cosine=0.8412: A delivery address can be changed only before the warehouse packs the order. After packing, support cannot change the address. After dispatch, the customer may contact the carrier using the tracking number to request an available delivery option.

**Baseline RAG — raw output:**

```text
Yes, support can change a delivery address after the warehouse packs the order.
```

Term coverage=0.0; citations=[]; citation valid=False; faithfulness proxy=False; style proxy=1.00; answer/abstain correct=True; projected guard=rejected; tokens=16; cap hit=False; answer latency=0.1113 s.

**Normal LoRA — raw output:**

```text
Yes, support can change the address before the warehouse packs the order. After packing, support cannot change the address.
```

Term coverage=0.5; citations=[]; citation valid=False; faithfulness proxy=False; style proxy=1.00; answer/abstain correct=True; projected guard=rejected; tokens=24; cap hit=False; answer latency=0.1833 s.

### eval-standard-shipping

**Question:** What is the usual standard shipping time?

**Label:** supported; expected sources: ['shipping']; required terms: ['three to five business days']; forbidden phrases: [].

**Accepted evidence supplied identically to both systems:**

- [shipping] paragraph 0; cosine=0.6538: Standard shipping usually takes three to five business days. Express shipping usually takes one to two business days.

**Baseline RAG — raw output:**

```text
The standard shipping time for Express shipping is typically one to two business days.
```

Term coverage=0.0; citations=[]; citation valid=False; faithfulness proxy=False; style proxy=1.00; answer/abstain correct=True; projected guard=rejected; tokens=16; cap hit=False; answer latency=0.1109 s.

**Normal LoRA — raw output:**

```text
Standard shipping usually takes three to five business days. Express shipping usually takes one to two business days. [shipping]
```

Term coverage=1.0; citations=['shipping']; citation valid=True; faithfulness proxy=True; style proxy=1.00; answer/abstain correct=True; projected guard=answered; tokens=25; cap hit=False; answer latency=0.1896 s.

### eval-annual-refund

**Question:** Can an annual subscription receive a partial refund after 45 days?

**Label:** supported; expected sources: ['subscriptions']; required terms: ['not', '30-day']; forbidden phrases: ['partial refund is available after 45 days'].

**Accepted evidence supplied identically to both systems:**

- [subscriptions] paragraph 1; cosine=0.7052: Annual subscriptions are not partially refunded after the 30-day refund window. Support may record cancellation so the plan does not renew.
- [refunds] paragraph 0; cosine=0.5723: Standard purchases are refundable within 30 calendar days when the customer provides the order ID. Approved refunds return to the original payment method.
- [subscriptions] paragraph 0; cosine=0.4839: Monthly subscriptions can be cancelled before the next renewal date. Access continues until the end of the paid billing period.

**Baseline RAG — raw output:**

```text
The context does not provide any information about the refunds or cancellations. It is unclear whether the refunds are available after 45 days or if they are only available after the 30-day refund window.
```

Term coverage=1.0; citations=[]; citation valid=False; faithfulness proxy=False; style proxy=1.00; answer/abstain correct=True; projected guard=rejected; tokens=46; cap hit=False; answer latency=0.2624 s.

**Normal LoRA — raw output:**

```text
No, an annual subscription cannot receive a partial refund after the 30-day refund window. The refund is only available within the 30-day window.
```

Term coverage=0.5; citations=[]; citation valid=False; faithfulness proxy=False; style proxy=1.00; answer/abstain correct=True; projected guard=rejected; tokens=35; cap hit=False; answer latency=0.2502 s.

### eval-pause

**Question:** How often and for how long can an active subscription be paused?

**Label:** supported; expected sources: ['subscriptions']; required terms: ['once per calendar quarter', '30 days']; forbidden phrases: [].

**Accepted evidence supplied identically to both systems:**

- [subscriptions] paragraph 2; cosine=0.8908: An active subscription may be paused once per calendar quarter for up to 30 days. The pause begins on the date selected by the customer.
- [subscriptions] paragraph 0; cosine=0.6718: Monthly subscriptions can be cancelled before the next renewal date. Access continues until the end of the paid billing period.
- [subscriptions] paragraph 1; cosine=0.4828: Annual subscriptions are not partially refunded after the 30-day refund window. Support may record cancellation so the plan does not renew.

**Baseline RAG — raw output:**

```text
The approved context states that an active subscription can be paused once per calendar quarter for up to 30 days. The pause begins on the date selected by the customer.

The next action is to give the next action.
```

Term coverage=1.0; citations=[]; citation valid=False; faithfulness proxy=False; style proxy=1.00; answer/abstain correct=True; projected guard=rejected; tokens=47; cap hit=False; answer latency=0.2670 s.

**Normal LoRA — raw output:**

```text
The approved context suggests a pause on the date selected by the customer, so the subscription can be cancelled before the next renewal date. [subscriptions]
```

Term coverage=0.0; citations=['subscriptions']; citation valid=True; faithfulness proxy=True; style proxy=1.00; answer/abstain correct=True; projected guard=answered; tokens=32; cap hit=False; answer latency=0.2333 s.

### eval-outage-updates

**Question:** How frequently should updates be published during a confirmed outage?

**Label:** supported; expected sources: ['incidents']; required terms: ['30 minutes']; forbidden phrases: [].

**Accepted evidence supplied identically to both systems:**

- [incidents] paragraph 0; cosine=0.7079: During a confirmed service outage, publish status updates at least every 30 minutes until recovery. Direct customers to the public status page for the latest confirmed information.
- [subscriptions] paragraph 2; cosine=0.3666: An active subscription may be paused once per calendar quarter for up to 30 days. The pause begins on the date selected by the customer.
- [subscriptions] paragraph 0; cosine=0.3169: Monthly subscriptions can be cancelled before the next renewal date. Access continues until the end of the paid billing period.

**Baseline RAG — raw output:**

```text
The approved context states that updates should be published at least every 30 minutes until recovery. This is a reasonable requirement, as it ensures that customers receive timely and accurate information about the outage.

However, the context does not specify the frequency of updates. It is possible that updates may be published at a rate that is too high or too low, depending on the customer's preferences. For
```

Term coverage=1.0; citations=[]; citation valid=False; faithfulness proxy=False; style proxy=1.00; answer/abstain correct=True; projected guard=rejected; tokens=80; cap hit=True; answer latency=0.4263 s.

**Normal LoRA — raw output:**

```text
Publish status updates at least every 30 minutes until recovery. Direct customers to the public status page for the latest confirmed information.
```

Term coverage=1.0; citations=[]; citation valid=False; faithfulness proxy=False; style proxy=1.00; answer/abstain correct=True; projected guard=rejected; tokens=28; cap hit=False; answer latency=0.2111 s.

### eval-log-secrets

**Question:** May diagnostic logs sent to support contain authentication tokens?

**Label:** supported; expected sources: ['incidents']; required terms: ['must not', 'authentication tokens']; forbidden phrases: ['authentication tokens may be included'].

**Accepted evidence supplied identically to both systems:**

- [incidents] paragraph 1; cosine=0.6541: Diagnostic logs may include request IDs and timestamps, but must not include passwords, authentication tokens, one-time passcodes, or full payment-card details.
- [account-security] paragraph 1; cosine=0.3440: For account recovery, ask for the registered email address and send a secure recovery link. If a customer lost access to an authenticator, identity verification must use the secure upload portal, never ordinary chat.
- [account-security] paragraph 2; cosine=0.3202: For a suspicious login, recommend resetting the password, signing out other sessions, and enabling multi-factor authentication.

**Baseline RAG — raw output:**

```text
Yes, diagnostic logs may contain authentication tokens. The context states that the logs may include request IDs and timestamps, but must not include passwords, authentication tokens, one-time passcodes, or full payment-card details.
```

Term coverage=1.0; citations=[]; citation valid=False; faithfulness proxy=False; style proxy=1.00; answer/abstain correct=True; projected guard=rejected; tokens=46; cap hit=False; answer latency=0.2634 s.

**Normal LoRA — raw output:**

```text
Yes, diagnostic logs may include password, authentication tokens, and one-time passcodes. For account recovery, ask for the registered email address and send a secure upload portal. For suspicious login, request resetting the password, signing out other sessions, and enabling multi-factor authentication.
```

Term coverage=0.5; citations=[]; citation valid=False; faithfulness proxy=False; style proxy=1.00; answer/abstain correct=True; projected guard=rejected; tokens=58; cap hit=False; answer latency=0.4051 s.

### eval-bank-chargeback

**Question:** How many days will my external bank take to resolve a chargeback for a duplicate payment?

**Label:** unsupported; should abstain; expected sources: []; required terms: []; forbidden phrases: [].

**Accepted evidence supplied identically to both systems:**

- [refunds] paragraph 2; cosine=0.6621: For a duplicate payment, collect the invoice number and both payment references. Finance reviews a complete duplicate-payment request within three business days.
- [refunds] paragraph 0; cosine=0.6048: Standard purchases are refundable within 30 calendar days when the customer provides the order ID. Approved refunds return to the original payment method.
- [refunds] paragraph 1; cosine=0.4727: Digital workshop tickets are refundable until 24 hours before the workshop starts. Requests received inside the final 24 hours are outside the refund window.

**Baseline RAG — raw output:**

```text
To determine the number of days the external bank will take to resolve a chargeback for a duplicate payment, we need to calculate the total time it will take for the external bank to resolve the chargeback.

The external bank will take 3 business days to resolve a chargeback for a duplicate payment. This means that the external bank will take 3 business days to resolve the chargeback for
```

Term coverage=n/a; citations=[]; citation valid=False; faithfulness proxy=False; style proxy=0.67; answer/abstain correct=False; projected guard=rejected; tokens=80; cap hit=True; answer latency=0.4324 s.

**Normal LoRA — raw output:**

```text
The standard refund window is three business days. Requests received inside the final 24 hours are outside the refund window. [refunds]
```

Term coverage=n/a; citations=['refunds']; citation valid=True; faithfulness proxy=False; style proxy=1.00; answer/abstain correct=False; projected guard=answered; tokens=31; cap hit=False; answer latency=0.2312 s.

### eval-shipping-compensation

**Question:** What compensation amount is guaranteed if express shipping arrives late?

**Label:** unsupported; should abstain; expected sources: []; required terms: []; forbidden phrases: [].

**Accepted evidence supplied identically to both systems:**

- [shipping] paragraph 0; cosine=0.5486: Standard shipping usually takes three to five business days. Express shipping usually takes one to two business days.
- [shipping] paragraph 1; cosine=0.3784: A delivery address can be changed only before the warehouse packs the order. After packing, support cannot change the address. After dispatch, the customer may contact the carrier using the tracking number to request an available delivery option.
- [refunds] paragraph 1; cosine=0.3609: Digital workshop tickets are refundable until 24 hours before the workshop starts. Requests received inside the final 24 hours are outside the refund window.

**Baseline RAG — raw output:**

```text
The provided context does not specify the amount of compensation that can be guaranteed if express shipping arrives late. It only mentions that express shipping can be refunded up to three to five business days after the workshop starts.
```

Term coverage=n/a; citations=[]; citation valid=False; faithfulness proxy=False; style proxy=1.00; answer/abstain correct=False; projected guard=rejected; tokens=43; cap hit=False; answer latency=0.2457 s.

**Normal LoRA — raw output:**

```text
Express shipping usually arrives three to five business days before the workshop starts. Express shipping usually takes one to two business days before the workshop starts. [shipping]
```

Term coverage=n/a; citations=['shipping']; citation valid=True; faithfulness proxy=False; style proxy=1.00; answer/abstain correct=False; projected guard=answered; tokens=34; cap hit=False; answer latency=0.2480 s.

### eval-cafeteria

**Question:** Where can I find next week's cafeteria menu?

**Label:** unsupported; should abstain; expected sources: []; required terms: []; forbidden phrases: [].

**Accepted evidence supplied identically to both systems:**

None. Both outputs below are the fixed retrieval fallback; no model generation occurred.

**Baseline RAG — raw output:**

```text
The approved material is insufficient. Escalate for manual review.
```

Term coverage=n/a; citations=[]; citation valid=False; faithfulness proxy=True; style proxy=1.00; answer/abstain correct=True; projected guard=abstained; tokens=0; cap hit=False; answer latency=0.0049 s.

**Normal LoRA — raw output:**

```text
The approved material is insufficient. Escalate for manual review.
```

Term coverage=n/a; citations=[]; citation valid=False; faithfulness proxy=True; style proxy=1.00; answer/abstain correct=True; projected guard=abstained; tokens=0; cap hit=False; answer latency=0.0049 s.

### eval-bicycle

**Question:** Can I reserve a bicycle parking space near the office?

**Label:** unsupported; should abstain; expected sources: []; required terms: []; forbidden phrases: [].

**Accepted evidence supplied identically to both systems:**

None. Both outputs below are the fixed retrieval fallback; no model generation occurred.

**Baseline RAG — raw output:**

```text
The approved material is insufficient. Escalate for manual review.
```

Term coverage=n/a; citations=[]; citation valid=False; faithfulness proxy=True; style proxy=1.00; answer/abstain correct=True; projected guard=abstained; tokens=0; cap hit=False; answer latency=0.0036 s.

**Normal LoRA — raw output:**

```text
The approved material is insufficient. Escalate for manual review.
```

Term coverage=n/a; citations=[]; citation valid=False; faithfulness proxy=True; style proxy=1.00; answer/abstain correct=True; projected guard=abstained; tokens=0; cap hit=False; answer latency=0.0036 s.

## Limitations and next decision

This is one model, one training seed, one fixed small corpus and 14 related cases. There is no independent judge, no blinded assessment, no new-policy test and no repeated training experiment. The current assistant also helped implement the pipeline, creating an additional source of reviewer bias. Scalar scores depend on the stated rubric; the full judgments and outputs are retained to allow correction. The numerical gain in correctness should not be treated as a reliable population improvement. ROUGE/BLEU are omitted because there is no canonical wording and lexical similarity would obscure polarity and invented-policy failures.

The raw comparison preserves all outputs rather than enforcing citation rejection at serving time; the projected guard analysis documents the consequence. Strict deployment behavior from section 4 is therefore not implemented as a serving layer by this report. Overlength inference prompts are checked against the base model context window, rather than the 256-token training limit. The two systems still receive identical prompts. This evaluates our scripts and saved adapter, not the separately checked-in reference report or its scores.

Recommended next work: (1) keep human review and a shared citation guard; (2) improve targeted training examples for negations, after/before boundaries, unsupported near-domain requests and correct citation use; (3) test a stronger small base model under the same retriever/prompt; (4) collect a fresh development set for threshold and answerability calibration plus a sealed test set including policy changes; (5) rerun with independent or blinded semantic review and multiple training seeds. Any changes made after seeing this benchmark need new held-out questions for an honest next comparison. Retain LoRA only if those tests show grounded, correctly cited improvements that justify the lifecycle cost.

## Reproduce and audit

Raw outputs: [comparison-20261029-171844-841895/both.md](../artifacts/comparison-20261029-171844-841895/both.md). Full trace: [results.json](../artifacts/comparison-20261029-171844-841895/results.json). Numeric metrics: [deterministic_metrics.json](deterministic_metrics.json).

```bash
# Recompute deterministic metrics without an API key or judge:
.venv/bin/python scripts/evaluate.py artifacts/comparison-20261029-171844-841895/results.json
# Obtain a new timed run with the existing normal adapter, without retraining:
TIMING_REPEATS=3 ADAPTER_PATH=artifacts/support-lora-adapter .venv/bin/python scripts/compare_answers.py
```

A new generation run needs new review binding and a check of answer hashes before reusing semantic judgments. Deterministic recomputation does not require re-running generation.