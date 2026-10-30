"""Score saved outputs and write the project's evaluation/decision report.

No API key or model run is needed. Deterministic metrics work on their own.
Optional --judge-reviews adds the assistant's saved, case-by-case judgments.
"""

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
from collections import Counter
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FALLBACK = "The approved material is insufficient. Escalate for manual review."


def contains_term(answer, term):
    # Match whole phrases: "not" should not match the beginning of "nothing".
    return re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", answer, re.IGNORECASE) is not None


def percentage(values):
    return 100 * statistics.mean(values) if values else None


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def wilson(successes, total):
    # A descriptive 95% interval for a binary rate, useful for showing small-n uncertainty.
    p = successes / total
    z = 1.96
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [100 * (center - margin), 100 * (center + margin)]


def evaluate(trace, cases, reviews=None):
    labels = {case["id"]: case for case in cases}
    if len(labels) != len(cases) or set(labels) != {row["id"] for row in trace["cases"]}:
        raise ValueError("Trace and evaluation IDs must match uniquely.")
    if len(trace["cases"]) != len(cases):
        raise ValueError("Duplicate or missing trace cases.")
    judge = {row["id"]: row for row in reviews["cases"]} if reviews else {}
    if reviews and set(judge) != set(labels):
        raise ValueError("Judge reviews must cover every case.")

    rows = []
    for row in trace["cases"]:
        label = labels[row["id"]]
        if row["question"] != label["question"]:
            raise ValueError("Question text changed since generation.")
        hits = row["all_retrieved"]
        expected = set(label["expected_sources"])
        accepted = {hit["source_id"] for hit in row["evidence"]}
        relevant_ranks = [i + 1 for i, hit in enumerate(hits) if hit["source_id"] in expected]
        first_rank = relevant_ranks[0] if relevant_ranks else None
        supported = not label["should_abstain"]
        item = {
            "id": row["id"], "supported": supported,
            "top_score": hits[0]["score"] if hits else None,
            "retrieval_gate_correct": bool(row["evidence"]) == supported,
            "first_relevant_rank": first_rank,
            "recall_at_1": len(expected & {hit["source_id"] for hit in hits[:1]}) / len(expected) if expected else None,
            "recall_at_3": len(expected & {hit["source_id"] for hit in hits}) / len(expected) if expected else None,
            "accepted_recall_at_3": len(expected & accepted) / len(expected) if expected else None,
            "precision_at_3": sum(hit["source_id"] in expected for hit in hits) / len(hits) if supported and hits else None,
            "reciprocal_rank": 1 / first_rank if first_rank else 0,
        }
        for system in ("baseline", "lora"):
            answer = row[system]["answer"]
            citations = re.findall(r"\[([^\[\]]+)\]", answer)
            citations = [citation.strip() for citation in citations]
            valid = bool(citations) and set(citations).issubset(accepted)
            abstained = row["status"] == "no-evidence" or answer.strip() == FALLBACK
            matches = sum(contains_term(answer, term) for term in label["required_terms"])
            forbidden = [term for term in label["forbidden_terms"] if contains_term(answer, term)]
            first_sentence = re.split(r"[.!?]", answer, maxsplit=1)[0]
            style_checks = [
                len(answer.split()) <= 80,
                len(first_sentence.split()) <= 30,
                not any(contains_term(answer, word) for word in ("maybe", "probably", "perhaps")),
            ]
            metrics = {
                "required_term_matches": matches,
                "required_term_count": len(label["required_terms"]),
                "term_coverage": matches / len(label["required_terms"]) if supported and label["required_terms"] else None,
                "forbidden_phrases": forbidden,
                "citations": citations,
                "citation_valid": valid,
                "citation_based_faithfulness_proxy": (valid and set(citations).issubset(expected) and not forbidden) if supported else abstained,
                "style_proxy": statistics.mean(style_checks),
                "all_style_checks": all(style_checks),
                "abstained": abstained,
                "decision_correct": (not abstained) if supported else abstained,
                "guard_status": "abstained" if abstained else "answered" if valid else "rejected",
                "generated": row["status"] == "generated",
                "cap_hit": row[system]["generated_tokens"] == trace["configuration"]["max_new_tokens"],
                "word_count": len(answer.split()),
                "generation_seconds": row[system]["generation_seconds"],
                "answer_latency_seconds": row[system]["answer_latency_seconds"],
                "generated_tokens": row[system]["generated_tokens"],
            }
            if reviews:
                review = judge[row["id"]][system]
                if hashlib.sha256(answer.encode()).hexdigest() != review["answer_sha256"]:
                    raise ValueError(f"Judgment does not match the output: {row['id']}/{system}")
                for name in ("correctness", "relevance", "faithfulness", "style"):
                    if review[name] not in (0, 1, 2):
                        raise ValueError("Judge scores must be 0, 1, or 2.")
                metrics["judge"] = review
                metrics["content_acceptable"] = review["correctness"] == 2 and review["faithfulness"] == 2
                metrics["fully_acceptable"] = metrics["content_acceptable"] and (abstained if not supported else valid and not metrics["cap_hit"])
            item[system] = metrics
        if reviews:
            item["pairwise_preference"] = judge[row["id"]]["pairwise_preference"]
        rows.append(item)

    supported_rows = [row for row in rows if row["supported"]]
    unsupported_rows = [row for row in rows if not row["supported"]]
    summary = {"case_count": len(rows), "supported_count": len(supported_rows), "unsupported_count": len(unsupported_rows)}
    summary["retrieval"] = {
        name: statistics.mean(row[name] for row in supported_rows)
        for name in ("recall_at_1", "recall_at_3", "accepted_recall_at_3", "precision_at_3", "reciprocal_rank")
    }
    summary["retrieval"]["gate_accuracy"] = percentage([row["retrieval_gate_correct"] for row in rows])
    summary["retrieval"]["unsupported_rejection_rate"] = percentage([row["retrieval_gate_correct"] for row in unsupported_rows])
    summary["systems"] = {}
    for system in ("baseline", "lora"):
        all_metrics = [row[system] for row in rows]
        supported_metrics = [row[system] for row in supported_rows]
        unsupported_metrics = [row[system] for row in unsupported_rows]
        generated = [item for item in all_metrics if item["generated"]]
        values = {
            "term_coverage_macro_pct": percentage([item["term_coverage"] for item in supported_metrics]),
            "term_coverage_micro_pct": 100 * sum(item["required_term_matches"] for item in supported_metrics) / sum(item["required_term_count"] for item in supported_metrics),
            "full_term_coverage_count": sum(item["term_coverage"] == 1 for item in supported_metrics),
            "citation_valid_count_supported": sum(item["citation_valid"] for item in supported_metrics),
            "citation_valid_count_generated": sum(item["citation_valid"] for item in generated),
            "citation_occurrences": sum(len(item["citations"]) for item in generated),
            "citation_proxy_count_supported": sum(item["citation_based_faithfulness_proxy"] for item in supported_metrics),
            "citation_proxy_count_all": sum(item["citation_based_faithfulness_proxy"] for item in all_metrics),
            "forbidden_phrase_case_count": sum(bool(item["forbidden_phrases"]) for item in all_metrics),
            "style_proxy_pct": percentage([item["style_proxy"] for item in generated]),
            "all_style_checks_count": sum(item["all_style_checks"] for item in generated),
            "decision_accuracy_pct": percentage([item["decision_correct"] for item in all_metrics]),
            "supported_answer_count": sum(not item["abstained"] for item in supported_metrics),
            "unsupported_abstain_count": sum(item["abstained"] for item in unsupported_metrics),
            "generated_unsupported_abstain_count": sum(item["abstained"] for item in unsupported_metrics if item["generated"]),
            "citation_guard_rejections": sum(item["guard_status"] == "rejected" for item in all_metrics),
            "unsupported_answers_passing_guard": sum(item["guard_status"] == "answered" for item in unsupported_metrics),
            "cap_hit_count": sum(item["cap_hit"] for item in generated),
            "mean_generated_words": statistics.mean(item["word_count"] for item in generated),
            "mean_generated_tokens": statistics.mean(item["generated_tokens"] for item in generated),
            "mean_answer_latency_seconds_all": statistics.mean(item["answer_latency_seconds"] for item in all_metrics),
            "mean_answer_latency_seconds_generated": statistics.mean(item["answer_latency_seconds"] for item in generated),
            "median_answer_latency_seconds_generated": statistics.median(item["answer_latency_seconds"] for item in generated),
            "p95_answer_latency_seconds_generated": percentile([item["answer_latency_seconds"] for item in generated], .95),
            "mean_generation_seconds": statistics.mean(item["generation_seconds"] for item in generated),
            "generation_tokens_per_second": sum(item["generated_tokens"] for item in generated) / sum(item["generation_seconds"] for item in generated),
        }
        if reviews:
            values["judge_supported_mean"] = {name: statistics.mean(item["judge"][name] for item in supported_metrics) for name in ("correctness", "relevance", "faithfulness", "style")}
            values["judge_generated_mean"] = {name: statistics.mean(item["judge"][name] for item in generated) for name in ("correctness", "relevance", "faithfulness", "style")}
            values["judge_correct_count_supported"] = sum(item["judge"]["correctness"] == 2 for item in supported_metrics)
            values["judge_correct_wilson95_supported"] = wilson(values["judge_correct_count_supported"], len(supported_rows))
            values["judge_faithful_count_supported"] = sum(item["judge"]["faithfulness"] == 2 for item in supported_metrics)
            values["content_acceptable_count_supported"] = sum(item["content_acceptable"] for item in supported_metrics)
            values["fully_acceptable_count_supported"] = sum(item["fully_acceptable"] for item in supported_metrics)
            values["fully_acceptable_count_all"] = sum(item["fully_acceptable"] for item in all_metrics)
        summary["systems"][system] = values

    thresholds = []
    for threshold in (0, .2, .3, .4, .5, .55, .6, .65, .67, .7, .75, .85, .9):
        true_positive = sum(row["top_score"] is not None and row["top_score"] >= threshold for row in supported_rows)
        false_positive = sum(row["top_score"] is not None and row["top_score"] >= threshold for row in unsupported_rows)
        true_negative = len(unsupported_rows) - false_positive
        thresholds.append({"threshold": threshold, "supported_accepted": true_positive, "unsupported_accepted": false_positive,
                           "accuracy_pct": 100 * (true_positive + true_negative) / len(rows),
                           "balanced_accuracy_pct": 50 * (true_positive / len(supported_rows) + true_negative / len(unsupported_rows))})
    summary["threshold_diagnostics"] = thresholds
    if reviews:
        summary["pairwise_generated"] = dict(Counter(row["pairwise_preference"] for row in rows if row["baseline"]["generated"]))
        summary["pairwise_all"] = dict(Counter(row["pairwise_preference"] for row in rows))
        gains = sum(row["baseline"]["judge"]["correctness"] != 2 and row["lora"]["judge"]["correctness"] == 2 for row in supported_rows)
        losses = sum(row["baseline"]["judge"]["correctness"] == 2 and row["lora"]["judge"]["correctness"] != 2 for row in supported_rows)
        discordant = gains + losses
        p_value = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(gains, losses) + 1)) / 2**discordant) if discordant else 1.0
        summary["paired_correctness"] = {"gains": gains, "losses": losses, "exact_mcnemar_two_sided_p": p_value}
    return {"summary": summary, "cases": rows}


def render_report(trace, evaluation, labels, reviews, trace_path):
    summary = evaluation["summary"]
    config = trace["configuration"]
    training = config["training"]
    baseline, lora = [summary["systems"][name] for name in ("baseline", "lora")]
    lines = []
    def add(text=""):
        lines.append(text)
    def table(header, body):
        add("| " + " | ".join(header) + " |")
        add("| " + " | ".join("---" for _ in header) + " |")
        for row in body:
            add("| " + " | ".join(str(value) for value in row) + " |")
        add()
    def metric(label, key, fmt="{:.1f}%"):
        return [label, fmt.format(baseline[key]), fmt.format(lora[key])]

    hardware = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"], text=True).splitlines()

    add("# Evaluation and decision report: baseline RAG versus normal LoRA")
    add()
    add("This report implements project sections 5 and 6. It evaluates only the baseline and the original adapter trained at learning rate **0.0002**. The 100× and 500× experiments are excluded.")
    add()
    add("## Decision")
    add()
    if reviews:
        add("**Revise the hybrid design; keep it as a learning experiment, not as an automatic support assistant.** LoRA improves some answers and citation frequency, but introduces regressions and retains serious policy reversals. The raw-answer comparison and the stricter citation-guard projection are shown separately.")
        add()
        add(f"The judge rates {baseline['judge_correct_count_supported']}/10 supported baseline answers and {lora['judge_correct_count_supported']}/10 LoRA answers fully correct. Only {baseline['fully_acceptable_count_supported']}/10 and {lora['fully_acceptable_count_supported']}/10 respectively are fully correct, grounded, cited and below the token cap. Both systems abstain on 2/4 unsupported cases; those two refusals are retrieval fallbacks, not generated judgments.")
    else:
        add("Deterministic checks alone cannot establish semantic answer quality. Retain both raw answer paths for review; a quality recommendation requires semantic judgments. Rerun with --judge-reviews to include the saved assistant review.")
    add()
    add("## Experimental scope and fairness")
    add()
    add("There are **14 held-out questions: 10 supported and 4 unsupported**, 6 Markdown sources, 16 paragraph chunks and 16 training examples. Retrieval is executed once per case. Both paths receive the same accepted passages, identical serialized prompt and generation settings. The baseline temporarily disables the adapter on the same frozen base model; the hybrid enables it. Dropout is off for both during evaluation. Expected sources, required terms, forbidden claims and judge references are never placed in generation prompts.")
    add()
    add("The timed rerun reproduced all 28 answers from the original normal-rate run exactly. Each generated answer was repeated three times and the script checked token-for-token equality across repeats. Latency repetitions are timing samples, not independent quality samples. No retraining or hyperparameter changes were performed for this evaluation.")
    add()
    add("Training and evaluation IDs and normalized questions are separate, but policies and scenarios overlap. For example, standard shipping is a training paraphrase, the annual-refund scenario already appears in training, and diagnostic-log restrictions appear in training. This is an in-domain exercise, not a sealed test of unseen policies. The evaluator's prior access to outputs and labels is disclosed below.")
    add()
    add("## Configuration and provenance")
    add()
    table(["Component", "Configuration"], [
        ["Base model", f"`{training['base_model']}`"],
        ["Base revision", f"`{training['base_model_revision']}`"],
        ["Embedding model", f"`{config['embedding_model']}`"],
        ["Embedding revision", f"`{config['embedding_revision']}`"],
        ["Retrieval", f"Qdrant local in-memory; cosine similarity; top_k={config['top_k']}; threshold={config['min_score']}; title + paragraph embedded; source ID/title/position/text preserved"],
        ["Chunking", "Blank-line paragraphs; Markdown heading retained as title; no overlap; rebuilt once per run"],
        ["Prompt", "Native chat template; shared system instruction; [source-id] passage text + question; assistant generation prefix"],
        ["Generation", f"Greedy (do_sample=False); maximum {config['max_new_tokens']} new tokens; float32; {config['device']}; seed={config['seed']}; model context-window check, no prompt truncation"],
        ["Evaluation machine", f"{hardware[0]}; {int(hardware[1]) / 1024**3:.0f} GiB unified memory; local macOS"],
        ["Adapter", f"`{training['adapter_path']}`"],
        ["LoRA", f"targets={training['lora_target_modules']}; rank={training['lora_rank']}; alpha={training['lora_alpha']}; dropout={training['lora_dropout']}"],
        ["Training", f"epochs={training['epochs']}; learning_rate={training['learning_rate']}; batch_size=1; 128 updates; AdamW; betas=(0.9, 0.999); epsilon=1e-8; weight_decay=0.01; no scheduler or gradient clipping; fixed example order"],
        ["Training tokens", f"Maximum {training['max_sequence_length']} prompt+answer tokens; answer-only loss; prompt labels -100; no silent truncation; float32"],
        ["Parameters", f"{training['trainable_parameters']:,} trainable of {training['total_parameters']:,} including adapter ({training['trainable_percentage']:.3f}%)"],
        ["Training duration", f"{training['training_seconds']:.3f} s; optimization + saving/summary work, excludes downloading, loading and tokenization"],
        ["Training loss", f"{training['epoch_losses'][0]:.4f} → {training['epoch_losses'][-1]:.4f}; diagnostic only, no held-out quality claim"],
        ["Training SHA-256", f"`{training['training_sha256']}`"],
        ["Evaluation SHA-256", f"`{config['evaluation_sha256']}`"],
        ["Timed run", f"`{trace_path.relative_to(ROOT)}`; UTC {config['created_at_utc']}"],
        ["Library versions observed during evaluation", "; ".join(f"{name}={version(name)}" for name in ("torch", "transformers", "peft", "sentence-transformers", "qdrant-client"))],
    ])
    add("The trace stores every accepted and pre-threshold top-3 passage, similarity score, exact prompt, raw answer, token count, per-repeat timing and knowledge-base file hashes. The raw answer files remain unchanged. Judgments are bound to the trace and individual answer hashes.")
    add()
    add("## Explicit deterministic rubric (no key required)")
    add()
    add("- **Recall@k:** fraction of labeled relevant source IDs appearing in the top-k paragraphs, macro-averaged over the 10 supported cases. All cases have one expected source; this reduces to source hit rate. MRR is the reciprocal rank of the first expected-source paragraph, averaged over those same 10 cases.")
    add("- **Precision@3:** fraction of the three retrieved paragraphs whose source ID is an expected source. This is source-label precision, not passage-level semantic relevance; useful style context can count as non-relevant under this label.")
    add("- **Required-term coverage:** case-insensitive whole-phrase matching, with word boundaries. Macro averages case coverage; micro divides all matched terms by all required terms. Only supported cases are included; empty labels do not earn free points. Synonyms and meaning are not inferred.")
    add("- **Citation validity:** at least one [source-id] citation and every cited ID belongs to accepted retrieved evidence. Citation-based faithfulness additionally requires cited IDs to belong to expected_sources and none of the exact forbidden phrases. This is a formatting/label proxy, not semantic entailment or claim-level citation coverage.")
    add("- **Style proxy:** mean of three checks: ≤80 whitespace words, first sentence ≤30 words, and no whole-word maybe/probably/perhaps. Reported over the 12 generated outputs. This deliberately simple proxy does not detect contradictions, wrong actions or instruction echoing.")
    add("- **Answer/abstain accuracy:** supported cases need an answer; unsupported cases need the exact canonical refusal or no-evidence fallback. A partly cautious statement followed by invented policy is not counted as abstention. An incorrect substantive answer can still pass this decision metric.")
    add("- **Projected citation guard:** answered if citations exist and all IDs were retrieved, rejected otherwise; clean refusals remain abstained. This is an offline projection of the project's citation requirement. It does not replace or rescore raw outputs and does not prove correctness.")
    add("- **Cap hit:** generated length equals the 80-token limit; this flags potential truncation, not certain truncation. Token counts include model-generated special tokens; text omits them.")
    add()
    add("## Shared retrieval quality and threshold evidence")
    add()
    table(["Metric", "Shared result"], [[name, f"{value:.3f}" if name not in ("gate_accuracy", "unsupported_rejection_rate") else f"{value:.1f}%"] for name, value in summary["retrieval"].items()])
    add("The 0.30 gate accepts all 10 supported questions and 2 unsupported questions, and rejects the other 2 unsupported questions: TP=10, FP=2, TN=2, FN=0; gate precision=83.3%, supported recall=100%, unsupported specificity=50%, balanced accuracy=75%. The two false positives concern bank chargebacks and late-shipping compensation: topical similarity is not evidence that the requested fact exists.")
    add()
    table(["Case", "Supported?", "Top cosine score", "First relevant rank", "0.30 gate"], [
        [row["id"], "yes" if row["supported"] else "no", f"{row['top_score']:.4f}" if row["top_score"] is not None else "none", row["first_relevant_rank"] or "n/a", "accept" if raw["evidence"] else "reject"]
        for row, raw in zip(evaluation["cases"], trace["cases"])
    ])
    table(["Threshold", "Supported accepted /10", "Unsupported accepted /4", "Gate accuracy", "Balanced accuracy"], [
        [f"{row['threshold']:.2f}", row["supported_accepted"], row["unsupported_accepted"], f"{row['accuracy_pct']:.1f}%", f"{row['balanced_accuracy_pct']:.1f}%"] for row in summary["threshold_diagnostics"]
    ])
    add("The weakest supported top score is 0.5246; the strongest unsupported score is 0.6621. Therefore no single similarity cutoff perfectly separates answerable from unanswerable cases. Raising the threshold enough to block both near-domain unsupported cases also rejects legitimate questions. The 0.30 threshold was inherited, not selected using a separate calibration set. This sweep is post-hoc benchmark diagnosis; do not tune on it and claim fresh held-out performance. Keep it provisional and calibrate with new development cases, including near-domain unknown facts.")
    add()
    add("## Aggregate automatic comparison")
    add()
    table(["Metric (denominator)", "Baseline RAG", "Normal LoRA"], [
        metric("Required-term macro coverage (10 supported)", "term_coverage_macro_pct"),
        metric("Required-term micro coverage (all supported labels)", "term_coverage_micro_pct"),
        metric("Full required-term coverage (count /10)", "full_term_coverage_count", "{}"),
        metric("Valid citation present (count /10 supported)", "citation_valid_count_supported", "{}"),
        metric("Valid citation present (count /12 generated)", "citation_valid_count_generated", "{}"),
        metric("Citation-based faithfulness proxy (count /10 supported)", "citation_proxy_count_supported", "{}"),
        metric("Citation-based faithfulness proxy including fallbacks (count /14)", "citation_proxy_count_all", "{}"),
        metric("Exact forbidden-phrase violations (count /14)", "forbidden_phrase_case_count", "{}"),
        metric("Style proxy average (12 generated)", "style_proxy_pct"),
        metric("All three style checks pass (count /12 generated)", "all_style_checks_count", "{}"),
        metric("Raw answer/abstain accuracy (14 cases)", "decision_accuracy_pct"),
        metric("Supported cases answered (count /10)", "supported_answer_count", "{}"),
        metric("Unsupported cases cleanly abstained (count /4)", "unsupported_abstain_count", "{}"),
        metric("Near-domain unsupported generated refusals (count /2)", "generated_unsupported_abstain_count", "{}"),
        metric("Projected citation-guard rejections (count /14)", "citation_guard_rejections", "{}"),
        metric("Unsupported answers passing citation guard (count /4)", "unsupported_answers_passing_guard", "{}"),
        metric("Generation token-cap hits (count /12)", "cap_hit_count", "{}"),
        metric("Mean words (12 generated)", "mean_generated_words", "{:.1f}"),
        metric("Mean generated tokens (12 generated)", "mean_generated_tokens", "{:.1f}"),
    ])
    add("All observed citation occurrences name retrieved sources. Baseline has no citations, so occurrence precision is undefined rather than 0% or 100%; LoRA's occurrence precision does not establish that its claims follow from those sources. Neither system hits the supplied exact forbidden phrases, although both contradict explicit prohibitions. Thus an exact forbidden-string test is particularly weak here.")
    add()
    add("## Assistant judge: separate semantic assessment")
    add()
    if reviews:
        add(f"Reviewer: {reviews['judge']}. **Not blinded:** the reviewer knows which system produced each answer and has seen the earlier outputs. This is a single assistant judgment, not independent human ground truth, a second-model API run, or a measured consensus. References were derived from the approved policies. No additional model service was called.")
        add()
        for name, definition in reviews["scale"].items():
            add(f"- **{name}:** {definition}")
        add()
        add("Citation compliance is measured separately from semantic faithfulness. A citation-free answer can be factually grounded, while a cited answer can contradict the policy. Fully acceptable supported answers require correctness=2, faithfulness=2, valid citations and no token-cap hit. Unsupported cases require a clean grounded abstention; fixed retrieval fallbacks are identified separately.")
        add()
        body = []
        for name in ("correctness", "relevance", "faithfulness", "style"):
            body.append([f"Mean {name} /2 (10 supported)", f"{baseline['judge_supported_mean'][name]:.2f}", f"{lora['judge_supported_mean'][name]:.2f}"])
        for name in ("correctness", "relevance", "faithfulness", "style"):
            body.append([f"Mean {name} /2 (12 generated, including 2 unsupported)", f"{baseline['judge_generated_mean'][name]:.2f}", f"{lora['judge_generated_mean'][name]:.2f}"])
        body.extend([
            metric("Fully correct supported answers (count /10)", "judge_correct_count_supported", "{}"),
            metric("Fully faithful supported answers (count /10)", "judge_faithful_count_supported", "{}"),
            metric("Correct and grounded, ignoring citations (count /10)", "content_acceptable_count_supported", "{}"),
            metric("Fully acceptable, including citations (count /10 supported)", "fully_acceptable_count_supported", "{}"),
            metric("Fully acceptable including 2 fixed fallbacks (count /14)", "fully_acceptable_count_all", "{}"),
        ])
        table(["Judge metric", "Baseline RAG", "Normal LoRA"], body)
        add("The supported-case fully-correct rate rises from 20% to 40% (+20 percentage points), while the mean correctness score rises from 0.70 to 1.00 on the 0–2 scale. The macro term-coverage gain is only +5 percentage points and micro coverage is unchanged. Mean semantic relevance falls from 1.80 to 1.50, reflecting off-topic or substituted answers on passport recovery and pauses; a style gain alone therefore does not settle the comparison.")
        add()
        pair = summary["pairwise_generated"]
        add(f"Among 12 generated pairs, LoRA is preferred in **{pair.get('lora', 0)}**, baseline in **{pair.get('baseline', 0)}**, with **{pair.get('tie', 0)} ties**. Both fixed fallback pairs tie. Preference means relative usefulness, not approval to serve the answer. A misleading but partially informative answer can win against a wholly incorrect one.")
        add()
        for system, name in (("baseline", "Baseline"), ("lora", "LoRA")):
            interval = summary["systems"][system]["judge_correct_wilson95_supported"]
            add(f"{name} fully-correct supported rate: {summary['systems'][system]['judge_correct_count_supported']}/10; descriptive 95% Wilson interval **{interval[0]:.1f}–{interval[1]:.1f}%**.")
        paired = summary["paired_correctness"]
        add(f"Paired fully-correct outcomes have {paired['gains']} LoRA gains and {paired['losses']} losses; exact two-sided McNemar p={paired['exact_mcnemar_two_sided_p']:.4f}. These calculations assume independent cases; shared policies and related scenarios weaken that assumption. They do not capture judge uncertainty. This tiny sample does not establish a statistically reliable quality improvement.")
    else:
        add("Not included. Deterministic evaluation was completed without a model judge or API key.")
    add()
    add("## Latency and resource costs")
    add()
    add(config["timing_protocol"])
    add()
    add("Three sequential repetitions per generated answer, one warmed model process on Apple MPS, one-token warm-up on each path, alternating system-first order by case. The baseline uses the same PEFT-wrapped model with its adapter disabled. Measurements include Python overhead and decoding, but exclude startup, report writing and network calls. Shared retrieval and tokenization are measured once per case and attributed to both paths; answer latency is this common work plus median generation time, an estimate rather than two separately timed end-to-end requests. No time-to-first-token measurement was made. Token-cap hits and shorter answers affect latency.")
    add()
    table(["Latency / throughput metric", "Baseline", "LoRA"], [
        metric("Mean answer latency, all 14 including fallbacks (s)", "mean_answer_latency_seconds_all", "{:.4f}"),
        metric("Mean answer latency, 12 generated (s)", "mean_answer_latency_seconds_generated", "{:.4f}"),
        metric("Median answer latency, 12 generated (s)", "median_answer_latency_seconds_generated", "{:.4f}"),
        metric("p95 answer latency, 12 generated (s; linear interpolation)", "p95_answer_latency_seconds_generated", "{:.4f}"),
        metric("Mean generation-only median time (s)", "mean_generation_seconds", "{:.4f}"),
        metric("Aggregate generated tokens / generation second", "generation_tokens_per_second", "{:.1f}"),
    ])
    add(f"Measured startup with cached model files: index build **{config['index_build_seconds']:.3f} s**, base+adapter load **{config['model_load_seconds']:.3f} s**, before warm-up. These are warm-cache observations, not cold-download benchmarks.")
    adapter = Path(training["adapter_path"])
    weight_bytes = sum(file.stat().st_size for file in adapter.glob("*.safetensors"))
    add(f"Adapter tensor file size: **{weight_bytes / 1024**2:.3f} MiB**. Float32 trainable tensor payload is approximately {training['trainable_parameters'] * 4 / 1024**2:.3f} MiB, but training also needs the frozen model, activations, gradients and optimizer state. The small trainable percentage is not total memory or compute savings. Peak memory, energy, full cold start, deployment concurrency and financial cost were not measured. No paid model API is used by these scripts; this report uses the current assistant conversation as judge.")
    add()
    add("Operationally the hybrid adds adapter storage, training-data/version tracking, validation and retraining work. Faster total responses here can reflect fewer generated tokens; compare throughput and output length before claiming an inference speedup.")
    add()
    add("## Failure and fragile-success analysis")
    add()
    add("1. **Correct evidence, wrong polarity:** workshop refunds and authentication tokens both have the needed rule in the first passage. Both systems nevertheless begin with yes. LoRA's refund answer even quotes the prohibition after saying yes. This is a reasoning/context-following failure after successful retrieval, not missing knowledge. The term proxy can reward a copied prohibition inside an incorrect answer.")
    add("2. **Citation-shaped hallucinations:** LoRA cites account-security while saying password recovery is not recommended, and cites subscriptions while replacing pause frequency/duration with cancellation advice. Both pass the source-ID check but fail semantic correctness and grounding. Claim-level review or an answerability/entailment check is needed beyond source membership.")
    add("3. **Unsupported near-domain questions:** bank chargeback and late-shipping compensation retrieve high-similarity policies but the requested facts are absent. Both generators invent details instead of cleanly abstaining. LoRA's incorrect answers contain valid source IDs, so a citation guard can let them through; the uncited baseline would be blocked. This prevents interpreting more accepted answers as safer answers.")
    add("4. **Some genuine improvements and important regressions:** LoRA fixes the standard-shipping interval, annual-refund decision and outage-update answer. Baseline is better on passport handling and subscription pauses. Shipping and annual refunds overlap strongly with training scenarios, so these successes do not prove generalization. Outage content is grounded but uncited; standard shipping is the clearest fully acceptable hybrid answer.")
    add("5. **Prompt and passage interaction:** top-3 retrieval includes related but inapplicable rules. The baseline transfers internal finance review time to an external bank deadline; LoRA mixes shipping timing with workshops. The shared design is fair for comparing adapters, but does not isolate whether fewer/more focused passages would improve either path. That needs a new controlled experiment.")
    add()
    add("## Per-case evidence and judgments")
    add()
    label_map = {label["id"]: label for label in labels}
    review_map = {row["id"]: row for row in reviews["cases"]} if reviews else {}
    table(["Case", "Supported?", "Baseline term coverage", "LoRA term coverage", "Baseline C/F /2", "LoRA C/F /2", "Preferred"], [
        [row["id"], "yes" if row["supported"] else "no", *[f"{row[name]['term_coverage']:.2f}" if row[name]["term_coverage"] is not None else "n/a" for name in ("baseline", "lora")],
         *[f"{row[name]['judge']['correctness']}/{row[name]['judge']['faithfulness']}" if reviews else "not judged" for name in ("baseline", "lora")], row.get("pairwise_preference", "not judged")]
        for row in evaluation["cases"]
    ])
    for raw, row in zip(trace["cases"], evaluation["cases"]):
        label = label_map[row["id"]]
        add(f"### {row['id']}")
        add()
        add(f"**Question:** {raw['question']}")
        add()
        add(f"**Label:** {'supported' if row['supported'] else 'unsupported; should abstain'}; expected sources: {label['expected_sources']}; required terms: {label['required_terms']}; forbidden phrases: {label['forbidden_terms']}.")
        if reviews:
            add()
            add(f"**Policy-derived reference:** {review_map[row['id']]['reference_answer']}")
        add()
        add("**Accepted evidence supplied identically to both systems:**")
        add()
        for hit in raw["evidence"]:
            add(f"- [{hit['source_id']}] paragraph {hit['position']}; cosine={hit['score']:.4f}: {hit['text']}")
        if not raw["evidence"]:
            add("None. Both outputs below are the fixed retrieval fallback; no model generation occurred.")
        add()
        for name in ("baseline", "lora"):
            item = row[name]
            add(f"**{'Baseline RAG' if name == 'baseline' else 'Normal LoRA'} — raw output:**")
            add()
            answer = raw[name]["answer"]
            fence = "`" * max(3, answer.count("`") + 1)
            add(f"{fence}text\n{answer}\n{fence}")
            add()
            add(f"Term coverage={item['term_coverage'] if item['term_coverage'] is not None else 'n/a'}; citations={item['citations']}; citation valid={item['citation_valid']}; faithfulness proxy={item['citation_based_faithfulness_proxy']}; style proxy={item['style_proxy']:.2f}; answer/abstain correct={item['decision_correct']}; projected guard={item['guard_status']}; tokens={item['generated_tokens']}; cap hit={item['cap_hit']}; answer latency={item['answer_latency_seconds']:.4f} s.")
            add()
            if reviews:
                judged = item["judge"]
                add(f"Judge scores /2: correctness={judged['correctness']}, relevance={judged['relevance']}, faithfulness={judged['faithfulness']}, style={judged['style']}. **Reason:** {judged['rationale']}")
                add()
        if reviews:
            add(f"**Pairwise preference:** {row['pairwise_preference']}.")
            add()
    add("## Limitations and next decision")
    add()
    add("This is one model, one training seed, one fixed small corpus and 14 related cases. There is no independent judge, no blinded assessment, no new-policy test and no repeated training experiment. The current assistant also helped implement the pipeline, creating an additional source of reviewer bias. Scalar scores depend on the stated rubric; the full judgments and outputs are retained to allow correction. The numerical gain in correctness should not be treated as a reliable population improvement. ROUGE/BLEU are omitted because there is no canonical wording and lexical similarity would obscure polarity and invented-policy failures.")
    add()
    add("The raw comparison preserves all outputs rather than enforcing citation rejection at serving time; the projected guard analysis documents the consequence. Strict deployment behavior from section 4 is therefore not implemented as a serving layer by this report. Overlength inference prompts are checked against the base model context window, rather than the 256-token training limit. The two systems still receive identical prompts. This evaluates our scripts and saved adapter, not the separately checked-in reference report or its scores.")
    add()
    add("Recommended next work: (1) keep human review and a shared citation guard; (2) improve targeted training examples for negations, after/before boundaries, unsupported near-domain requests and correct citation use; (3) test a stronger small base model under the same retriever/prompt; (4) collect a fresh development set for threshold and answerability calibration plus a sealed test set including policy changes; (5) rerun with independent or blinded semantic review and multiple training seeds. Any changes made after seeing this benchmark need new held-out questions for an honest next comparison. Retain LoRA only if those tests show grounded, correctly cited improvements that justify the lifecycle cost.")
    add()
    add("## Reproduce and audit")
    add()
    add(f"Raw outputs: [{trace_path.parent.name}/both.md](../{trace_path.parent.relative_to(ROOT)}/both.md). Full trace: [results.json](../{trace_path.relative_to(ROOT)}). Numeric metrics: [evaluation_metrics.json](evaluation_metrics.json).")
    add()
    if reviews:
        add("Semantic scores and reasons: [judge_reviews.json](judge_reviews.json). They were entered by the current assistant; the Python evaluator verifies and aggregates them, not generates new model judgments.")
        add()
    add("```bash")
    add(f"# Recompute deterministic metrics without an API key or judge:")
    add(f".venv/bin/python scripts/evaluate.py {trace_path.relative_to(ROOT)}")
    if reviews:
        add("# Include the saved assistant judgments and regenerate this decision report:")
        add(f".venv/bin/python scripts/evaluate.py {trace_path.relative_to(ROOT)} --judge-reviews reports/judge_reviews.json")
    add("# Obtain a new timed run with the existing normal adapter, without retraining:")
    add("TIMING_REPEATS=3 ADAPTER_PATH=artifacts/support-lora-adapter .venv/bin/python scripts/compare_answers.py")
    add("```")
    add()
    add("A new generation run needs new review binding and a check of answer hashes before reusing semantic judgments. Deterministic recomputation does not require re-running generation.")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--judge-reviews", type=Path)
    args = parser.parse_args()
    trace_path = args.results.resolve()
    trace_bytes = trace_path.read_bytes()
    trace = json.loads(trace_bytes)
    if trace["configuration"]["training"]["learning_rate"] != 0.0002:
        raise ValueError("This report is scoped to the original 0.0002 adapter, not the high-rate experiments.")
    labels_bytes = (ROOT / "data/evaluation/cases.json").read_bytes()
    if hashlib.sha256(labels_bytes).hexdigest() != trace["configuration"]["evaluation_sha256"]:
        raise ValueError("Evaluation labels changed since generation.")
    labels = json.loads(labels_bytes)
    reviews = json.loads(args.judge_reviews.read_text()) if args.judge_reviews else None
    if reviews and hashlib.sha256(trace_bytes).hexdigest() != reviews["results_sha256"]:
        raise ValueError("Judgments belong to a different trace.")
    evaluation = evaluate(trace, labels, reviews)
    evaluation["results_sha256"] = hashlib.sha256(trace_bytes).hexdigest()
    evaluation["configuration"] = trace["configuration"]
    report_folder = ROOT / "reports"
    report_folder.mkdir(exist_ok=True)
    prefix = "evaluation" if reviews else "deterministic"
    (report_folder / f"{prefix}_metrics.json").write_text(json.dumps(evaluation, indent=2))
    report = render_report(trace, evaluation, labels, reviews, trace_path)
    if not reviews:
        report = report.replace("evaluation_metrics.json", "deterministic_metrics.json")
    report_path = report_folder / f"{prefix}_report.md"
    report_path.write_text(report)
    print(f"Wrote {report_path}")
    print(json.dumps(evaluation["summary"], indent=2))
