import json
from collections import Counter
from pathlib import Path

import pytest

from app.utils import unified_rag_eval as eval_mod
from app.utils import perf_tracker as perf_mod
from app.utils.eval_dataset_builder import _query_type_targets
from app.utils.eval_job_utils import (
    _dataset_catalog,
    create_evaluation_job,
    merge_appended_evaluation_report,
)
from app.utils.path_util import PROJECT_ROOT
from app.utils.unified_rag_eval import (
    _build_ragas_row,
    _build_ground_truth_alignment,
    _llm_quality_scores,
    _summarize_pipeline,
    _summarize_retrieval,
    _summarize_variant,
    build_feature_variant_definition,
)
from app.query_process.agent.nodes import node_item_name_confirm as item_confirm_mod
from app.query_process.agent.nodes import node_query_decompose as query_decompose_mod
from app.query_process.agent.nodes import node_retrieval_grader as retrieval_grader_mod


def test_perf_session_records_first_token_latency(monkeypatch):
    timestamps = iter([100.0, 100.2, 100.5, 101.0])
    monkeypatch.setattr(perf_mod.time, "time", lambda: next(timestamps))

    session = perf_mod.PerfSession("session-1", "query")
    session.mark_first_token()
    session.mark_first_answer()
    doc = session.to_document()

    assert doc["first_token_ms"] == 200.0
    assert doc["first_answer_ms"] == 500.0
    assert doc["total_duration_ms"] == 1000.0


def test_eval_percentile_uses_nearest_rank():
    assert eval_mod._pct([10, 20, 30, 40], 0.50) == 20
    assert eval_mod._pct([10, 20, 30, 40], 0.95) == 40


def test_summarize_performance_includes_first_token_cold_hot_and_stage_p50():
    cold_cases = [
        {
            "latency_ms": 1000,
            "first_token_ms": 250,
            "first_answer_ms": 900,
            "cache_temperature": "cold",
            "stage_durations_ms": {"node_answer_output": 500},
        },
        {
            "latency_ms": 1200,
            "first_token_ms": 300,
            "first_answer_ms": 1000,
            "cache_temperature": "cold",
            "stage_durations_ms": {"node_answer_output": 700},
        },
    ]
    hot_cases = [
        {
            "latency_ms": 400,
            "first_token_ms": 80,
            "first_answer_ms": 350,
            "cache_temperature": "hot",
            "stage_durations_ms": {"node_answer_output": 220},
        },
        {
            "latency_ms": 500,
            "first_token_ms": 120,
            "first_answer_ms": 420,
            "cache_temperature": "hot",
            "stage_durations_ms": {"node_answer_output": 260},
        },
    ]

    summary = eval_mod._summarize_performance(
        hot_cases,
        cold_case_results=cold_cases,
        hot_case_results=hot_cases,
    )

    assert summary["avg_first_token_ms"] == 100
    assert summary["p50_first_token_ms"] == 80
    assert summary["p95_first_token_ms"] == 120
    assert summary["cold_p50_total_duration_ms"] == 1000
    assert summary["cold_p95_total_duration_ms"] == 1200
    assert summary["hot_p50_total_duration_ms"] == 400
    assert summary["hot_p95_total_duration_ms"] == 500
    assert summary["stages"][0]["p50_duration_ms"] == 220
    assert summary["stages"][0]["p95_duration_ms"] == 260


def test_run_single_case_via_service_carries_first_token(monkeypatch):
    captured = {}

    class FakeResponse:
        content = b"{}"

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "answer": "回答",
                "latency_ms": 500,
                "first_token_ms": 90,
                "first_answer_ms": 450,
                "stage_durations_ms": {"node_answer_output": 300},
                "metadata": {"cache_summary": {}},
            }

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(eval_mod.requests, "post", fake_post)
    result = eval_mod._run_single_case_via_service(
        {"case_id": "case_1", "query": "问题"},
        "baseline_rag",
        "http://query-service",
        cache_temperature="hot",
    )

    assert captured["json"]["streaming"] is True
    assert result["first_token_ms"] == 90
    assert result["cache_temperature"] == "hot"


def test_item_name_fast_extract_uses_forced_names_without_llm():
    result = item_confirm_mod._fast_extract_info(
        "HUAWEIMateBookB7-420 如何设置快捷键？",
        history=[],
        forced_item_names=["HUAWEIMateBookB7-420"],
    )

    assert result["strategy"] == "forced_item_names"
    assert result["item_names"] == ["HUAWEIMateBookB7-420"]
    assert result["use_rag"] is True


def test_query_decompose_fast_gate_skips_when_subquery_routing_disabled():
    result = query_decompose_mod._fast_detect_compound(
        {"evaluation_overrides": {"agentic_features": {"subquery_routing": False}}},
        "HUAWEIMateBookB7-420 如何设置快捷键？",
        ["HUAWEIMateBookB7-420"],
    )

    assert result["is_compound"] is False
    assert "subquery_routing_disabled" in result["reason"]


def test_retrieval_grader_gate_skips_high_confidence_docs():
    reason = retrieval_grader_mod._retrieval_grader_skip_reason(
        {
            "evaluation_overrides": {},
            "evidence_coverage_summary": {},
            "target_coverage": {},
            "retry_count": 0,
        },
        [{"score": 0.91, "text": "相关"}, {"score": 0.70, "text": "次相关"}],
    )

    assert reason.startswith("high_confidence_retrieval:")


def test_summary_includes_final_context_precision_and_rerank_diagnostics():
    cases = [
        {
            "retrieved_context_ids": ["a", "noise"],
            "relevant_chunk_ids": ["a"],
            "rerank_diagnostics": {
                "fallback": True,
                "heuristic": True,
                "cache_hit": False,
                "candidate_count": 10,
                "selected_count": 2,
            },
            "final_context_summary": {"included_docs": 2, "used_chars": 1200},
            "retrieval_judge_skipped_reason": "high_confidence_retrieval",
            "hallucination_judge_skipped_reason": "grounded_evidence_low_risk",
            "cache_summary": {},
        }
    ]

    retrieval_metrics, retrieval_coverage, _, _ = _summarize_retrieval(cases)
    pipeline_metrics = _summarize_pipeline(cases)

    assert retrieval_metrics["final_context_precision"] == 0.5
    assert retrieval_coverage["final_context_precision"] == 1
    assert pipeline_metrics["rerank_fallback_rate"] == 1.0
    assert pipeline_metrics["rerank_heuristic_rate"] == 1.0
    assert pipeline_metrics["avg_rerank_candidate_count"] == 10
    assert pipeline_metrics["avg_rerank_selected_count"] == 2
    assert pipeline_metrics["retrieval_judge_skipped_rate"] == 1.0
    assert pipeline_metrics["hallucination_judge_skipped_rate"] == 1.0


def test_feature_variant_builds_expected_overrides():
    payload = build_feature_variant_definition(
        {"label": "Baseline + HyDE + BM25 + Cache", "features": ["hyde", "bm25", "cache"]}
    )

    config = payload["config"]
    overrides = config["evaluation_overrides"]
    plan = overrides["retrieval_plan_overrides"]

    assert payload["name"].startswith("combo_")
    assert config["compare_to"] == "combo_baseline"
    assert config["use_case_query_type"] is True
    assert config["warmup_rounds"] == 1
    assert config["reset_cache_before_run"] is True
    assert overrides["cache_enabled"] is True
    assert overrides["bm25_enabled"] is True
    assert plan["run_embedding"] is True
    assert plan["run_bm25"] is True
    assert plan["run_hyde"] is True
    assert plan["run_kg"] is False


def test_feature_variant_auto_enables_dependencies():
    payload = build_feature_variant_definition({"features": ["retrieval_rescue"]})
    feature_variant = payload["config"]["feature_variant"]
    agentic = payload["config"]["evaluation_overrides"]["agentic_features"]

    assert feature_variant["requested_features"] == ["retrieval_rescue"]
    assert feature_variant["auto_enabled_features"] == [
        "subquery_routing",
        "evidence_coverage",
    ]
    assert agentic["subquery_routing"] is True
    assert agentic["evidence_coverage"] is True
    assert agentic["retrieval_rescue"] is True


def test_feature_variant_name_is_stable_for_same_feature_set():
    left = build_feature_variant_definition({"features": ["cache", "bm25", "hyde"]})
    right = build_feature_variant_definition({"features": ["hyde", "cache", "bm25"]})

    assert left["name"] == right["name"]


def test_feature_variant_rejects_unknown_feature():
    with pytest.raises(ValueError, match="未知评测功能"):
        build_feature_variant_definition({"features": ["not_a_feature"]})


def test_create_evaluation_job_does_not_auto_add_baseline(tmp_path):
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text('{"cases": []}', encoding="utf-8")

    job = create_evaluation_job(
        str(dataset_path),
        variants=[],
        feature_variants=[{"label": "HyDE Only", "features": ["hyde"]}],
    )

    assert "combo_baseline" not in job["variants"]
    assert len(job["variants"]) == 1
    assert job["variants"][0].startswith("combo_")


def test_create_evaluation_job_only_runs_baseline_when_explicit(tmp_path):
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text('{"cases": []}', encoding="utf-8")

    job = create_evaluation_job(
        str(dataset_path),
        variants=[],
        feature_variants=[{"label": "Baseline", "features": []}],
    )

    assert job["variants"] == ["combo_baseline"]


def test_merge_appended_report_compares_new_variant_to_existing_variants():
    def variant(name, features, score):
        return {
            "description": name,
            "technique": name,
            "feature_variant": {
                "name": name,
                "label": name,
                "requested_features": features,
                "resolved_features": features,
                "feature_labels": features,
                "auto_enabled_features": [],
                "auto_enabled_feature_labels": [],
            },
            "summary": {
                "variant": name,
                "description": name,
                "technique": name,
                "case_count": 2,
                "ragas_metrics": {"factual_correctness": score},
                "ragas_coverage": {"factual_correctness": 2},
                "ragas_errors": {},
                "retrieval_metrics": {"recall@5": score},
                "retrieval_coverage": {"recall@5": 2},
                "pipeline_metrics": {},
                "performance_metrics": {"avg_total_duration_ms": 1000 / score},
                "headline_metrics": {"factual_correctness": score},
                "warnings": [],
            },
            "by_query_type": {},
        }

    base_report = {
        "generated_at": "2026-04-24T00:00:00",
        "dataset_path": "/tmp/eval.json",
        "dataset_name": "demo",
        "case_count": 2,
        "evaluation_method": {
            "mode": "controlled_ablation",
            "execution_order": ["combo_baseline", "combo_hyde"],
            "feature_variants": [
                {"name": "combo_baseline", "resolved_features": []},
                {"name": "combo_hyde", "resolved_features": ["hyde"]},
            ],
        },
        "variants": {
            "combo_baseline": variant("combo_baseline", [], 0.5),
            "combo_hyde": variant("combo_hyde", ["hyde"], 0.7),
        },
        "final_variant": "combo_hyde",
        "final_system_metrics": {},
        "comparisons": {
            "combo_hyde_vs_combo_baseline": {
                "variant": "combo_hyde",
                "compare_to": "combo_baseline",
                "technique": "combo_hyde",
                "overall": {},
                "by_query_type": {},
            }
        },
    }
    appended_report = {
        "dataset_path": "/tmp/eval.json",
        "dataset_name": "demo",
        "case_count": 2,
        "variants": {
            "combo_cache": variant("combo_cache", ["cache"], 0.9),
        },
        "comparisons": {},
    }

    merged = merge_appended_evaluation_report(
        base_report,
        appended_report,
        appended_variant_names=["combo_cache"],
    )

    assert list(merged["variants"].keys()) == [
        "combo_baseline",
        "combo_hyde",
        "combo_cache",
    ]
    assert merged["final_variant"] == "combo_cache"
    assert "combo_hyde_vs_combo_baseline" in merged["comparisons"]
    assert "combo_cache_vs_combo_baseline" in merged["comparisons"]
    assert "combo_cache_vs_combo_hyde" in merged["comparisons"]


def test_25_case_dataset_distribution_prioritizes_constraint():
    assert _query_type_targets(25) == {
        "constraint": 5,
        "explain": 4,
        "relation": 4,
        "general": 4,
        "comparison": 4,
        "navigation": 4,
    }


def test_ground_truth_alignment_reports_item_name_filter_miss_separately():
    summary, warnings = _build_ground_truth_alignment(
        [
            {
                "eligible": True,
                "resolved_ids": ["chunk_a"],
                "source": "reference_answer_bm25",
                "item_name_filter_miss": True,
            }
        ],
        1,
    )

    assert summary["resolved_cases"] == 1
    assert summary["unresolved_cases"] == 0
    assert summary["item_name_filter_miss_cases"] == 1
    assert "item_names 与当前知识库名称不一致" in warnings[0]


def test_ragas_row_carries_structured_judge_context():
    row = _build_ragas_row(
        {
            "query": "如何开启智慧多窗？",
            "query_type": "navigation",
            "item_names": ["华为平板 C7"],
            "answerable": False,
            "response": "资料中未提供该功能。",
            "reference_answer": "资料中未提供该功能开启方式。",
            "retrieved_contexts": ["智慧多窗相关说明"],
            "retrieved_context_ids": ["chunk_1"],
            "relevant_chunk_ids": ["chunk_1"],
        }
    )

    assert row["query_type"] == "navigation"
    assert row["item_names"] == ["华为平板 C7"]
    assert row["answerable"] is False
    assert row["reference_context_ids"] == ["chunk_1"]


def test_llm_judge_uses_strict_v2_rubric_and_score_buckets(monkeypatch):
    captured = {}

    class FakeJudge:
        def invoke(self, prompt):
            captured["prompt"] = prompt

            class Response:
                content = json.dumps(
                    {
                        "factual_correctness": 0.76,
                        "faithfulness": 0.59,
                        "response_relevancy": 0.22,
                        "llm_context_recall": 1.0,
                        "diagnostics": {"factual": "partial"},
                    }
                )

            return Response()

    def fake_get_llm_client(*, model=None, json_mode=False):
        captured["json_mode"] = json_mode
        return FakeJudge()

    monkeypatch.setattr(eval_mod, "get_llm_client", fake_get_llm_client)
    scores = _llm_quality_scores(
        {
            "user_input": "如何开启智慧多窗？",
            "query_type": "navigation",
            "item_names": ["华为平板 C7"],
            "answerable": False,
            "reference": "资料中未提供该功能开启方式。",
            "response": "进入设置即可开启智慧多窗。",
            "retrieved_contexts": ["资料中未提供该功能开启方式。"],
        }
    )

    assert captured["json_mode"] is True
    assert "固定档位" in captured["prompt"]
    assert '"query_type": "navigation"' in captured["prompt"]
    assert '"answerable": false' in captured["prompt"]
    assert scores == {
        "factual_correctness": 0.8,
        "faithfulness": 0.6,
        "response_relevancy": 0.2,
        "llm_context_recall": 1.0,
    }


def test_variant_summary_no_longer_contains_business_metric_sections():
    summary = _summarize_variant(
        "combo_baseline",
        [
            {
                "case_id": "case_1",
                "query_type": "general",
                "response": "回答",
                "retrieved_context_ids": ["chunk_1"],
                "relevant_chunk_ids": ["chunk_1"],
                "cache_summary": {},
                "latency_ms": 120.0,
                "stage_durations_ms": {},
                "error": "",
            }
        ],
        ragas_bundle={
            "summary": {"factual_correctness": 0.8, "faithfulness": 0.6},
            "coverage": {"factual_correctness": 1, "faithfulness": 1},
            "errors": {},
            "metadata": {},
        },
    )

    removed_sections = {f"business_{suffix}" for suffix in ("metrics", "coverage")}
    assert removed_sections.isdisjoint(summary)
    assert all(not key.startswith("business_") for key in summary["headline_metrics"])


def test_business_benchmark_30_dataset_shape():
    dataset_path = Path(PROJECT_ROOT) / "docs" / "business_benchmark_30.docs.json"
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    required_fields = {
        "case_id",
        "query",
        "query_type",
        "item_names",
        "answerable",
        "reference_answer",
        "required_facts",
        "forbidden_facts",
        "relevant_chunk_ids",
        "expected_source_chunk_ids",
        "evaluation_tags",
    }

    assert payload["dataset_name"] == "business_benchmark_30"
    assert len(cases) == 30
    assert Counter(case["query_type"] for case in cases) == {
        "general": 5,
        "navigation": 5,
        "constraint": 5,
        "explain": 5,
        "relation": 5,
        "comparison": 5,
    }
    assert sum(1 for case in cases if not case.get("answerable", True)) >= 3
    for case in cases:
        assert required_fields <= set(case.keys())
        assert case["required_facts"]
        assert case["relevant_chunk_ids"]
        assert "business_benchmark" in case["evaluation_tags"]
        assert "auto_generated" not in case["evaluation_tags"]


def test_business_benchmark_100_dataset_shape_and_catalog():
    dataset_path = Path(PROJECT_ROOT) / "docs" / "business_benchmark_100.docs.json"
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    required_fields = {
        "case_id",
        "query",
        "query_type",
        "item_names",
        "answerable",
        "reference_answer",
        "required_facts",
        "forbidden_facts",
        "relevant_chunk_ids",
        "expected_source_chunk_ids",
        "evaluation_tags",
    }

    assert payload["dataset_name"] == "business_benchmark_100"
    assert len(cases) == 100
    assert Counter(case["query_type"] for case in cases) == {
        "comparison": 17,
        "relation": 17,
        "navigation": 17,
        "general": 17,
        "explain": 16,
        "constraint": 16,
    }
    assert Counter(case["item_names"][0] for case in cases) == {
        "HUAWEIMateBookB7-420": 50,
        "华为平板 C7 用户指南-(DBY-W09,HarmonyOS 2_01,ZH-CN)": 50,
    }
    assert sum(1 for case in cases if not case.get("answerable", True)) >= 9
    assert len({case["case_id"] for case in cases}) == 100
    for case in cases:
        assert required_fields <= set(case.keys())
        assert case["query"].strip()
        assert case["reference_answer"].strip()
        assert case["required_facts"]
        assert case["relevant_chunk_ids"]
        assert case["expected_source_chunk_ids"]
        assert "business_benchmark" in case["evaluation_tags"]
        assert "business_benchmark_100" in case["evaluation_tags"]
        assert "auto_generated" not in case["evaluation_tags"]

    catalog_item = next(
        item for item in _dataset_catalog() if item["key"] == "business_benchmark_100"
    )
    assert catalog_item["case_count"] == 100
    assert catalog_item["exists"] is True
    assert catalog_item["path"].endswith("business_benchmark_100.docs.json")
