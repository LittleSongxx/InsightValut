import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv

from app.clients.milvus_schema import CHUNKS_OUTPUT_FIELDS, extract_chunk_id
from app.clients.milvus_utils import get_milvus_client, query_chunks_by_filter
from app.conf.milvus_config import milvus_config
from app.utils.path_util import PROJECT_ROOT

QUERY_TYPES = [
    "navigation",
    "constraint",
    "explain",
    "relation",
    "general",
    "comparison",
]
DEFAULT_CASE_COUNT = 100


def _clean_text(text: str, limit: int | None = None) -> str:
    value = str(text or "")
    value = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\$\\?textcircled\{?\d+\}?\$", "", value)
    value = re.sub(r"[#*_`]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" ：:。")
    if limit and len(value) > limit:
        return value[:limit].rstrip(" ，,；;。") + "。"
    return value


def _title(doc: Dict[str, Any]) -> str:
    return _clean_text(doc.get("title") or doc.get("parent_title") or "相关内容", 60)


def _parent_title(doc: Dict[str, Any]) -> str:
    parent = _clean_text(doc.get("parent_title") or "", 60)
    return parent or _title(doc)


def _summary(doc: Dict[str, Any], limit: int = 260) -> str:
    title = _title(doc)
    content = _clean_text(doc.get("content") or "", 900)
    content = re.sub(rf"^{re.escape(title)}\s*", "", content).strip()
    if not content:
        return f"该片段主要说明{title}。"

    sentences = [
        part.strip()
        for part in re.split(r"(?<=[。！？；])\s+|[•●]\s*", content)
        if part.strip()
    ]
    selected: List[str] = []
    total_len = 0
    for sentence in sentences:
        if len(sentence) < 8 and selected:
            continue
        selected.append(sentence)
        total_len += len(sentence)
        if total_len >= limit:
            break
    answer = "；".join(selected) if selected else content
    return _clean_text(answer, limit)


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", str(text or "")).strip("_")
    return cleaned[:36] or "item"


def _query_for(
    doc: Dict[str, Any], query_type: str, partner: Dict[str, Any] | None = None
) -> str:
    item = str(doc.get("item_name") or "该产品")
    title = _title(doc)
    parent = _parent_title(doc)
    if query_type == "navigation":
        return f"在{item}中，{title}应该怎么操作？"
    if query_type == "constraint":
        return f"{item}使用{title}时有哪些注意事项或限制？"
    if query_type == "explain":
        return f"{item}的{title}主要解决什么问题？请简要说明。"
    if query_type == "relation":
        if parent and parent != title:
            return f"{item}里“{title}”和“{parent}”之间是什么关系？"
        return f"{item}中{title}涉及哪些相关功能或步骤？"
    if query_type == "comparison":
        partner_title = _title(partner or doc)
        return f"{item}中“{title}”和“{partner_title}”分别讲什么，使用场景有什么区别？"
    return f"请概括{item}中关于{title}的说明。"


def _case_from_doc(
    doc: Dict[str, Any],
    *,
    case_index: int,
    query_type: str,
    partner: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    item = str(doc.get("item_name") or "").strip()
    chunk_id = str(extract_chunk_id(doc) or "").strip()
    partner_id = str(extract_chunk_id(partner or {}) or "").strip()
    title = _title(doc)
    case_id = f"auto_{_slug(item)}_{query_type}_{case_index:03d}"
    relevant_ids = [chunk_id]
    reference = _summary(doc)
    source_titles = [title]

    if (
        query_type == "comparison"
        and partner is not None
        and partner_id
        and partner_id != chunk_id
    ):
        relevant_ids.append(partner_id)
        partner_title = _title(partner)
        source_titles.append(partner_title)
        reference = (
            f"“{title}”部分：{_summary(doc, 180)} "
            f"“{partner_title}”部分：{_summary(partner, 180)}"
        )

    return {
        "case_id": case_id,
        "query_type": query_type,
        "query": _query_for(doc, query_type, partner),
        "item_names": [item] if item else [],
        "reference_answer": reference,
        "relevant_chunk_ids": relevant_ids,
        "source_titles": source_titles,
        "evaluation_tags": ["auto_generated", "from_current_milvus", query_type],
    }


def _load_existing_cases(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    return [dict(case) for case in cases or [] if isinstance(case, dict)]


def _fetch_docs() -> List[Dict[str, Any]]:
    client = get_milvus_client()
    if client is None:
        raise RuntimeError("无法连接 Milvus，不能从当前知识库生成评测集。")
    docs = query_chunks_by_filter(
        client,
        milvus_config.chunks_collection,
        output_fields=list(CHUNKS_OUTPUT_FIELDS),
        limit=-1,
    )
    usable = []
    for doc in docs:
        if not extract_chunk_id(doc):
            continue
        if not str(doc.get("content") or "").strip():
            continue
        if not str(doc.get("item_name") or "").strip():
            continue
        usable.append(doc)
    if not usable:
        raise RuntimeError("当前 Milvus 知识库没有可用于生成评测集的 chunk。")
    return usable


def _target_item_counts(
    docs: Sequence[Dict[str, Any]],
    target_count: int,
    seed_cases: Sequence[Dict[str, Any]],
) -> Dict[str, int]:
    item_names = sorted(
        {
            str(doc.get("item_name") or "").strip()
            for doc in docs
            if str(doc.get("item_name") or "").strip()
        }
    )
    if not item_names:
        return {}
    if len(item_names) == 1:
        return {item_names[0]: target_count}

    raw_counts = Counter(str(doc.get("item_name") or "").strip() for doc in docs)
    if len(item_names) == 2:
        smaller, larger = sorted(item_names, key=lambda name: raw_counts[name])
        smaller_target = min(target_count // 2, max(30, target_count * 2 // 5))
        targets = {smaller: smaller_target, larger: target_count - smaller_target}
    else:
        base = target_count // len(item_names)
        targets = {item_name: base for item_name in item_names}
        for item_name in item_names[: target_count - base * len(item_names)]:
            targets[item_name] += 1

    seeded_counts = Counter(
        str((case.get("item_names") or [""])[0]).strip()
        for case in seed_cases
        if (case.get("item_names") or [])
    )
    for item_name, seeded_count in seeded_counts.items():
        if item_name in targets:
            targets[item_name] = max(targets[item_name], seeded_count)
    return targets


def _query_type_targets(target_count: int) -> Dict[str, int]:
    base = target_count // len(QUERY_TYPES)
    targets = {query_type: base for query_type in QUERY_TYPES}
    for query_type in QUERY_TYPES[: target_count - base * len(QUERY_TYPES)]:
        targets[query_type] += 1
    return targets


def _next_query_type(
    cases: Sequence[Dict[str, Any]], target_count: int, offset: int
) -> str:
    targets = _query_type_targets(target_count)
    counts = Counter(str(case.get("query_type") or "general") for case in cases)
    candidates = sorted(
        QUERY_TYPES,
        key=lambda query_type: (
            counts[query_type] / max(targets[query_type], 1),
            (QUERY_TYPES.index(query_type) - offset) % len(QUERY_TYPES),
        ),
    )
    return candidates[0]


def build_dataset(
    *,
    output_path: Path,
    target_count: int = DEFAULT_CASE_COUNT,
    include_existing: bool = False,
) -> Dict[str, Any]:
    load_dotenv(Path(PROJECT_ROOT) / ".env")
    docs = _fetch_docs()
    seed_cases = _load_existing_cases(output_path) if include_existing else []
    cases: List[Dict[str, Any]] = []
    seen_ids = set()

    for case in seed_cases:
        case_id = str(case.get("case_id") or case.get("id") or "").strip()
        if not case_id or case_id in seen_ids:
            continue
        if not case.get("reference_answer") or not case.get("relevant_chunk_ids"):
            continue
        case.setdefault("evaluation_tags", ["manual_seed"])
        cases.append(case)
        seen_ids.add(case_id)
        if len(cases) >= target_count:
            break

    by_item: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for doc in sorted(
        docs,
        key=lambda item: (
            str(item.get("item_name") or ""),
            str(item.get("title") or ""),
            str(extract_chunk_id(item) or ""),
        ),
    ):
        by_item[str(doc.get("item_name") or "").strip()].append(doc)

    targets = _target_item_counts(docs, target_count, cases)
    case_index = 1
    while len(cases) < target_count:
        progressed = False
        for item_name, item_docs in sorted(by_item.items()):
            if len(cases) >= target_count:
                break
            current_item_count = sum(
                1 for case in cases if item_name in (case.get("item_names") or [])
            )
            if current_item_count >= targets.get(item_name, target_count):
                continue
            doc = item_docs[case_index % len(item_docs)]
            query_type = _next_query_type(cases, target_count, case_index)
            partner = None
            if query_type == "comparison" and len(item_docs) > 1:
                partner = item_docs[(case_index + 1) % len(item_docs)]
            case = _case_from_doc(
                doc,
                case_index=case_index,
                query_type=query_type,
                partner=partner,
            )
            if case["case_id"] in seen_ids:
                case_index += 1
                continue
            cases.append(case)
            seen_ids.add(case["case_id"])
            case_index += 1
            progressed = True
        if not progressed:
            break

    cases = cases[:target_count]
    notes = [
        "所有样本都绑定当前 Milvus stable chunk_id，便于统计 Recall/MRR。",
        "覆盖 navigation/constraint/explain/relation/general/comparison 等题型。",
    ]
    if include_existing and seed_cases:
        notes.insert(0, "保留输出文件中的既有人工样本，并从当前 chunk 补足剩余样本。")
    else:
        notes.insert(0, "样本从当前 Milvus chunk 自动构造，便于和当前知识库保持一致。")

    payload = {
        "dataset_name": "insightvault_100_case_eval",
        "generated_at": datetime.now().isoformat(),
        "description": "基于当前 InsightVault Milvus 知识库生成的 100 条多产品 RAG 消融评测集。",
        "notes": notes,
        "sources": {
            "chunks_collection": milvus_config.chunks_collection,
            "case_count_by_item": dict(
                Counter(tuple(case.get("item_names") or [""])[0] for case in cases)
            ),
            "case_count_by_query_type": dict(
                Counter(case.get("query_type") or "general" for case in cases)
            ),
        },
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从当前知识库生成统一 RAG 评测数据集")
    parser.add_argument(
        "--output",
        default=str(Path(PROJECT_ROOT) / "docs" / "graph_eval_cases.docs.json"),
        help="输出数据集路径",
    )
    parser.add_argument(
        "--count", type=int, default=DEFAULT_CASE_COUNT, help="目标样本数"
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="保留输出文件里已有的人工样本，并从当前知识库补足到目标数量",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_dataset(
        output_path=Path(args.output).expanduser().resolve(),
        target_count=args.count,
        include_existing=args.include_existing,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "case_count": len(payload["cases"]),
                "case_count_by_item": payload["sources"]["case_count_by_item"],
                "case_count_by_query_type": payload["sources"][
                    "case_count_by_query_type"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
