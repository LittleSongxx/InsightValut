import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from app.core.logger import logger
from app.lm.lm_utils import get_llm_client

_SEMANTIC_LABELS = {
    "Step",
    "Parameter",
    "Component",
    "Fault",
    "Cause",
    "Solution",
    "Warning",
}

_RELATION_TYPES = {
    "HAS_STEP",
    "HAS_PARAMETER",
    "HAS_COMPONENT",
    "HAS_FAULT",
    "HAS_CAUSE",
    "HAS_SOLUTION",
    "HAS_WARNING",
    "CAUSED_BY",
    "RESOLVED_BY",
    "RELATED_TO",
    "NEXT_STEP",
}

_COMPONENT_SUFFIXES = (
    "面板",
    "按钮",
    "开关",
    "电源",
    "传感器",
    "电机",
    "模块",
    "接口",
    "组件",
    "部件",
    "轴",
    "阀",
    "泵",
    "滚轮",
    "丝杆",
    "探头",
    "保险丝",
)

_LLM_EXTRACTION_ENABLED = os.getenv("KG_LLM_EXTRACTION_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _slug(text: Any) -> str:
    cleaned = _clean_text(text).lower()
    cleaned = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:120] or "unknown"


def _doc_key(item_name: str, file_title: str) -> str:
    return f"product::{_slug(item_name)}::document::{_slug(file_title)}"


def _section_key(item_name: str, file_title: str, section_title: str) -> str:
    return f"product::{_slug(item_name)}::section::{_slug(file_title)}::{_slug(section_title)}"


def _entity_key(item_name: str, label: str, name: str, section_key: str = "") -> str:
    if label == "Step":
        return f"product::{_slug(item_name)}::{label.lower()}::{_slug(section_key)}::{_slug(name)}"
    if label in {"Parameter", "Warning"}:
        return f"product::{_slug(item_name)}::{label.lower()}::{_slug(section_key)}::{_slug(name)}"
    return f"product::{_slug(item_name)}::{label.lower()}::{_slug(name)}"


def _unique_append(items: List[Dict[str, Any]], value: Dict[str, Any], key: str = "node_key") -> None:
    existing_index = next((idx for idx, item in enumerate(items) if item.get(key) == value.get(key)), -1)
    if existing_index < 0:
        items.append(value)
        return

    existing = items[existing_index]
    for field in ("source_chunk_ids", "source_titles", "source_section_keys"):
        merged = list(dict.fromkeys((existing.get(field) or []) + (value.get(field) or [])))
        existing[field] = merged
    if not existing.get("description") and value.get("description"):
        existing["description"] = value.get("description")
    if not existing.get("value") and value.get("value"):
        existing["value"] = value.get("value")
    if existing.get("order") is None and value.get("order") is not None:
        existing["order"] = value.get("order")


def _add_relation(relations: List[Dict[str, Any]], relation: Dict[str, Any]) -> None:
    rel_type = relation.get("type")
    if rel_type not in _RELATION_TYPES:
        return
    relation_key = (
        relation.get("source_key"),
        rel_type,
        relation.get("target_key"),
        relation.get("evidence_chunk_id"),
    )
    for item in relations:
        item_key = (
            item.get("source_key"),
            item.get("type"),
            item.get("target_key"),
            item.get("evidence_chunk_id"),
        )
        if item_key == relation_key:
            return
    relations.append(relation)


def _extract_numbered_lines(content: str) -> List[Tuple[int, str]]:
    results: List[Tuple[int, str]] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^(?:步骤\s*)?(\d{1,2})[.)、\-\s]+(.+)$", stripped, flags=re.IGNORECASE)
        if not match:
            match = re.match(r"^([一二三四五六七八九十]{1,3})[、.．]\s*(.+)$", stripped)
        if not match:
            continue
        order_text, body = match.groups()
        try:
            order = int(order_text)
        except ValueError:
            numerals = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
            order = numerals.get(order_text[:1], len(results) + 1)
        body = _clean_text(body)
        if body:
            results.append((order, body[:180]))
    return results


def _extract_parameter_pairs(content: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for line in content.splitlines():
        stripped = line.strip().lstrip("-*•")
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9\u4e00-\u9fff()（）%./_-]{1,32})\s*[:：=]\s*([^\n]{1,80})$", stripped)
        if match:
            name, value = match.groups()
            cleaned_name = _clean_text(name)
            cleaned_value = _clean_text(value)
            if cleaned_name and cleaned_value:
                pairs.append((cleaned_name, cleaned_value))
                continue
        inline_match = re.findall(
            r"([A-Za-z\u4e00-\u9fff]{2,20})(?:为|是|保持在|设置为|默认建议设置在)\s*([0-9]+(?:\.[0-9]+)?\s*[A-Za-z%℃°VvAaWwMmKkΩ/.-]{0,12})",
            stripped,
        )
        for name, value in inline_match:
            cleaned_name = _clean_text(name)
            cleaned_value = _clean_text(value)
            if cleaned_name and cleaned_value:
                pairs.append((cleaned_name, cleaned_value))
    deduped: List[Tuple[str, str]] = []
    seen = set()
    for name, value in pairs:
        key = (name, value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped[:8]


def _extract_warning_lines(content: str) -> List[str]:
    warnings: List[str] = []
    for line in content.splitlines():
        stripped = line.strip().lstrip("-*•")
        if not stripped or stripped.startswith("#"):
            continue
        if any(keyword in stripped for keyword in ("警告", "注意", "危险", "严禁", "禁止", "必须", "勿")):
            warnings.append(_clean_text(stripped)[:180])
    deduped: List[str] = []
    for item in warnings:
        if item not in deduped:
            deduped.append(item)
    return deduped[:6]


def _extract_component_candidates(content: str, title: str) -> List[str]:
    candidates: List[str] = []
    text = f"{title}\n{content}"
    for match in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,20}(?:面板|按钮|开关|电源|传感器|电机|模块|接口|组件|部件|轴|阀|泵|滚轮|丝杆|探头|保险丝)", text):
        normalized = _clean_text(match)
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    for suffix in _COMPONENT_SUFFIXES:
        if suffix in title and title not in candidates:
            candidates.append(_clean_text(title))
    return candidates[:6]


def _extract_fault_cause_solution(content: str) -> Dict[str, List[str]]:
    faults: List[str] = []
    causes: List[str] = []
    solutions: List[str] = []
    for line in content.splitlines():
        stripped = line.strip().lstrip("-*•")
        if not stripped or stripped.startswith("#"):
            continue
        normalized = _clean_text(stripped)
        if any(keyword in normalized for keyword in ("故障", "异常", "报警", "无法", "不能", "不工作")) and normalized not in faults:
            faults.append(normalized[:180])
        if any(keyword in normalized for keyword in ("原因", "导致", "由于", "检查", "可能是")) and normalized not in causes:
            causes.append(normalized[:180])
        if any(keyword in normalized for keyword in ("解决", "处理", "排除", "更换", "恢复", "重新设置")) and normalized not in solutions:
            solutions.append(normalized[:180])
    return {
        "faults": faults[:4],
        "causes": causes[:6],
        "solutions": solutions[:6],
    }


def _infer_semantics(chunk: Dict[str, Any], item_name: str, section_key: str) -> Dict[str, Any]:
    title = _clean_text(chunk.get("title") or chunk.get("parent_title") or "")
    content = str(chunk.get("content") or "")
    source_chunk_id = str(chunk.get("chunk_id") or "")
    section_title = title or chunk.get("file_title") or "section"

    nodes: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []

    step_candidates = _extract_numbered_lines(content)
    if not step_candidates and any(keyword in title for keyword in ("步骤", "操作", "安装", "调试", "设置", "流程")):
        body = _clean_text(content[:180])
        if body:
            step_candidates = [(1, body)]

    ordered_step_nodes: List[Dict[str, Any]] = []
    for order, body in step_candidates:
        name = body[:80]
        node_key = _entity_key(item_name, "Step", name, section_key)
        step_node = {
            "node_key": node_key,
            "label": "Step",
            "name": name,
            "description": body,
            "value": "",
            "order": order,
            "source_chunk_ids": [source_chunk_id] if source_chunk_id else [],
            "source_titles": [section_title],
            "source_section_keys": [section_key],
        }
        _unique_append(nodes, step_node)
        ordered_step_nodes.append(step_node)

    ordered_step_nodes.sort(key=lambda item: item.get("order") or 0)
    for current, nxt in zip(ordered_step_nodes, ordered_step_nodes[1:]):
        _add_relation(
            relations,
            {
                "source_key": current["node_key"],
                "target_key": nxt["node_key"],
                "type": "NEXT_STEP",
                "evidence_chunk_id": source_chunk_id,
            },
        )

    parameter_pairs = _extract_parameter_pairs(content)
    for name, value in parameter_pairs:
        node_key = _entity_key(item_name, "Parameter", name, section_key)
        _unique_append(
            nodes,
            {
                "node_key": node_key,
                "label": "Parameter",
                "name": name,
                "description": f"{name}: {value}",
                "value": value,
                "order": None,
                "source_chunk_ids": [source_chunk_id] if source_chunk_id else [],
                "source_titles": [section_title],
                "source_section_keys": [section_key],
            },
        )

    warnings = _extract_warning_lines(content)
    for warning in warnings:
        node_key = _entity_key(item_name, "Warning", warning[:80], section_key)
        _unique_append(
            nodes,
            {
                "node_key": node_key,
                "label": "Warning",
                "name": warning[:80],
                "description": warning,
                "value": "",
                "order": None,
                "source_chunk_ids": [source_chunk_id] if source_chunk_id else [],
                "source_titles": [section_title],
                "source_section_keys": [section_key],
            },
        )

    components = _extract_component_candidates(content, title)
    for component in components:
        node_key = _entity_key(item_name, "Component", component, section_key)
        _unique_append(
            nodes,
            {
                "node_key": node_key,
                "label": "Component",
                "name": component,
                "description": component,
                "value": "",
                "order": None,
                "source_chunk_ids": [source_chunk_id] if source_chunk_id else [],
                "source_titles": [section_title],
                "source_section_keys": [section_key],
            },
        )

    fault_bundle = _extract_fault_cause_solution(content)
    fault_keys: List[str] = []
    cause_keys: List[str] = []
    solution_keys: List[str] = []

    for label, bucket, target_keys in (
        ("Fault", fault_bundle.get("faults") or [], fault_keys),
        ("Cause", fault_bundle.get("causes") or [], cause_keys),
        ("Solution", fault_bundle.get("solutions") or [], solution_keys),
    ):
        for text in bucket:
            node_key = _entity_key(item_name, label, text[:80], section_key)
            _unique_append(
                nodes,
                {
                    "node_key": node_key,
                    "label": label,
                    "name": text[:80],
                    "description": text,
                    "value": "",
                    "order": None,
                    "source_chunk_ids": [source_chunk_id] if source_chunk_id else [],
                    "source_titles": [section_title],
                    "source_section_keys": [section_key],
                },
            )
            target_keys.append(node_key)

    for fault_key in fault_keys:
        for cause_key in cause_keys:
            _add_relation(
                relations,
                {
                    "source_key": fault_key,
                    "target_key": cause_key,
                    "type": "CAUSED_BY",
                    "evidence_chunk_id": source_chunk_id,
                },
            )
        for solution_key in solution_keys:
            _add_relation(
                relations,
                {
                    "source_key": fault_key,
                    "target_key": solution_key,
                    "type": "RESOLVED_BY",
                    "evidence_chunk_id": source_chunk_id,
                },
            )

    node_index = {node.get("node_key"): node for node in nodes}
    section_relation_map = {
        "Step": "HAS_STEP",
        "Parameter": "HAS_PARAMETER",
        "Component": "HAS_COMPONENT",
        "Fault": "HAS_FAULT",
        "Cause": "HAS_CAUSE",
        "Solution": "HAS_SOLUTION",
        "Warning": "HAS_WARNING",
    }
    for node_key, node in node_index.items():
        relation_type = section_relation_map.get(node.get("label"))
        if not relation_type:
            continue
        _add_relation(
            relations,
            {
                "source_key": section_key,
                "target_key": node_key,
                "type": relation_type,
                "evidence_chunk_id": source_chunk_id,
            },
        )

    return {"nodes": nodes, "relations": relations}


def _safe_json_loads(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "").replace("```", "")
    return json.loads(cleaned)


def _llm_extract_semantics(chunk: Dict[str, Any], item_name: str, section_key: str) -> Dict[str, Any]:
    if not _LLM_EXTRACTION_ENABLED:
        return {"nodes": [], "relations": []}

    content = str(chunk.get("content") or "").strip()
    if not content:
        return {"nodes": [], "relations": []}

    prompt = f"""
你是工业手册知识图谱抽取助手。请只基于给定文本抽取适合 Neo4j 的实体与关系。

产品：{item_name}
章节：{chunk.get('title') or chunk.get('parent_title') or ''}
chunk_id：{chunk.get('chunk_id') or ''}
文本：
{content[:1800]}

仅允许以下实体类型：Step, Parameter, Component, Fault, Cause, Solution, Warning。
仅允许以下关系类型：HAS_STEP, HAS_PARAMETER, HAS_COMPONENT, HAS_FAULT, HAS_CAUSE, HAS_SOLUTION, HAS_WARNING, CAUSED_BY, RESOLVED_BY, RELATED_TO, NEXT_STEP。
如果没有就返回空数组。
请返回 JSON：
{{
  "nodes": [
    {{"label": "Parameter", "name": "温度", "description": "默认建议设置在110℃", "value": "110℃", "order": null}},
    {{"label": "Step", "name": "设置温度", "description": "开机后先设置温度", "value": "", "order": 1}}
  ],
  "relations": [
    {{"source_name": "设置温度", "source_label": "Step", "target_name": "温度", "target_label": "Parameter", "type": "RELATED_TO"}}
  ]
}}
""".strip()

    try:
        client = get_llm_client(json_mode=True)
        response = client.invoke(prompt)
        data = _safe_json_loads(response.content)
    except Exception as exc:
        logger.warning(f"图谱语义抽取降级为启发式: {exc}")
        return {"nodes": [], "relations": []}

    source_chunk_id = str(chunk.get("chunk_id") or "")
    section_title = _clean_text(chunk.get("title") or chunk.get("parent_title") or "")
    nodes: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []

    for raw_node in data.get("nodes") or []:
        label = raw_node.get("label")
        name = _clean_text(raw_node.get("name"))
        if label not in _SEMANTIC_LABELS or not name:
            continue
        node_key = _entity_key(item_name, label, name, section_key)
        _unique_append(
            nodes,
            {
                "node_key": node_key,
                "label": label,
                "name": name,
                "description": _clean_text(raw_node.get("description") or name)[:220],
                "value": _clean_text(raw_node.get("value") or "")[:120],
                "order": raw_node.get("order"),
                "source_chunk_ids": [source_chunk_id] if source_chunk_id else [],
                "source_titles": [section_title] if section_title else [],
                "source_section_keys": [section_key],
            },
        )

    node_map = {(node.get("label"), node.get("name")): node.get("node_key") for node in nodes}
    for raw_relation in data.get("relations") or []:
        rel_type = raw_relation.get("type")
        source_key = node_map.get((raw_relation.get("source_label"), _clean_text(raw_relation.get("source_name"))))
        target_key = node_map.get((raw_relation.get("target_label"), _clean_text(raw_relation.get("target_name"))))
        if not source_key or not target_key or rel_type not in _RELATION_TYPES:
            continue
        _add_relation(
            relations,
            {
                "source_key": source_key,
                "target_key": target_key,
                "type": rel_type,
                "evidence_chunk_id": source_chunk_id,
            },
        )

    return {"nodes": nodes, "relations": relations}


def build_graph_payload(item_name: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    documents: List[Dict[str, Any]] = []
    sections: List[Dict[str, Any]] = []
    semantic_nodes: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []
    images: List[Dict[str, Any]] = []

    for chunk in sorted(chunks or [], key=lambda item: item.get("part", 0)):
        chunk_id = str(chunk.get("chunk_id") or "")
        file_title = _clean_text(chunk.get("file_title") or item_name or "document")
        section_title = _clean_text(chunk.get("title") or chunk.get("parent_title") or file_title)
        document_key = _doc_key(item_name, file_title)
        section_key = _section_key(item_name, file_title, section_title)

        _unique_append(
            documents,
            {
                "node_key": document_key,
                "title": file_title,
                "product_name": item_name,
            },
        )
        _unique_append(
            sections,
            {
                "node_key": section_key,
                "title": section_title,
                "parent_title": _clean_text(chunk.get("parent_title") or section_title),
                "file_title": file_title,
                "product_name": item_name,
            },
        )

        for image_url in chunk.get("image_urls") or []:
            image_url = _clean_text(image_url)
            if not image_url:
                continue
            _unique_append(
                images,
                {
                    "node_key": f"product::{_slug(item_name)}::image::{_slug(image_url)}",
                    "url": image_url,
                    "product_name": item_name,
                    "source_chunk_id": chunk_id,
                },
            )

        heuristic = _infer_semantics(chunk, item_name, section_key)
        llm_semantics = _llm_extract_semantics(chunk, item_name, section_key)
        for node in heuristic.get("nodes") or []:
            _unique_append(semantic_nodes, node)
        for node in llm_semantics.get("nodes") or []:
            _unique_append(semantic_nodes, node)
        for relation in heuristic.get("relations") or []:
            _add_relation(relations, relation)
        for relation in llm_semantics.get("relations") or []:
            _add_relation(relations, relation)

    grouped_steps: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for node in semantic_nodes:
        if node.get("label") == "Step":
            for section_key in node.get("source_section_keys") or []:
                grouped_steps[section_key].append(node)

    for _, step_nodes in grouped_steps.items():
        step_nodes.sort(key=lambda item: item.get("order") or 0)
        for current, nxt in zip(step_nodes, step_nodes[1:]):
            _add_relation(
                relations,
                {
                    "source_key": current.get("node_key"),
                    "target_key": nxt.get("node_key"),
                    "type": "NEXT_STEP",
                    "evidence_chunk_id": ((current.get("source_chunk_ids") or [None])[0]),
                },
            )

    return {
        "documents": documents,
        "sections": sections,
        "semantic_nodes": semantic_nodes,
        "semantic_relations": relations,
        "images": images,
    }
