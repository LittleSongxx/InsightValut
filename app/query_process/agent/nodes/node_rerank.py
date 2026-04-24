from app.utils.task_utils import add_running_task, add_done_task
from app.lm.reranker_utils import get_reranker_model
from app.conf.query_threshold_config import query_threshold_config
from app.clients.milvus_schema import get_entity_field, extract_chunk_id
from app.core.logger import logger
import sys
import hashlib

from app.utils.query_cache_utils import query_cache_get, query_cache_set
from app.utils.anchor_context_utils import reorder_docs_for_target_coverage


def _contextual_rerank_text(entity, content):
    prefix_parts = [
        get_entity_field(entity, "section_path"),
        get_entity_field(entity, "chunk_context"),
        get_entity_field(entity, "title"),
        get_entity_field(entity, "parent_title"),
        get_entity_field(entity, "file_title"),
    ]
    prefix = "\n".join(str(part).strip() for part in prefix_parts if str(part or "").strip())
    body = str(content or "").strip()
    if prefix and prefix not in body[:500]:
        return f"{prefix}\n\n{body}"
    return body


def step_1_merge_docs(state):
    """
    阶段一：文档合并与标准化

    目标：将多路召回（本地知识库 + 联网搜索）的异构数据，统一合并为 Reranker 模型可处理的标准格式。
    """
    rrf_docs = state.get("rrf_chunks") or []
    web_docs = state.get("web_search_docs") or []

    logger.info(
        f"Step 1: 开始合并文档 - 本地RRF源: {len(rrf_docs)}条, 联网Web源: {len(web_docs)}条"
    )
    doc_items = []

    for i, doc in enumerate(rrf_docs):
        entity = get_entity_field(doc, None) or doc
        if not isinstance(entity, dict):
            logger.warning(f"本地文档格式异常 (index={i}): {type(entity)}")
            continue

        content = (
            get_entity_field(entity, "expanded_text")
            or get_entity_field(entity, "content")
            or get_entity_field(entity, "text")
        )
        if not content:
            logger.debug(f"跳过无内容文档 (index={i}, keys={list(entity.keys())})")
            continue

        chunk_id = extract_chunk_id(entity)
        title = (
            get_entity_field(entity, "title")
            or get_entity_field(entity, "document_title")
            or get_entity_field(entity, "item_name")
            or ""
        )
        source = get_entity_field(entity, "source") or "local"

        doc_items.append(
            {
                "text": _contextual_rerank_text(entity, content),
                "doc_id": chunk_id,
                "chunk_id": chunk_id,
                "title": title,
                "url": "",
                "source": source,
                "image_urls": get_entity_field(entity, "image_urls") or [],
                "graph_fact": get_entity_field(entity, "graph_fact") or "",
                "graph_entities": get_entity_field(entity, "graph_entities") or [],
                "section_title": get_entity_field(entity, "section_title") or "",
                "section_path": get_entity_field(entity, "section_path") or "",
                "chunk_context": get_entity_field(entity, "chunk_context") or "",
                "anchor_terms": get_entity_field(entity, "anchor_terms") or [],
                "document_title": get_entity_field(entity, "document_title") or "",
                "expanded_chunk_ids": get_entity_field(entity, "expanded_chunk_ids")
                or [],
                "context_expanded": bool(
                    get_entity_field(entity, "context_expanded", default=False)
                ),
                "evidence_chunk_ids": get_entity_field(entity, "evidence_chunk_ids")
                or [],
                "evidence_text": get_entity_field(entity, "evidence_text") or "",
            }
        )

    for i, doc in enumerate(web_docs):
        text = (doc.get("snippet") or doc.get("content") or "").strip()
        url = (doc.get("url") or "").strip()
        title = (doc.get("title") or "").strip()

        if not text:
            logger.debug(f"跳过无内容联网结果 (index={i})")
            continue

        doc_items.append(
            {
                "text": text,
                "doc_id": None,
                "chunk_id": None,
                "title": title,
                "url": url,
                "source": "web",
                "image_urls": doc.get("image_urls") or [],
                "graph_fact": "",
                "graph_entities": [],
                "section_title": "",
                "document_title": "",
                "expanded_chunk_ids": [],
                "context_expanded": False,
                "evidence_chunk_ids": [],
                "evidence_text": "",
            }
        )

    logger.info(f"Step 1: 文档合并完成，共输出 {len(doc_items)} 条标准化文档")
    return doc_items


def step_2_rerank_docs(state, doc_items):
    """
    阶段二：对文档进行重排序
    """
    question = state.get("rewritten_query") or state.get("original_query") or ""

    if not doc_items or not question:
        logger.warning("Step 2: 跳过重排序 (无文档或无问题)")
        return []

    logger.info(f"Step 2: 开始重排序 (Rerank), 待排序文档数: {len(doc_items)}")
    descriptor_docs = []
    for item in doc_items:
        descriptor_docs.append(
            {
                "chunk_id": item.get("chunk_id"),
                "doc_id": item.get("doc_id"),
                "source": item.get("source"),
                "title": item.get("title"),
                "text_hash": hashlib.sha256(str(item.get("text") or "").encode("utf-8")).hexdigest(),
            }
        )
    cache_descriptor = {"question": question, "docs": descriptor_docs}
    cached = query_cache_get("rerank", cache_descriptor)
    if isinstance(cached, list):
        logger.info(f"Step 2: 重排序缓存命中，直接返回 {len(cached)} 条结果")
        return cached

    texts = [x["text"] for x in doc_items]
    try:
        reranker = get_reranker_model()
        sentence_pairs = [[question, t] for t in texts]
        logger.info("Step 2: 正在计算相关性得分...")
        scores = reranker.compute_score(sentence_pairs)

        scored_docs = []
        for item, text, score in zip(doc_items, texts, scores):
            score_val = float(score)
            scored_docs.append(
                {
                    "text": text,
                    "score": score_val,
                    "source": item.get("source") or "",
                    "chunk_id": item.get("chunk_id"),
                    "doc_id": item.get("doc_id"),
                    "url": item.get("url") or "",
                    "title": item.get("title") or "",
                    "image_urls": item.get("image_urls") or [],
                    "graph_fact": item.get("graph_fact") or "",
                    "graph_entities": item.get("graph_entities") or [],
                    "section_title": item.get("section_title") or "",
                    "section_path": item.get("section_path") or "",
                    "chunk_context": item.get("chunk_context") or "",
                    "anchor_terms": item.get("anchor_terms") or [],
                    "document_title": item.get("document_title") or "",
                    "expanded_chunk_ids": item.get("expanded_chunk_ids") or [],
                    "context_expanded": bool(item.get("context_expanded", False)),
                    "evidence_chunk_ids": item.get("evidence_chunk_ids") or [],
                    "evidence_text": item.get("evidence_text") or "",
                }
            )
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        query_cache_set("rerank", cache_descriptor, scored_docs)
        return scored_docs
    except Exception:
        logger.exception("Step 2: 重排序过程发生异常")
        fallback_docs = [
            {
                "text": x.get("text"),
                "score": 0.0,
                "source": x.get("source") or "",
                "chunk_id": x.get("chunk_id"),
                "doc_id": x.get("doc_id"),
                "url": x.get("url") or "",
                "title": x.get("title") or "",
                "image_urls": x.get("image_urls") or [],
                "graph_fact": x.get("graph_fact") or "",
                "graph_entities": x.get("graph_entities") or [],
                "section_title": x.get("section_title") or "",
                "section_path": x.get("section_path") or "",
                "chunk_context": x.get("chunk_context") or "",
                "anchor_terms": x.get("anchor_terms") or [],
                "document_title": x.get("document_title") or "",
                "expanded_chunk_ids": x.get("expanded_chunk_ids") or [],
                "context_expanded": bool(x.get("context_expanded", False)),
                "evidence_chunk_ids": x.get("evidence_chunk_ids") or [],
                "evidence_text": x.get("evidence_text") or "",
            }
            for x in doc_items
        ]
        return fallback_docs


def step_3_topk(
    scored_docs,
    query_anchor_targets=None,
    query_type: str = "general",
    query_family: str = "general",
):
    """
    阶段三：动态 TopK
    基于 scored_docs（已按 score 降序排序）进行智能截断，
    核心逻辑：结合固定上下限+断崖阈值判断，避免机械取前N条，保留语义相关的连续文档集合
    """
    cfg = query_threshold_config
    max_topk = min(cfg.rerank_max_topk, len(scored_docs))
    min_topk = cfg.rerank_min_topk
    if query_family in {"comparison", "multi_hop_relation"} or (
        query_type in {"comparison", "relation", "constraint", "explain"}
        and query_family not in {"section_summary", "section_lookup", "procedure_lookup"}
    ):
        min_topk = min(max_topk, max(min_topk, 4))
    gap_ratio = cfg.rerank_gap_ratio
    gap_abs = cfg.rerank_gap_abs

    topk = max_topk
    if topk > min_topk:
        for i in range(min_topk - 1, max_topk - 1):
            s1 = scored_docs[i].get("score")
            s2 = scored_docs[i + 1].get("score")
            gap = s1 - s2
            rel = gap / (abs(s1) + 1e-6)
            if gap >= gap_abs or rel >= gap_ratio:
                logger.info(
                    f"Step 3: 触发断崖截断 @ index={i} (Score {s1:.4f} -> {s2:.4f}, Gap={gap:.4f})"
                )
                topk = i + 1
                break

    target_count = len([target for target in query_anchor_targets or [] if str(target or "").strip()])
    if target_count:
        topk = max(topk, min(len(scored_docs), target_count))

    coverage_ordered_docs = reorder_docs_for_target_coverage(
        scored_docs,
        query_anchor_targets or [],
    )
    topk_docs = coverage_ordered_docs[:topk]
    logger.info(f"Step 3: 截断完成，保留前 {len(topk_docs)} 条文档 (TopK={topk})")

    if topk_docs:
        preview = ", ".join(
            [
                f"{d.get('chunk_id') or 'Web'}({d.get('score'):.3f})"
                for d in topk_docs[:3]
            ]
        )
        logger.debug(f"Step 3: Top3 文档预览: {preview}")

    return topk_docs


def node_rerank(state):
    """
    Rerank节点
    对检索到的文档进行重新排序，提高相关性
    """
    logger.info("---Rerank (重排序) 节点开始处理---")
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    doc_items = step_1_merge_docs(state)
    scored_docs = step_2_rerank_docs(state, doc_items)
    topk_docs = step_3_topk(
        scored_docs,
        state.get("query_anchor_targets") or [],
        query_type=str(state.get("query_type") or "general"),
        query_family=str(state.get("router_query_family") or "general"),
    )

    logger.info(f"Rerank 节点处理结束, 最终输出 {len(topk_docs)} 条文档")

    add_done_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )
    return {"reranked_docs": topk_docs}


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rerank 本地测试")
    print("=" * 50)

    mock_rrf_chunks = [
        {
            "chunk_id": "local_1",
            "content": "RRF是一种倒数排名融合算法",
            "title": "算法介绍",
            "score": 0.9,
        },
        {
            "chunk_id": "local_2",
            "content": "BGE是一个强大的重排序模型",
            "title": "模型介绍",
            "score": 0.8,
        },
        {
            "chunk_id": "local_3",
            "content": "无关的测试文档内容",
            "title": "测试文档",
            "score": 0.1,
        },
    ]

    mock_web_docs = [
        {
            "title": "Rerank技术详解",
            "url": "http://web.com/1",
            "snippet": "Rerank即重排序，常用于RAG系统的第二阶段",
        },
        {
            "title": "无关网页",
            "url": "http://web.com/2",
            "snippet": "今天天气不错，适合出去游玩",
        },
    ]

    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？",
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False,
    }

    try:
        result = node_rerank(mock_state)
        reranked = result.get("reranked_docs", [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入文档总数: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"输出文档总数: {len(reranked)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(reranked, 1):
            print(
                f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:20]}..."
            )

        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] Rerank 打分正常")
        else:
            print("\n[FAIL] Rerank 打分异常 (均为0或负数)")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
