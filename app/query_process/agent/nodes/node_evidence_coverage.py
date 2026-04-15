import sys
from typing import Any, Dict

from app.core.logger import logger
from app.query_process.agent.agentic_utils import (
    analyze_evidence_coverage,
    is_agentic_feature_enabled,
)
from app.utils.task_utils import add_done_task, add_running_task


def node_evidence_coverage(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("---node_evidence_coverage (证据覆盖检查) 开始处理---")
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    if not is_agentic_feature_enabled(state, "evidence_coverage"):
        summary = {
            "enabled": False,
            "coverage_score": 1.0,
            "needs_rescue": False,
            "doc_count": len(state.get("reranked_docs") or []),
        }
    else:
        summary = analyze_evidence_coverage(state)
        summary["enabled"] = True

    add_done_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )
    logger.info(
        f"node_evidence_coverage: 完成，coverage_score={summary.get('coverage_score')}, "
        f"needs_rescue={summary.get('needs_rescue')}"
    )
    return {"evidence_coverage_summary": summary}
