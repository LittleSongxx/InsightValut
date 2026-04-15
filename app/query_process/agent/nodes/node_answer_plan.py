import sys
from typing import Any, Dict

from app.core.logger import logger
from app.query_process.agent.agentic_utils import build_answer_plan
from app.utils.task_utils import add_done_task, add_running_task


def node_answer_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("---node_answer_plan (结构化回答规划) 开始处理---")
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    answer_plan = build_answer_plan(state)

    add_done_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )
    logger.info(
        f"node_answer_plan: 完成，query_type={answer_plan.get('query_type')}, "
        f"response_format={answer_plan.get('response_format')}"
    )
    return {"answer_plan": answer_plan}
