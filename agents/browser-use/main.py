import asyncio
import json
import logging
import os
import urllib.parse
from typing import Any, Dict, Optional, Tuple

from browser_use import Agent as BrowserUseAgent
from browser_use import ChatGoogle
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

WAIT_FOR_MENTIONS_TOOL = "coral_wait_for_mentions"
SEND_MESSAGE_TOOL = "coral_send_message"
DEFAULT_TIMEOUT_MS = 60000
ERROR_RETRY_DELAY = 5


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_runtime_config() -> Dict[str, Any]:
    runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME")
    if runtime is None:
        load_dotenv()

    config = {
        "runtime": runtime,
        "coral_sse_url": os.getenv("CORAL_SSE_URL"),
        "agent_id": os.getenv("CORAL_AGENT_ID"),
        "agent_description": os.getenv(
            "CORAL_AGENT_DESCRIPTION",
            "Browser-use agent that can browse the web and report results.",
        ),
        "timeout_ms": int(os.getenv("TIMEOUT_MS", DEFAULT_TIMEOUT_MS)),
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "model": os.getenv("BROWSER_USE_MODEL", "gemini-2.5-flash"),
    }

    missing = [
        key
        for key in ("coral_sse_url", "agent_id", "google_api_key")
        if not config[key]
    ]
    if missing:
        raise ValueError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    return config


def normalize_tool_output(tool_output: Any) -> str:
    raw_result = tool_output[0] if isinstance(tool_output, tuple) else tool_output
    if isinstance(raw_result, list):
        for item in raw_result:
            if isinstance(item, str):
                return item
        return ""
    return raw_result or ""


async def wait_for_instruction(tool, timeout_ms: int) -> Tuple[str, Dict[str, Optional[str]]]:
    while True:
        tool_output = await tool.ainvoke({"timeoutMs": timeout_ms})
        result_text = normalize_tool_output(tool_output)

        if not result_text:
            await asyncio.sleep(1)
            continue

        try:
            payload = json.loads(result_text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to decode wait_for_mentions payload: %s", exc)
            await asyncio.sleep(1)
            continue

        result_type = payload.get("result")
        if result_type == "error_timeout":
            await asyncio.sleep(1)
            continue

        if result_type != "wait_for_mentions_success":
            logger.warning("Unexpected wait_for_mentions result type: %s", result_type)
            await asyncio.sleep(1)
            continue

        messages = payload.get("messages") or []
        if not messages:
            await asyncio.sleep(1)
            continue

        message = messages[0]
        content = (message.get("content") or "").strip()
        if not content:
            await asyncio.sleep(1)
            continue

        context = {
            "thread_id": message.get("threadId"),
            "sender_id": message.get("senderId"),
        }

        return content, context


async def execute_browser_task(task: str, model: str) -> str:
    def run_task() -> str:
        agent = BrowserUseAgent(task=task, llm=ChatGoogle(model=model))
        result = agent.run_sync()
        return str(result)

    return await asyncio.to_thread(run_task)


async def send_response(tool, context: Dict[str, Optional[str]], content: str) -> None:
    payload = {
        "threadId": context.get("thread_id"),
        "content": content,
        "mentions": [context.get("sender_id")] if context.get("sender_id") else [],
    }
    await tool.ainvoke(payload)


async def main() -> None:
    config = load_runtime_config()

    coral_params = {
        "agentId": config["agent_id"],
        "agentDescription": config["agent_description"],
    }

    query_string = urllib.parse.urlencode(coral_params)
    if "?" in config["coral_sse_url"]:
        coral_url = f"{config['coral_sse_url']}&{query_string}"
    else:
        coral_url = f"{config['coral_sse_url']}?{query_string}"

    logger.info("Connecting to Coral Server: %s", coral_url)

    client = MultiServerMCPClient(
        connections={
            "coral": {
                "transport": "sse",
                "url": coral_url,
                "timeout": float(config["timeout_ms"]),
                "sse_read_timeout": float(config["timeout_ms"]),
            }
        }
    )

    coral_tools = await client.get_tools(server_name="coral")
    tool_map = {tool.name: tool for tool in coral_tools}

    for required in (WAIT_FOR_MENTIONS_TOOL, SEND_MESSAGE_TOOL):
        if required not in tool_map:
            raise ValueError(f"Required tool '{required}' not available from Coral server")

    wait_tool = tool_map[WAIT_FOR_MENTIONS_TOOL]
    send_tool = tool_map[SEND_MESSAGE_TOOL]

    logger.info("Browser-use agent ready. Awaiting instructions...")

    while True:
        try:
            instruction, context = await wait_for_instruction(
                wait_tool,
                config["timeout_ms"],
            )
            logger.info(
                "Received instruction on thread %s from %s",
                context.get("thread_id"),
                context.get("sender_id"),
            )

            try:
                result = await execute_browser_task(instruction, config["model"])
                response = result or "Task completed with no additional output."
            except Exception as task_error:
                logger.exception("Browser-use task failed")
                response = f"Browser-use agent encountered an error: {task_error}"

            await send_response(send_tool, context, response)

        except Exception as loop_error:
            logger.exception("Error in browser-use agent loop: %s", loop_error)
            await asyncio.sleep(ERROR_RETRY_DELAY)


if __name__ == "__main__":
    asyncio.run(main())
