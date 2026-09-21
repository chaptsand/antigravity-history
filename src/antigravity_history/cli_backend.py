"""
Antigravity CLI (agy) data backend.

Supports direct reading of Antigravity CLI conversations:
- Summaries: ~/.gemini/antigravity-cli/conversation_summaries.db
- Transcripts: ~/.gemini/antigravity-cli/brain/<id>/.system_generated/logs/transcript_full.jsonl
"""

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional


def get_cli_dir() -> Path:
    """Return the Antigravity CLI application data directory."""
    env_dir = os.environ.get("ANTIGRAVITY_CLI_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".gemini" / "antigravity-cli"


def has_cli_data(cli_dir: Optional[Path] = None) -> bool:
    """Check if Antigravity CLI database exists."""
    base = cli_dir or get_cli_dir()
    db_file = base / "conversation_summaries.db"
    return db_file.is_file()


def get_cli_trajectories(cli_dir: Optional[Path] = None) -> dict[str, Any]:
    """Read conversation summaries from the CLI SQLite database.

    Returns:
        {cascade_id: summary_dict}
    """
    base = cli_dir or get_cli_dir()
    db_file = base / "conversation_summaries.db"
    if not db_file.is_file():
        return {}

    summaries = {}
    try:
        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                conversation_id,
                title,
                preview,
                step_count,
                last_modified_time,
                workspace_uris,
                status,
                last_user_input_time
            FROM conversation_summaries
            ORDER BY last_modified_time DESC
            """
        )
        rows = cursor.fetchall()
        conn.close()
    except Exception:
        try:
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    conversation_id,
                    title,
                    preview,
                    step_count,
                    last_modified_time,
                    workspace_uris,
                    status,
                    last_user_input_time
                FROM conversation_summaries
                ORDER BY last_modified_time DESC
                """
            )
            rows = cursor.fetchall()
            conn.close()
        except Exception:
            return {}

    for row in rows:
        cid = row[0]
        title = row[1] or ""
        preview = row[2] or ""
        step_count = row[3] or 0
        last_modified = str(row[4] or "")
        workspace_uris_raw = row[5] or ""
        status = row[6] or "DONE"
        last_user_input = str(row[7] or "")

        summary_text = title if title else (preview[:50] if preview else f"Conversation {cid[:8]}")

        workspaces = []
        if workspace_uris_raw:
            try:
                uris = json.loads(workspace_uris_raw)
                if isinstance(uris, list):
                    workspaces = [{"workspaceFolderAbsoluteUri": u} for u in uris]
            except Exception:
                pass

        summaries[cid] = {
            "summary": summary_text,
            "stepCount": step_count,
            "createdTime": last_user_input or last_modified,
            "lastModifiedTime": last_modified,
            "lastUserInputTime": last_user_input,
            "status": status,
            "workspaces": workspaces,
            "source": "cli",
        }

    return summaries


def get_cli_conversation_messages(
    cid: str,
    level: str = "default",
    cli_dir: Optional[Path] = None,
) -> list[dict]:
    """Parse messages from a CLI conversation transcript log.

    Args:
        cid: Conversation UUID
        level: "default" / "thinking" / "full"
        cli_dir: Optional base directory

    Returns:
        List of message dicts matching parser.parse_steps output format.
    """
    base = cli_dir or get_cli_dir()
    logs_dir = base / "brain" / cid / ".system_generated" / "logs"

    full_log = logs_dir / "transcript_full.jsonl"
    compact_log = logs_dir / "transcript.jsonl"

    log_file = full_log if full_log.is_file() else compact_log
    if not log_file.is_file():
        return []

    include_thinking = level in ("thinking", "full")
    include_full = level == "full"

    messages = []
    pending_tool = None

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    step = json.loads(line)
                except json.JSONDecodeError:
                    continue

                stype = step.get("type", "")
                created_at = step.get("created_at", "")
                content = step.get("content") or ""
                thinking = step.get("thinking") or ""
                tool_calls = step.get("tool_calls") or []

                # Tool output handling (GENERIC step following a tool call)
                if pending_tool and stype == "GENERIC":
                    if include_full:
                        pending_tool["output"] = content
                    if pending_tool.get("tool_name") == "run_command":
                        m = re.search(r"exited with code (\d+)", content)
                        if m:
                            pending_tool["exit_code"] = int(m.group(1))
                    elif pending_tool.get("tool_name") == "search_web":
                        if include_full:
                            pending_tool["search_summary"] = content
                    elif pending_tool.get("tool_name") == "view_file":
                        m_lines = re.search(r"Total Lines:\s*(\d+)", content)
                        if m_lines:
                            pending_tool["num_lines"] = int(m_lines.group(1))
                        m_bytes = re.search(r"Total Bytes:\s*(\d+)", content)
                        if m_bytes:
                            pending_tool["num_bytes"] = int(m_bytes.group(1))

                    messages.append(pending_tool)
                    pending_tool = None
                    continue
                elif pending_tool:
                    messages.append(pending_tool)
                    pending_tool = None

                if stype == "USER_INPUT":
                    m = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", content, re.DOTALL)
                    clean_content = m.group(1).strip() if m else content.strip()
                    msg = {"role": "user", "content": clean_content}
                    if include_thinking and created_at:
                        msg["timestamp"] = created_at
                    messages.append(msg)

                elif stype == "PLANNER_RESPONSE":
                    if include_thinking and thinking.strip():
                        msg = {
                            "role": "assistant",
                            "content": content.strip() if content else "",
                            "thinking": thinking.strip(),
                        }
                        if created_at:
                            msg["timestamp"] = created_at
                        messages.append(msg)
                    elif content and content.strip():
                        msg = {
                            "role": "assistant",
                            "content": content.strip(),
                        }
                        if created_at:
                            msg["timestamp"] = created_at
                        messages.append(msg)

                    for tc in tool_calls:
                        name = tc.get("name", "unknown")
                        args = tc.get("args") or {}
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {}

                        tool_msg = {"role": "tool", "tool_name": name}
                        if include_thinking and created_at:
                            tool_msg["timestamp"] = created_at

                        if name == "run_command":
                            tool_msg["content"] = args.get("CommandLine", "")
                            if include_thinking and args.get("Cwd"):
                                tool_msg["cwd"] = args.get("Cwd")
                        elif name in ("replace_file_content", "write_to_file"):
                            tool_msg["tool_name"] = "code_edit"
                            target = args.get("TargetFile", "")
                            desc = args.get("Description", "")
                            tool_msg["content"] = f"[Code Edit] {target}\n{desc}".strip() if desc else f"[Code Edit] {target}"
                            tool_msg["file_path"] = target
                            if include_full:
                                if "ReplacementContent" in args:
                                    tool_msg["diff"] = args["ReplacementContent"]
                                elif "CodeContent" in args:
                                    tool_msg["diff"] = args["CodeContent"]
                        elif name == "view_file":
                            tool_msg["content"] = args.get("AbsolutePath", "")
                        elif name == "search_web":
                            tool_msg["content"] = args.get("query", "")
                        elif name == "list_dir":
                            tool_msg["content"] = args.get("DirectoryPath", "")
                        elif name == "grep_search":
                            tool_msg["content"] = f"{args.get('SearchPath', '')} : {args.get('Query', '')}"
                        elif name == "find_by_name":
                            tool_msg["content"] = f"{args.get('SearchPath', '')} : {args.get('Pattern', '')}"
                        elif name == "ask_question":
                            questions = args.get("questions", [])
                            tool_msg["content"] = "\n".join(q.get("question", "") for q in questions)
                        else:
                            tool_msg["content"] = args.get("toolSummary") or args.get("toolAction") or str(args)

                        pending_tool = tool_msg

        if pending_tool:
            messages.append(pending_tool)

    except Exception:
        pass

    return messages
