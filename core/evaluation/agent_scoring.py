"""Agent 评测的自动评分规则。

原则：**能用程序判的，绝不交给 LLM 打分。**
  * 工具选择、参数是否正确：从执行记录里逐项比对，完全确定。
  * 任务是否完成、必须包含/不得包含哪些结论：规则匹配，可复现。
  * 数值是否准确：答案里的数字要么能在工具结果中找到，要么在题目的
    允许值（derived numbers）里；找不到就计为"无依据数值"。
  * 依据是否来自正确手册：检查答案是否同时出现手册名与期望章节号。

局限：答案忠实度里"定性结论是否有依据"无法完全程序化，因此本模块只对
**数值**做无依据率统计，并在结果里保留缺失项，供人工复核。不要把 LLM 的
自评分当作唯一标准。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_LIST_MARKER_RE = re.compile(r"^\s*\d+[.、)）]\s*", re.MULTILINE)


def normalize(text: str) -> str:
    text = (text or "").lower()
    # 日期里的 T 分隔符与空格等价："2026-03-13T10:00" vs "2026-03-13 10:00"
    text = re.sub(r"(?<=\d)t(?=\d)", "", text)
    return re.sub(r"\s+", "", text)


DOC_ALIASES = {
    "离心泵故障排查手册": ["离心泵故障排查手册", "离心泵手册"],
    "网络设备告警处置指南": ["网络设备告警处置指南", "网络设备告警"],
    "旋转设备振动诊断与润滑管理规范": [
        "旋转设备振动诊断与润滑管理规范",
        "旋转设备振动诊断",
        "旋转设备维护规范",
    ],
}


def load_dataset(path: Path | str) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def call_name(call: dict) -> str:
    return str(call.get("tool_name") or call.get("name") or "")


def call_args(call: dict) -> dict:
    return call.get("arguments") or call.get("args") or {}


def selected_tools(record: dict) -> list[str]:
    return [call_name(c) for c in (record.get("tool_calls") or [])]


def _arg_matches(expected, actual) -> bool:
    exp = normalize(str(expected))
    act = normalize(str(actual))
    if not exp:
        return True
    return exp == act or exp in act or act in exp


def score_tool_selection(case: dict, record: dict) -> dict:
    names = selected_tools(record)
    name_set = set(names)
    missing = [t for t in case.get("required_tools", []) if t not in name_set]
    any_required = case.get("required_any", [])
    any_ok = (not any_required) or any(t in name_set for t in any_required)
    forbidden = [t for t in case.get("forbidden_tools", []) if t in name_set]
    return {
        "ok": not missing and any_ok and not forbidden,
        "selected": names,
        "missing": missing,
        "required_any_ok": any_ok,
        "forbidden_hit": forbidden,
    }


def score_tool_args(case: dict, record: dict) -> dict:
    specs = case.get("required_args", {})
    names = set(selected_tools(record))
    any_required = set(case.get("required_any", []))
    required = set(case.get("required_tools", []))

    details: list[dict] = []
    checked = 0
    ok = True
    for tool, expected_args in specs.items():
        if tool not in names:
            # required_any 里的工具没被选，但另一个替代工具被选了：不强制它的参数
            if tool in any_required and any(t in names for t in any_required):
                details.append({"tool": tool, "status": "skipped_alternative"})
                continue
            if tool in required:
                ok = False
                details.append({"tool": tool, "status": "missing_call"})
            continue

        calls = [c for c in (record.get("tool_calls") or []) if call_name(c) == tool]
        matched_call = None
        for call in calls:
            args = call_args(call)
            if all(
                key in args and _arg_matches(value, args.get(key))
                for key, value in expected_args.items()
            ):
                matched_call = args
                break
        checked += 1
        if matched_call is None:
            ok = False
            details.append({"tool": tool, "status": "arg_mismatch", "calls": [call_args(c) for c in calls]})
        else:
            details.append({"tool": tool, "status": "ok", "args": matched_call})

    return {
        "ok": ok,
        "applicable": checked > 0,
        "checked_tools": checked,
        "details": details,
    }


def score_task(case: dict, record: dict) -> dict:
    answer = record.get("final_answer") or ""
    norm = normalize(answer)
    missing = [text for text in case.get("must_include", []) if normalize(text) not in norm]
    any_required = case.get("must_include_any", [])
    any_hit = [text for text in any_required if normalize(text) in norm]
    any_ok = (not any_required) or bool(any_hit)
    forbidden = [text for text in case.get("must_not_include", []) if normalize(text) in norm]
    return {
        "ok": not missing and any_ok and not forbidden,
        "missing": missing,
        "must_include_any_ok": any_ok,
        "must_include_any_hit": any_hit,
        "forbidden_hit": forbidden,
    }


def extract_numbers(text: str) -> list[float]:
    cleaned = _LIST_MARKER_RE.sub("", text or "")
    values: list[float] = []
    for token in _NUM_RE.findall(cleaned):
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _number_in(value: float, candidates: list[float], rel_tol: float = 0.02) -> bool:
    for cand in candidates:
        if value == cand:
            return True
        if abs(value - cand) <= max(0.01, abs(cand) * rel_tol):
            return True
    return False


def score_numbers(case: dict, record: dict) -> dict:
    answer = record.get("final_answer") or ""
    answer_numbers = extract_numbers(answer)
    expected = [float(v) for v in case.get("expected_numbers", [])]
    correct = [v for v in expected if _number_in(v, answer_numbers)]
    missing = [v for v in expected if not _number_in(v, answer_numbers)]
    return {
        "applicable": bool(expected),
        "expected": expected,
        "correct": correct,
        "missing": missing,
        "accuracy": len(correct) / len(expected) if expected else None,
    }


def score_unsupported(case: dict, record: dict) -> dict:
    """答案中的数字能否追溯到工具结果、问题或允许的派生值。"""
    answer = record.get("final_answer") or ""
    answer_numbers = extract_numbers(answer)

    reference_text = [case.get("question", "")]
    reference_text.extend(str(c.get("result") or "") for c in record.get("tool_calls") or [])
    reference_numbers = extract_numbers("\n".join(reference_text))
    reference_numbers.extend(float(v) for v in case.get("expected_numbers", []))
    reference_numbers.extend(float(v) for v in case.get("allowed_numbers", []))

    unsupported = [v for v in answer_numbers if not _number_in(v, reference_numbers)]
    return {
        "total_claims": len(answer_numbers),
        "unsupported": unsupported,
        "supported": len(answer_numbers) - len(unsupported),
        "rate": (len(unsupported) / len(answer_numbers)) if answer_numbers else 0.0,
    }


def score_evidence(case: dict, record: dict) -> dict:
    doc = case.get("expected_doc")
    sections = case.get("expected_sections", [])
    if not doc and not sections:
        return {"applicable": False, "ok": True}

    answer = normalize(record.get("final_answer") or "")
    aliases = case.get("expected_doc_aliases") or DOC_ALIASES.get(doc, [doc] if doc else [])
    doc_ok = (not doc) or any(normalize(alias) in answer for alias in aliases)
    section_hit = [s for s in sections if normalize(s) in answer]
    section_ok = (not sections) or bool(section_hit)
    return {
        "applicable": True,
        "ok": doc_ok and section_ok,
        "doc_ok": doc_ok,
        "section_hit": section_hit,
        "sections_ok": section_ok,
    }


def failure_reasons(score: dict) -> list[str]:
    reasons: list[str] = []
    if score.get("error"):
        reasons.append(f"执行错误：{score['error']}")
    selection = score["tool_selection"]
    if not selection["ok"]:
        if selection["missing"]:
            reasons.append(f"缺少工具：{','.join(selection['missing'])}")
        if not selection["required_any_ok"]:
            reasons.append("未调用任一可接受工具")
        if selection["forbidden_hit"]:
            reasons.append(f"调用了禁用工具：{','.join(selection['forbidden_hit'])}")
    args = score["tool_args"]
    if args["applicable"] and not args["ok"]:
        reasons.append("工具参数不匹配")
    task = score["task"]
    if not task["ok"]:
        if task["missing"]:
            reasons.append(f"答案缺少：{','.join(task['missing'])}")
        if not task["must_include_any_ok"]:
            reasons.append("答案未包含任一可接受表述")
        if task["forbidden_hit"]:
            reasons.append(f"答案出现禁止表述：{','.join(task['forbidden_hit'])}")
    numbers = score["numbers"]
    if numbers["applicable"] and numbers["missing"]:
        reasons.append(f"缺少关键数值：{numbers['missing']}")
    evidence = score["evidence"]
    if evidence["applicable"] and not evidence["ok"]:
        reasons.append("手册依据不完整（手册名或章节号缺失）")
    return reasons


def score_case(case: dict, record: dict) -> dict:
    score = {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "answer": record.get("final_answer") or "",
        "error": record.get("error"),
        "trace_id": record.get("trace_id"),
        "tool_selection": score_tool_selection(case, record),
        "tool_args": score_tool_args(case, record),
        "task": score_task(case, record),
        "numbers": score_numbers(case, record),
        "unsupported": score_unsupported(case, record),
        "evidence": score_evidence(case, record),
    }
    score["failure_reasons"] = failure_reasons(score)
    score["success"] = not score["failure_reasons"]
    return score


def aggregate(scores: list[dict]) -> dict:
    if not scores:
        return {"n": 0}

    def applicable_mean(key: str) -> float | None:
        rows = [s[key] for s in scores if s[key].get("applicable", True)]
        return mean(1.0 if r["ok"] else 0.0 for r in rows) if rows else None

    number_rows = [s for s in scores if s["numbers"]["applicable"]]
    total_expected = sum(len(s["numbers"]["expected"]) for s in number_rows)
    total_correct = sum(len(s["numbers"]["correct"]) for s in number_rows)
    total_numeric_claims = sum(s["unsupported"]["total_claims"] for s in scores)
    total_unsupported = sum(len(s["unsupported"]["unsupported"]) for s in scores)

    return {
        "n": len(scores),
        "tool_selection_accuracy": applicable_mean("tool_selection"),
        "tool_argument_accuracy": (
            mean(1.0 if s["tool_args"]["ok"] else 0.0
                 for s in scores if s["tool_args"]["applicable"])
            if any(s["tool_args"]["applicable"] for s in scores) else None
        ),
        "task_success_rate": applicable_mean("task"),
        "numeric_accuracy": (total_correct / total_expected) if total_expected else None,
        "evidence_accuracy": applicable_mean("evidence"),
        "unsupported_claim_rate": (
            total_unsupported / total_numeric_claims if total_numeric_claims else None
        ),
        "error_rate": mean(1.0 if s["error"] else 0.0 for s in scores),
        "success_rate": mean(1.0 if s["success"] else 0.0 for s in scores),
        "numeric_claims_total": total_numeric_claims,
        "numeric_claims_unsupported": total_unsupported,
    }


def validate_dataset(cases: list[dict], available_tools: set[str]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for case in cases:
        cid = case.get("id", "")
        if not cid:
            problems.append("存在缺少 id 的用例")
        elif cid in seen:
            problems.append(f"{cid}: id 重复")
        seen.add(cid)

        for key in ("question", "category"):
            if not case.get(key):
                problems.append(f"{cid}: 缺少 {key}")

        tools = (
            case.get("required_tools", [])
            + case.get("required_any", [])
            + case.get("forbidden_tools", [])
            + list(case.get("required_args", {}))
        )
        for tool in tools:
            if tool not in available_tools:
                problems.append(f"{cid}: 未知工具 {tool}")

        if not case.get("required_tools") and not case.get("required_any"):
            problems.append(f"{cid}: 未指定任何期望工具")
    return problems
