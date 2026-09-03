"""
Turns a flagged window's raw features + deterministic reasons into a short,
merchant-facing / analyst-facing natural language explanation, using an LLM.

Why an LLM here at all, given the reasons are already deterministic?
Because the audit trail needs to read like something a compliance reviewer
or a merchant support agent can act on immediately, not a list of z-scores.
The LLM NEVER decides the action - `rules_engine.decide()` already did that
deterministically. The LLM only narrates a decision that was already made,
which keeps the system's core money-decision path deterministic and testable.

Supports TWO providers, tried in this order:
  1. ANTHROPIC_API_KEY  -> Claude (claude-sonnet-4-6)
  2. GEMINI_API_KEY      -> Gemini (gemini-3.6-flash), via the current
                            `google-genai` SDK (the old `google-generativeai`
                            package is deprecated)
If neither key is set, both `explain()` and `answer_question()` fall back to
a clean offline template so the whole pipeline still runs end-to-end without
any API key (useful for CI / grading without secrets).
"""

import os
import json

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass  # python-dotenv not installed - env vars must be set manually


def _call_llm(system_prompt: str, user_payload: dict, max_tokens: int = 250) -> str:
    """Provider-agnostic LLM call. Tries Claude first, then Gemini, based on
    whichever API key is present. Returns raw response text (expected to be
    a JSON string per the system prompt's instructions) or raises - callers
    are responsible for their own try/except + offline fallback."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    if anthropic_key:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": json.dumps(user_payload)}],
        )
        return "".join(b.text for b in msg.content if hasattr(b, "text"))

    if gemini_key:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=gemini_key)
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=json.dumps(user_payload),
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                # Give plenty of headroom above the visible-answer length:
                # Gemini spends part of this budget on internal reasoning
                # before writing the JSON, and thinking_budget=0 isn't
                # supported the same way across all Gemini model versions,
                # so we don't try to disable it - we just budget generously.
                max_output_tokens=max_tokens * 4,
                temperature=0.3,
            ),
        )
        text = response.text
        if not text:
            raise RuntimeError(f"Gemini returned no text (finish_reason={response.candidates[0].finish_reason if response.candidates else 'unknown'})")
        return text

    raise RuntimeError("No LLM API key configured (set ANTHROPIC_API_KEY or GEMINI_API_KEY)")


def _parse_json_response(text: str) -> dict:
    """Both providers occasionally wrap JSON in ```json fences, or prepend
    stray preamble text, despite instructions not to - strip defensively
    before parsing. As a last resort, extract the substring between the
    first '{' and last '}', which recovers JSON even if the model added a
    sentence before or after it."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned[4:] if cleaned.startswith("json") else cleaned
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise json.JSONDecodeError(
            f"No valid JSON found in response. Raw text: {text[:300]!r}", cleaned, 0)


SYSTEM_PROMPT = """You are a fraud-risk explanation assistant for a payments platform.
You are given a flagged transaction window (aggregated stats, not raw card data)
and the deterministic reasons a risk model flagged it, plus the bounded action
that was already decided by a separate rules engine.
Write a concise (<=60 words) explanation for a human risk analyst.
State: what looked abnormal, how confident to be, and what the analyst should
check first. Do not invent numbers not given to you. Do not recommend an
action - the action is already decided; you are only narrating why.
Return strict JSON only, no markdown fences: {"explanation": "...", "check_first": "..."}"""


def _template_explain(reasons, decision, row):
    return {
        "explanation": (
            f"Flagged with risk score {row['risk_score']:.2f}. "
            + "; ".join(reasons) + f". Action taken: {decision.action}."
        ),
        "check_first": reasons[0] if reasons else "review aggregated window stats",
    }


def explain(row, reasons, decision) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        return _template_explain(reasons, decision, row)

    try:
        payload = {
            "window_start": str(row["ts"]),
            "merchant_id": row["merchant_id"],
            "txn_count": int(row["txn_count"]),
            "total_amount_inr": round(float(row["total_amount"]), 2),
            "fail_rate": round(float(row["fail_rate"]), 2),
            "distinct_bins": int(row["distinct_bins"]),
            "distinct_countries": int(row["distinct_countries"]),
            "risk_score": round(float(row["risk_score"]), 3),
            "deterministic_reasons": reasons,
            "bounded_action_already_decided": decision.action,
        }
        text = _call_llm(SYSTEM_PROMPT, payload, max_tokens=400)
        return _parse_json_response(text)
    except json.JSONDecodeError as e:
        fallback = _template_explain(reasons, decision, row)
        fallback["llm_error"] = f"{e} | raw_text_snippet={text[:200]!r}" if "text" in dir() else str(e)
        return fallback
    except Exception as e:
        fallback = _template_explain(reasons, decision, row)
        fallback["llm_error"] = str(e)
        return fallback


ASK_SYSTEM_PROMPT = """You are a fraud-risk analyst assistant for a payments
platform, answering a human reviewer's follow-up question about ONE already-
flagged transaction window. You are given the window's aggregated stats, the
deterministic reasons it was flagged, the bounded action already decided,
and any explanation already given.

Answer the reviewer's specific question directly and concisely (<=80 words).
Ground every claim in the numbers you were given - never invent a card
number, customer name, IP address, or any other specific identifier that
was not provided to you; this data is aggregated window statistics, not raw
PII, and you do not have access to anything beyond what's in the payload.
If the question asks for something you don't have data for (e.g. "what is
this customer's full name"), say plainly that this system only sees
aggregated, anonymized window statistics and does not have that detail.
Do not change or second-guess the action that was already decided - you are
answering a question, not making a new risk decision.
Return strict JSON only, no markdown fences: {"answer": "..."}"""


def _template_answer(question, row, reasons, decision):
    return {
        "answer": (
            f"(Offline mode - no live LLM connection) This window has a "
            f"risk score of {row['risk_score']:.2f} and was actioned as "
            f"{decision.action}. Flagged reasons: {'; '.join(reasons)}. "
            f"For a full analysis, set ANTHROPIC_API_KEY or GEMINI_API_KEY."
        )
    }


def answer_question(question: str, row, reasons, decision_action: str, bounded_exposure: float = 0, prior_explanation: str = None) -> dict:
    """Free-form Q&A over an already-scored window - the interactive layer on
    top of explain(). Same offline-fallback contract: never crashes without
    a live API key, always returns a usable (if generic) answer.

    `row` can be a pandas Series (from the live scoring path) or a plain
    dict (from a stored audit record) - only dict-style .get()/[] access is
    used below so both work without extra plumbing."""

    class _FakeDecision:
        action = decision_action

    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        return _template_answer(question, row, reasons, _FakeDecision())

    try:
        payload = {
            "window_start": str(row["ts"]) if "ts" in row else row.get("window_start"),
            "merchant_id": row.get("merchant_id"),
            "txn_count": int(row.get("txn_count", 0)),
            "total_amount_inr": round(float(row.get("total_amount", row.get("total_amount_inr", 0))), 2),
            "avg_amount_inr": round(float(row.get("avg_amount", 0)), 2),
            "fail_rate": round(float(row.get("fail_rate", 0)), 2),
            "distinct_bins": int(row.get("distinct_bins", 0)),
            "distinct_countries": int(row.get("distinct_countries", 0)),
            "risk_score": round(float(row.get("risk_score", 0)), 3),
            "deterministic_reasons": reasons,
            "bounded_action_already_decided": decision_action,
            "bounded_exposure_inr": bounded_exposure,
            "prior_explanation": prior_explanation,
            "reviewer_question": question,
        }
        text = _call_llm(ASK_SYSTEM_PROMPT, payload, max_tokens=450)
        return _parse_json_response(text)
    except Exception as e:
        fallback = _template_answer(question, row, reasons, _FakeDecision())
        fallback["llm_error"] = str(e)
        return fallback
