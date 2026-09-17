"""
engine/agents.py
-------------------
Groq-powered agentic workflow (OpenAI-compatible SDK) for the Resume
Screening Copilot. Three agents:

  1. Parsing Agent        -> structured candidate fields (name, skills, etc.)
  2. Evaluation Agent      -> 0-100 match score vs. the job description
  3. Explainability Agent  -> plain-language recruiter report + recommendation
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Any

try:
    import groq
except ImportError:  # pragma: no cover - groq is declared in requirements
    groq = None

from openai import APIStatusError, NotFoundError, OpenAI

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "qwen/qwen3.8-27b",
]

VALID_RECOMMENDATIONS = ("Shortlist", "Hold", "Reject")


class AgentError(Exception):
    """Raised for any Groq API / agent-level failure."""
    pass


def normalize_model_choice(model: str) -> str:
    cleaned = (model or "").strip()
    if cleaned in AVAILABLE_MODELS:
        return cleaned
    return DEFAULT_MODEL


def _is_model_not_found_error(error: Exception) -> bool:
    if isinstance(error, NotFoundError):
        return True
    if isinstance(error, APIStatusError) and getattr(error, "status_code", None) == 404:
        return True
    if groq is not None and isinstance(error, getattr(groq, "NotFoundError", ())):
        return True

    error_text = str(error).lower()
    return any(
        marker in error_text
        for marker in (
            "404",
            "model_not_found",
            "does not exist",
            "do not have access",
            "not found",
        )
    )


# --------------------------------------------------------------------------
# Client setup
# --------------------------------------------------------------------------

def get_groq_client(api_key: str) -> OpenAI:
    if not api_key or not api_key.strip():
        raise AgentError("Groq API key is missing. Set GROQ_API_KEY in .env or Streamlit secrets.")
    try:
        return OpenAI(api_key=api_key.strip(), base_url=GROQ_BASE_URL)
    except Exception as e:
        raise AgentError(f"Failed to initialize Groq client: {e}")


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------

def _extract_json(raw_text: str) -> Dict[str, Any]:
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty response from model.")

    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _call_groq_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    candidate_models = []
    for candidate in [normalize_model_choice(model), DEFAULT_MODEL, FALLBACK_MODEL, *AVAILABLE_MODELS]:
        if candidate and candidate not in candidate_models:
            candidate_models.append(candidate)

    last_error = None
    for candidate_model in candidate_models:
        try:
            response = client.chat.completions.create(
                model=candidate_model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
                response_format={"type": "json_object"},
            )
            break
        except Exception as e:
            last_error = e
            error_text = str(e).lower()

            if "response_format" not in error_text and "json" not in error_text:
                if _is_model_not_found_error(e):
                    continue
                raise AgentError(f"Groq API request failed: {e}")

            try:
                fallback_messages = [
                    {
                        "role": "system",
                        "content": system_prompt + "\nRespond ONLY with valid JSON. No prose, no markdown fences.",
                    },
                    {"role": "user", "content": user_prompt},
                ]
                response = client.chat.completions.create(
                    model=candidate_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=fallback_messages,
                )
                break
            except Exception as inner_e:
                last_error = inner_e
                if _is_model_not_found_error(inner_e):
                    continue
                raise AgentError(f"Groq API request failed: {inner_e}")
    else:
        attempted = ", ".join(candidate_models)
        raise AgentError(
            "Groq model request failed for all configured models "
            f"({attempted}). Last error: {last_error}"
        )

    try:
        raw = response.choices[0].message.content
    except (AttributeError, IndexError) as e:
        raise AgentError(f"Unexpected Groq API response structure: {e}")

    try:
        return _extract_json(raw)
    except Exception as e:
        raise AgentError(f"Failed to parse JSON response from model: {e}")


# --------------------------------------------------------------------------
# Agent 1: Parsing Agent
# --------------------------------------------------------------------------

def parsing_agent(client: OpenAI, model: str, resume_text: str) -> Dict[str, Any]:
    system_prompt = (
        "You are a precise resume parsing assistant used in an ATS pipeline. "
        "Extract structured candidate information from raw resume text. "
        "Always respond with valid JSON only, no commentary."
    )
    user_prompt = f"""Extract the following fields from this resume text and return them as JSON with EXACTLY these keys:
- "name": string (candidate's full name, or "Unknown Candidate" if not found)
- "contact": object with "email" (string) and "phone" (string), use "Not found" if missing
- "top_skills": list of up to 10 strings, the candidate's strongest / most relevant skills
- "experience_years": integer, approximate total years of professional experience
- "education": list of strings, each like "Degree - Institution"

Resume text:
\"\"\"
{resume_text[:6000]}
\"\"\"

Return only the JSON object.
"""
    data = _call_groq_json(client, model, system_prompt, user_prompt, temperature=0.1, max_tokens=800)

    data.setdefault("name", "Unknown Candidate")
    if not data.get("name"):
        data["name"] = "Unknown Candidate"

    contact = data.get("contact")
    if not isinstance(contact, dict):
        contact = {}
    contact.setdefault("email", "Not found")
    contact.setdefault("phone", "Not found")
    data["contact"] = contact

    skills = data.get("top_skills")
    data["top_skills"] = skills if isinstance(skills, list) else []

    try:
        data["experience_years"] = int(float(data.get("experience_years", 0)))
    except (TypeError, ValueError):
        data["experience_years"] = 0

    education = data.get("education")
    data["education"] = education if isinstance(education, list) else []

    return data


# --------------------------------------------------------------------------
# Agent 2: Evaluation Agent
# --------------------------------------------------------------------------

def evaluation_agent(
    client: OpenAI,
    model: str,
    jd_text: str,
    resume_chunks: List[str],
    vector_score: float,
) -> Dict[str, Any]:
    system_prompt = (
        "You are an expert technical recruiter evaluating candidate-to-job fit. "
        "You combine retrieved semantic evidence with your own domain judgement. "
        "Respond with valid JSON only."
    )
    joined_chunks = "\n---\n".join(resume_chunks[:8]) if resume_chunks else "No relevant chunks retrieved."

    user_prompt = f"""Job Description:
\"\"\"
{jd_text[:4000]}
\"\"\"

Most relevant resume excerpts (retrieved via vector similarity search; the retrieval system already computed a semantic similarity score of {vector_score:.1f}/100 for this candidate):
\"\"\"
{joined_chunks[:4000]}
\"\"\"

Evaluate how well this candidate matches the job description overall, weighing both the semantic similarity signal and your own reading of the excerpts (required skills present, seniority fit, domain relevance).

Return JSON with EXACTLY these keys:
- "match_score": integer from 0 to 100 (overall fit score)
- "reasoning": a short string (1-2 sentences) explaining the score

Return only the JSON object.
"""
    data = _call_groq_json(client, model, system_prompt, user_prompt, temperature=0.1, max_tokens=400)

    try:
        score = float(data.get("match_score", vector_score))
    except (TypeError, ValueError):
        score = vector_score
    score = max(0.0, min(100.0, score))
    data["match_score"] = round(score, 1)
    data.setdefault("reasoning", "")

    return data


# --------------------------------------------------------------------------
# Agent 3: Explainability Agent
# --------------------------------------------------------------------------

def explainability_agent(
    client: OpenAI,
    model: str,
    jd_text: str,
    candidate_info: Dict[str, Any],
    match_score: float,
    resume_chunks: List[str],
) -> Dict[str, Any]:
    system_prompt = (
        "You are a senior hiring manager producing a concise, plain-language "
        "candidate screening report for a recruiting team. Respond with valid JSON only."
    )
    joined_chunks = "\n---\n".join(resume_chunks[:8]) if resume_chunks else "N/A"

    user_prompt = f"""Job Description:
\"\"\"
{jd_text[:3000]}
\"\"\"

Candidate profile (structured):
{json.dumps(candidate_info, indent=2)[:2000]}

Computed match score: {match_score}/100

Relevant resume excerpts:
\"\"\"
{joined_chunks[:3000]}
\"\"\"

Write a recruiter screening report. Return JSON with EXACTLY these keys:
- "matched_prerequisites": list of strings, requirements/skills the candidate clearly satisfies
- "skill_gaps": list of strings, missing qualifications or weak areas relative to the JD
- "recommendation": exactly one of "Shortlist", "Hold", or "Reject"
- "justification": string, 2-4 sentences in plain, non-technical language explaining the recommendation

Recommendation guidance: "Shortlist" if match_score >= 75 and no critical gaps; "Hold" if match_score is 45-74 or there are moderate gaps; "Reject" if match_score < 45 or there are major/critical gaps.

Return only the JSON object.
"""
    data = _call_groq_json(client, model, system_prompt, user_prompt, temperature=0.3, max_tokens=700)

    matched = data.get("matched_prerequisites")
    data["matched_prerequisites"] = matched if isinstance(matched, list) else []

    gaps = data.get("skill_gaps")
    data["skill_gaps"] = gaps if isinstance(gaps, list) else []

    rec = data.get("recommendation")
    if rec not in VALID_RECOMMENDATIONS:
        if match_score >= 75:
            rec = "Shortlist"
        elif match_score < 45:
            rec = "Reject"
        else:
            rec = "Hold"
    data["recommendation"] = rec

    data.setdefault("justification", "")

    return data
