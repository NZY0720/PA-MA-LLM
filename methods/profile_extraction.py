"""Heterogeneity profile extraction pipeline.

Implements the schema-aligned RAG procedure described in the manuscript:
for each LCP and each subjective behavioral dimension m, retrieve TopK
evidence chunks from the park-specific corpus and call a schema-constrained
LLM extractor to produce a tuple (theta, kappa, tau) representing the
preference score, confidence indicator, and evidence-grounded rationale.

The extractor is wrapped so that:
  - keyword-based retrieval is deterministic and offline,
  - the LLM step uses the same DeepSeekJSONClient used elsewhere,
  - results are cached on disk so the case study does not re-pay the API
    cost on every experiment run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.constants import BASE_DIR, DEEPSEEK_API_URL, DEFAULT_MODEL
from utils.io_utils import load_json, write_json

CORPUS_DIR = BASE_DIR / "data" / "profile_corpus"
PROFILE_CACHE_DIR = BASE_DIR / "outputs" / "profile_extraction"

SUBJECTIVE_DIMENSIONS = (
    {
        "name": "risk",
        "label": "Risk Tolerance",
        "definition": "Willingness to accept forecast and trade-execution risk for "
                       "potential economic upside. Higher score means more risk-tolerant.",
        "query_terms": ("risk", "conservative", "aggressive", "speculative", "buffer", "exposure"),
    },
    {
        "name": "carbon",
        "label": "Carbon Preference",
        "definition": "Strength of carbon-driven preference over pure cost minimization. "
                       "Higher score means more willingness to pay a small cost premium for carbon displacement.",
        "query_terms": ("carbon", "emission", "low-carbon", "decarbon", "green", "ambition"),
    },
    {
        "name": "service",
        "label": "Service Priority",
        "definition": "How strongly tenant comfort and service-quality requirements bind the operator. "
                       "Higher score means service is more strictly protected.",
        "query_terms": ("service", "comfort", "tenant", "fulfilment", "quality", "HVAC"),
    },
    {
        "name": "autonomy",
        "label": "Operator Autonomy",
        "definition": "Day-to-day discretion of the operator within governance constraints. "
                       "Higher score means more autonomous operator.",
        "query_terms": ("autonomy", "authority", "discretion", "committee", "approval", "mandate"),
    },
    {
        "name": "fair",
        "label": "Fairness Priority",
        "definition": "Emphasis on benefit-sharing and equitable carbon-burden allocation across "
                       "tenants and counterparties. Higher score means stronger fairness orientation.",
        "query_terms": ("fair", "fairness", "allocation", "share", "burden", "equitable"),
    },
    {
        "name": "neg",
        "label": "Negotiation Concession",
        "definition": "How concessive the operator is in inter-park bidding. Higher score means "
                       "more willing to concede price or quantity to clear a trade.",
        "query_terms": ("negotiation", "concession", "bid", "ask", "patient", "firm", "revise"),
    },
)

EVIDENCE_TYPES = (
    "operating_charter",
    "esg_excerpt",
    "historical_ops_note",
    "expert_review",
    "policy_alignment",
)

_DEFAULT_TOPK = 2


@dataclass(frozen=True)
class EvidenceChunk:
    source: str
    text: str


@dataclass
class ExtractedDimension:
    theta: float
    confidence: float
    rationale: str
    source_chunks: list[str]


def _load_park_corpus(park_id: str) -> list[EvidenceChunk]:
    park_dir = CORPUS_DIR / park_id
    if not park_dir.exists():
        raise FileNotFoundError(f"Profile corpus directory missing: {park_dir}")
    chunks: list[EvidenceChunk] = []
    for md_file in sorted(park_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        chunks.append(EvidenceChunk(source=md_file.name, text=text))
    return chunks


def _retrieve_topk(chunks: list[EvidenceChunk], query_terms: tuple[str, ...], top_k: int) -> list[EvidenceChunk]:
    scored: list[tuple[int, EvidenceChunk]] = []
    for chunk in chunks:
        lower = chunk.text.lower()
        score = sum(len(re.findall(rf"\b{re.escape(term.lower())}\b", lower)) for term in query_terms)
        scored.append((score, chunk))
    scored.sort(key=lambda item: (-item[0], item[1].source))
    selected = [chunk for score, chunk in scored if score > 0][:top_k]
    if not selected:
        selected = [chunks[0]] if chunks else []
    return selected


def _build_extraction_prompt(
    park_id: str,
    park_type: str,
    dimension: dict[str, Any],
    evidence: list[EvidenceChunk],
) -> dict[str, Any]:
    return {
        "task": "Extract a single subjective profile dimension for a low-carbon park (LCP).",
        "park_id": park_id,
        "park_type": park_type,
        "dimension_name": dimension["name"],
        "dimension_label": dimension["label"],
        "dimension_definition": dimension["definition"],
        "evidence": [{"source": chunk.source, "text": chunk.text} for chunk in evidence],
        "instructions": (
            "Read the evidence carefully. Output a single JSON object with fields "
            "theta, confidence, and rationale. Theta is a real number in [0,1] "
            "summarizing how strongly this dimension applies to the park "
            "(0 = not at all, 1 = very strongly). Confidence is a real number in "
            "[0,1] reflecting how unambiguously the evidence supports the score. "
            "Rationale is one short sentence citing the relevant evidence file(s) "
            "by their source name."
        ),
        "output_schema": {
            "theta": "float in [0,1]",
            "confidence": "float in [0,1]",
            "rationale": "one short sentence",
        },
    }


def _clip_unit(value: Any, fallback: float = 0.5) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(0.0, min(1.0, x))


def _heuristic_extract(park_id: str, dimension: dict[str, Any]) -> ExtractedDimension:
    """Deterministic fallback used in mock-LLM mode and when the API fails.

    Uses simple lexical cues from the park's evidence; designed to be plausible
    but not as nuanced as the real LLM extraction.
    """
    chunks = _load_park_corpus(park_id)
    retrieved = _retrieve_topk(chunks, dimension["query_terms"], _DEFAULT_TOPK)
    text = " ".join(chunk.text.lower() for chunk in retrieved)

    def _hit(words: tuple[str, ...]) -> int:
        return sum(text.count(w) for w in words)

    name = dimension["name"]
    if name == "risk":
        high = _hit(("aggressive", "speculative", "short-notice", "near the upper end"))
        low = _hit(("conservative", "risk-averse", "buffer", "moderate-conservative"))
    elif name == "carbon":
        high = _hit(("carbon-first", "carbon priority", "above-baseline", "low-carbon", "decarbon", "green"))
        low = _hit(("cost-driven", "cost-priority", "no carbon premium", "regional floor"))
    elif name == "service":
        high = _hit(("tenant service", "comfort band", "service-tenant", "service quality"))
        low = _hit(("regional default", "participatory", "wider"))
    elif name == "autonomy":
        high = _hit(("high autonomy", "high; the", "full authority", "broad authority"))
        low = _hit(("committee", "approval", "must consult", "subject to"))
    elif name == "fair":
        high = _hit(("fair", "fairness", "shared", "equitable", "benefit"))
        low = _hit(("socialised", "metered-consumption", "no carbon-burden"))
    else:  # neg
        high = _hit(("concessive", "patient", "willing to wait", "moderately concessive"))
        low = _hit(("firm", "holds its position", "withholding"))
    raw = 0.5 + 0.06 * (high - low)
    theta = _clip_unit(raw)
    confidence = _clip_unit(0.55 + 0.05 * (high + low))
    rationale = f"Heuristic estimate from {', '.join(chunk.source for chunk in retrieved)}."
    return ExtractedDimension(
        theta=round(theta, 4),
        confidence=round(confidence, 4),
        rationale=rationale,
        source_chunks=[chunk.source for chunk in retrieved],
    )


def extract_park_profile(
    park_id: str,
    park_type: str,
    client: Any | None,
    use_mock: bool,
    top_k: int = _DEFAULT_TOPK,
) -> dict[str, ExtractedDimension]:
    chunks = _load_park_corpus(park_id)
    profile: dict[str, ExtractedDimension] = {}
    for dimension in SUBJECTIVE_DIMENSIONS:
        retrieved = _retrieve_topk(chunks, dimension["query_terms"], top_k)
        if use_mock or client is None:
            profile[dimension["name"]] = _heuristic_extract(park_id, dimension)
            continue
        prompt_user = _build_extraction_prompt(park_id, park_type, dimension, retrieved)
        try:
            raw = client.generate_json(
                "You extract a structured behavioral profile for a low-carbon park. Return JSON only.",
                prompt_user,
            )
            theta = _clip_unit(raw.get("theta"))
            confidence = _clip_unit(raw.get("confidence"))
            rationale = str(raw.get("rationale", "")).strip()[:240] or "No rationale provided."
            profile[dimension["name"]] = ExtractedDimension(
                theta=round(theta, 4),
                confidence=round(confidence, 4),
                rationale=rationale,
                source_chunks=[chunk.source for chunk in retrieved],
            )
        except Exception:
            profile[dimension["name"]] = _heuristic_extract(park_id, dimension)
    return profile


def _profile_to_dict(profile: dict[str, ExtractedDimension]) -> dict[str, Any]:
    return {
        name: {
            "theta": value.theta,
            "confidence": value.confidence,
            "rationale": value.rationale,
            "source_chunks": value.source_chunks,
        }
        for name, value in profile.items()
    }


def load_or_extract_profile(
    park_id: str,
    park_type: str,
    client: Any | None,
    use_mock: bool,
    force_refresh: bool = False,
) -> dict[str, ExtractedDimension]:
    cache_path = PROFILE_CACHE_DIR / f"{park_id}.json"
    if cache_path.exists() and not force_refresh:
        payload = load_json(cache_path)
        cached_dims = payload.get("dimensions", {})
        if all(dim["name"] in cached_dims for dim in SUBJECTIVE_DIMENSIONS):
            return {
                name: ExtractedDimension(
                    theta=float(cached_dims[name]["theta"]),
                    confidence=float(cached_dims[name]["confidence"]),
                    rationale=str(cached_dims[name]["rationale"]),
                    source_chunks=list(cached_dims[name].get("source_chunks", [])),
                )
                for name in cached_dims
            }
    profile = extract_park_profile(park_id, park_type, client, use_mock)
    write_json(
        cache_path,
        {
            "park_id": park_id,
            "park_type": park_type,
            "mode": "mock" if use_mock else "llm",
            "dimensions": _profile_to_dict(profile),
        },
    )
    return profile


def profile_signature(profile: dict[str, ExtractedDimension]) -> dict[str, float]:
    """Return a compact (theta-only) dict suitable for prompt injection."""
    return {name: value.theta for name, value in profile.items()}
