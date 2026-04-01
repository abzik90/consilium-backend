"""RAG (Retrieval-Augmented Generation) pipeline.

Orchestrates:
1. Semantic search over the knowledge base (vectorstore)
2. Context assembly from retrieved chunks
3. LLM call with augmented prompt
4. Citation extraction
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from app.llm import chat as llm_chat
from app.vectorstore import RetrievedChunk, search as vector_search

logger = logging.getLogger(__name__)

PROTOCOLS_CATEGORY = "Protocols"
TEXTBOOKS_CATEGORY = "Textbooks"
ARCHIVE_CATEGORY = "Previous Histories"
ARCHIVE_SIMILARITY_THRESHOLD = 0.55

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RAGCitation:
    """A citation reference extracted from the LLM response."""

    ref_num: int
    document_id: str
    document_name: str
    category: str
    page: int | None
    excerpt: str


@dataclass
class RAGResult:
    """The full result of a RAG query."""

    answer: str
    citations: list[RAGCitation] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)


# ---------------------------------------------------------------------------
# System prompt with RAG context
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """\
You are Consilium, a medical AI assistant specializing in clinical decision \
support for doctors in a hospital setting. Your primary domain is \
gastrointestinal (GI) internal bleeding management.

INSTRUCTIONS:
- Provide evidence-based, concise medical guidance.
- A set of knowledge base excerpts is provided below in the SOURCES block. \
  These excerpts come from your hospital's clinical knowledge base — \
  textbooks, protocols, and guidelines curated by the medical team.
- Source-priority policy:
    - MAIN diagnosis, routing, and treatment plan should be based primarily on \
        protocol documents when applicable.
    - Additional suggestions, caveats, explanatory notes, and background can be \
        supported by textbooks.
    - If archived previous cases are meaningfully similar, surface them as \
        similar cases and explain what is comparable.
- Evaluate the provided sources for applicability to the user's question. \
    Use and cite a source when it meaningfully supports the answer, such as \
    when it contains relevant pathophysiology, diagnostic criteria, treatment \
    algorithms, drug dosages, classification systems, or clinical guidance \
    that applies to the case.
- Do NOT cite a source only because it was retrieved. If a source is not \
    applicable, ignore it.
- When a source is applicable, prefer referring to the document explicitly \
    in the answer, for example: "Согласно клиническому протоколу [2], ..." \
    or "По данным руководства [1][3], ...".
- Cite sources with bracketed numbers like [1], [2], etc. Each citation \
  number corresponds to a source listed in the SOURCES block.
- Integrate citations naturally throughout your answer where they support \
    the clinical recommendation or factual statement.
- Every concrete recommendation in your answer should make its provenance \
    clear. If the recommendation is supported by the knowledge base, attach \
    the relevant citation directly to that recommendation. If the recommendation \
    is based on general medical reasoning and not directly supported by the \
    provided sources, explicitly mark it as "Общее клиническое соображение" \
    or "На основе общих медицинских знаний", and do not attach a fake citation.
- Never present a recommendation as if it came from the knowledge base unless \
    there is an applicable cited source supporting it.
- Prefer wording such as: "Рекомендуется срочная ЭГДС [2]" for KB-backed \
    points, and "Рассмотреть консультацию реаниматолога (общее клиническое \
    соображение)" for model-only points.
- If one or more sources from category "Previous Histories" are similar, add \
    a short section "### Похожие случаи из архива" and summarize only the \
    clinically relevant parallels with citations.
- At the very end of your response, list ALL cited references under a \
  "### Источники" heading, e.g.:\
  \n  ### Источники\
  \n  [1] Название документа (стр. X)\
  \n  [3] Название документа — раздел\
  \n  (only list the ones you actually cited in the text).
- If NONE of the provided sources have any relevance whatsoever, answer \
  from your general medical knowledge and add this at the very end of \
  your response:\
  \n  ---\
  \n  ⚠️ Не удалось найти релевантные данные в базе знаний. Ответ основан \
  на общих медицинских знаниях.
- Always clarify that your responses are for informational purposes only \
  and should not replace clinical judgement.
- Structure your response with clear headings and bullet points when \
  appropriate.
- Respond in the same language the user writes in. If the clinical data is \
  in Russian, respond in Russian.

WHEN THE USER PROVIDES CLINICAL ADMISSION DATA (patient examination notes, \
lab results, imaging findings, etc.), you MUST provide a structured response \
with the following sections:

### Предварительный диагноз
State the preliminary diagnosis based on the clinical picture, complaints, \
exam findings, and test results.

### Маршрутизация пациента
Recommend which hospital department the patient should be routed to \
(e.g., хирургия, реанимация, терапия, гастроэнтерология) with justification \
based on severity and clinical findings. Consider:
- Hemodynamic stability (BP, HR)
- Hemoglobin level and degree of anemia
- Active vs. completed bleeding signs
- Need for urgent surgical/endoscopic intervention
- ICU criteria (hemodynamic instability, Hb < 70 g/L, ongoing active bleeding)

### План обследования
Recommended diagnostic workup (labs, imaging, procedures).

### План лечения
Recommended treatment plan including:
- Diet and activity level (режим)
- IV fluid resuscitation and medications with dosing
- Blood transfusion recommendations if indicated (based on Hb, clinical status)
- Monitoring parameters (BP, HR, diuresis, repeat labs)
- Specialist consultations needed

Within the management plan sections, each bullet or recommendation should \
make clear whether it is:
- based on the knowledge base, with citation(s), or
- based on general medical reasoning without direct knowledge base support.

### Степень тяжести кровотечения
Classify the bleeding severity and anemia grade based on provided lab values \
and clinical findings.

{context_block}"""


def _dedupe_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Remove duplicate chunks while preserving order."""
    seen: set[tuple[str, int | None, str | None, str]] = set()
    out: list[RetrievedChunk] = []
    for chunk in chunks:
        key = (chunk.document_id, chunk.page, chunk.heading, chunk.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return out


def _retrieve_context_chunks(user_message: str, n_results: int) -> list[RetrievedChunk]:
    """Retrieve chunks with category-aware prioritization.

    Priority:
    1) Protocols for core recommendations
    2) Textbooks for supplemental suggestions/explanations
    3) Similar archived cases when similarity is strong
    """
    protocol_k = max(4, n_results // 2)
    textbook_k = max(3, n_results // 3)
    archive_k = max(2, n_results // 4)

    protocol_chunks = vector_search(
        query=user_message,
        n_results=protocol_k,
        category=PROTOCOLS_CATEGORY,
    )
    textbook_chunks = vector_search(
        query=user_message,
        n_results=textbook_k,
        category=TEXTBOOKS_CATEGORY,
    )
    archive_chunks = vector_search(
        query=user_message,
        n_results=archive_k,
        category=ARCHIVE_CATEGORY,
    )
    archive_chunks = [
        c for c in archive_chunks if c.score >= ARCHIVE_SIMILARITY_THRESHOLD
    ]

    merged = _dedupe_chunks(protocol_chunks + textbook_chunks + archive_chunks)

    if len(merged) < n_results:
        fallback_chunks = vector_search(query=user_message, n_results=n_results * 2)
        merged = _dedupe_chunks(merged + fallback_chunks)

    return merged[:n_results]


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered SOURCES block."""
    if not chunks:
        return ""

    lines = ["SOURCES:"]
    for i, chunk in enumerate(chunks, 1):
        source_info = f"[{i}] [{chunk.category}] {chunk.document_name}"
        if chunk.page:
            source_info += f" (p. {chunk.page})"
        if chunk.heading:
            source_info += f" — {chunk.heading}"
        source_info += f" | relevance={chunk.score:.2f}"
        lines.append(source_info)
        lines.append(chunk.text)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def query(
    user_message: str,
    history: list[dict[str, str]],
    *,
    n_results: int = 10,
    category: str | None = None,
) -> RAGResult:
    """Run the full RAG pipeline: retrieve → augment → generate.

    Parameters
    ----------
    user_message : str
        The current user question.
    history : list
        Prior conversation turns ``[{"role": ..., "content": ...}, ...]``.
    n_results : int
        Number of knowledge chunks to retrieve.
    category : str, optional
        Restrict retrieval to a specific knowledge category.

    Returns
    -------
    RAGResult with the answer, citations, and retrieved chunks.
    """
    # 1. Retrieve relevant chunks
    if category:
        chunks = vector_search(query=user_message, n_results=n_results, category=category)
    else:
        chunks = _retrieve_context_chunks(user_message, n_results)
    logger.info("RAG: retrieved %d chunks for query: %.80s…", len(chunks), user_message)

    # 2. Build augmented system prompt
    context_block = _build_context_block(chunks)
    system_prompt = RAG_SYSTEM_PROMPT.format(context_block=context_block)

    # 3. Call the LLM
    from app.llm import _get_api_key, OPENROUTER_URL, MODEL  # noqa: avoid circular
    import httpx

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    api_key = _get_api_key()
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.4,  # lower temp for factual RAG
        "max_tokens": 3072,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://consilium.kz",
        "X-Title": "Consilium",
    }

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(OPENROUTER_URL, json=payload, headers=headers)
        resp.raise_for_status()

    answer = resp.json()["choices"][0]["message"]["content"]

    # 4. Extract citation references from the answer  [1], [2], etc.
    cited_nums = set(int(m) for m in re.findall(r"\[(\d+)\]", answer))

    citations: list[RAGCitation] = []
    for num in sorted(cited_nums):
        idx = num - 1  # 1-based → 0-based
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            citations.append(
                RAGCitation(
                    ref_num=num,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    category=chunk.category,
                    page=chunk.page,
                    excerpt=chunk.text,
                )
            )

    return RAGResult(answer=answer, citations=citations, retrieved_chunks=chunks)


def query_stream(
    user_message: str,
    history: list[dict[str, str]],
    *,
    n_results: int = 10,
    category: str | None = None,
) -> tuple[list[RetrievedChunk], Iterator[str]]:
    """Streaming RAG pipeline: retrieve → augment → stream generate.

    Returns
    -------
    (chunks, token_iterator) — the retrieved chunks (for citation extraction
    after streaming completes) and an iterator that yields content deltas.
    """
    import json
    from collections.abc import Iterator

    if category:
        chunks = vector_search(query=user_message, n_results=n_results, category=category)
    else:
        chunks = _retrieve_context_chunks(user_message, n_results)
    logger.info("RAG stream: retrieved %d chunks for query: %.80s…", len(chunks), user_message)

    context_block = _build_context_block(chunks)
    system_prompt = RAG_SYSTEM_PROMPT.format(context_block=context_block)

    from app.llm import _get_api_key, OPENROUTER_URL, MODEL
    import httpx

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    api_key = _get_api_key()
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 3072,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # "HTTP-Referer": "https://consilium.kz",
        # "X-Title": "Consilium",
    }

    def _iter_tokens() -> Iterator[str]:
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", OPENROUTER_URL, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content

    return chunks, _iter_tokens()


def extract_citations(answer: str, chunks: list[RetrievedChunk]) -> list[RAGCitation]:
    """Extract citation references from a completed answer text."""
    cited_nums = set(int(m) for m in re.findall(r"\[(\d+)\]", answer))
    citations: list[RAGCitation] = []
    for num in sorted(cited_nums):
        idx = num - 1
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            citations.append(
                RAGCitation(
                    ref_num=num,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    category=chunk.category,
                    page=chunk.page,
                    excerpt=chunk.text,
                )
            )
    return citations
