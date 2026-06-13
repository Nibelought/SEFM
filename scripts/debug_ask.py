"""Print raw LLM output, retrieved chunks, and the validation verdict."""

from __future__ import annotations

import sys

from app.factory import build_app_service
from app.generation.citation_guard import REFUSAL, validate_answer
from app.generation.prompts import build_prompt


def main() -> int:
    question = " ".join(sys.argv[1:]) or "What amperage range does the JTN600 fuse holder support?"
    svc = build_app_service(with_llm=True)
    if svc.llm is None:
        print("No LLM configured")
        return 1

    hits = svc.search(question)
    chunks = [h.chunk for h in hits]
    print(f"=== Retrieved {len(chunks)} chunks ===")
    for i, c in enumerate(chunks, 1):
        snippet = " ".join(c.text.split())[:200]
        print(f"[{i}] {c.source} p.{c.page} ({c.kind}): {snippet}")

    prompt = build_prompt(question, chunks)
    print("\n=== Prompt (last 400 chars) ===")
    print(prompt[-400:])

    print("\n=== Raw LLM output ===")
    raw = svc.llm.generate(prompt, max_tokens=300, temperature=0.1)
    print(raw)

    print("\n=== Validation verdict ===")
    validated = validate_answer(raw, chunks)
    if validated == REFUSAL and raw.strip() != REFUSAL:
        print("REJECTED by citation guard")
    elif validated == REFUSAL:
        print("Model itself replied 'Not found'")
    else:
        print("ACCEPTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
