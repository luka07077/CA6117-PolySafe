"""
Build the drug-safety evidence vector store from data/evidence/ (idempotent).

    python scripts/build_rag.py            # add new/changed documents
    python scripts/build_rag.py --rebuild  # wipe data/chroma_db and re-index everything

Needs the embedding model from configs/agent_config.yaml (llm.embedding_model; "local" needs no key).
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vector_stores import CHROMA_DIR, EVIDENCE_DIR, ingest_document, search  # noqa: E402

PROBES = ["warfarin ibuprofen bleeding risk", "simvastatin clarithromycin rhabdomyolysis",
          "opioid benzodiazepine respiratory depression", "lisinopril spironolactone hyperkalaemia"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    if args.rebuild and os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    files = sorted(f for f in os.listdir(EVIDENCE_DIR) if f.endswith((".md", ".txt")))
    total = sum(ingest_document(os.path.join(EVIDENCE_DIR, f)) for f in files)
    print(f"{len(files)} documents, {total} new chunks -> {CHROMA_DIR}")
    for q in PROBES:
        hits = search(q, k=2)
        print(f"\nQ: {q}\n" + "\n".join(
            f"  [{h.metadata['source']} · {h.metadata.get('section', '')}] {h.page_content[:90]!r}" for h in hits))


if __name__ == "__main__":
    main()
