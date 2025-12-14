from pathlib import Path
import json
import re
from typing import List, Optional


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def lambda_body_to_dsl(body: str) -> str:
    """Translate lambda string to flattened DSL tokens."""
    s = body
    s = re.sub(r"left\s+(\d+)", lambda m: f"l_{m.group(1)}", s)
    s = re.sub(r"right\s+(\d+)", lambda m: f"r_{m.group(1)}", s)
    s = s.replace("2x1", "h").replace("1x2", "v")
    s = re.sub(r"\blambda\b", " ", s)
    s = s.replace("#", " ")
    s = re.sub(r"\$\d+", " ", s)
    s = s.translate(str.maketrans({"(": " ", ")": " "}))
    tokens = [tok for tok in s.split() if tok]
    filtered = [t for t in tokens if t == "h" or t == "v" or re.match(r"[lr]_\d+", t)]
    return " ".join(filtered)


def extract_chunk_lambda(entry: dict, chunk_name: str) -> Optional[str]:
    """Find the lambda string for a chunk within a single entry."""
    dsl_list = entry.get("dsl", [])
    dsl_lambda = entry.get("dsl_lambda", [])
    try:
        idx = dsl_list.index(chunk_name)
    except ValueError:
        return None
    if idx < len(dsl_lambda):
        return dsl_lambda[idx]
    offset_from_end = (len(dsl_list) - 1) - idx
    alt_idx = (len(dsl_lambda) - 1) - offset_from_end
    if 0 <= alt_idx < len(dsl_lambda):
        return dsl_lambda[alt_idx]
    return None


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    programs_dir = repo_root / "data" / "model" / "programs_for_you"
    output_path = repo_root / "data" / "model" / "task" / "chunk.json"

    results = []
    for json_file in sorted(programs_dir.glob("programs_ppt_*.json")):
        ppt = int(json_file.stem.split("_")[-1])
        data = json.loads(json_file.read_text())
        if not data:
            continue

        # Take all chunks observed in this ppt, preserving first-seen order.
        chunks_ordered = unique_preserve_order(
            [c for entry in data for c in entry.get("chunks", [])]
        )

        # Collect candidate lambdas per chunk across all trials, keep the shortest string.
        chunk_lambdas: List[Optional[str]] = []
        for ch in chunks_ordered:
            candidates = []
            for entry in data:
                lam = extract_chunk_lambda(entry, ch)
                if lam is not None:
                    candidates.append(lam)
            if candidates:
                chunk_lambdas.append(min(candidates, key=len))
            else:
                chunk_lambdas.append(None)

        chunk_dsl = [lambda_body_to_dsl(l) if l is not None else None for l in chunk_lambdas]

        # Deduplicate by chunk_dsl, keeping the shorter chunk name when duplicates arise.
        dedup_order = []
        dedup_map = {}
        for name, lam, dsl in zip(chunks_ordered, chunk_lambdas, chunk_dsl):
            if dsl not in dedup_map:
                dedup_map[dsl] = (name, lam)
                dedup_order.append(dsl)
            else:
                prev_name, prev_lam = dedup_map[dsl]
                if len(name) < len(prev_name) or (len(name) == len(prev_name) and name < prev_name):
                    dedup_map[dsl] = (name, lam)
        chunks_ordered = [dedup_map[d][0] for d in dedup_order]
        chunk_lambdas = [dedup_map[d][1] for d in dedup_order]
        chunk_dsl = dedup_order

        results.append(
            {
                "ppt": ppt,
                "chunks": chunks_ordered,
                "chunk_lambdas": chunk_lambdas,
                "chunk_dsl": chunk_dsl,
            }
        )

    results.sort(key=lambda x: x["ppt"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
