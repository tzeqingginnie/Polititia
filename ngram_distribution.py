"""
Build per-speaker n-gram distributions from extracted speech corpora.

Expected input
--------------
Use the normalized files produced by extract_speeches.py:

    <output_dir>/by_speaker/<speaker_slug>.txt

Each line is treated as one speech, so n-grams never cross speech boundaries.

Examples
--------
Auto-discover all by_speaker directories under extracted_texts:

    uv run python ngram_distribution.py

Use one or more explicit by_speaker directories:

    uv run python ngram_distribution.py extracted_texts/2022/by_speaker extracted_texts/2023/by_speaker

Count only bigrams through four-grams:

    uv run python ngram_distribution.py --ngram-sizes 2 3 4 --top-k 100

Generate both surface and lemma n-grams with spaCy:

    uv run python ngram_distribution.py --token-mode both --spacy-model fr_core_news_md
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


DEFAULT_ROOT = Path("extracted_texts")
DEFAULT_OUTPUT = Path("ngram_distribution.csv")
DEFAULT_NGRAM_SIZES = (1, 2, 3, 4)
DEFAULT_SPACY_MODEL = "fr_core_news_md"


@dataclass
class SpeakerStats:
    counts: dict[int, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    speech_counts: dict[int, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    total_ngrams: Counter[int] = field(default_factory=Counter)
    total_speeches: int = 0
    source_files: set[Path] = field(default_factory=set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ranked per-speaker n-gram distributions."
    )
    parser.add_argument(
        "speaker_dirs",
        nargs="*",
        type=Path,
        help="Optional by_speaker directories. If omitted, directories are discovered under --root.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Root searched for by_speaker directories. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--ngram-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_NGRAM_SIZES),
        help="N-gram sizes to count. Default: 1 2 3 4",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Rows to keep per speaker and n. Use 0 to keep all. Default: 50",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum raw count required for an n-gram to be emitted. Default: 1",
    )
    parser.add_argument(
        "--min-speeches",
        type=int,
        default=1,
        help="Minimum number of speeches a speaker must have. Default: 1",
    )
    parser.add_argument(
        "--token-mode",
        choices=("surface", "lemma", "both"),
        default="surface",
        help="Token representation to count. Default: surface",
    )
    parser.add_argument(
        "--spacy-model",
        default=DEFAULT_SPACY_MODEL,
        help=f"spaCy model used for lemma mode. Default: {DEFAULT_SPACY_MODEL}",
    )
    parser.add_argument(
        "--drop-stopwords",
        action="store_true",
        help="Drop spaCy stopwords in lemma mode.",
    )
    parser.add_argument(
        "--keep-pos",
        nargs="+",
        default=None,
        help="Optional spaCy POS tags to keep in lemma mode, e.g. NOUN VERB ADJ PROPN.",
    )
    return parser.parse_args()


def validate_ngram_sizes(sizes: list[int]) -> tuple[int, ...]:
    invalid = [size for size in sizes if size < 1]
    if invalid:
        raise SystemExit(f"N-gram sizes must be positive integers: {invalid}")
    return tuple(sorted(set(sizes)))


def discover_speaker_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise SystemExit(f"Input root does not exist: {root}")
    dirs = sorted(path for path in root.rglob("by_speaker") if path.is_dir())
    if root.name == "by_speaker" and root.is_dir():
        dirs.insert(0, root)
    return sorted(set(dirs))


def resolve_speaker_dirs(explicit_dirs: list[Path], root: Path) -> list[Path]:
    dirs = explicit_dirs or discover_speaker_dirs(root)
    missing = [path for path in dirs if not path.is_dir()]
    if missing:
        raise SystemExit(
            "Speaker corpus directories not found:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    if not dirs:
        raise SystemExit(
            f"No by_speaker directories found under {root}. "
            "Run extract_speeches.py first, then rerun this script."
        )
    return dirs


def iter_speaker_files(speaker_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for speaker_dir in speaker_dirs:
        files.extend(sorted(speaker_dir.glob("*.txt")))
    if not files:
        raise SystemExit(
            "No speaker .txt files found in:\n"
            + "\n".join(f"  - {path}" for path in speaker_dirs)
        )
    return files


def ngrams(tokens: list[str], n: int) -> list[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def surface_tokens(line: str) -> list[str]:
    return line.split()


def load_spacy_model(model_name: str):
    try:
        import spacy
    except ImportError as exc:
        raise SystemExit(
            "spaCy is required for --token-mode lemma/both. "
            "Install it with: uv add spacy"
        ) from exc

    try:
        return spacy.load(model_name, disable=["ner"])
    except OSError as exc:
        raise SystemExit(
            f"spaCy model not found: {model_name}\n"
            f"Install it with: uv run python -m spacy download {model_name}"
        ) from exc


def build_lemma_tokenizer(
    model_name: str,
    drop_stopwords: bool,
    keep_pos: list[str] | None,
) -> Callable[[str], list[str]]:
    nlp = load_spacy_model(model_name)
    keep_pos_set = {pos.upper() for pos in keep_pos} if keep_pos else None

    def tokenize(line: str) -> list[str]:
        tokens: list[str] = []
        for token in nlp(line):
            if token.is_space or token.is_punct or token.like_num:
                continue
            if drop_stopwords and token.is_stop:
                continue
            if keep_pos_set is not None and token.pos_ not in keep_pos_set:
                continue

            lemma = token.lemma_.strip().lower() or token.text.strip().lower()
            if lemma == "_":
                lemma = token.text.strip().lower()
            if not any(char.isalpha() for char in lemma):
                continue
            tokens.append(lemma)
        return tokens

    return tokenize


def build_tokenizers(args: argparse.Namespace) -> dict[str, Callable[[str], list[str]]]:
    tokenizers: dict[str, Callable[[str], list[str]]] = {}
    if args.token_mode in {"surface", "both"}:
        tokenizers["surface"] = surface_tokens
    if args.token_mode in {"lemma", "both"}:
        tokenizers["lemma"] = build_lemma_tokenizer(
            model_name=args.spacy_model,
            drop_stopwords=args.drop_stopwords,
            keep_pos=args.keep_pos,
        )
    return tokenizers


def update_speaker_stats(
    stats: SpeakerStats,
    lines: list[str],
    ngram_sizes: tuple[int, ...],
    source_file: Path,
    tokenize: Callable[[str], list[str]],
) -> None:
    stats.source_files.add(source_file)

    for line in lines:
        tokens = tokenize(line)
        if not tokens:
            continue

        stats.total_speeches += 1
        for size in ngram_sizes:
            grams = ngrams(tokens, size)
            if not grams:
                continue
            stats.counts[size].update(grams)
            stats.speech_counts[size].update(set(grams))
            stats.total_ngrams[size] += len(grams)


def collect_stats(
    speaker_files: list[Path],
    ngram_sizes: tuple[int, ...],
    tokenizers: dict[str, Callable[[str], list[str]]],
) -> dict[str, dict[str, SpeakerStats]]:
    by_mode: dict[str, dict[str, SpeakerStats]] = {
        mode: defaultdict(SpeakerStats) for mode in tokenizers
    }

    for path in speaker_files:
        speaker = path.stem
        lines = path.read_text(encoding="utf-8").splitlines()
        for mode, tokenize in tokenizers.items():
            update_speaker_stats(
                by_mode[mode][speaker],
                lines,
                ngram_sizes,
                path,
                tokenize,
            )

    return {mode: dict(stats) for mode, stats in by_mode.items()}


def iter_rows(
    by_mode: dict[str, dict[str, SpeakerStats]],
    ngram_sizes: tuple[int, ...],
    top_k: int,
    min_count: int,
    min_speeches: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for token_mode in sorted(by_mode):
        by_speaker = by_mode[token_mode]
        for speaker in sorted(by_speaker):
            stats = by_speaker[speaker]
            if stats.total_speeches < min_speeches:
                continue

            for size in ngram_sizes:
                total_for_size = stats.total_ngrams[size]
                if total_for_size == 0:
                    continue

                ranked = sorted(
                    (
                        (ngram, count)
                        for ngram, count in stats.counts[size].items()
                        if count >= min_count
                    ),
                    key=lambda item: (-item[1], item[0]),
                )
                if top_k > 0:
                    ranked = ranked[:top_k]

                for rank, (ngram, count) in enumerate(ranked, start=1):
                    speech_count = stats.speech_counts[size][ngram]
                    rows.append({
                        "token_mode": token_mode,
                        "speaker": speaker,
                        "n": size,
                        "rank": rank,
                        "ngram": ngram,
                        "count": count,
                        "frequency": count / total_for_size,
                        "total_ngrams_for_speaker_n": total_for_size,
                        "speech_count": speech_count,
                        "speech_frequency": speech_count / stats.total_speeches,
                        "total_speeches_for_speaker": stats.total_speeches,
                        "source_file_count": len(stats.source_files),
                    })

    return rows


def write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "token_mode",
        "speaker",
        "n",
        "rank",
        "ngram",
        "count",
        "frequency",
        "total_ngrams_for_speaker_n",
        "speech_count",
        "speech_frequency",
        "total_speeches_for_speaker",
        "source_file_count",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    ngram_sizes = validate_ngram_sizes(args.ngram_sizes)

    if args.top_k < 0:
        raise SystemExit("--top-k must be >= 0")
    if args.min_count < 1:
        raise SystemExit("--min-count must be >= 1")
    if args.min_speeches < 1:
        raise SystemExit("--min-speeches must be >= 1")

    speaker_dirs = resolve_speaker_dirs(args.speaker_dirs, args.root)
    speaker_files = iter_speaker_files(speaker_dirs)
    tokenizers = build_tokenizers(args)
    by_mode = collect_stats(speaker_files, ngram_sizes, tokenizers)
    rows = iter_rows(
        by_mode=by_mode,
        ngram_sizes=ngram_sizes,
        top_k=args.top_k,
        min_count=args.min_count,
        min_speeches=args.min_speeches,
    )

    if not rows:
        raise SystemExit("No n-gram rows matched the selected filters.")

    write_csv(rows, args.out)
    speaker_count = len({speaker for stats in by_mode.values() for speaker in stats})
    print(
        f"Wrote {len(rows)} rows for {speaker_count} speaker(s), "
        f"{', '.join(sorted(tokenizers))} token mode(s), "
        f"from {len(speaker_files)} file(s) to {args.out}"
    )


if __name__ == "__main__":
    main()
