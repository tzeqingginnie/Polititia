#!/usr/bin/env python3
"""Build a compact data bundle for the Assemblee dashboard."""

from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyze_project_ngrams import CONTENT_EXCLUDE_TERMS  # noqa: E402

OUTPUT_DIR = ROOT / "dashboard" / "data"

ANALYSIS_DIR = ROOT / "analysis_outputs" / "plain_project_content_stable"
SUMMARY_PATH = ANALYSIS_DIR / "analysis_summary.json"
SPEAKER_ACTIVITY_PATH = ANALYSIS_DIR / "speaker_activity.csv"
PARTY_ACTIVITY_PATH = ANALYSIS_DIR / "party_activity.csv"
GLOBAL_NGRAMS_PATH = ANALYSIS_DIR / "global_common_ngrams.csv"
PARTY_COMMON_PATH = ANALYSIS_DIR / "party_common_ngrams.csv"
PARTY_DISTINCTIVE_PATH = ANALYSIS_DIR / "party_distinctive_ngrams.csv"
SPEAKER_TFIDF_PATH = ANALYSIS_DIR / "speaker_tfidf_ngrams.csv"

NGRAM_SIZES = (1, 2, 3, 4)
MARKER_NGRAM_SIZES = (2, 3, 4, 5)
TOP_CONTENT = 14
TOP_MARKERS = 14

PARTY_CONFIG = [
    {"id": "LFI_NFP", "label": "LFI / NFP", "family": "Left", "color": "#c43b58"},
    {"id": "GDR", "label": "GDR", "family": "Left", "color": "#8e4bb5"},
    {"id": "EcoS", "label": "EcoS", "family": "Green", "color": "#2d9b68"},
    {"id": "SOC", "label": "SOC", "family": "Left", "color": "#e05a87"},
    {"id": "LIOT", "label": "LIOT", "family": "Independent", "color": "#4f8ec8"},
    {"id": "Dem", "label": "Dem", "family": "Center", "color": "#e4a72c"},
    {"id": "EPR", "label": "EPR", "family": "Center", "color": "#2e75d4"},
    {"id": "HOR", "label": "HOR", "family": "Center-right", "color": "#42a9b8"},
    {"id": "DR", "label": "DR", "family": "Right", "color": "#3156a3"},
    {"id": "UDR", "label": "UDR", "family": "Right", "color": "#6f5b9e"},
    {"id": "RN", "label": "RN", "family": "Far right", "color": "#26324d"},
    {"id": "UNLABELED", "label": "Unlabeled", "family": "Unknown", "color": "#9aa1aa"},
]

FRENCH_STOPWORDS = {
    "a",
    "afin",
    "ai",
    "aie",
    "aient",
    "ait",
    "alors",
    "apres",
    "après",
    "as",
    "au",
    "aucun",
    "aussi",
    "autre",
    "aux",
    "avaient",
    "avais",
    "avait",
    "avant",
    "avec",
    "avez",
    "avons",
    "ayant",
    "bon",
    "car",
    "ce",
    "cela",
    "ces",
    "cet",
    "cette",
    "ceux",
    "chaque",
    "ci",
    "comme",
    "comment",
    "dans",
    "de",
    "des",
    "du",
    "donc",
    "dont",
    "elle",
    "elles",
    "en",
    "encore",
    "entre",
    "est",
    "et",
    "etaient",
    "etais",
    "etait",
    "ete",
    "être",
    "eu",
    "eux",
    "fait",
    "faites",
    "fois",
    "font",
    "hors",
    "ici",
    "il",
    "ils",
    "je",
    "juste",
    "l",
    "la",
    "le",
    "les",
    "leur",
    "leurs",
    "lors",
    "lui",
    "ma",
    "mais",
    "me",
    "meme",
    "même",
    "mes",
    "moins",
    "mon",
    "ne",
    "ni",
    "nos",
    "notre",
    "nous",
    "on",
    "ont",
    "ou",
    "où",
    "par",
    "parce",
    "pas",
    "peu",
    "peut",
    "plus",
    "pour",
    "pourquoi",
    "qu",
    "quand",
    "que",
    "quel",
    "quelle",
    "quelles",
    "quels",
    "qui",
    "quoi",
    "sa",
    "sans",
    "se",
    "sera",
    "serait",
    "seront",
    "ses",
    "si",
    "sont",
    "sous",
    "soyez",
    "sur",
    "ta",
    "te",
    "tel",
    "telle",
    "tes",
    "toi",
    "ton",
    "tous",
    "tout",
    "toute",
    "toutes",
    "tres",
    "très",
    "tu",
    "un",
    "une",
    "va",
    "vers",
    "vos",
    "votre",
    "vous",
    "y",
}

ADDRESS_TERMS = {
    "cher",
    "chers",
    "chere",
    "chère",
    "collegue",
    "collègues",
    "collègue",
    "madame",
    "mesdames",
    "ministre",
    "monsieur",
    "president",
    "président",
    "presidente",
    "présidente",
    "rapporteur",
    "rapporteure",
}
PROCEDURE_TERMS = {
    "amendement",
    "amendements",
    "article",
    "avis",
    "commission",
    "gouvernement",
    "loi",
    "projet",
    "rapport",
    "scrutin",
    "seance",
    "séance",
    "texte",
}
STANCE_TERMS = {
    "crois",
    "demande",
    "demandons",
    "devons",
    "doit",
    "faut",
    "pense",
    "propose",
    "proposons",
    "refuse",
    "souhaite",
    "voter",
    "votons",
}
NEGATION_TERMS = {"aucun", "jamais", "ne", "ni", "non", "pas", "personne", "plus", "rien"}
PRONOUN_TERMS = {"je", "j", "nous", "notre", "nos", "vous", "votre", "vos"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def int_value(value: str | int | None) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def float_value(value: str | float | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def slugify(value: str) -> str:
    simple = strip_accents(value).lower()
    simple = re.sub(r"[^a-z0-9]+", "-", simple).strip("-")
    return simple or "unknown"


def ngrams(tokens: list[str], size: int) -> list[str]:
    if len(tokens) < size:
        return []
    return [" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]


def content_tokens(line: str) -> list[str]:
    tokens: list[str] = []
    for raw in line.split():
        token = raw.strip().lower()
        if len(token) <= 1:
            continue
        if token in CONTENT_EXCLUDE_TERMS:
            continue
        if not any(char.isalpha() for char in token):
            continue
        tokens.append(token)
    return tokens


def classify_marker(tokens: list[str]) -> str | None:
    token_set = set(tokens)
    if token_set & ADDRESS_TERMS:
        return "address"
    if token_set & PROCEDURE_TERMS:
        return "procedure"
    if token_set & STANCE_TERMS:
        return "stance"
    if token_set & NEGATION_TERMS:
        return "negation"
    if token_set & PRONOUN_TERMS:
        return "pronoun"
    return None


def rank_counter(
    counts: Counter[str],
    totals: int,
    speech_counts: Counter[str] | None = None,
    total_speeches: int = 0,
    top_k: int = 10,
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for rank, (ngram, count) in enumerate(counts.most_common(top_k), start=1):
        row: dict[str, int | float | str] = {
            "rank": rank,
            "ngram": ngram,
            "count": count,
            "frequency": count / totals if totals else 0,
        }
        if speech_counts is not None:
            row["speechCount"] = speech_counts[ngram]
            row["speechFrequency"] = speech_counts[ngram] / total_speeches if total_speeches else 0
        rows.append(row)
    return rows


def build_politician_records() -> tuple[list[dict[str, object]], dict[str, dict[str, object]], dict[str, str]]:
    grouped: dict[str, dict[str, object]] = {}
    slug_to_politician: dict[str, str] = {}

    for row in read_csv(SPEAKER_ACTIVITY_PATH):
        name = row["speaker"]
        party = row["party"]
        politician_id = f"{slugify(name)}--{slugify(party)}"
        source_path = ROOT / row["source_path"]

        if politician_id not in grouped:
            grouped[politician_id] = {
                "id": politician_id,
                "name": name,
                "party": party,
                "partySource": set(),
                "speechCount": 0,
                "surfaceTokenCount": 0,
                "sourceSlugs": [],
                "sourcePaths": [],
            }

        record = grouped[politician_id]
        record["partySource"].add(row["party_source"])  # type: ignore[index,union-attr]
        record["speechCount"] = int(record["speechCount"]) + int_value(row["speech_count"])
        record["surfaceTokenCount"] = int(record["surfaceTokenCount"]) + int_value(row["surface_token_count"])
        record["sourceSlugs"].append(row["speaker_slug"])  # type: ignore[index,union-attr]
        record["sourcePaths"].append(source_path)  # type: ignore[index,union-attr]
        slug_to_politician[row["speaker_slug"]] = politician_id

    politicians: list[dict[str, object]] = []
    for record in grouped.values():
        source_flags = sorted(record["partySource"])  # type: ignore[arg-type]
        record["partySource"] = " + ".join(source_flags)
        record["sourceSlugs"] = sorted(record["sourceSlugs"])  # type: ignore[arg-type]
        record["sourcePathCount"] = len(record["sourcePaths"])  # type: ignore[arg-type]
        public_record = {
            key: value
            for key, value in record.items()
            if key != "sourcePaths"
        }
        politicians.append(public_record)

    politicians.sort(
        key=lambda item: (
            next((i for i, party in enumerate(PARTY_CONFIG) if party["id"] == item["party"]), 999),
            -int(item["surfaceTokenCount"]),
            str(item["name"]),
        )
    )
    return politicians, grouped, slug_to_politician


def build_politician_phrases(grouped: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}

    for politician_id, record in grouped.items():
        source_paths = record["sourcePaths"]  # type: ignore[index]
        content_counts: dict[int, Counter[str]] = {size: Counter() for size in NGRAM_SIZES}
        content_speech_counts: dict[int, Counter[str]] = {size: Counter() for size in NGRAM_SIZES}
        content_totals: Counter[int] = Counter()
        marker_counts: dict[str, Counter[str]] = defaultdict(Counter)
        marker_speech_counts: dict[str, Counter[str]] = defaultdict(Counter)
        marker_totals: Counter[str] = Counter()
        total_speeches = 0

        for source_path in source_paths:
            path = Path(source_path)
            if not path.exists():
                continue

            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                total_speeches += 1

                c_tokens = content_tokens(line)
                for size in NGRAM_SIZES:
                    grams = ngrams(c_tokens, size)
                    content_counts[size].update(grams)
                    content_speech_counts[size].update(set(grams))
                    content_totals[size] += len(grams)

                s_tokens = line.split()
                seen_markers: dict[str, set[str]] = defaultdict(set)
                for size in MARKER_NGRAM_SIZES:
                    for phrase in ngrams(s_tokens, size):
                        tokens = phrase.split()
                        category = classify_marker(tokens)
                        if category is None:
                            continue
                        marker_counts[category][phrase] += 1
                        marker_totals[category] += 1
                        seen_markers[category].add(phrase)
                for category, phrases in seen_markers.items():
                    marker_speech_counts[category].update(phrases)

        result[politician_id] = {
            "content": {
                str(size): rank_counter(
                    content_counts[size],
                    content_totals[size],
                    content_speech_counts[size],
                    total_speeches,
                    TOP_CONTENT,
                )
                for size in NGRAM_SIZES
            },
            "markers": {
                category: rank_counter(
                    counts,
                    marker_totals[category],
                    marker_speech_counts[category],
                    total_speeches,
                    TOP_MARKERS,
                )
                for category, counts in sorted(marker_counts.items())
            },
        }

    return result


def group_ngram_rows(
    rows: list[dict[str, str]],
    group_key: str | None = None,
    value_keys: tuple[str, ...] = ("frequency",),
) -> dict[str, object]:
    if group_key is None:
        grouped: dict[str, list[dict[str, object]]] = {str(size): [] for size in NGRAM_SIZES}
        for row in rows:
            size = row["n"]
            entry: dict[str, object] = {
                "rank": int_value(row["rank"]),
                "ngram": row["ngram"],
                "count": int_value(row["count"]),
            }
            for key in value_keys:
                if key in row:
                    entry[key] = float_value(row[key])
            grouped.setdefault(size, []).append(entry)
        return grouped

    grouped_by_key: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: {str(size): [] for size in NGRAM_SIZES}
    )
    for row in rows:
        key = row[group_key]
        size = row["n"]
        entry = {
            "rank": int_value(row["rank"]),
            "ngram": row["ngram"],
            "count": int_value(row["count"]),
        }
        for value_key in value_keys:
            if value_key in row:
                entry[value_key] = float_value(row[value_key])
        grouped_by_key[key].setdefault(size, []).append(entry)
    return dict(grouped_by_key)


def build_tfidf_phrases() -> dict[str, dict[str, list[dict[str, object]]]]:
    grouped: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: {str(size): [] for size in NGRAM_SIZES}
    )
    if not SPEAKER_TFIDF_PATH.exists():
        return {}

    for row in read_csv(SPEAKER_TFIDF_PATH):
        rank = int_value(row["rank"])
        if rank > TOP_CONTENT:
            continue
        politician_id = f"{slugify(row['speaker'])}--{slugify(row['party'])}"
        size = row["n"]
        grouped[politician_id].setdefault(size, []).append({
            "rank": rank,
            "ngram": row["ngram"],
            "count": int_value(row["count"]),
            "term_frequency": float_value(row.get("term_frequency")),
            "tf_idf_vs_rest": float_value(row.get("tf_idf_vs_rest")),
            "idf_vs_rest": float_value(row.get("idf_vs_rest")),
            "rest_document_share": float_value(row.get("rest_document_share")),
        })

    return dict(grouped)


def build_party_data(politicians: list[dict[str, object]]) -> list[dict[str, object]]:
    party_activity = {row["party"]: row for row in read_csv(PARTY_ACTIVITY_PATH)}
    politician_counts = Counter(str(item["party"]) for item in politicians)
    party_sources: dict[str, Counter[str]] = defaultdict(Counter)
    for politician in politicians:
        party_sources[str(politician["party"])][str(politician["partySource"])] += 1

    parties: list[dict[str, object]] = []
    for config in PARTY_CONFIG:
        party_id = config["id"]
        activity = party_activity.get(party_id, {})
        parties.append(
            {
                **config,
                "speakerFileCount": int_value(activity.get("speaker_count")),
                "politicianCount": politician_counts[party_id],
                "speechCount": int_value(activity.get("speech_count")),
                "analysisTokenCount": int_value(activity.get("analysis_token_count")),
                "partySourceBreakdown": dict(party_sources[party_id]),
            }
        )
    return parties


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    politicians, grouped, _slug_to_politician = build_politician_records()
    phrases_by_politician = build_politician_phrases(grouped)
    tfidf_by_politician = build_tfidf_phrases()
    for politician_id, phrases in phrases_by_politician.items():
        phrases["tfidf"] = tfidf_by_politician.get(
            politician_id,
            {str(size): [] for size in NGRAM_SIZES},
        )
    parties = build_party_data(politicians)

    party_common = group_ngram_rows(
        read_csv(PARTY_COMMON_PATH),
        group_key="party",
        value_keys=("frequency",),
    )
    party_distinctive = group_ngram_rows(
        read_csv(PARTY_DISTINCTIVE_PATH),
        group_key="party",
        value_keys=("party_frequency", "log_odds_vs_rest"),
    )
    global_common = group_ngram_rows(
        read_csv(GLOBAL_NGRAMS_PATH),
        value_keys=("frequency",),
    )

    party_phrases: dict[str, dict[str, object]] = {}
    for party in [config["id"] for config in PARTY_CONFIG]:
        party_phrases[party] = {
            "common": party_common.get(party, {str(size): [] for size in NGRAM_SIZES}),
            "distinctive": party_distinctive.get(party, {str(size): [] for size in NGRAM_SIZES}),
        }

    data = {
        "meta": {
            "speakerFiles": summary.get("speaker_files"),
            "politicians": len(politicians),
            "partiesDetected": summary.get("parties_detected", []),
            "eligibleParties": summary.get("eligible_parties", []),
            "tokenMode": summary.get("token_mode"),
            "ngramSizes": summary.get("ngram_sizes", list(NGRAM_SIZES)),
            "minTfidfCount": summary.get("min_tfidf_count"),
            "totalSpeeches": summary.get("total_speeches"),
            "totalSurfaceTokens": summary.get("total_surface_tokens"),
            "totalAnalysisTokens": summary.get("total_analysis_tokens"),
            "partySourceSpeechCounts": summary.get("party_source_speech_counts", {}),
            "sources": {
                "summary": str(SUMMARY_PATH.relative_to(ROOT)),
                "speakerActivity": str(SPEAKER_ACTIVITY_PATH.relative_to(ROOT)),
                "partyActivity": str(PARTY_ACTIVITY_PATH.relative_to(ROOT)),
                "globalCommonNgrams": str(GLOBAL_NGRAMS_PATH.relative_to(ROOT)),
                "partyCommonNgrams": str(PARTY_COMMON_PATH.relative_to(ROOT)),
                "partyDistinctiveNgrams": str(PARTY_DISTINCTIVE_PATH.relative_to(ROOT)),
                "speakerTfidfNgrams": str(SPEAKER_TFIDF_PATH.relative_to(ROOT)),
            },
        },
        "partyOrder": [config["id"] for config in PARTY_CONFIG],
        "parties": parties,
        "politicians": politicians,
        "phrasesByPolitician": phrases_by_politician,
        "partyPhrases": party_phrases,
        "globalPhrases": global_common,
    }

    json_path = OUTPUT_DIR / "dashboard-data.json"
    js_path = OUTPUT_DIR / "dashboard-data.js"
    json_text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    json_path.write_text(json_text + "\n", encoding="utf-8")
    js_path.write_text("window.DASHBOARD_DATA = " + json_text + ";\n", encoding="utf-8")
    print(f"Wrote {json_path.relative_to(ROOT)} and {js_path.relative_to(ROOT)}")
    print(f"Politicians: {len(politicians)}")


if __name__ == "__main__":
    main()
