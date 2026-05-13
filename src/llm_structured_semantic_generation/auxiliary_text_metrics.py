from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import yaml


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.:/-]+|[^\sA-Za-z0-9_.:/-]")


def normalize_yaml_for_text_metrics(yaml_text: str) -> str:
    """Render parseable YAML through a stable dumper before weak text metrics."""

    try:
        documents = tuple(yaml.safe_load_all(yaml_text))
    except yaml.YAMLError:
        return yaml_text.strip()
    return yaml.safe_dump_all(
        documents,
        allow_unicode=True,
        sort_keys=True,
        explicit_start=len(documents) > 1,
    ).strip()


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def _ngram_counts(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    if order <= 0 or len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def bleu_score(reference_text: str, prediction_text: str, *, max_order: int = 4) -> float:
    reference_tokens = _tokenize(reference_text)
    prediction_tokens = _tokenize(prediction_text)
    if not reference_tokens and not prediction_tokens:
        return 1.0
    if not reference_tokens or not prediction_tokens:
        return 0.0

    precisions: list[float] = []
    for order in range(1, max_order + 1):
        prediction_counts = _ngram_counts(prediction_tokens, order)
        reference_counts = _ngram_counts(reference_tokens, order)
        if not prediction_counts:
            precisions.append(1.0)
            continue
        overlap = sum(
            min(count, reference_counts[ngram])
            for ngram, count in prediction_counts.items()
        )
        # Add-one smoothing keeps short, partially correct YAML from collapsing
        # to zero only because one higher-order n-gram is absent.
        precisions.append((overlap + 1.0) / (sum(prediction_counts.values()) + 1.0))

    brevity_penalty = 1.0
    if len(prediction_tokens) < len(reference_tokens):
        brevity_penalty = math.exp(1.0 - (len(reference_tokens) / len(prediction_tokens)))
    return brevity_penalty * math.exp(sum(math.log(value) for value in precisions) / max_order)


def _f1(reference_count: int, prediction_count: int, overlap: int) -> float:
    if reference_count == 0 and prediction_count == 0:
        return 1.0
    if reference_count == 0 or prediction_count == 0 or overlap == 0:
        return 0.0
    precision = overlap / prediction_count
    recall = overlap / reference_count
    return 2 * precision * recall / (precision + recall)


def rouge_n_f1(reference_text: str, prediction_text: str, *, order: int) -> float:
    reference_tokens = _tokenize(reference_text)
    prediction_tokens = _tokenize(prediction_text)
    reference_counts = _ngram_counts(reference_tokens, order)
    prediction_counts = _ngram_counts(prediction_tokens, order)
    overlap = sum(
        min(count, prediction_counts[ngram])
        for ngram, count in reference_counts.items()
    )
    return _f1(sum(reference_counts.values()), sum(prediction_counts.values()), overlap)


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0] * (len(right) + 1)
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[-1]


def rouge_l_f1(reference_text: str, prediction_text: str) -> float:
    reference_tokens = _tokenize(reference_text)
    prediction_tokens = _tokenize(prediction_text)
    return _f1(
        len(reference_tokens),
        len(prediction_tokens),
        _lcs_length(reference_tokens, prediction_tokens),
    )


def compute_auxiliary_text_metrics(
    reference_yaml: str,
    prediction_yaml: str,
    *,
    perplexity: float | None = None,
) -> dict[str, Any]:
    reference_text = normalize_yaml_for_text_metrics(reference_yaml)
    prediction_text = normalize_yaml_for_text_metrics(prediction_yaml)
    return {
        "bleu_score": bleu_score(reference_text, prediction_text),
        "rouge1_f1": rouge_n_f1(reference_text, prediction_text, order=1),
        "rouge2_f1": rouge_n_f1(reference_text, prediction_text, order=2),
        "rougeL_f1": rouge_l_f1(reference_text, prediction_text),
        "perplexity": perplexity,
        "perplexity_available": perplexity is not None,
    }
