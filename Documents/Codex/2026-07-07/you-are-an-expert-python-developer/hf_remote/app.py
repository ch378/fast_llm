"""Hugging Face Space demo for evaluating extracted banking CSV files."""

from __future__ import annotations

import csv
import io
import pickle
import re
from collections import Counter
from pathlib import Path

import gradio as gr
import xgboost as xgb


MASTERCard_COMPACT = (
    "provider", "billing_event_id", "service_id", "service_name",
    "tier_event_id", "rate_type", "currency", "acquirer", "issuer",
    "rate_tier", "tier_starting_value", "tier_ending_value", "rate",
    "frequency", "unit", "page",
)
MASTERCARD_LEGACY = (
    "provider", "service_id", "service_name", "billing_event_id",
    "billing_event_name", "tier_event_id", "rate_type", "currency",
    "acquirer", "issuer", "rate_tier", "tier_starting_value",
    "tier_ending_value", "rate", "frequency", "unit", "page",
    "confidence", "source_chunk", "source_text",
)
MODEL_PATH = Path(__file__).with_name("model.pkl")
MODEL = pickle.loads(MODEL_PATH.read_bytes()) if MODEL_PATH.is_file() else None


def _clean(value: str | None) -> str:
    return " ".join((value or "").split())


def _contains_number(value: str) -> bool:
    return bool(re.search(r"[-+]?\d+(?:[.,]\d+)?", value))


def _stable_code(value: str, modulus: int = 1000003) -> float:
    result = 0
    for character in value:
        result = (result * 31 + ord(character)) % modulus
    return result / modulus


def _numeric_rate(value: str) -> float:
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", value or "")
    return float(match.group(0).replace(",", ".")) if match else 0.0


def _model_features(row: dict[str, str]) -> list[float]:
    rate = _clean(row.get("rate"))
    page_match = re.match(r"(\d+)", _clean(row.get("page")))
    return [
        _stable_code(_clean(row.get("provider"))),
        _stable_code(_clean(row.get("service_id"))),
        _stable_code(_clean(row.get("billing_event_id"))),
        _stable_code(_clean(row.get("tier_event_id"))),
        float(bool(_clean(row.get("service_name")))),
        float(bool(_clean(row.get("currency")))),
        float(bool(_clean(row.get("acquirer")))),
        float(bool(_clean(row.get("issuer")))),
        float(bool(_clean(row.get("rate_tier")))),
        float(bool(_clean(row.get("tier_ending_value")))),
        float(bool(rate)),
        float(rate.casefold() == "variable"),
        _numeric_rate(rate),
        float(page_match.group(1)) if page_match else 0.0,
    ]


def evaluate_csv(file: object, provider: str) -> tuple[str, list[list[str]]]:
    if file is None:
        return "Upload a CSV file.", []

    path = getattr(file, "name", file)
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = tuple(reader.fieldnames or ())
        rows = list(reader)

    if provider == "mastercard":
        expected = MASTERCARD_COMPACT if "billing_event_id" in headers and "billing_event_name" not in headers else MASTERCARD_LEGACY
        required = ("provider", "billing_event_id", "page")
    else:
        expected = headers
        required = ("provider",)

    missing = [column for column in expected if column not in headers]
    duplicate_keys = Counter(
        tuple(_clean(row.get(column)) for column in headers) for row in rows
    )
    duplicate_rows = sum(count - 1 for count in duplicate_keys.values() if count > 1)
    issues: list[list[str]] = []
    valid = 0
    model_needs_review = 0

    for number, row in enumerate(rows, start=2):
        errors: list[str] = []
        for column in required:
            if not _clean(row.get(column)):
                errors.append(f"missing:{column}")
        if _clean(row.get("provider")).casefold() != provider:
            errors.append("provider_mismatch")
        page = _clean(row.get("page") or row.get("source_page"))
        if page and not re.fullmatch(r"\d+(?:\s*-\s*\d+)?", page):
            errors.append("invalid_page")
        rate = _clean(row.get("rate"))
        if provider == "mastercard" and rate and rate.casefold() != "variable" and not _contains_number(rate):
            errors.append("invalid_rate")
        status = "valid" if not errors else "needs_review"
        valid += status == "valid"
        probability = None
        if MODEL is not None and provider == "mastercard":
            probability = float(MODEL.predict(xgb.DMatrix([_model_features(row)]))[0])
            model_needs_review += probability < 0.5
        if errors:
            model_text = f"model_probability={probability:.3f}" if probability is not None else "model=unavailable"
            issues.append([str(number), status, f"{'; '.join(errors)}; {model_text}"])

    total = len(rows)
    structure = 1.0 if not missing else (len(expected) - len(missing)) / len(expected)
    row_score = valid / total if total else 0.0
    duplicate_score = (total - duplicate_rows) / total if total else 0.0
    score = (0.4 * structure) + (0.4 * row_score) + (0.2 * duplicate_score)
    status = "valid" if score >= 0.9 else "needs_review"
    report = (
        f"### {status}\n"
        f"Rows: **{total}**\n\n"
        f"Valid rows: **{valid}**\n\n"
        f"Needs review: **{total - valid}**\n\n"
        f"Unique columns: **{len(headers)}**\n\n"
        f"Duplicate rows: **{duplicate_rows}**\n\n"
        f"Quality score: **{score:.4f}**\n\n"
        f"Missing columns: `{', '.join(missing) or 'none'}`\n\n"
        f"XGBoost model: **{'loaded' if MODEL is not None else 'unavailable'}**\n\n"
        f"XGBoost rows needing review: **{model_needs_review}**"
    )
    return report, issues


with gr.Blocks(title="Banking CSV Evaluation") as demo:
    gr.Markdown(
        "# Banking CSV Evaluation\n"
        "Upload an extracted CSV and inspect its structure, missing values, rates, pages, and duplicates."
    )
    with gr.Row():
        csv_file = gr.File(label="Extracted CSV", file_types=[".csv"], type="filepath")
        provider = gr.Dropdown(
            choices=["mastercard", "cb", "visa", "other_banks"],
            value="mastercard",
            label="Provider",
        )
    run = gr.Button("Evaluate", variant="primary")
    report = gr.Markdown()
    issues = gr.Dataframe(
        headers=["row", "status", "errors"],
        label="Rows requiring review",
        datatype=["str", "str", "str"],
    )
    run.click(evaluate_csv, inputs=[csv_file, provider], outputs=[report, issues])


if __name__ == "__main__":
    demo.launch()
