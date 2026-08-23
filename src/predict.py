import argparse
import json
import sys
from pathlib import Path

import data_prep as dp
from train import load_fraud_stack

LABELS = {0: "Legitimate", 1: "Fraud"}
BATCH_KEYS = ("transactions", "records", "inputs")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score credit-card transactions with the experiment 06 stack."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["input.json"],
        help="JSON or JSONL file(s): one transaction, a list, or {\"transactions\": [...]}",
    )
    parser.add_argument("-o", "--output", default="output.json")
    parser.add_argument("--models", default=None, help="directory with model.pkl")
    return parser.parse_args()


def load_json_file(path):
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"input file is empty: {path}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line_no, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_no}: {exc}") from exc
        if not rows:
            raise ValueError(f"no JSON objects found in {path}")
        return rows


def payload_rows(payload):
    if isinstance(payload, dict):
        for key in BATCH_KEYS:
            if key in payload:
                rows = payload[key]
                if not isinstance(rows, list):
                    raise ValueError(f'"{key}" must be a list of transactions')
                return rows, False
        return [payload], True
    if isinstance(payload, list):
        return payload, False
    raise ValueError("input must be a transaction object or a list")


def collect_payload(paths):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("input file not found: " + ", ".join(missing))

    rows = []
    single = len(paths) == 1
    for path in paths:
        part, part_single = payload_rows(load_json_file(path))
        rows.extend(part)
        if not part_single:
            single = False
    if not rows:
        raise ValueError("no transactions in input")
    if single and len(rows) == 1:
        return rows[0]
    return rows


def score_frame(stack, frame, single):
    class_ids, proba = stack.predict(frame)
    rows = []
    for index, (cid, p) in enumerate(zip(class_ids.tolist(), proba.tolist())):
        item = {
            "prediction": LABELS[int(cid)],
            "class_id": int(cid),
            "probability": round(float(p), 4),
            "threshold": float(stack.threshold),
            "status": "success",
        }
        if not single:
            item["index"] = index
        rows.append(item)
    if single:
        return rows[0]
    return {"predictions": rows, "count": len(rows), "status": "success"}


def main():
    args = parse_args()
    in_paths = [Path(path) for path in args.inputs]
    out_path = Path(args.output)
    try:
        payload = collect_payload(in_paths)
        frame, single = dp.records_to_frame(payload)
        stack = load_fraud_stack(args.models)
        result = score_frame(stack, frame, single)
    except Exception as exc:
        result = {"status": "error", "message": str(exc)}
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 1

    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
