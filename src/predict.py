import argparse
import json
import sys
from pathlib import Path

import data_prep as dp
from train import load_fraud_stack

LABELS = {0: "Legitimate", 1: "Fraud"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score credit-card transactions with the experiment 06 stack."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="input.json",
        help="JSON file (one transaction or a list)",
    )
    parser.add_argument("-o", "--output", default="output.json")
    parser.add_argument("--models", default=None, help="directory with model.pkl")
    return parser.parse_args()


def score_frame(stack, frame, single):
    class_ids, proba = stack.predict(frame)
    rows = []
    for cid, p in zip(class_ids.tolist(), proba.tolist()):
        rows.append(
            {
                "prediction": LABELS[int(cid)],
                "class_id": int(cid),
                "probability": round(float(p), 4),
                "threshold": float(stack.threshold),
                "status": "success",
            }
        )
    if single:
        return rows[0]
    return {"predictions": rows, "status": "success"}


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    try:
        if not in_path.exists():
            raise FileNotFoundError(f"input file not found: {in_path}")
        payload = json.loads(in_path.read_text())
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
