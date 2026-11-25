import json
from typing import Any, Tuple, Union


def flip_candidate_label(label: str) -> str:
    return "B" if label == "A" else "A"


def resolve_human_winner(row_data: Any) -> str:
    if "actual_score" in row_data:
        val = str(row_data["actual_score"]).strip().upper()
        if "ASSISTANT A" in val or val in {"A", "1", "1.0"}:
            return "A"
        if "ASSISTANT B" in val or val in {"B", "2", "2.0"}:
            return "B"
        try:
            parsed_val = json.loads(str(row_data["actual_score"]).replace("'", '"'))
            if isinstance(parsed_val, (list, tuple)) and len(parsed_val) >= 2:
                return "A" if float(parsed_val[0]) >= float(parsed_val[1]) else "B"
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    if "model_1" in row_data and "chosen_model" in row_data:
        if str(row_data["chosen_model"]).strip() == str(row_data["model_1"]).strip():
            return "A"
        if "rejected_model" in row_data and str(row_data["rejected_model"]).strip() == str(row_data["model_1"]).strip():
            return "B"

    if "label" in row_data:
        label_val = str(row_data["label"]).strip().upper()
        if "ASSISTANT A" in label_val or label_val in {"1", "0", "A"}:
            return "A"
        if "ASSISTANT B" in label_val or label_val in {"2", "B"}:
            return "B"

    return "A"


def order_candidates(row_data: Any, position: str) -> Tuple[str, str, str]:
    human_winner = resolve_human_winner(row_data)
    if position == "AB":
        return str(row_data["response_1"]), str(row_data["response_2"]), human_winner
    return str(row_data["response_2"]), str(row_data["response_1"]), flip_candidate_label(human_winner)


def map_positional_to_logical(choice: str, position: str) -> Union[int, str]:
    if choice == "INVALID":
        return "INVALID"
    if position == "AB":
        return 1 if choice == "A" else 2
    return 2 if choice == "A" else 1
