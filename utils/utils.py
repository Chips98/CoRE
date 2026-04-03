import json
from collections import defaultdict


def load_data(data_path):
    """Load a JSON dataset from disk."""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_and_sort_by_user(data):
    """Group samples by user and sort each user's data by time."""
    user_data = defaultdict(list)

    for item in data:
        user_data[item["user_id"]].append(item)

    for user_id in user_data:
        user_data[user_id].sort(
            key=lambda x: (
                x.get("timestep", ""),
                x.get("conversation_id", -1),
                x.get("sub_id", -1),
            )
        )

    return user_data
