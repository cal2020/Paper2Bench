"""Dynamic programming candidate; the protected checker enumerates subsets."""
import json
from pathlib import Path

data = json.loads(Path("input.json").read_text())
states = {0: (0, [])}
for index, (weight, value) in enumerate(zip(data["weights"], data["values"])):
    updated = dict(states)
    for current_weight, (current_value, selected) in states.items():
        total = current_weight + weight
        if total <= data["capacity"] and (total not in updated or current_value + value > updated[total][0]):
            updated[total] = (current_value + value, selected + [index])
    states = updated
best = max(states.values(), key=lambda pair: pair[0])
Path("predictions.json").write_text(json.dumps({"selected": best[1]}))
