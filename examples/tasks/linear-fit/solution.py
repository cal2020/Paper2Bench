"""Fit ordinary least squares using only the public training observations."""
import json
from pathlib import Path

data = json.loads(Path("input.json").read_text())
train = data["train"]
x_mean = sum(row[0] for row in train) / len(train)
y_mean = sum(row[1] for row in train) / len(train)
slope = sum((x-x_mean)*(y-y_mean) for x, y in train) / sum((x-x_mean)**2 for x, y in train)
intercept = y_mean - slope*x_mean
Path("predictions.json").write_text(json.dumps({"predictions": [slope*x+intercept for x in data["test"]]}))
