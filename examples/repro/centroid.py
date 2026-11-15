import json, math, sys
data = json.load(sys.stdin)
labels = sorted({row["y"] for row in data["train"]})
centroids = {}
for label in labels:
    rows = [row["x"] for row in data["train"] if row["y"] == label]
    centroids[label] = [math.fsum(col) / len(rows) for col in zip(*rows)]
result = []
for row in data["test"]:
    result.append(min(labels, key=lambda label: (
        math.dist(row["x"], centroids[label]), label)))
print(json.dumps({"ids": [row["id"] for row in data["test"]], "labels": result}))
