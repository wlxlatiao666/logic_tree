import json

with open("uncertainty_results_7B_aime.json", "r") as f:
    data = json.load(f)

complexities = [item["complexity"] for item in data if "complexity" in item]
mean_complexity = sum(complexities) / len(complexities)
print("Mean complexity:", mean_complexity)