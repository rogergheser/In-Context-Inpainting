import json

with open('dataset.json') as f:
    data = json.load(f)

print(data)