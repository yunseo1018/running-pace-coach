import json
import pandas as pd

try:
    with open('endomondoHR_proper.json', 'r') as f:
        data = json.load(f)
except json.JSONDecodeError:
    data = []
    with open('endomondoHR_proper.json', 'r') as f:
        for line in f:
            data.append(json.loads(line))

print(f"전체 기록 수: {len(data)}")
print("첫 번째 기록:")
print(data[0])