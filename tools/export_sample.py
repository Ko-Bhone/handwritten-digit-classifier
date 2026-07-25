import json
from pathlib import Path

from src.data.loader import DigitDataLoader
from src.config import DATA_PATH

loader = DigitDataLoader(DATA_PATH)
x, y = loader.load()

sample = {
    "pixels": x[0].tolist()
}

output_file = Path("sample.json")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(sample, f, indent=4)

print(f"Saved to: {output_file.resolve()}")
print(f"Number of pixels: {len(sample['pixels'])}")