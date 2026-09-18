import os
import json
from wordfreq import zipf_frequency

# ========== Configuration ==========
JSON_DIR = "path/to/extraction_results"

# ========== Utility functions ==========
def normalize_text(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(x)
    return str(x).strip()


def is_incomplete_words(text):
    """
    Incomplete-word test specific to the projection field
    """
    if text.strip() == "":
        return False

    words = text.split()

    bad_words = 0
    total_words = 0

    for w in words:
        if len(w) < 3:
            continue

        total_words += 1

        # Projection strings contain many fully capitalized tokens (allowed)
        if w.isupper():
            continue

        # Skip numeric and symbolic tokens
        if not any(c.isalpha() for c in w):
            continue

        # Language-frequency test
        if zipf_frequency(w.lower(), "en") < 1.0:
            bad_words += 1

    if total_words == 0:
        return False

    return bad_words >= 1


# ========== Main statistics ==========
total = 0
error = 0

for fname in os.listdir(JSON_DIR):
    if not fname.endswith(".json"):
        continue

    path = os.path.join(JSON_DIR, fname)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    projection = normalize_text(data.get("projection"))

    if projection == "":
        continue

    total += 1

    if is_incomplete_words(projection):
        error += 1

# ========== Output ==========
ratio = error / total if total > 0 else 0

print("\n===== Projection Incomplete Words =====")
print(f"total: {total}")
print(f"incomplete: {error}")
print(f"ratio: {ratio:.4f}")
print(f"percentage: {ratio*100:.2f}%")