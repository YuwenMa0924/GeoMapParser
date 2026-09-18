import pandas as pd
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

# ================= Parameters =================
INPUT_EXCEL = "path/to/maps_table_e8.xlsx"
OUTPUT_EXCEL = "path/to/maps_table_e10.xlsx"
JSON_FOLDER = "path/to/metadata_jsons"

TITLE_COLUMN = "title"
FILE_COLUMN = "file"

K = 25
TOTAL_SAMPLE = 100

# ================= Read Excel =================
print("📄 Reading Excel...")
df = pd.read_excel(INPUT_EXCEL)

# Clean title
df = df[df[TITLE_COLUMN].notna()].copy()
df[TITLE_COLUMN] = df[TITLE_COLUMN].astype(str).str.strip()

# ================= Build image_id =================
def extract_image_id(x):
    try:
        return str(int(''.join(filter(str.isdigit, str(x)))))
    except:
        return None

df["image_id"] = df[FILE_COLUMN].apply(extract_image_id)
df = df[df["image_id"].notna()].copy()

print(f"✔ Valid samples: {len(df)}")

# ================= Build JSON index =================
print("\n📂 Building the JSON file index...")

json_map = {}
for fname in os.listdir(JSON_FOLDER):
    if not fname.lower().endswith(".json"):
        continue
    key = ''.join(filter(str.isdigit, fname))
    if key != "":
        json_map[key] = fname

print(f"✔ Number of JSON files: {len(json_map)}")

# ================= Extract boxes =================
def extract_box_from_file(image_id):
    try:
        if image_id not in json_map:
            print(f"❌ JSON not found: {image_id}")
            return 0

        path = os.path.join(JSON_FOLDER, json_map[image_id])

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "total_detection_boxes" in data:
            return data["total_detection_boxes"]
        elif "boxes" in data:
            return len(data["boxes"])
        elif "detections" in data and "boxes" in data["detections"]:
            return len(data["detections"]["boxes"])
        else:
            print(f"⚠ Unrecognized structure: {image_id}")
            return 0

    except Exception as e:
        print(f"❌ Read failed: {image_id}, {e}")
        return 0

print("\n📦 Extracting box_count...")
df["box_count"] = [
    extract_box_from_file(i) for i in tqdm(df["image_id"])
]

# ================= Box statistics =================
print("\n📊 Box statistics:")
print(df["box_count"].describe())

# ================= embedding =================
print("\n🧠 Computing title embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    df[TITLE_COLUMN].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)

# ================= Feature fusion =================
box_feature = df["box_count"].values.reshape(-1, 1)

scaler = StandardScaler()
box_scaled = scaler.fit_transform(box_feature)

X = np.hstack([embeddings, box_scaled])

# ================= PCA =================
print("\n📉 PCA dimensionality reduction...")
pca = PCA(n_components=20, random_state=42)
X_reduced = pca.fit_transform(X)

# ================= KMeans =================
print("\n🔵 KMeans clustering...")
kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=20
)

df["cluster_id"] = kmeans.fit_predict(X_reduced)

# ================= Distance computation =================
centers = kmeans.cluster_centers_

distances = []
for x, cid in zip(X_reduced, df["cluster_id"]):
    d = cosine_distances(
        x.reshape(1, -1),
        centers[cid].reshape(1, -1)
    )[0, 0]
    distances.append(d)

df["distance_to_center"] = distances

# ================= Sampling (fixed version) =================
print("\n🎯 Sampling...")

cluster_sizes = df["cluster_id"].value_counts()
selected_indices = []

# ===== Round 1: proportional sampling =====
for cid, size in cluster_sizes.items():
    sub = df[df["cluster_id"] == cid]

    n_sample = int(size / len(df) * TOTAL_SAMPLE)
    n_sample = max(1, n_sample)
    n_sample = min(n_sample, len(sub))

    top_center = sub.nsmallest(n_sample // 2, "distance_to_center")
    top_diverse = sub.nlargest(n_sample - len(top_center), "distance_to_center")

    selected_indices.extend(top_center.index.tolist())
    selected_indices.extend(top_diverse.index.tolist())

# Deduplicate
selected_indices = list(set(selected_indices))

# ===== Round 2: top-up =====
if len(selected_indices) < TOTAL_SAMPLE:
    print(f"⚠ Current sample count {len(selected_indices)}; top-up to {TOTAL_SAMPLE}")

    remaining = df.drop(index=selected_indices)

    # Pick "medium-difficulty" samples (more principled)
    remaining = remaining.sort_values("distance_to_center")

    needed = TOTAL_SAMPLE - len(selected_indices)
    supplement = remaining.iloc[:needed].index.tolist()

    selected_indices.extend(supplement)

# ===== Final count control =====
selected_indices = selected_indices[:TOTAL_SAMPLE]

# Flag
df["is_selected"] = False
df.loc[selected_indices, "is_selected"] = True

# ================= Output =================
df_sorted = df.sort_values(["cluster_id", "distance_to_center"])
df_sorted.to_excel(OUTPUT_EXCEL, index=False)

# ================= Summary =================
print("\n✅ Done")
print(f"Total samples: {len(df)}")
print(f"Selected samples: {df['is_selected'].sum()}")
print(f"Output path: {OUTPUT_EXCEL}")