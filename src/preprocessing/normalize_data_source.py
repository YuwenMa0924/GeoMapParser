import pandas as pd
import re

# ===============================
# Path configuration
# ===============================
INPUT_EXCEL = "path/to/maps_table_e2.xlsx"
OUTPUT_EXCEL = "path/to/maps_table_e3.xlsx"

COLUMN = "data_source"


# ===============================
# Dictionaries (canonical mapping)
# ===============================

MISSION_DICT = {

    "Apollo": ["apollo"],

    "LRO": [
        "lunar reconnaissance orbiter",
        "lro",
    ],

    "MGS": [
        "mars global surveyor",
        "mgs",
    ],

    "MRO": [
        "mars reconnaissance orbiter",
        "mro",
    ],

    "Mars Odyssey": ["mars odyssey"],

    "Viking": ["viking"],

    "Clementine": ["clementine"],

    "MESSENGER": ["messenger"],

    "Magellan": ["magellan"],

    "Galileo": ["galileo"],

    "Cassini": ["cassini"],

    "Voyager": ["voyager"],

    "Chandrayaan": ["chandrayaan"],

    "Mariner": ["mariner"],

}


SENSOR_DICT = {

    "MOLA": [
        "mola",
        "mars orbiter laser altimeter",
    ],

    "THEMIS": [
        "themis",
        "thermal emission imaging system",
    ],

    "CTX": [
        "ctx",
        "context camera",
    ],

    "HiRISE": [
        "hirise",
        "high resolution imaging science experiment",
    ],

    "MOC": [
        "moc",
        "mars orbiter camera",
    ],

    "HRSC": [
        "hrsc",
        "high resolution stereo camera",
    ],

    "LOLA": [
        "lola",
        "lunar orbiter laser altimeter",
        "lunar reconnaissance orbiter laser altimeter",
    ],

    "LROC": [
        "lroc",
        "lunar reconnaissance orbiter camera",
    ],

    "WAC": [
        "wac",
        "wide angle camera",
    ],

    "NAC": [
        "nac",
        "narrow angle camera",
    ],

    "SAR": [
        "sar",
        "synthetic aperture radar",
        "radar",
    ],

    "VIS": [
        "vis",
        "visible imaging",
        "visible image",
    ],

}


PRODUCT_DICT = {

    "DEM": [
        "dem",
        "digital elevation model",
    ],

    "DTM": [
        "dtm",
        "digital terrain model",
    ],

    "Imagery": ["imagery"],

    "Mosaic": [
        "mosaic",
        "photomosaic",
    ],

    "Topography": [
        "topography",
        "elevation",
        "topographic",
    ],

    "Albedo": ["albedo"],

    "MDIM": [
        "mdim",
        "mars digital image mosaic",
    ],

}


REFERENCE_PATTERNS = [
    r"USGS\s+I[-–]\d+",
    r"USGS\s+Map\s+\d+",
    r"DOI[:\s]\S+",
    r"PDS",
    r"NASA\s+PDS"
]


# ===============================
# Text normalization
# ===============================

def normalize_text(text):

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text


# ===============================
# Dictionary matching
# ===============================

def dict_match(text, dictionary):

    text = normalize_text(text)

    results = []

    for canonical, aliases in dictionary.items():

        for alias in aliases:

            if re.search(rf"\b{alias}\b", text):

                results.append(canonical)
                break

    return "; ".join(sorted(set(results))) if results else None


# ===============================
# reference extraction
# ===============================

def extract_reference(text):

    refs = []

    for pat in REFERENCE_PATTERNS:

        found = re.findall(pat, text, flags=re.IGNORECASE)

        refs.extend(found)

    return "; ".join(refs) if refs else None


# ===============================
# Read the data
# ===============================

df = pd.read_excel(INPUT_EXCEL)

df["data_source_raw"] = df[COLUMN].astype(str)


# ===============================
# Information extraction
# ===============================

missions = []
sensors = []
products = []
references = []

for text in df["data_source_raw"]:

    missions.append(dict_match(text, MISSION_DICT))
    sensors.append(dict_match(text, SENSOR_DICT))
    products.append(dict_match(text, PRODUCT_DICT))
    references.append(extract_reference(text))


df["mission_norm"] = missions
df["sensor_norm"] = sensors
df["product_norm"] = products
df["reference"] = references


# ===============================
# Output
# ===============================

df.to_excel(OUTPUT_EXCEL, index=False)

print("✔ Data source normalization completed")
print("✔ Fields generated: mission_norm / sensor_norm / product_norm")
print(f"✔ Output saved to: {OUTPUT_EXCEL}")