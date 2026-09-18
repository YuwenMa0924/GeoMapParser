import pandas as pd
import re
from rapidfuzz import fuzz

# =========================
# File paths
# =========================
INPUT_EXCEL = "path/to/maps_table_e1.xlsx"
OUTPUT_EXCEL = "path/to/maps_table_e2.xlsx"


# =========================
# Agency dictionary (Canonical → aliases)
# =========================
AGENCY_DICT = {

    "USGS": [
        "usgs",
        "us geological survey",
        "u s geological survey",
        "us geol survey"
    ],

    "NASA": [
        "nasa",
        "national aeronautics space administration",
        "national aeronautics and space administration",
        "NASA Goddard Space Flight Center (GSFC)"
    ],

    "JPL": [
        "jet propulsion laboratory",
        "jpl"
    ],

    "DMA": [
        "defense mapping agency",
        "defense mapping agency topographic center",
        "defense mapping topographic center",
        "dma"
    ],

    "ACIC": [
        "aeronautical chart information center",
        "aeronautical chart and information center",
        "usic",
        "acic"
    ],

    "USAF": [
        "us air force",
        "united states air force",
        "usaf"
    ],

    "USATC": [
        "us army topographic command",
        "u s army topographic command"
    ],

    "LPI": [
        "lunar planetary institute",
        "lpi",
        "center for lunar science exploration"
    ],

    "DOI": [
        "us department interior",
        "u s department interior"
    ]
}


# =========================
# Parent-agency hierarchy
# =========================
AGENCY_PARENT = {

    "USGS": "DOI",
    "NASA": "NASA",
    "JPL": "NASA",
    "DMA": "DoD",
    "ACIC": "USAF",
    "USAF": "DoD",
    "USATC": "US Army",
    "LPI": "NASA-funded",
    "DOI": "US Government"
}


# =========================
# Country hierarchy
# =========================
AGENCY_COUNTRY = {

    "USGS": "USA",
    "NASA": "USA",
    "JPL": "USA",
    "DMA": "USA",
    "ACIC": "USA",
    "USAF": "USA",
    "USATC": "USA",
    "LPI": "USA",
    "DOI": "USA"
}


# =========================
# Text normalization
# =========================
def normalize_text(x):

    x = str(x).lower()
    x = re.sub(r"[^a-z0-9 ]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    return x


# =========================
# Single-agency matching
# =========================
def match_agency(text, threshold=60):

    text = normalize_text(text)

    best_match = None
    best_score = 0

    for canonical, aliases in AGENCY_DICT.items():

        for alias in aliases:

            score = fuzz.token_sort_ratio(text, alias)

            if score > best_score:
                best_score = score
                best_match = canonical

    if best_score >= threshold:
        return best_match
    else:
        return "Other"


# =========================
# Multi-agency splitting + matching
# =========================
def process_agencies(text):

    if pd.isna(text):
        return ["Unknown"]

    parts = re.split(r"[;,/&]| and ", str(text))

    results = []

    for p in parts:

        p = p.strip()

        if p == "":
            continue

        match = match_agency(p)

        results.append(match)

    return list(set(results))


# =========================
# Read the data
# =========================
df = pd.read_excel(INPUT_EXCEL)


# =========================
# Agency splitting + normalization
# =========================
df["agency_list"] = df["publication_agency"].apply(process_agencies)

df["agency_norm"] = df["agency_list"].apply(
    lambda x: "; ".join(x)
)


# =========================
# Parent hierarchy
# =========================
def map_parent(agency_list):

    parents = []

    for a in agency_list:

        parent = AGENCY_PARENT.get(a, "Other")

        parents.append(parent)

    return "; ".join(list(set(parents)))


df["agency_parent"] = df["agency_list"].apply(map_parent)


# =========================
# Country hierarchy
# =========================
def map_country(agency_list):

    countries = []

    for a in agency_list:

        c = AGENCY_COUNTRY.get(a, "Unknown")

        countries.append(c)

    return "; ".join(list(set(countries)))


df["agency_country"] = df["agency_list"].apply(map_country)


# =========================
# Output
# =========================
df.to_excel(OUTPUT_EXCEL, index=False)

print("✔ Agency normalization completed")
print("✔ Fields generated: agency_norm / agency_parent / agency_country")
print("✔ Output file:", OUTPUT_EXCEL)