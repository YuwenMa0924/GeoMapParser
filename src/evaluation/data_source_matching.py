import pandas as pd
import re


# ============================================================
# 1. Path configuration
# ============================================================

INPUT_EXCEL = (
    "path/to/GMIE_data/"
    "6.maps_table_e6.xlsx"
)

OUTPUT_EXCEL = (
    "path/to/GMIE_data/"
    "maps_table_e8.xlsx"
)

REVIEW_EXCEL = (
    "path/to/GMIE_data/"
    "data_source_manual_review.xlsx"
)

COLUMN = "data_source"


# ============================================================
# 2. Canonical dictionaries
# ============================================================

# ------------------------------------------------------------
# Mission
# ------------------------------------------------------------

MISSION_DICT = {

    "Apollo": [
        "apollo"
    ],

    "LRO": [
        "lunar reconnaissance orbiter",
        "lro"
    ],

    "MGS": [
        "mars global surveyor",
        "mgs"
    ],

    "MRO": [
        "mars reconnaissance orbiter",
        "mro"
    ],

    "Mars Odyssey": [
        "mars odyssey"
    ],

    "Viking": [
        "viking"
    ],

    "Clementine": [
        "clementine"
    ],

    "MESSENGER": [
        "messenger"
    ],

    "Magellan": [
        "magellan"
    ],

    "Galileo": [
        "galileo"
    ],

    "Cassini": [
        "cassini"
    ],

    "Voyager": [
        "voyager"
    ],

    "Chandrayaan": [
        "chandrayaan"
    ],

    "Mariner": [
        "mariner"
    ]
}


# ------------------------------------------------------------
# Sensor / Instrument
#
# Full names are used as aliases
# Canonical names are kept consistent
# ------------------------------------------------------------

SENSOR_DICT = {

    "MOLA": [
        "mola",
        "mars orbiter laser altimeter"
    ],

    "THEMIS": [
        "themis",
        "thermal emission imaging system"
    ],

    "CTX": [
        "ctx",
        "context camera"
    ],

    "HiRISE": [
        "hirise",
        "high resolution imaging science experiment"
    ],

    "MOC": [
        "moc",
        "mars orbiter camera"
    ],

    "HRSC": [
        "hrsc",
        "high resolution stereo camera"
    ],

    "LOLA": [
        "lola",
        "lunar orbiter laser altimeter",
        "lunar reconnaissance orbiter laser altimeter"
    ],

    "LROC": [
        "lroc",
        "lunar reconnaissance orbiter camera"
    ],

    "WAC": [
        "wac",
        "wide angle camera"
    ],

    "NAC": [
        "nac",
        "narrow angle camera"
    ],

    "SAR": [
        "sar",
        "synthetic aperture radar",
        "radar"
    ],

    "VIS": [
        "vis",
        "visible imaging",
        "visible image"
    ]
}


# ------------------------------------------------------------
# Product
#
# IMPORTANT:
# "image" intentionally removed.
# ------------------------------------------------------------

PRODUCT_DICT = {

    "DEM": [
        "dem",
        "digital elevation model"
    ],

    "DTM": [
        "dtm",
        "digital terrain model"
    ],

    "Imagery": [
        "imagery"
    ],

    "Mosaic": [
        "mosaic",
        "photomosaic"
    ],

    "Topography": [
        "topography",
        "elevation",
        "topographic"
    ],

    "Albedo": [
        "albedo"
    ],

    "MDIM": [
        "mdim",
        "mars digital image mosaic"
    ]
}


# ============================================================
# 3. Reference patterns
# ============================================================

REFERENCE_PATTERNS = [
    r"USGS\s+I[-–]\d+",
    r"USGS\s+Map\s+\d+",
    r"DOI[:\s]\S+",
    r"PDS",
    r"NASA\s+PDS"
]


# ============================================================
# 4. Known mission-sensor relationships
#
# Used to identify combinations that "clearly deserve manual inspection".
#
# Note:
# This only flags "potential inconsistencies" and never automatically deletes or modifies anything.
# Multi-source maps are allowed.
# ============================================================

MISSION_SENSOR_EXPECTED = {

    # Mars Global Surveyor
    "MGS": {
        "MOC",
        "MOLA"
    },

    # Mars Reconnaissance Orbiter
    "MRO": {
        "CTX",
        "HiRISE",
        "MOC"
    },

    # Mars Odyssey
    "Mars Odyssey": {
        "THEMIS"
    },

    # LRO
    "LRO": {
        "LROC",
        "LOLA",
        "NAC",
        "WAC"
    }
}


# ============================================================
# 5. Text normalization
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Keep letters, digits and spaces
    text = re.sub(
        r"[^a-z0-9 ]",
        " ",
        text
    )

    # Collapse consecutive spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# 6. Alias matching
# ============================================================

def alias_match(text, alias):

    return re.search(
        rf"\b{re.escape(alias.lower())}\b",
        text
    ) is not None


# ============================================================
# 7. Dictionary matching
# ============================================================

def dict_match(text, dictionary):

    text_norm = normalize_text(text)

    results = []
    matched_aliases = []

    for canonical, aliases in dictionary.items():

        for alias in aliases:

            alias_norm = normalize_text(alias)

            if alias_match(
                text_norm,
                alias_norm
            ):

                results.append(
                    canonical
                )

                matched_aliases.append(
                    f"{canonical}:{alias}"
                )

                # Record each canonical name only once
                break

    results = sorted(
        set(results)
    )

    matched_aliases = sorted(
        set(matched_aliases)
    )

    canonical_result = (
        "; ".join(results)
        if results
        else None
    )

    alias_result = (
        "; ".join(matched_aliases)
        if matched_aliases
        else None
    )

    return (
        canonical_result,
        alias_result
    )


# ============================================================
# 8. Reference extraction
# ============================================================

def extract_reference(text):

    if pd.isna(text):
        return None

    text = str(text)

    refs = []

    for pattern in REFERENCE_PATTERNS:

        found = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        refs.extend(found)

    if not refs:
        return None

    return "; ".join(
        sorted(set(refs))
    )


# ============================================================
# 9. Get list from normalized field
# ============================================================

def split_values(value):

    if value is None:
        return []

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        x.strip()
        for x in value.split(";")
        if x.strip()
    ]


# ============================================================
# 10. Detect suspicious mission-sensor combinations
#
# IMPORTANT:
# Multiple sensors are NOT automatically errors.
#
# Example:
# CTX + THEMIS
# MOLA + THEMIS
#
# These are valid multi-source situations and are NOT flagged.
#
# Only when a mission is explicitly identified and a sensor
# appears that is strongly associated with a different planetary
# mission family, we create a "potential_inconsistency" flag.
# ============================================================

def detect_mission_sensor_inconsistency(
    mission_norm,
    sensor_norm
):

    missions = split_values(
        mission_norm
    )

    sensors = set(
        split_values(sensor_norm)
    )

    if not missions or not sensors:
        return None

    suspicious = []

    for mission in missions:

        if mission not in MISSION_SENSOR_EXPECTED:
            continue

        expected = (
            MISSION_SENSOR_EXPECTED[
                mission
            ]
        )

        # ----------------------------------------------------
        # Only consider clearly incompatible sensors.
        #
        # If a map contains several missions, the sensor may
        # belong to another explicitly listed mission.
        # Therefore we only flag the combination when there is
        # exactly one mission.
        # ----------------------------------------------------

        if len(missions) == 1:

            incompatible = (
                sensors - expected
            )

            if incompatible:

                suspicious.append(
                    f"{mission} -> "
                    f"{', '.join(sorted(incompatible))}"
                )

    if suspicious:

        return "; ".join(
            suspicious
        )

    return None


# ============================================================
# 11. Detect completely unmatched source
#
# Only when:
#
# data_source contains actual text
# AND
# none of the three dimensions (mission / sensor / product) match
#
# does it enter manual review.
#
# A missing match in a single dimension does not trigger review.
# ============================================================

def detect_unmatched_source(
    text,
    mission_norm,
    sensor_norm,
    product_norm
):

    text_norm = normalize_text(
        text
    )

    if not text_norm:
        return None

    if (
        not mission_norm
        and not sensor_norm
        and not product_norm
    ):

        return "unmatched_source"

    return None


# ============================================================
# 12. Determine manual review
#
# Only the following cases enter manual review:
#
# A. Completely impossible to normalize
# B. A potential mission-sensor inconsistency appears
#
# Multiple values alone do not enter manual review.
# ============================================================

def determine_manual_review(
    unmatched_source,
    mission_sensor_inconsistency
):

    reasons = []

    if unmatched_source:

        reasons.append(
            unmatched_source
        )

    if mission_sensor_inconsistency:

        reasons.append(
            "potential_mission_sensor_inconsistency"
        )

    if reasons:

        return True, "; ".join(
            reasons
        )

    return False, None


# ============================================================
# 13. Read input
# ============================================================

df = pd.read_excel(
    INPUT_EXCEL
)

if COLUMN not in df.columns:

    raise ValueError(
        f"Field not found: {COLUMN}"
    )


# ============================================================
# 14. Preserve original data_source
# ============================================================

df["data_source_raw"] = (
    df[COLUMN]
    .fillna("")
    .astype(str)
)


# ============================================================
# 15. Extraction containers
# ============================================================

missions = []
mission_matches = []

sensors = []
sensor_matches = []

products = []
product_matches = []

references = []

multi_source_flags = []

mission_sensor_flags = []

unmatched_flags = []

manual_review_flags = []
manual_review_reasons = []


# ============================================================
# 16. Main processing
# ============================================================

for text in df["data_source_raw"]:

    # --------------------------------------------------------
    # Mission
    # --------------------------------------------------------

    mission, mission_alias = dict_match(
        text,
        MISSION_DICT
    )


    # --------------------------------------------------------
    # Sensor
    # --------------------------------------------------------

    sensor, sensor_alias = dict_match(
        text,
        SENSOR_DICT
    )


    # --------------------------------------------------------
    # Product
    # --------------------------------------------------------

    product, product_alias = dict_match(
        text,
        PRODUCT_DICT
    )


    # --------------------------------------------------------
    # Reference
    # --------------------------------------------------------

    reference = extract_reference(
        text
    )


    # --------------------------------------------------------
    # Multi-source information
    #
    # Recorded only; not treated as an error.
    # --------------------------------------------------------

    multi_flags = []

    if len(split_values(mission)) > 1:
        multi_flags.append(
            "multiple_missions"
        )

    if len(split_values(sensor)) > 1:
        multi_flags.append(
            "multiple_sensors"
        )

    if len(split_values(product)) > 1:
        multi_flags.append(
            "multiple_products"
        )

    multi_source_flag = (
        "; ".join(multi_flags)
        if multi_flags
        else None
    )


    # --------------------------------------------------------
    # Mission-sensor consistency
    # --------------------------------------------------------

    mission_sensor_flag = (
        detect_mission_sensor_inconsistency(
            mission,
            sensor
        )
    )


    # --------------------------------------------------------
    # Completely unmatched
    # --------------------------------------------------------

    unmatched_flag = (
        detect_unmatched_source(
            text,
            mission,
            sensor,
            product
        )
    )


    # --------------------------------------------------------
    # Manual review
    # --------------------------------------------------------

    manual_review, review_reason = (
        determine_manual_review(
            unmatched_flag,
            mission_sensor_flag
        )
    )


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    missions.append(
        mission
    )

    mission_matches.append(
        mission_alias
    )

    sensors.append(
        sensor
    )

    sensor_matches.append(
        sensor_alias
    )

    products.append(
        product
    )

    product_matches.append(
        product_alias
    )

    references.append(
        reference
    )

    multi_source_flags.append(
        multi_source_flag
    )

    mission_sensor_flags.append(
        mission_sensor_flag
    )

    unmatched_flags.append(
        unmatched_flag
    )

    manual_review_flags.append(
        manual_review
    )

    manual_review_reasons.append(
        review_reason
    )


# ============================================================
# 17. Write normalized fields
# ============================================================

df["mission_norm"] = missions

df["mission_match"] = mission_matches

df["sensor_norm"] = sensors

df["sensor_match"] = sensor_matches

df["product_norm"] = products

df["product_match"] = product_matches

df["reference"] = references


# ============================================================
# 18. Quality-control fields
# ============================================================

df["multi_source_flag"] = (
    multi_source_flags
)

df["mission_sensor_inconsistency"] = (
    mission_sensor_flags
)

df["unmatched_source"] = (
    unmatched_flags
)

df["manual_review"] = (
    manual_review_flags
)

df["manual_review_reason"] = (
    manual_review_reasons
)


# ============================================================
# 19. Save complete normalized dataset
# ============================================================

df.to_excel(
    OUTPUT_EXCEL,
    index=False
)


# ============================================================
# 20. Generate manual review table
# ============================================================

review_columns = [

    "data_source_raw",

    "mission_norm",
    "mission_match",

    "sensor_norm",
    "sensor_match",

    "product_norm",
    "product_match",

    "reference",

    "multi_source_flag",

    "mission_sensor_inconsistency",

    "unmatched_source",

    "manual_review_reason"
]


review_df = df[
    df["manual_review"] == True
][
    review_columns
].copy()


# Save manual review table
review_df.to_excel(
    REVIEW_EXCEL,
    index=False
)


# ============================================================
# 21. Statistics
# ============================================================

total_records = len(df)

review_count = len(
    review_df
)

multi_source_count = (
    df["multi_source_flag"]
    .notna()
    .sum()
)

inconsistency_count = (
    df["mission_sensor_inconsistency"]
    .notna()
    .sum()
)

unmatched_count = (
    df["unmatched_source"]
    .notna()
    .sum()
)


print(
    "\n============================================================"
)

print(
    "DATA SOURCE NORMALIZATION COMPLETED"
)

print(
    "============================================================"
)

print(
    f"Total records: {total_records}"
)

print(
    f"Records containing multiple sources: "
    f"{multi_source_count}"
)

print(
    f"Potential mission-sensor inconsistencies: "
    f"{inconsistency_count}"
)

print(
    f"Completely unmatched records: "
    f"{unmatched_count}"
)

print(
    f"Records requiring manual review: "
    f"{review_count}"
)

print(
    "\nFull normalized dataset:"
)

print(
    OUTPUT_EXCEL
)

print(
    "\nManual review dataset:"
)

print(
    REVIEW_EXCEL
)


# ============================================================
# 22. Print manual review records
# ============================================================

if review_count > 0:

    print(
        "\n============================================================"
    )

    print(
        "RECORDS REQUIRING MANUAL REVIEW"
    )

    print(
        "============================================================"
    )

    print(
        review_df.to_string(
            index=False
        )
    )

else:

    print(
        "\n✔ No records require manual review."
    )


# ============================================================
# 23. Summary by review reason
# ============================================================

if review_count > 0:

    print(
        "\n============================================================"
    )

    print(
        "MANUAL REVIEW REASON SUMMARY"
    )

    print(
        "============================================================"
    )

    print(
        df[
            df["manual_review"] == True
        ][
            "manual_review_reason"
        ]
        .value_counts()
        .to_string()
    )


# ============================================================
# 24. Multi-source summary
#
# Multiple sources are not an error; recorded for statistics only.
# ============================================================

print(
    "\n============================================================"
)

print(
    "MULTI-SOURCE SUMMARY"
)

print(
    "============================================================"
)

if multi_source_count > 0:

    print(
        df[
            "multi_source_flag"
        ]
        .dropna()
        .value_counts()
        .to_string()
    )

else:

    print(
        "No multi-source records detected."
    )