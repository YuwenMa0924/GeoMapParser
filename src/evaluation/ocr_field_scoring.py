import os

# ================= HuggingFace mirror =================

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"


import json
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

import re
from wordfreq import zipf_frequency


# ================= Parameters =================

JSON_DIR = (
    "path/to/example_outputs"
)


OUTPUT_EXCEL = (
    "path/to/semantic_analysis_results/"
    "C/semantic_evaluationC_sigmoid_OCR_final.xlsx"
)



FIELDS = [
    "title",
    "scale",
    "projection",
    "publication_agency",
    "publication_date",
    "data_source"
]


# semantic evaluation fields

SEMANTIC_FIELDS = {
    "title",
    "projection",
    "publication_agency",
    "data_source"
}


# OCR incomplete words evaluation fields

OCR_FIELDS = {
    "title",
    "projection",
    "publication_agency",
    "data_source"
}



# semantic sigmoid parameters

Z0 = 2.5
K = 1.0



# ================= Embedding model =================


embedder = SentenceTransformer(
    "path/to/models/scibert_scivocab_uncased"
)



# ================= Text normalization =================


def normalize_text(x):

    if x is None:
        return ""

    if isinstance(x, list):
        return "; ".join(x)

    return str(x).strip()



# ================= Rule penalty =================


def rule_violation(field, text):

    """
    Rule consistency penalty

    Pr = 1 - rule confidence

    correct:
        Pr = 0

    incorrect:
        Pr = 1
    """

    if text is None or text.strip()=="":
        return 1.0, ["empty_field"]


    # scale

    if field=="scale":

        if re.search(
            r"1\s*[:：]\s*\d[\d,]*",
            text.strip()
        ):

            return 0.0,[]

        else:

            return 1.0,[
                "invalid_scale_format"
            ]



    # publication date

    if field=="publication_date":

        if re.fullmatch(
            r"\d{4}",
            text.strip()
        ):

            return 0.0,[]

        else:

            return 1.0,[
                "invalid_date_format"
            ]


    return 0.0,[]




# ================= OCR incomplete word penalty =================


def ocr_penalty(field,text):



    if field not in OCR_FIELDS:

        return 0.0,[]



    if text is None or text.strip()=="":

        return 1.0,[
            "empty_field"
        ]



    words=text.split()


    if len(words)==0:

        return 0.0,[]



    bad_words=[]



    for w in words:


        clean_word=re.sub(
            r"[^a-zA-Z]",
            "",
            w
        )


        if len(clean_word)<4:

            continue



        if clean_word.isupper():

            continue



        freq=zipf_frequency(
            clean_word.lower(),
            "en"
        )


        if freq < 1.0:

            bad_words.append(
                clean_word
            )



    ratio=len(bad_words)/len(words)


    reasons=[]


    if len(bad_words)>0:

        reasons.append(
            "incomplete_words"
        )


    return ratio,reasons





# ================= Continuous semantic penalty =================


def semantic_penalty(z,z0=Z0,k=K):

    penalty = (
        1 /
        (
            1+
            np.exp(
                -k*(z-z0)
            )
        )
    )

    return penalty




# ================= Read JSON =================


records=[]


for fname in os.listdir(JSON_DIR):


    if not (
        fname.endswith(".json")
        or
        fname.endswith(".JSON")
    ):

        continue



    path=os.path.join(
        JSON_DIR,
        fname
    )



    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data=json.load(f)



    for field in FIELDS:


        records.append(
            {
                "file":fname,

                "field":field,

                "text":
                normalize_text(
                    data.get(field)
                )
            }
        )



if len(records)==0:

    raise ValueError(
        "No JSON files were read"
    )



df=pd.DataFrame(records)



print(
    "Number of field records:",
    len(df)
)




# ================= Embedding =================


print(
    "Computing embeddings..."
)


embeddings=embedder.encode(

    df["text"].tolist(),

    normalize_embeddings=True,

    show_progress_bar=True

)


df["embedding"]=list(
    embeddings
)




# ================= semantic z-score =================


print(
    "Computing semantic z-score..."
)


semantic_z=[]



for field in FIELDS:


    sub=df[
        df["field"]==field
    ]



    if field not in SEMANTIC_FIELDS:


        semantic_z.extend(
            [0.0]*len(sub)
        )

        continue




    X=np.vstack(
        sub["embedding"].values
    )



    center=X.mean(
        axis=0,
        keepdims=True
    )



    distances=cosine_distances(
        X,
        center
    ).flatten()



    z=(

        distances-distances.mean()

    )/(

        distances.std()+1e-6

    )



    semantic_z.extend(
        z.tolist()
    )



df["semantic_z"]=semantic_z





# ================= Rule penalty =================


rule_results=df.apply(

    lambda row:
    rule_violation(
        row["field"],
        row["text"]
    ),

    axis=1

)


df["rule_score"]=rule_results.apply(
    lambda x:x[0]
)


df["rule_reasons"]=rule_results.apply(
    lambda x:x[1]
)





# ================= OCR penalty =================


print(
    "Computing OCR incomplete-word penalty..."
)



ocr_results=df.apply(

    lambda row:
    ocr_penalty(
        row["field"],
        row["text"]
    ),

    axis=1

)



df["ocr_score"]=ocr_results.apply(
    lambda x:x[0]
)



df["ocr_reasons"]=ocr_results.apply(
    lambda x:x[1]
)





# ================= anomaly =================


df["is_anomaly"]=(
(df["semantic_z"]>Z0)
|
(df["rule_score"]>0)
|
(df["ocr_score"]>0)
)




def explain(row):

    reasons=[]


    if row["semantic_z"]>Z0:

        reasons.append(
            "semantic_outlier"
        )


    reasons.extend(
        row["rule_reasons"]
    )


    reasons.extend(
        row["ocr_reasons"]
    )


    return ";".join(
        sorted(set(reasons))
    )



df["anomaly_reason"]=df.apply(
    explain,
    axis=1
)





# ================= Final quality score =================


# ================= Final quality score =================

def compute_quality(row):

    """
    Final field extraction quality.

    Q_i = (1-Ps)*(1-Pr)*(1-Po)

    Ps:
        semantic inconsistency penalty

    Pr:
        rule violation penalty

    Po:
        OCR incomplete word penalty
    """


    # ----------------
    # Semantic penalty
    # ----------------

    semantic_p=0.0


    if row["field"] in SEMANTIC_FIELDS:

        semantic_p=semantic_penalty(
            row["semantic_z"]
        )



    # ----------------
    # Rule penalty
    # ----------------

    rule_p=min(
        1.0,
        row["rule_score"]
    )



    # ----------------
    # OCR penalty
    # ----------------

    ocr_p=min(
        1.0,
        row["ocr_score"]
    )



    # ----------------
    # Final quality
    # Q=(1-Ps)(1-Pr)(1-Po)
    # ----------------

    quality=(

        1-semantic_p

    )*(

        1-rule_p

    )*(

        1-ocr_p

    )



    return round(
        max(0,quality),
        3
    )




df["quality_score"]=df.apply(
    compute_quality,
    axis=1
)





# ================= quality level =================


def quality_level(score):


    if score>=0.8:

        return "good"


    elif score>=0.5:

        return "medium"


    else:

        return "poor"



df["quality_level"]=df[
    "quality_score"
].apply(
    quality_level
)





# ================= Output =================


df.drop(
    columns=["embedding"],
    inplace=True
)



df.to_excel(
    OUTPUT_EXCEL,
    index=False
)



print(
    "✔ Evaluation finished"
)



print(
    "\nAverage quality:"
)



for field in FIELDS:


    sub=df[
        df["field"]==field
    ]


    print(

        field,

        round(
            sub["quality_score"].mean(),
            3
        )

    )