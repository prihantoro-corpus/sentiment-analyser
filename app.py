import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import matplotlib.pyplot as plt
from textblob import TextBlob
from scipy.cluster.hierarchy import dendrogram, linkage
import re

st.set_page_config(page_title="Multi-file Sentiment Analyzer", layout="wide")
st.title("Multi-file Sentiment Analyzer")

# =========================================================
# Sentence segmentation utility
# =========================================================
def split_into_sentences(text):
    if not text:
        return []
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Rule-based splitter: . ! ? followed by space and capital/number
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


# =========================================================
# Language selection (sentiment only EN for now)
# =========================================================
language = st.selectbox("Choose language", ["EN", "ID", "JP"], index=0)
if language != "EN":
    st.info("Sentiment analysis is currently supported for English only. Other languages are accepted as input but not analyzed.")

# =========================================================
# Input method
# =========================================================
input_mode = st.radio("Input method", ["Direct Input", "Upload Files"])
files_data = {}

if input_mode == "Direct Input":
    text_input = st.text_area(
        "Enter text (sentences will be auto-detected if not separated)",
        height=200
    )

    if text_input.strip():
        # If user already used line breaks, respect them; otherwise auto-split
        if "\n" in text_input.strip():
            sentences = [line.strip() for line in text_input.splitlines() if line.strip()]
        else:
            sentences = split_into_sentences(text_input)

        files_data["direct_input.txt"] = sentences

else:
    uploaded_files = st.file_uploader(
        "Upload multiple files (txt, tsv, xlsx)",
        type=["txt", "tsv", "xlsx"],
        accept_multiple_files=True
    )

    for file in uploaded_files or []:
        raw_text = ""

        if file.name.endswith(".txt"):
            raw_text = file.read().decode("utf-8", errors="ignore")

        elif file.name.endswith(".tsv"):
            df = pd.read_csv(file, sep="\t", header=None)
            raw_text = " ".join(df.iloc[:, 0].astype(str).tolist())

        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file, header=None)
            raw_text = " ".join(df.iloc[:, 0].astype(str).tolist())

        else:
            continue

        # If already one sentence per line, keep; otherwise auto-split
        if "\n" in raw_text.strip():
            sentences = [line.strip() for line in raw_text.splitlines() if line.strip()]
        else:
            sentences = split_into_sentences(raw_text)

        files_data[file.name] = sentences


# =========================================================
# Sentiment function (scaled -5 to +5)
# =========================================================
def sentiment_score(sentence):
    if not sentence or language != "EN":
        return 0.0
    polarity = TextBlob(sentence).sentiment.polarity  # -1 to +1
    return round(polarity * 5, 2)


# =========================================================
# Process
# =========================================================
results = {}
summary_rows = []

for fname, sentences in files_data.items():
    scores = [sentiment_score(s) for s in sentences]
    df = pd.DataFrame({"score": scores, "sentence": sentences})
    results[fname] = df

    neg = (df["score"] < -1).sum()
    neu = ((df["score"] >= -1) & (df["score"] <= 1)).sum()
    pos = (df["score"] > 1).sum()

    summary_rows.append({
        "file": fname,
        "negative": neg,
        "neutral": neu,
        "positive": pos,
        "mean": df["score"].mean() if len(df) else 0
    })


# =========================================================
# Preview
# =========================================================
if results:
    st.subheader("Preview (first 5 and last 5)")
    for fname, df in results.items():
        st.markdown(f"**{fname}**")
        preview_df = pd.concat([df.head(5), df.tail(5)])
        st.dataframe(preview_df, use_container_width=True)


# =========================================================
# Download Excel zipped
# =========================================================
if results:
    st.subheader("Download Results")
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, df in results.items():
            out = io.BytesIO()
            df.to_excel(out, index=False)
            zf.writestr(fname + ".xlsx", out.getvalue())

    st.download_button(
        label="Download all as ZIP (Excel)",
        data=zip_buffer.getvalue(),
        file_name="sentiment_results.zip",
        mime="application/zip"
    )


# =========================================================
# Charts
# =========================================================
if results:
    st.subheader("Overall Comparison Charts")
    summary_df = pd.DataFrame(summary_rows)

    # ---------- 100% stacked bar ----------
    st.markdown("**100% Stacked Bar (Negative / Neutral / Positive)**")

    perc_df = summary_df.copy()
    total = perc_df[["negative", "neutral", "positive"]].sum(axis=1)
    perc_df["negative"] = perc_df["negative"] / total * 100
    perc_df["neutral"] = perc_df["neutral"] / total * 100
    perc_df["positive"] = perc_df["positive"] / total * 100

    fig, ax = plt.subplots()
    ax.bar(perc_df["file"], perc_df["negative"], bottom=0)
    ax.bar(perc_df["file"], perc_df["neutral"], bottom=perc_df["negative"])
    ax.bar(
        perc_df["file"],
        perc_df["positive"],
        bottom=perc_df["negative"] + perc_df["neutral"]
    )
    ax.set_ylabel("Percentage")
    ax.set_xticklabels(perc_df["file"], rotation=45, ha="right")
    st.pyplot(fig)

    # ---------- Dendrogram ----------
    st.markdown("**Dendrogram (group files by sentiment similarity)**")
    if len(summary_df) > 1:
        X = summary_df[["mean"]].values
        Z = linkage(X, method="ward")

        fig2, ax2 = plt.subplots()
        dendrogram(Z, labels=summary_df["file"].values, ax=ax2)
        st.pyplot(fig2)
    else:
        st.info("Need at least 2 files for dendrogram.")

    # ---------- Individual line charts ----------
    st.subheader("Individual File Sentiment Flow")
    for fname, df in results.items():
        st.markdown(f"**{fname}**")
        fig3, ax3 = plt.subplots()
        ax3.plot(range(len(df)), df["score"])
        ax3.set_ylim(-5, 5)
        ax3.set_ylabel("Sentiment (-5 to +5)")
        ax3.set_xlabel("Sentence Index")
        st.pyplot(fig3)
