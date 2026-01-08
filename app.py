import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import re

# ---- matplotlib safe backend ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.backends.backend_pdf import PdfPages


# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="Multi-file Sentiment Analyzer", layout="wide")
st.title("Multi-file Sentiment Analyzer")

analyzer = SentimentIntensityAnalyzer()


# =========================================================
# Sentence segmentation utility
# =========================================================
def split_into_sentences(text):
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


# =========================================================
# Language selection
# =========================================================
language = st.selectbox("Choose language", ["EN", "ID", "JP"], index=0)
if language != "EN":
    st.info("Sentiment analysis is currently supported for English only.")


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

        if raw_text.strip():
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

    vs = analyzer.polarity_scores(sentence)
    compound = vs["compound"]  # -1 to +1
    return round(compound * 5, 2)


# =========================================================
# Process files
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
# Single file preview + download
# =========================================================
selected_file = None

if results:
    selected_file = st.selectbox("Select file to preview", list(results.keys()))
    df_selected = results[selected_file]

    st.subheader(f"Preview: {selected_file}")
    preview_df = pd.concat([df_selected.head(5), df_selected.tail(5)])
    st.dataframe(preview_df, use_container_width=True)

    out_single = io.BytesIO()
    df_selected.to_excel(out_single, index=False)

    st.download_button(
        label="Download this file (Excel)",
        data=out_single.getvalue(),
        file_name=selected_file + ".xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# =========================================================
# Download ALL files as ONE Excel (multi-sheet)
# =========================================================
if results:
    all_excel_buffer = io.BytesIO()
    with pd.ExcelWriter(all_excel_buffer, engine="openpyxl") as writer:
        for fname, df in results.items():
            safe_sheet = re.sub(r"[\\/*?:\[\]]", "_", fname)[:31]
            df.to_excel(writer, sheet_name=safe_sheet, index=False)

    st.download_button(
        label="Download ALL files as ONE Excel (multi-sheet)",
        data=all_excel_buffer.getvalue(),
        file_name="sentiment_all_files.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# =========================================================
# OVERALL CHARTS — SIDE BY SIDE
# =========================================================
if results:
    st.subheader("Overall Comparison")

    summary_df = pd.DataFrame(summary_rows)
    col1, col2 = st.columns(2)

    # ---------- 100% stacked bar ----------
    with col1:
        st.markdown("**100% Stacked Bar**")

        perc_df = summary_df.copy()
        total = perc_df[["negative", "neutral", "positive"]].sum(axis=1)
        perc_df["negative"] = perc_df["negative"] / total * 100
        perc_df["neutral"] = perc_df["neutral"] / total * 100
        perc_df["positive"] = perc_df["positive"] / total * 100

        fig_bar, ax_bar = plt.subplots(figsize=(4, 3))
        ax_bar.bar(perc_df["file"], perc_df["negative"], label="Negative")
        ax_bar.bar(perc_df["file"], perc_df["neutral"], bottom=perc_df["negative"], label="Neutral")
        ax_bar.bar(
            perc_df["file"],
            perc_df["positive"],
            bottom=perc_df["negative"] + perc_df["neutral"],
            label="Positive"
        )

        ax_bar.set_ylabel("Percentage", fontsize=9)
        ax_bar.set_xticklabels(perc_df["file"], rotation=30, ha="right", fontsize=8)
        ax_bar.legend(fontsize=7, frameon=False)
        ax_bar.set_title("Sentiment Distribution", fontsize=10)

        st.pyplot(fig_bar)

    # ---------- Dendrogram ----------
    with col2:
        st.markdown("**Dendrogram (Similarity)**")

        if len(summary_df) > 1:
            X = summary_df[["mean", "negative", "neutral", "positive"]].values
            Z = linkage(X, method="ward")

            fig_den, ax_den = plt.subplots(figsize=(4, 3))
            dendrogram(
                Z,
                labels=summary_df["file"].values,
                orientation="right",
                leaf_font_size=8,
                ax=ax_den
            )

            ax_den.set_xlabel("Distance", fontsize=9)
            ax_den.set_title("Hierarchical Clustering", fontsize=10)

            st.pyplot(fig_den)
        else:
            st.info("Need at least 2 files for dendrogram.")


# =========================================================
# INDIVIDUAL FILE CHARTS — 4 PER ROW
# =========================================================
if results:
    st.subheader("Individual Sentiment Flows")

    file_names = list(results.keys())
    rows = [file_names[i:i+4] for i in range(0, len(file_names), 4)]

    for row_files in rows:
        cols = st.columns(4)

        for idx, fname in enumerate(row_files):
            df = results[fname]

            fig_i, ax_i = plt.subplots(figsize=(3, 2))
            ax_i.plot(range(len(df)), df["score"], linewidth=1)

            ax_i.set_ylim(-5, 5)
            ax_i.set_title(fname, fontsize=9)
            ax_i.set_xlabel("Idx", fontsize=7)
            ax_i.set_ylabel("Score", fontsize=7)
            ax_i.tick_params(axis='both', labelsize=7)

            cols[idx].pyplot(fig_i)


# =========================================================
# EXPORT ALL INDIVIDUAL CHARTS AS ONE PDF (multi-page)
# =========================================================
if results:
    pdf_buffer = io.BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        for fname, df in results.items():
            fig, ax = plt.subplots(figsize=(6, 3.5))

            ax.plot(range(len(df)), df["score"], linewidth=1.2)
            ax.set_ylim(-5, 5)
            ax.set_title(fname, fontsize=11)
            ax.set_xlabel("Sentence Index", fontsize=9)
            ax.set_ylabel("Sentiment (-5 to +5)", fontsize=9)
            ax.tick_params(axis='both', labelsize=8)

            pdf.savefig(fig, dpi=150, bbox_inches="tight")
            plt.close(fig)

    st.download_button(
        label="Download ALL individual charts as PDF",
        data=pdf_buffer.getvalue(),
        file_name="sentiment_individual_charts.pdf",
        mime="application/pdf"
    )
