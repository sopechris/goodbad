# Progress Log: Building **goodbad**

## Overview  
This document tracks the key steps and decisions in building **goodbad**, a project aiming to explore concepts of good and evil through philosophy, religion, and psychology using NLP and Retrieval-Augmented Generation (RAG).

---

## Phase 1: Philosophy Dataset

I started by searching for datasets and was fortunate to find a well-prepared philosophy dataset on Kaggle. It was already cleaned, segmented, and pruned — perfect for testing NLP techniques right away. This dataset became my initial playground to experiment with RAG and to validate the processing pipeline.

---

## Phase 2: Religious Texts

Next, I sought to replicate the same approach with religious texts. I found a comprehensive Kaggle dataset containing 34 major religious books and added the Quran to complete the collection. To clean these texts, I adapted the pipeline from the philosophy dataset creator’s GitHub.  

However, some texts, especially the Vedas and Hindu scriptures, posed serious challenges. They contained complex elements like footnotes, Sanskrit verses alongside English commentary, and numerous reference markers. Cleaning these required careful manual inspection and multiple pipeline adjustments.

---

## Phase 3: Psychology Texts

For psychology, I selected 15 foundational works representing the main schools of thought. I downloaded these one by one from Project Gutenberg in plain text format. These texts were generally easier to clean, so I applied the adapted pipeline and stored the results in CSV files for downstream processing.

---

## Trial Run: First RAG Prototype

With the philosophy dataset cleaned and ready, I ran a first trial of a Retrieval-Augmented Generation model using only this data. This proof-of-concept demonstrated the viability of the approach before scaling to include religious and psychological texts.  

First RAG with overall embeddings, not with per school tokenization. See if it works, if yes, continue with per school tokenization.
    Seems to work, I also added in religion, and concluded religion needs to be further cleaned (not ready for use).

---

## Trial Run: First Streamlit Prototype if possible

---

## 📁 Next Steps

- Finalize cleaning of religious and psychology datasets.
- Merge all sources into a unified knowledge base.
- Develop a scalable and efficient RAG pipeline.
- Begin prototyping the app to present **multiple moral perspectives** on a given question.
"""
