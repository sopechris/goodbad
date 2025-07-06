# Progress Log: Building **goodbad**

## 📌 Overview
This document summarizes the key steps and decisions taken in the development of **goodbad**, a project aimed at exploring the concepts of good and evil across philosophy, religion, and psychology using NLP and Retrieval-Augmented Generation (RAG).

---

## 🔍 Phase 1: Philosophy Dataset

- ✅ **Source**: A well-structured and cleaned dataset from Kaggle containing the **history of philosophy**.
- ✅ **Status**: Ready-to-use — already **segmented, pruned, and cleaned**.
- ✅ **Action Taken**: No preprocessing required. Ideal dataset to perform natural language processing and trial RAG models.
- 💡 **Note**: This dataset served as the **initial playground** for model testing and pipeline validation.

---

## 🕊️ Phase 2: Religious Texts

- 🔍 **Discovery**: Found a **Kaggle dataset** with **34 major religious texts**.
- ➕ **Added**: Included the **Quran** manually to complete the set.
- 🧹 **Cleaning Strategy**:  
  - Reused the **text processing pipeline** from the **philosophy dataset creator's GitHub repo**.
  - Adapted the pipeline to fit the **unique structure** of religious texts.

- ⚠️ **Challenges**:
  - **Vedas and Hindu texts** were especially difficult to clean:
    - Contained **footnotes**, **Sanskrit phrases**, **bilingual commentary**, **reference numbers**, and **poetic hymn structures**.
  - Required **manual inspection and pipeline adjustments**.

---

## 🧠 Phase 3: Psychology Texts

- 📚 **Selection**: Identified **15 foundational works** representing core psychological schools.
- 🧲 **Source**: Downloaded from **Project Gutenberg** in `.txt` format.
- 🧼 **Cleaning Plan**:
  - Applied the **adapted cleaning pipeline** from earlier steps.
  - Stored cleaned results in `.csv` format for further processing.
  - Texts were relatively **easy to clean**, making this phase more straightforward.

---

## 🚀 Trial Run: First RAG Prototype

- ✅ **Input**: Used only the **philosophy_data.csv**.
- ⚙️ **Objective**: Test Retrieval-Augmented Generation (RAG) using a **single domain** to validate model behavior before scaling to religion and psychology.
- 🔬 **Insight**: This trial served as the **proof-of-concept** for the goodbad architecture.

---

## 📁 Next Steps

- Finalize cleaning of religious and psychology datasets.
- Merge all sources into a unified knowledge base.
- Develop a scalable and efficient RAG pipeline.
- Begin prototyping the app to present **multiple moral perspectives** on a given question.
"""
