# Progress Log: Building **goodbad**



### Philosophy Dataset

I started by searching for datasets and was fortunate to find a well-prepared philosophy dataset on Kaggle. It was already cleaned, segmented, and pruned — perfect for testing NLP techniques right away. This dataset became my initial playground to experiment with RAG and to validate the processing pipeline.

---

### Religious Texts

Next, I sought to replicate the same approach with religious texts. I found a comprehensive Kaggle dataset containing 34 major religious books and added the Quran to complete the collection. To clean these texts, I adapted the pipeline from the philosophy dataset creator’s GitHub.  

However, some texts, especially the Vedas and Hindu scriptures, posed serious challenges. They contained complex elements like footnotes, Sanskrit verses alongside English commentary, and numerous reference markers. Cleaning these required careful manual inspection and multiple pipeline adjustments. Looked for information, discarded a bunch of books not that much in our scope (regarding occultism, zoroastrianism...); and added a bunch from important philosophies (confucianism, tao, sikhism, tanakh).

Sentences with many capital letters and no philosophical meaning were some to be cleaned; for that we put a threshold on percentage of capital starting words and philosophical words. With that, many phrases were discarded.

Ad-Hoc cleaning looking at things to remove (headers, notes, footnotes, chapter names...). Longest part, had to go over all texts one by one, left a couple untouched to see if fixed. Checked whether previos pipeline took care of them and for those who didn't integrated to philosophy cleaning pipeline. Bible and Quran already cleaned .csv, separate from pipeline and put after.

Post cleaning ad hoc extra cleaning. After deeply checking .csv and spotting extra slipped noise, added new extra cleaning to the notebook and rerun to get final .csv. Extra cleaning of a couple phrases I had noted before in the first .csv check.

Even with all this, still had to clean a bunch, I identified the problem that the religious texts usually have a ton of names and some sentences were lacking philosophy and were generating noise (something like: "Jesus said to Micah that he should call Mark"). So I sropped massively and filetered aggresively for these kinds of phrases to get the dropped_phrases.csv (columns:[sentence, school, reason]). Inserted this to Claude Sonnet 5 and let it modify the aggresive script to not filter core and useful phrases. Re-ran the previous .csv through the new filter and done.

---

### Psychology Texts

For psychology, I selected 15 foundational works representing the main schools of thought. I downloaded these one by one from Project Gutenberg in plain text format. These texts were generally easier to clean, so I applied the adapted pipeline and stored the results in CSV files for downstream processing.

Ad Hoc cleaning

---

### Trial Run: First Prototype

With the philosophy dataset cleaned and ready, I ran a first trial. Mi idea was to first do an analysis to see how good the datasets were, and if they aligned with what I expected. Philosophy seems to get ideas straight, but concluded religion needs to be further cleaned (not ready for use), didn't try psychology. Did some further analysis in analysis.ipynb, and plotted the data using PCA and other techniques.

First try of concept with overall embeddings, not with per school tokenization. See if it works, if yes, continue with per school tokenization.
    Seems to work, but a bit slower

---

### Trial Run: First Streamlit Prototype if possible

Got the first streamlit prototype up and running (upgraded logo1.png for greater quality and did some designing). A bit slow when loading the page, but pretty fast afterwards.

Did the sidebar, and tried a bunch of options but finally went for the checked list. In the future, I'd like to get dropdowns for religion and psychology, as well as philosophy.

---

Realized embeddings were not as good as expected with "all-MiniLM-L6-v2"; so decided to go for a better sentence transformer, one more designed towards qa and more sophisticated "intfloat/e5-large-v2", a huggingface model by intfloat (.npz files went from 400Mb to >1Gb). Got a lot better results. 

Had to download the model, to do proper paralelization in the cluster. Parallelized over 50 nodes and got it in 1h. 

---

Work to get some useful relationship for the app with the PCA and UMAP projections etc. and correlations between religion vs philosophy. 
Idea after cleaning religion and psychology texts is first, to get all the texts in my own repo or something (probably github).    
Line redirect to text and specific line.  
Work with RAG. Get a LLM to work:  

- Toggle checked list on sideline to get an overall answer with the checked options  
- Do a match between philo/religion/psycho to see how much they align

Prototype with Streamlit, then move to Django/Vercel

### Extra Ideas

- Phrase of the day
- Problem of the day
    - Comments and likes
- Personality/philosophy test
---

## 📁 Next Steps

- Finalize cleaning of religious and psychology datasets.
- Merge all sources into a unified knowledge base.
- Develop a scalable and efficient RAG pipeline.
- Begin prototyping the app to present **multiple moral perspectives** on a given question.
"""
