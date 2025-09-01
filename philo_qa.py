import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from collections import defaultdict
import os
from PIL import Image
book_urls = {
    "A Treatise Concerning The Principles Of Human Knowledge": "https://www.gutenberg.org/cache/epub/4723/pg4723-images.html",
    "A Treatise Of Human Nature": "https://www.gutenberg.org/cache/epub/4705/pg4705-images.html",
    "Anti-Oedipus": "#",
    "Aristotle - Complete Works": "https://classics.mit.edu/Browse/browse-Aristotle.html",
    "Being And Time": "#",
    "Beyond Good And Evil": "https://www.gutenberg.org/cache/epub/4363/pg4363-images.html",
    "Capital": "https://www.marxists.org/archive/marx/works/1867-c1/",
    "Critique Of Judgement": "https://archive.org/details/immanuel-kant-critique-of-judgement",
    "Critique Of Practical Reason": "https://www.gutenberg.org/cache/epub/5683/pg5683-images.html",
    "Critique Of Pure Reason": "https://www.gutenberg.org/cache/epub/4280/pg4280-images.html",
    "Dialogues Concerning Natural Religion": "https://www.gutenberg.org/cache/epub/4583/pg4583-images.html",
    "Difference And Repetition": "#",
    "Discourse On Method": "https://www.gutenberg.org/cache/epub/59/pg59-images.html",
    "Ecce Homo": "https://www.gutenberg.org/cache/epub/52319/pg52319-images.html",
    "Elements Of The Philosophy Of Right": "https://www.marxists.org/reference/archive/hegel/works/pr/",
    "Enchiridion": "https://www.gutenberg.org/cache/epub/45109/pg45109-images.html",
    "Essay Concerning Human Understanding": "https://www.gutenberg.org/cache/epub/10615/pg10615-images.html",
    "Essential Works Of Lenin": "#",
    "Ethics": "https://www.gutenberg.org/cache/epub/3800/pg3800-images.html",
    "History Of Madness": "#",
    "Lewis - Papers": "#",
    "Meditations": "https://www.gutenberg.org/cache/epub/2680/pg2680-images.html",
    "Meditations On First Philosophy": "https://www.gutenberg.org/cache/epub/23306/pg23306-images.html",
    "Naming And Necessity": "#",
    "Off The Beaten Track": "#",
    "On Certainty": "#",
    "On The Improvement Of Understanding": "https://www.gutenberg.org/cache/epub/16800/pg16800-images.html",
    "On The Principles Of Political Economy And Taxation": "https://www.econlib.org/library/Ricardo/ricP.html",
    "Philosophical Investigations": "#",
    "Philosophical Studies": "#",
    "Philosophical Troubles": "#",
    "Plato - Complete Works": "https://classics.mit.edu/Browse/browse-Plato.html",
    "Quintessence": "#",
    "Science Of Logic": "https://www.marxists.org/reference/archive/hegel/works/sl/index.htm",
    "Second Treatise On Government": "https://www.gutenberg.org/cache/epub/7370/pg7370-images.html",
    "The Analysis Of Mind": "https://www.gutenberg.org/cache/epub/2529/pg2529-images.html",
    "The Antichrist": "https://www.gutenberg.org/cache/epub/19322/pg19322-images.html",
    "The Birth Of The Clinic": "#",
    "The Communist Manifesto": "https://www.gutenberg.org/cache/epub/61/pg61-images.html",
    "The Crisis Of The European Sciences And Phenomenology": "#",
    "The Idea Of Phenomenology": "https://archive.org/details/ideaofphenomenol00hussuoft",
    "The Logic Of Scientific Discovery": "#",
    "The Order Of Things": "#",
    "The Phenomenology Of Perception": "#",
    "The Phenomenology Of Spirit": "https://www.marxists.org/reference/archive/hegel/works/ph/phconten.htm",
    "The Problems Of Philosophy": "https://www.gutenberg.org/cache/epub/5827/pg5827-images.html",
    "The Search After Truth": "#",
    "The Second Sex": "#",
    "The System Of Ethics": "#",
    "The Wealth Of Nations": "https://www.gutenberg.org/cache/epub/3300/pg3300-images.html",
    "Theodicy": "https://archive.org/details/theodicyessayson00leibuoft",
    "Three Dialogues": "https://www.gutenberg.org/cache/epub/4723/pg4723-images.html",  # same as Treatise
    "Thus Spake Zarathustra": "https://www.gutenberg.org/cache/epub/1998/pg1998-images.html",
    "Tractatus Logico-Philosophicus": "https://www.gutenberg.org/cache/epub/5740/pg5740-images.html",
    "Twilight Of The Idols": "https://www.gutenberg.org/cache/epub/52263/pg52263-images.html",
    "Vindication Of The Rights Of Woman": "https://www.gutenberg.org/cache/epub/3420/pg3420-images.html",
    "Women, Race, And Class": "#",
    "Writing And Difference": "#"
}


# Load model and data
@st.cache_resource
def load_data():
    model = SentenceTransformer("intfloat/e5-large-v2") #may need to change this to e5

    # Absolute paths
    base_path = os.path.dirname(__file__)
    index_path = os.path.join(base_path, "philosophy_faiss.index")
    data_path = os.path.join(base_path, "philosophy_embeddings_merged.npz")

    # Load FAISS index and metadata
    index = faiss.read_index(index_path)
    data = np.load(data_path, allow_pickle=True)
    metadata = data['metadata']
    embeddings = data['embeddings']

    return model, index, metadata, embeddings

model, index, metadata, embeddings = load_data()

# Show banner image
base_path = os.path.dirname(__file__)
banner_path = os.path.join(base_path, "./logos/logo1.png")
banner = Image.open(banner_path)
st.image(banner, use_container_width=True)

# Input field with placeholder only
query = st.text_input(
    "Your Question",  # required non-empty label
    placeholder="🫣 Ask a philosophical question...",
    label_visibility="collapsed"  # hides it from UI
)


# Custom emojis for each school
school_emojis = {
    "analytic": "🔍",
    "aristotle": "🏛️",
    "capitalism": "🏦",
    "communism": "☭",
    "continental": "🎭",
    "empiricism": "🔬",
    "feminism": "♀️",
    "german_idealism": "🧠",
    "nietzsche": "🦅",
    "phenomenology": "🌀",
    "plato": "🏺",
    "rationalism": "📐",
    "stoicism": "🗿"
}
with st.sidebar:
    st.markdown("## 🧭 Filter Schools")
    st.markdown("Uncheck to hide a school from the results:")

    selected_schools = []

    for school in sorted(school_emojis.keys()):
        is_checked = st.checkbox(f"{school_emojis[school]} {school.replace('_', ' ').title()}", value=True)
        if is_checked:
            selected_schools.append(school)


st.markdown("""
<style>
.main .block-container {
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

.card-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 1.5rem;
}

.school-card {
    flex: 1 1 calc(33% - 1.5rem); /* 3 cards per row */
    border-radius: 20px;
    background-color: #fefaf5;
    padding: 1.5em;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 16px rgba(0,0,0,0.07);
    animation: fadeIn 0.6s ease-in-out;
    min-width: 300px;
}

.school-header {
    font-size: 1.25em;
    font-weight: 600;
    margin-bottom: 1em;
}

.sentence {
    margin-bottom: 1em;
    padding-left: 0.8em;
    border-left: 4px solid #ccc;
    font-size: 1rem;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(15px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)


# If user submits query
if query:
    query_vec = model.encode(["query: " + query], normalize_embeddings=True).astype("float32")
    k = 5000
    D, I = index.search(query_vec, k)

    # Collect top 2 results per school (above similarity threshold)
    school_hits = defaultdict(list)
    for dist, idx in zip(D[0], I[0]):
        m = metadata[idx]
        school = m.get('school', 'Unknown School')
        author = m.get('author', 'Unknown Author')
        book = m.get('title', 'Unknown Book')
        book_url = book_urls.get(book, '#')  # fallback if unknown
        sentence = m.get('sentence_str', 'No sentence available')
        similarity = dist
        if similarity < 0.2:
            continue
        if len(school_hits[school]) < 2:
            formatted = formatted = f'<em>“{sentence}”</em><br><a href="{book_url}" target="_blank" title="Click and Ctrl+F to search this sentence." style="text-decoration:none;">({book})</a> — {author}'
            school_hits[school].append((similarity, formatted))

    # Sort schools by highest matching sentence similarity
    # Sort by most relevant school
    schools_with_results = sorted(
        school_hits.keys(),
        key=lambda s: max(sim for sim, _ in school_hits[s]),
        reverse=True
    )

    # Create one large HTML block for all cards together
    cards_html = '<div class="card-container">'

    for school in schools_with_results:
        if school not in selected_schools:
            continue
        else:
            hits = sorted(school_hits[school], reverse=True)[:2]
            emoji = school_emojis.get(school, "📖")

            # Start card
            card_html = f"""
            <div class="school-card">
                <div class="school-header">{emoji} {school.replace('_', ' ').title()}</div>
            """

            for _, sentence in hits:
                card_html += f"""<div class="sentence">{sentence}</div>"""

            card_html += "</div>"  # close school-card
            cards_html += card_html

    cards_html += "</div>"  # close card-container

    # Render all at once
    st.markdown(cards_html, unsafe_allow_html=True)
