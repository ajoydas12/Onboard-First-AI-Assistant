# scrape.py
import requests
from bs4 import BeautifulSoup
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Scrape the website (simplified example for one page)
URL = "https://www.occamsadvisory.com/"
response = requests.get(URL)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract clean text from meaningful tags
text_chunks = [p.get_text(strip=True) for p in soup.find_all(['p', 'h1', 'h2', 'li']) if p.get_text(strip=True)]

# Save raw text for fallback
with open('knowledge_base.json', 'w') as f:
    json.dump(text_chunks, f)

# 2. Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2') # Lightweight and effective model
embeddings = model.encode(text_chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

# 3. Build and save FAISS index
d = embeddings.shape[1] # Dimension of vectors
index = faiss.IndexFlatL2(d)
index.add(embeddings)
faiss.write_index(index, 'knowledge_base.index')

print("Scraping and indexing complete.")