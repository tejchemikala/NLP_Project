
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ToxicityDetector:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Load the sentence encoder model
        self.model = SentenceTransformer(model_name)
        self.toxic_desc = "This text includes harmful, offensive, or abusive language."
        self.benign_desc = "This text is respectful and contains no offensive content."
    def encode(self, texts):
        return self.model.encode(texts)
    def is_toxic(self, text):
        embeddings = self.encode([text, self.toxic_desc, self.benign_desc])
        sim_toxic = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        sim_benign = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
        return sim_toxic > sim_benign

detector = ToxicityDetector()
test_text = "I hate you!"
if detector.is_toxic(test_text):
    print("The text is considered toxic.")
else:
    print("The text is not considered toxic.")
