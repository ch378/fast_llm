from sentence_transformers import SentenceTransformer

# Modèle pour générer les embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embedding(text: str):
    """
    Encode le texte en embedding et retourne un vecteur list
    """
    return model.encode(text).tolist()