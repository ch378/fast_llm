import redis
import uuid
import numpy as np

# Connexion à Redis Docker
r = redis.Redis(host="localhost", port=6379)

def save_embedding(text, embedding):
    """
    Sauvegarde le texte et son embedding dans Redis
    """
    key = f"text:{uuid.uuid4()}"
    r.hset(
        key,
        mapping={
            "text": text,
            "embedding": np.array(embedding, dtype=np.float32).tobytes()
        }
    )
    return key