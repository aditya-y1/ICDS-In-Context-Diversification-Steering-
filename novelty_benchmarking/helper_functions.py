from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from openai import OpenAI
import tiktoken, os, numpy as np


# Authentication
client = OpenAI()


embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = SemanticChunker(
    embedding,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=25 # change this percentile value to be lower for smaller chunks
    )

def split_text_into_chunks(
    text: str,
    model_name: str = "all-MiniLM-L6-v2",
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: int = 25
):
    """
    Splits a given text into semantic chunks using HuggingFace embeddings.

    Args:
        text (str): The input text to split.
        model_name (str): HuggingFace embedding model name.
        breakpoint_threshold_type (str): Method for determining split points. 
            Options: "percentile" or "standard_deviation".
        breakpoint_threshold_amount (int): Threshold value to adjust chunk sizes.
            Lower values => smaller, more granular chunks.

    Returns:
        List[str]: A list of semantically meaningful text chunks.
    """

    # Split the text into chunks
    chunks = text_splitter.split_text(text)

    return chunks

def tokenise(text: str, model: str = "gpt-4o-mini"):
    """
    Tokenizes text for given model using tiktoken.
    """
    encoding = tiktoken.encoding_for_model(model)
    token_ids = encoding.encode(text)
    return token_ids

def greedy_with_topm(prompt: str, n: int, m: int, model):
    """
    Greedy-generate n tokens and capture the top-m alternatives at each step.
    Uses Chat Completions (temperature=0) so the logprobs schema is consistent.

    Returns:
        text: str                      # the generated text
        steps: list[dict]              # per-token details:
            [{
                "step": int,
                "taken_token": str,
                "taken_logprob": float,
                "top_m": [{"token": str, "logprob": float, "bytes": list[int]}]
            }, ...]
    """
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,            # greedy
        max_tokens=n,
        logprobs=True,            # ask for token-level logprobs
        top_logprobs=m            # and top-m alternatives per position
    )

    choice = resp.choices[0]
    text = choice.message.content or ""

    # Defensive: logprobs may be None in rare cases; handle gracefully
    steps = []
    lp = getattr(choice, "logprobs", None)
    content_lp = getattr(lp, "content", None) if lp is not None else None

    if content_lp:
        for i, step in enumerate(content_lp, start=1):
            taken_token = step.token
            taken_logprob = step.logprob
            # Top-m alternatives (may include the taken token as well)
            topm = []
            if step.top_logprobs:
                for t in step.top_logprobs:
                    topm.append({
                        "token": t.token,
                        "logprob": t.logprob,
                        "bytes": t.bytes
                    })
            steps.append({
                "step": i,
                "taken_token": taken_token,
                "taken_logprob": taken_logprob,
                "top_m": topm
            })

    return text, steps


def encode_tokens(text: str, model: str = "gpt-4o-mini"):
    """
    Converts text into token IDs for a given OpenAI model.

    Args:
        text (str): The input text to tokenize.
        model (str): OpenAI model name (e.g., "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo").

    Returns:
        list: List of integer token IDs.
    """
    try:
        # Load the encoding for the specified model
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for unknown models
        print(f"Model '{model}' not found in tiktoken; using cl100k_base as fallback.")
        encoding = tiktoken.get_encoding("cl100k_base")

    return encoding.encode(text)




# -------------------------------------------------------------------------------------------------------------------- #


# Initialize the embedding model once (recommended)

def get_embeddings(texts):
    """
    Generate embeddings for a string or list of strings using HuggingFaceEmbeddings.

    Args:
        texts (str or list): Input text or list of texts to embed.

    Returns:
        np.ndarray or list[np.ndarray]: A single embedding vector (np.ndarray)
                                        or a list of vectors for multiple texts.
    """
    if isinstance(texts, str):
        return np.array(embedding.embed_query(texts), dtype=np.float32)
    elif isinstance(texts, list):
        return [np.array(vec, dtype=np.float32) for vec in embedding.embed_documents(texts)]
    else:
        raise TypeError("Input must be a string or list of strings.")


# ---- Utility: cosine similarity -------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1D vectors.
    Returns a float in [-1, 1].
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)

