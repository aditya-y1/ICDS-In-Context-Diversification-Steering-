# Imports
import os, time
from openai import OpenAI
from helper_functions import *
import numpy as np
import tiktoken

# hyperparameters 
LB_FUNCTION = lambda x: 55.0537634409/((x**2 + 1)**10) - 50.0537634409 # min: -50, max: 5


def print_bias_extremes(logit_bias: dict, model: str = "gpt-4o-mini", k: int = 5):
    if not logit_bias:
        print("[logit_bias] (empty)")
        return
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    items = [(int(tid), float(v)) for tid, v in logit_bias.items()]
    items.sort(key=lambda x: x[1])                       # ascending by bias
    neg = items[:k]
    pos = sorted(items, key=lambda x: x[1], reverse=True)[:k]

    def dec(tid):
        try: return enc.decode([tid])
        except Exception: return f"<{tid}>"

    print("\n[logit_bias extremes]")
    for label, group in (("most negative", neg), ("most positive", pos)):
        print(f"{label}:")
        for tid, val in group:
            print(f"  id={tid:>6}  bias={val:>6.1f}  tok={repr(dec(tid))}")



# Authentication
client = OpenAI()

# Functions

"""
Function to generate k tokens (OpenAI)
"""
def generate_k_tokens(
    prompt,
    model="gpt-4o-mini",
    temperature=0.7,
    k=16,
    logit_bias={}
):
    resp = client.completions.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=k,
        echo=False,
        logit_bias=logit_bias
    )
    return resp.choices[0].text


"""
Function to generate reseponse without steering.
Stream until we reach the finished flag from the LLM.
"""
def generate_no_steering(
    prompt,
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=500,
    finished_flag="###"
):
    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=finished_flag,              # let API stop at the flag
    )
    out = response.choices[0].text or ""
    # defensive trim if model echoed the flag anyway
    if finished_flag in out:
        out = out.split(finished_flag)[0]
    return out                           # continuation only



"""
Function to generate response with steering
"""

def generate_with_steering(
    prompt,
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=500,
    finished_flag="###",
    tokens_per_batch=16,
    timeout=120,
    next_n_vision=16,
    top_m_each=15,
    logit_bias_from_similarity = LB_FUNCTION
):
    # initialise running variables
    start = time.time()
    chunks = [prompt]
    n_chunks = len(chunks)
    max_n_chunks = len(chunks)
    current_text = prompt
    logit_bias = {}

    while (len(tokenise(current_text)) < max_tokens) and (time.time() - start < timeout) and (finished_flag not in current_text[len(prompt):]):

        # Generate the next tokens continuing the prompt
        next = generate_k_tokens(
            prompt=current_text,
            model=model,
            temperature=temperature,
            k=tokens_per_batch,
            logit_bias=logit_bias
        )
        current_text = current_text + next

        # chunk the text
        chunks = split_text_into_chunks(current_text)
        n_chunks = len(chunks)

        # if new chunk, then start the logit bias updating
        if n_chunks > max_n_chunks:
            logit_bias = {}
            max_n_chunks = n_chunks
            _, steps = greedy_with_topm(current_text, next_n_vision, top_m_each, model)
            future_tokens = [j["token"] for i in steps for j in i["top_m"]]
            chunk_vecs = get_embeddings(chunks)
            chunk_mat = np.asarray(chunk_vecs)

            for token in future_tokens:
                token_vec = get_embeddings(token)
                sims = np.array([cosine_similarity(token_vec, v) for v in chunk_mat])
                score = logit_bias_from_similarity(float(sims.mean())) # calculate the function on the average similarity
                for tid in encode_tokens(token):
                    logit_bias[tid] = score
            # print("\n".join(chunks))
            # print("--------------------------------------------")
            # print_bias_extremes(logit_bias, k=3)
            # input()


    return current_text[len(prompt):]