import argparse
import asyncio
import bisect
import functools
import json
import os

import datasets
import numpy as np
import torch
from aiofiles import open as aio_open
from datasets import load_dataset
from pydantic import BaseModel
from tqdm.asyncio import tqdm

# Authentication
os.environ["OPENAI_API_KEY"] = "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"

# --- Try to import transformers only if we need local RM ---
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CONCURRENT_REQUESTS = 1

reward_thresholds = [
    -7.71875,
    -6.28125,
    -6.0,
    -5.71875,
    -5.5,
    -5.0,
    -4.375,
    -3.4375,
    -2.046875,
]

def transform_raw_reward(reward: float) -> int:
    # score of 1 to 10
    return bisect.bisect_left(reward_thresholds, reward) + 1

# -------------- LOCAL RM MODE (original) --------------
@functools.cache
def rm_and_tokenizer():
    """
    WARNING: This loads a huge model by default. On Mac (no CUDA),
    prefer using --judge gpt4 to avoid downloads & FlashAttention.
    """
    model_name = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # Disable FlashAttention on Mac/CPU or when you don't have flash_attn:
        attn_implementation="eager",
        num_labels=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return rm, tokenizer

class Rating(BaseModel):
    rating: int

@torch.inference_mode()
async def score_partition_rm(prompt: str, generations: list[str], partition: list[int]):
    """Asynchronously scores the partition using a local reward model."""
    rm, tokenizer = rm_and_tokenizer()
    convs = [
        [
            {"content": prompt, "role": "user"},
            {"content": generation, "role": "assistant"},
        ]
        for generation in generations
    ]
    batch = tokenizer.apply_chat_template(
        convs,
        tokenize=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    ).to(rm.device)

    with torch.no_grad():
        raw_rewards = rm(**batch).logits[:, 0].tolist()
    scores = [transform_raw_reward(r) for r in raw_rewards]

    generation_scores = []
    partition_scores = []
    for s, p in zip(scores, partition, strict=False):
        if p == len(partition_scores):
            generation_scores.append(s)
            partition_scores.append(s)
        else:
            generation_scores.append(0)

    assert len(partition_scores) == (max(partition) + 1), (
        f"partition_scores: {partition_scores}, partition: {partition}"
    )
    return generation_scores, partition_scores

# -------------- OPENAI JUDGE MODE (new) --------------
# Requires: pip install openai ; export OPENAI_API_KEY=...
from openai import OpenAI
openai_client = OpenAI()

EVAL_SYSTEM_PROMPT = (
    "You are a strict evaluator. Given a task prompt and a model response, "
    "output ONLY a JSON object with keys: score (float 0.0-1.0) and rationale (short string). "
    "Score for correctness, helpfulness, and instruction-following."
)

async def score_partition_openai(prompt: str, generations: list[str], partition: list[int], model: str):
    """
    Scores generations with an OpenAI model (e.g., gpt-4o or gpt-4o-mini) to avoid local downloads.
    Returns (generation_scores[0..n-1], partition_scores[0..k-1]) with 1..10 scale to match RM mode.
    """
    # Ask OpenAI to score each generation independently
    # To keep things simple and within rate limits, do them sequentially here (CONCURRENT_REQUESTS handles outer tasks).
    raw_scores_0to1 = []
    for generation in generations:
        out = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": f"PROMPT:\n{prompt}\n\nRESPONSE:\n{generation}\n\nReturn JSON only."}
            ],
            temperature=0.0,
            max_tokens=256,
        )
        text = out.choices[0].message.content.strip()
        try:
            j = json.loads(text)
            s = float(j.get("score", 0.0))
        except Exception:
            s = 0.0  # fallback if judge output malformed
        raw_scores_0to1.append(s)

    # Map 0..1 to 1..10 to stay consistent with transform_raw_reward buckets
    # (simple linear mapping; tweak if you prefer different scaling)
    scores_1to10 = [int(min(10, max(1, round(s * 10)))) for s in raw_scores_0to1]

    generation_scores = []
    partition_scores = []
    for s, p in zip(scores_1to10, partition, strict=False):
        if p == len(partition_scores):
            generation_scores.append(s)
            partition_scores.append(s)
        else:
            generation_scores.append(0)

    assert len(partition_scores) == (max(partition) + 1), (
        f"partition_scores: {partition_scores}, partition: {partition}"
    )
    return generation_scores, partition_scores

# -------------- SHARED PIPELINE --------------
async def process_instances(instances, output_file, patience, judge_mode, openai_model):
    """Processes all instances concurrently and writes results to a file."""
    # Skip if already complete
    if os.path.exists(output_file):
        try:
            existing_output = load_dataset("json", data_files=output_file, split="train")
            if not set(instances["id"]) - set(existing_output["id"]):
                print("All prompts are scored. Skipping.")
                return
        except datasets.exceptions.DatasetGenerationError:
            pass

    async with aio_open(output_file, "w", buffering=1) as f:
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

        async def process_single_instance(instance):
            async with semaphore:
                if judge_mode == "rm":
                    generation_scores, partition_scores = await score_partition_rm(
                        instance["prompt"], instance["generations"], instance["partition"]
                    )
                else:
                    generation_scores, partition_scores = await score_partition_openai(
                        instance["prompt"], instance["generations"], instance["partition"], model=openai_model
                    )

                utility = np.average(
                    generation_scores,
                    weights=patience ** np.arange(len(instance["generations"])),
                )
                return {
                    **instance,
                    "generation_scores": generation_scores,
                    "partition_scores": partition_scores,
                    "utility": float(utility),
                }

        tasks = [process_single_instance(instance) for instance in instances]
        for result in tqdm(await asyncio.gather(*tasks), total=len(instances)):
            await f.write(json.dumps(result) + "\n")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, help="Directory with partitions.jsonl")
    parser.add_argument("--patience", type=float, default=0.8, help="Discount factor for cumulative utility")
    parser.add_argument(
        "--judge",
        choices=["rm", "gpt4", "gpt4o-mini"],
        default="rm",
        help="Scoring backend: 'rm' uses local reward model; 'gpt4' or 'gpt4o-mini' uses OpenAI API."
    )
    args = parser.parse_args()

    instances = load_dataset("json", data_files=os.path.join(args.eval_dir, "partitions.jsonl"), split="train")
    os.makedirs(args.eval_dir, exist_ok=True)
    output_file = os.path.join(args.eval_dir, "scores.jsonl")

    openai_model = "gpt-4o" if args.judge == "gpt4" else "gpt-4o-mini"
    await process_instances(instances, output_file, args.patience, judge_mode=args.judge, openai_model=openai_model)

if __name__ == "__main__":
    asyncio.run(main())
