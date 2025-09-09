# scripts/run_custom_inference.py
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import trange

from generate import generate_with_steering, generate_no_steering

def run_subset(subset: str, out_dir: Path, num_generations: int = 10):
    """
    subset: "curated" or "wildchat"
    Writes out_dir/generations.jsonl in the schema NoveltyBench expects.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load NoveltyBench prompts from Hugging Face
    # curated has ~100 items; wildchat ~1,000 items.
    ds = load_dataset("yimingzhang/novelty-bench", split=subset)
    # ds = ds.select(range(10)) # if we want a subset of the dataset

    out_path = out_dir / "generations.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i in trange(len(ds), desc=f"Generating ({subset})"):
            ex = ds[i]
            # The dataset exposes 'id' and 'prompt'
            ex_id = ex["id"]
            prompt = ex["prompt"]

            prompt = prompt + "\nWhen finished, write exactly ###. I.e., write three hash charachers when finished."

            gens = []
            for _ in range(num_generations):
                text = generate_with_steering(
                    prompt=prompt,
                    model="gpt-4o-mini",
                    temperature=0.7,
                    max_tokens=500,
                    finished_flag="###",
                )
                gens.append(text)

            row = {"example_id": ex_id, "prompt": prompt, "generations": gens}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path}")

def main():
    # Where NoveltyBench expects files for the pipeline
    base = Path("results")
    # Feel free to change "steering" to your run name
    run_name = "steering"

    # 1) NB-Curated
    run_subset("curated", base / "curated" / run_name, num_generations=5)

    # 2) NB-WildChat (optional; comment out if you only want curated first)
    # run_subset("wildchat", base / "wildchat" / run_name, num_generations=5)

if __name__ == "__main__":
    main()
