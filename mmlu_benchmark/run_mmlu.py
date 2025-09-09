from datasets import load_dataset, DatasetDict
from generate import generate_with_steering, generate_no_steering
import re, json
from typing import Optional, Tuple, List

LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}

def _norm(s: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation-ish edges."""
    return re.sub(r"\s+", " ", s.strip().lower())

def parse_model_answer(output: str, choices: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse the model's output to extract (letter, choice_text).
    Returns (None, None) if nothing is found confidently.
    """
    if not output:
        return None, None

    # 1) Easiest case: "Answer: X" where X is a letter (A/B/C/D) — allow punctuation after.
    m = re.search(r"(?i)answer\s*:\s*([ABCD])\b", output)
    if m:
        letter = m.group(1).upper()
        return letter, choices[LETTER_TO_IDX[letter]]

    # 2) Try to grab the chunk after "Answer:" up to a terminator (### or newline)
    m = re.search(r"(?is)answer\s*:\s*(.+?)(?:###|\n|$)", output)
    if m:
        chunk = m.group(1).strip()
        # 2a) Look for an initial letter pattern in that chunk (e.g., "C", "C)", "C.")
        m2 = re.match(r"^\s*([ABCD])\s*[\).:]?\s*", chunk, flags=re.I)
        if m2:
            letter = m2.group(1).upper()
            return letter, choices[LETTER_TO_IDX[letter]]

        # 2b) Otherwise, try to match the chunk to one of the full choice texts.
        norm_chunk = _norm(chunk)
        norm_choices = [_norm(c) for c in choices]
        # Exact normalized match
        for idx, nc in enumerate(norm_choices):
            if norm_chunk == nc or nc in norm_chunk or norm_chunk in nc:
                letter = "ABCD"[idx]
                return letter, choices[idx]

    # 3) As a final fallback, pick the first standalone A/B/C/D mentioned anywhere.
    m = re.search(r"\b([ABCD])\b", output, flags=re.I)
    if m:
        letter = m.group(1).upper()
        return letter, choices[LETTER_TO_IDX[letter]]

    # Nothing confident found.
    return None, None


SUBJECTS = [
    "abstract_algebra", 
    "college_computer_science", 
    "high_school_computer_science", 
    "college_mathematics", 
    "high_school_mathematics", 
    "elementary_mathematics"
    ]

# Example: concatenate the test split across all subjects
from datasets import concatenate_datasets
test_splits = [load_dataset("cais/mmlu", s, split="test") for s in SUBJECTS]
mmlu_test_all = concatenate_datasets(test_splits)

prompts = []
subjects = []
model_answers = []
model_answer_letters = []
true_answers = []

# querying
for i in mmlu_test_all:
    options_str = f"A: {i["choices"][0]}, B: {i["choices"][1]}, C: {i["choices"][2]}, D: {i["choices"][3]}"
    prompt = i["question"] + "\n" + "options: " + options_str + "\nWrite your answer in the form: Answer: <insert answer letter here> ###.\nKeep your reasoning to a few sentences."
    true_ans = i["choices"][i["answer"]]
    true_letter = "ABCD"[i["answer"]]

    resp = generate_no_steering(
        prompt,
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=500,
        finished_flag="###",
        # tokens_per_batch=16,
        # timeout=120,
        # next_n_vision=16,
        # top_m_each=15,
        # logit_bias_from_similarity = lambda x: 50.0488758553/((x**2 + 1)**10) - 50.0488758553 # min: -50, max: 0
    )
    # model_output = <string returned by your generation function>
    letter, choice_text = parse_model_answer(resp, i["choices"])
    prompts.append(prompt)
    subjects.append(i["subject"])
    model_answers.append(choice_text)
    model_answer_letters.append(letter)
    true_answers.append(true_ans)
    
    with open("mmlu_results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "subject": i["subject"],
            "prompt": prompt,
            "true_letter": true_letter,
            "true_answer": true_ans,
            "pred_letter": letter,
            "pred_answer": choice_text,
            "response": resp
        }, ensure_ascii=False) + "\n")




# right after, run python summarize_mmlu_jsonl.py --input mmlu_results.jsonl --outdir summary_out
