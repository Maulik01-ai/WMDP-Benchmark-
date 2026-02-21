import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
RMU_PATH     = "./rmu_model"   # from previous cell

MMLU_SUBJECTS = [
    "college_biology",
    "virology",
    "college_computer_science",
    "computer_security",
]

CHOICE_LETTERS = ["A", "B", "C", "D"]

print("Using device:", DEVICE)

print("Loading cais/mmlu (all config)…")
mmlu_all = load_dataset("cais/mmlu", "all")
mmlu_test = mmlu_all["test"]   # has fields: question, subject, choices, answer

print("Total MMLU test rows:", len(mmlu_test))

def build_mmlu_prompt(question, choices, subject_name):
    text = f"The following are multiple choice questions (with answers) about {subject_name}.\n\n"
    text += question.strip() + "\n"
    for lab, ch in zip(CHOICE_LETTERS, choices):
        text += f"{lab}. {ch}\n"
    text += "Answer:"
    return text

@torch.no_grad()
def predict_choice_index(tokenizer, model, question, choices, subject_name):
    prompt = build_mmlu_prompt(question, choices, subject_name)
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = model(**enc)
    logits = out.logits[0, -1, :]
    token_ids = [
        tokenizer(l, add_special_tokens=False).input_ids[0]
        for l in CHOICE_LETTERS
    ]
    scores = logits[token_ids].float().cpu().numpy()
    return int(np.argmax(scores))

def subject_nice_name(subject_key):
    return subject_key.replace("_", " ")

def eval_mmlu_model(model_name_or_path, tag):
    print(f"\n=== Loading {tag} model from: {model_name_or_path} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
    )
    model.to(DEVICE)
    model.eval()

    results = {}
    total_correct = 0
    total_count   = 0

    for subject in MMLU_SUBJECTS:
        ds_sub = mmlu_test.filter(lambda ex, s=subject: ex["subject"] == s)
        n = len(ds_sub)
        correct = 0

        for ex in ds_sub:
            q        = ex["question"]
            choices  = ex["choices"]
            gold_idx = int(ex["answer"])   # already 0..3

            pred_idx = predict_choice_index(
                tokenizer, model, q, choices, subject_nice_name(subject)
            )
            if pred_idx == gold_idx:
                correct += 1

        acc = 100.0 * correct / n
        results[subject] = acc
        total_correct += correct
        total_count   += n

        print(f"{tag} {subject:26s}: {acc:6.2f}% (n={n})")

    mean_4 = float(np.mean(list(results.values())))
    micro  = 100.0 * total_correct / total_count
    print(f"{tag} mean over 4 subjects      : {mean_4:6.2f}%")
    print(f"{tag} micro-avg over all samples: {micro:6.2f}%")

    # free GPU memory for safety
    del model
    torch.cuda.empty_cache()

    return results, mean_4

base_task, base_mean = eval_mmlu_model(MODEL_NAME, "BASE")
rmu_task,  rmu_mean  = eval_mmlu_model(RMU_PATH, "RMU")

print("\nMMLU 4-subject results:")
print("BASE per-subject :", base_task, "| mean =", base_mean)
print("RMU  per-subject :", rmu_task,  "| mean =", rmu_mean)

# Optional: compare drops with WMDP results from previous cell
def drop(old, new): return float(old - new)

bio_drop   = drop(wmdp_base["bio"],   wmdp_rmu["bio"])
chem_drop  = drop(wmdp_base["chem"],  wmdp_rmu["chem"])
cyber_drop = drop(wmdp_base["cyber"], wmdp_rmu["cyber"])
mmlu_drop  = drop(base_mean,          rmu_mean)

print("\n=== Summary of drops (hazard vs benign) ===")
print(f"WMDP-Bio   drop: {bio_drop:6.2f} points")
print(f"WMDP-Chem  drop: {chem_drop:6.2f} points")
print(f"WMDP-Cyber drop: {cyber_drop:6.2f} points")
print(f"MMLU-4 mean drop: {mmlu_drop:6.2f} points")
