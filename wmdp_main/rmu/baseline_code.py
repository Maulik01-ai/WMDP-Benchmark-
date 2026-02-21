
# Colab WMDP–RMU Experiment (Zephyr-7B + MMLU + Wikitext)

!pip install -q "transformers[torch]>=4.39" "accelerate>=0.30" "datasets>=2.14" bitsandbytes peft matplotlib

import os, json, csv, torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model_name = "HuggingFaceH4/zephyr-7b-beta"
DATA_DIR = "/content"      # Uploaded folder path
BASE_SAVE = "/content/models/base_model"
RMU_SAVE  = "/content/models/rmu_all_wmdp"
os.makedirs(BASE_SAVE, exist_ok=True)
os.makedirs(RMU_SAVE, exist_ok=True)

use_bnb = True
bnb_config = None
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    print(" bitsandbytes found — using 4-bit quantization.")
except Exception as e:
    print(" bitsandbytes unavailable, falling back to float32:", e)
    use_bnb = False

print(f" Loading {model_name} on {device} …")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if use_bnb:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
else:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()
print(" Model loaded successfully!\n")

# Save base model
tokenizer.save_pretrained(BASE_SAVE)
model.save_pretrained(BASE_SAVE)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

wmdp_bio   = load_json(f"{DATA_DIR}/bio_questions.json")
wmdp_chem  = load_json(f"{DATA_DIR}/chem_questions.json")
wmdp_cyber = load_json(f"{DATA_DIR}/cyber_questions.json")
print(f" Loaded {len(wmdp_bio)} Bio, {len(wmdp_chem)} Chem, {len(wmdp_cyber)} Cyber questions.\n")

wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")
mmlu_bio  = load_dataset("cais/mmlu", "college_biology", split="test[:100]")
mmlu_viro = load_dataset("cais/mmlu", "virology", split="test[:100]")

# Convert HF datasets to list of dicts
def dataset_to_dict_list(ds):
    return [dict(zip(ds.column_names, row)) for row in zip(*[ds[c] for c in ds.column_names])]

mmlu_bio  = dataset_to_dict_list(mmlu_bio)
mmlu_viro = dataset_to_dict_list(mmlu_viro)

def extract_qa_fields(ex):
    q = ex.get("question", ex.get("input", ""))
    a = ex.get("answer", "unknown")
    if isinstance(a, int) and "choices" in ex:
        ch = ex["choices"]
        a = ch[a] if a < len(ch) else "unknown"
    return q, str(a)

def eval_accuracy(dataset, prefix, limit=20, use_model=None):
    m = use_model or model
    correct = total = 0
    subset = dataset[:limit]
    print(f"🔹 Evaluating {prefix} ({len(subset)} samples)…")
    for i, ex in enumerate(subset):
        if i % 5 == 0: print(f"→ {i}/{len(subset)}")
        q, ref = extract_qa_fields(ex)
        if not q:
            continue
        prompt = f"Question: {q}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(m.device)
        with torch.no_grad():
            out = m.generate(**inputs, max_new_tokens=50)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        if ref.lower() in text.lower():
            correct += 1
        total += 1
    acc = correct / total if total else 0
    print(f" {prefix} accuracy = {acc:.3f}\n")
    return acc

print("\n Base Model Evaluation")
base_bio   = eval_accuracy(wmdp_bio,   "WMDP-Bio")
base_chem  = eval_accuracy(wmdp_chem,  "WMDP-Chem")
base_cyber = eval_accuracy(wmdp_cyber, "WMDP-Cyber")
base_mmlu_bio  = eval_accuracy(mmlu_bio,  "MMLU-Biology")
base_mmlu_viro = eval_accuracy(mmlu_viro, "MMLU-Virology")


from peft import LoraConfig, get_peft_model

peft_cfg = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.15, bias="none", task_type="CAUSAL_LM"
)
peft_model = get_peft_model(model, peft_cfg)

def add_rmu_text(e):
    return {"text": f"This text is irrelevant to hazardous materials or biology. Forget all hazardous, biological, and chemical knowledge. {e['text']}"}

wiki_rmu = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10%]")
wiki_rmu = wiki_rmu.map(add_rmu_text)

tok_rmu = wiki_rmu.map(
    lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=256)
)

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
train_args = TrainingArguments(
    output_dir=RMU_SAVE,
    per_device_train_batch_size=1,
    num_train_epochs=4,        # Increased from 1 → 4
    learning_rate=1e-4,        # Stronger forgetting rate
    save_strategy="epoch",
    logging_steps=200,
    report_to="none",
)

trainer = Trainer(model=peft_model, args=train_args,
                  data_collator=collator, train_dataset=tok_rmu)

print("\n Running STRONG RMU fine-tuning (4 epochs, Wikitext 10%) …")
trainer.train()
trainer.save_model(RMU_SAVE)
tokenizer.save_pretrained(RMU_SAVE)
print(f" Strong RMU model saved → {RMU_SAVE}\n")


rmu_model = AutoModelForCausalLM.from_pretrained(RMU_SAVE).to(device)
rmu_model.eval()

print(" RMU Model Evaluation\n")
rmu_bio   = eval_accuracy(wmdp_bio,   "WMDP-Bio (RMU)",   use_model=rmu_model)
rmu_chem  = eval_accuracy(wmdp_chem,  "WMDP-Chem (RMU)",  use_model=rmu_model)
rmu_cyber = eval_accuracy(wmdp_cyber, "WMDP-Cyber (RMU)", use_model=rmu_model)
rmu_mmlu_bio  = eval_accuracy(mmlu_bio,  "MMLU-Biology (RMU)",  use_model=rmu_model)
rmu_mmlu_viro = eval_accuracy(mmlu_viro, "MMLU-Virology (RMU)", use_model=rmu_model)

summary = [
    ["Dataset","Base Acc","RMU Acc"],
    ["WMDP-Bio",   base_bio,   rmu_bio],
    ["WMDP-Chem",  base_chem,  rmu_chem],
    ["WMDP-Cyber", base_cyber, rmu_cyber],
    ["MMLU-Bio",   base_mmlu_bio,  rmu_mmlu_bio],
    ["MMLU-Viro",  base_mmlu_viro, rmu_mmlu_viro],
]
csv_path = "/content/results_summary.csv"
with open(csv_path,"w",newline="") as f:
    csv.writer(f).writerows(summary)
print(f" Results saved → {csv_path}\n")

labels = [r[0] for r in summary[1:]]
base   = [r[1] for r in summary[1:]]
rmu    = [r[2] for r in summary[1:]]
plt.figure(figsize=(8,5))
plt.barh(labels, base, label="Base Model", alpha=0.6)
plt.barh(labels, rmu,  label="RMU Model", alpha=0.6)
plt.xlabel("Accuracy")
plt.title(" WMDP + MMLU — Base vs RMU Accuracy")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("/content/results_chart.png", dpi=300, bbox_inches="tight")
plt.show()
