from datasets import load_dataset
from pathlib import Path

def main():
    data_dir = Path("data/mmlu")
    data_dir.mkdir(parents=True, exist_ok=True)

    BIO_SUBJECTS = ["college_biology", "high_school_biology"]
    VIR_SUBJECTS = ["virology"]

    print("Downloading MMLU (all subsets)...")
    mmlu = load_dataset("cais/mmlu", "all", cache_dir=str(data_dir))

    def save_subset(subjects, name):
        out = data_dir / f"{name}.jsonl"
        if out.exists():
            print(f"{out} already exists — skipping.")
            return
        ds = mmlu["test"].filter(lambda x: x["subject"] in subjects)
        ds.to_json(out)
        print(f"Saved {name} subset → {out} ({len(ds)} entries)")

    save_subset(BIO_SUBJECTS, "biology")
    save_subset(VIR_SUBJECTS, "virology")

if __name__ == "__main__":
    main()
