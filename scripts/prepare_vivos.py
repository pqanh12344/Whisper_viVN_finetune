import csv
from pathlib import Path

def build_csv(split):
    base = Path("data/vivos") / split
    wav_dir = base / "waves"
    txt_path = base / "prompts.txt"
    out_csv = f"vivos_{split}.csv"

    mapping = {}
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            utt_id, text = line.strip().split(" ", 1)
            mapping[utt_id] = text

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "text"])
        for wav in wav_dir.rglob("*.wav"):
            utt_id = wav.stem
            if utt_id in mapping:
                writer.writerow([str(wav), mapping[utt_id]])

for split in ["train", "test"]:
    build_csv(split)
print("Created vivos_train.csv and vivos_test.csv")
