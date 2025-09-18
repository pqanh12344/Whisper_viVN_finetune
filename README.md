# Whisper Vietnamese Fine-tune (VIVOS dataset)

Dự án này giúp bạn fine-tune [OpenAI Whisper Large v3](https://huggingface.co/openai/whisper-large-v3) cho **nhận dạng giọng nói tiếng Việt** bằng cách sử dụng [VIVOS Corpus](https://www.kaggle.com/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr).

---

## 📂 Cấu trúc project

```bash
whisper-vietnamese-finetune/
│── data/ # chứa dataset (Kaggle VIVOS)
│ └── vivos/...
│
│── configs/
│ └── training_config.yaml # cấu hình huấn 
│
│── scripts/
│ ├── prepare_vivos.py # xử lý dữ liệu thành CSV
│ ├── train.py # fine-tune Whisper
│ └── inference.py # chạy inference
│
│── modules/
│ ├── dataset.py # load dataset
│ ├── preprocessing.py # xử lý dữ liệu đầu vào
│ ├── model.py # load Whisper + freeze encoder
│
│── requirements.txt
└── README.md
```

---

## 🚀 Cài đặt

```bash
git clone https://github.com/yourusername/whisper-vietnamese-finetune.git
cd whisper-vietnamese-finetune
pip install -r requirements.txt
```

## 🏋️ Huấn luyện

Chạy lệnh:
```bash
python scripts/train.py
```
Checkpoint sẽ được lưu tại ./checkpoints/whisper-vn.



## 🎙️ Inference

Thử chạy inference với audio mới (sample.wav):
```bash
python scripts/inference.py
```

Ví dụ output:
```bash
xin chào các bạn
```

## 📊 Đánh giá

Sau khi huấn luyện, có thể tính WER/CER với script evaluate.py.
```bash
from jiwer import wer
wer(refs, hyps)
```
