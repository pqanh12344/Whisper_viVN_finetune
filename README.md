# Whisper Vietnamese Fine-tune (VIVOS dataset)

Dự án này giúp bạn fine-tune [OpenAI Whisper Large v3](https://huggingface.co/openai/whisper-large-v3) cho **nhận dạng giọng nói tiếng Việt** bằng cách sử dụng [VIVOS Corpus](https://www.kaggle.com/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr).

---

## 📂 Cấu trúc project

whisper-vietnamese-finetune/ <br>
│── data/ # chứa dataset (Kaggle VIVOS) <br>
│ └── vivos/... <br>
│ <br>
│── configs/ <br>
│ └── training_config.yaml # cấu hình huấn  <br>
│ <br>
│── scripts/ <br>
│ ├── download_data.sh # tải dữ liệu từ Kaggle <br>
│ ├── prepare_vivos.py # xử lý dữ liệu thành CSV <br>
│ ├── train.py # fine-tune Whisper <br>
│ ├── evaluate.py # đánh giá WER/CER <br>
│ └── inference.py # chạy inference <br>
│ <br>
│── modules/ <br>
│ ├── dataset.py # load dataset <br>
│ ├── preprocessing.py # xử lý dữ liệu đầu vào <br>
│ ├── model.py # load Whisper + freeze encoder <br>
│ └── utils.py # hàm phụ trợ <br>
│ <br>
│── requirements.txt <br>
└── README.md <br>


---

## 🚀 Cài đặt

```bash
git clone https://github.com/yourusername/whisper-vietnamese-finetune.git
cd whisper-vietnamese-finetune
pip install -r requirements.txt
```

## 🏋️ Huấn luyện

Chạy lệnh:

python scripts/train.py

Checkpoint sẽ được lưu tại ./checkpoints/whisper-vn.



## 🎙️ Inference

Thử chạy inference với audio mới (sample.wav):

python scripts/inference.py


Ví dụ output:

xin chào các bạn

## 📊 Đánh giá

Sau khi huấn luyện, có thể tính WER/CER với script evaluate.py.

from jiwer import wer
wer(refs, hyps)
