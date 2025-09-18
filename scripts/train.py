import yaml
from modules.dataset import load_vivos_dataset
from modules.preprocessing import prepare_example
from modules.model import load_model
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

def main():
    # load config
    with open("configs/training_config.yaml") as f:
        cfg = yaml.safe_load(f)

    # load data
    train_ds, test_ds = load_vivos_dataset(cfg["train_csv"], cfg["test_csv"])

    # load model + processor
    model, processor = load_model(cfg["model_name"], freeze_encoder=True)

    # preprocess
    train_ds = train_ds.map(lambda x: prepare_example(x, processor), remove_columns=train_ds.column_names)
    test_ds  = test_ds.map(lambda x: prepare_example(x, processor), remove_columns=test_ds.column_names)

    # training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        fp16=cfg["fp16"],
        logging_steps=cfg["logging_steps"],
        evaluation_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        predict_with_generate=True,
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

if __name__ == "__main__":
    main()
