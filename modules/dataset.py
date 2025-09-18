from datasets import load_dataset, Audio

def load_vivos_dataset(train_csv, test_csv):
    train_ds = load_dataset("csv", data_files=train_csv)["train"]
    test_ds  = load_dataset("csv", data_files=test_csv)["train"]

    train_ds = train_ds.cast_column("path", Audio(sampling_rate=16000))
    test_ds  = test_ds.cast_column("path", Audio(sampling_rate=16000))

    return train_ds, test_ds
