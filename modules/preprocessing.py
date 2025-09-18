def prepare_example(batch, processor):
    audio = batch["path"]["array"]
    inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    with processor.as_target_processor():
        labels = processor.tokenizer(batch["text"], return_tensors="pt").input_ids
    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = labels[0]
    return batch
