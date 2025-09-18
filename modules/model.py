from transformers import WhisperProcessor, WhisperForConditionalGeneration

def load_model(model_name, freeze_encoder=True):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    if freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
    return model, processor
