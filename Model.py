import transformers
from transformers import T5Model, T5ForConditionalGeneration, RobertaTokenizer, PretrainedConfig, AutoModelForSeq2SeqLM


def t5_architect():
    model = T5Model.from_pretrained('model/T5')
    print("====================================T5-encoder====================================")
    print(model.encoder)
    print("====================================T5-decoder====================================")
    print(model.decoder)


def codet5_architect():
    model = T5ForConditionalGeneration.from_pretrained('model/code-T5-base')
    print(model)
    print("====================================CodeT5-encoder====================================")
    print(model.encoder)
    print("====================================CodeT5-decoder====================================")
    print(model.decoder)


def codet5_attributes():
    model = T5ForConditionalGeneration.from_pretrained('model/code-T5-base')
    tokenizer = RobertaTokenizer.from_pretrained('model/code-T5-base')
    # Count the total number of parameters
    total_parameters = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in CodeT5: {total_parameters}")

    # Get the length of the longest input sequence
    max_input_length = tokenizer.model_max_length
    print(f"Longest input sequence length for CodeT5: {max_input_length}")


if __name__ == '__main__':
    # t5_architect()
    codet5_architect()
    # codet5_attributes()
