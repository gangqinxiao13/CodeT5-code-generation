import json
from transformers import T5ForConditionalGeneration, RobertaTokenizer
model = T5ForConditionalGeneration.from_pretrained('model/code-T5-base')
tokenizer = RobertaTokenizer.from_pretrained('model/code-T5-base')
print("====================================code-T5====================================")
# params = list(model.named_parameters())
# print('The CodeT5 model has {:} different named parameters.'.format(len(params)))
# print('====================================Encoder====================================')
# print(model.encoder)
# print('====================================Decoder====================================')
# print(model.decoder)

# Count the total number of parameters
total_parameters = sum(p.numel() for p in model.parameters())
print(f"Total parameters in CodeT5: {total_parameters}")

# Get the length of the longest input sequence
max_input_length = tokenizer.model_max_length
print(f"Longest input sequence length for CodeT5: {max_input_length}")