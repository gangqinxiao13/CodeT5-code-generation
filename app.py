import os
import torch
from flask_cors import CORS
from flask import Flask, render_template, request
from transformers import RobertaTokenizer, T5ForConditionalGeneration

app = Flask(__name__)
CORS(app)

model_path = 'model/experiment_5/final_model'
assert os.path.exists(model_path), "Model path does not exist..."

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Your device is {device}.')
# Set up model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = model.to(device)
model.eval()

def generation(task_prefix, input):
    input = tokenizer(
        [task_prefix + input],
        return_tensors="pt",
        padding='max_length',
        max_length=512,
        truncation=True
    )

    outputs = model.generate(
        input_ids=input['input_ids'],
        attention_mask=input["attention_mask"],
        do_sample=False,
        max_new_tokens=512
        )
    generated_code = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return generated_code[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_option = request.form['dropdown']
        text = request.form['textarea']
        generated_code = 'Please input English enquiry!'

        if text == '':
            return render_template('index.html', input_textarea_content=text, textarea_content=generated_code)

        if selected_option == 'conala':
            generated_code = generation(task_prefix="Generate Python code from natural language:", input=text)
            return render_template('index.html', input_textarea_content=text, textarea_content=generated_code)
        elif selected_option == 'api':
            generated_code = generation(task_prefix="Generate Python code from natural language: (from Pythonapi)", input=text)
            return render_template('index.html', input_textarea_content=text, textarea_content=generated_code)
        elif selected_option == 'mbpp':
            generated_code = generation(task_prefix="Generate Python code from natural language: (from Mbpp)", input=text)
            return render_template('index.html', input_textarea_content=text, textarea_content=generated_code)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)