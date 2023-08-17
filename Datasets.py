from torch.utils.data import Dataset
from transformers import RobertaTokenizer
import pandas as pd
import urllib.request


def get_longest_sequence_length_conala():
    data = pd.read_csv('data/conala/conala-mined/conala-mined.csv')
    max_text_length = 0
    max_code_length = 0
    longest_text = ''
    longest_code = ''
    tokenizer = RobertaTokenizer.from_pretrained("model/code-T5-base")
    for i in range(len(data)):
        text = tokenizer.tokenize(data.iloc[i]["intent"])
        code = tokenizer.tokenize(data.iloc[i]["snippet"])
        if len(text) > max_text_length:
            max_text_length = len(text)
            longest_text = data.iloc[i]["intent"]
        if len(code) > max_code_length:
            max_code_length = len(code)
            longest_code = data.iloc[i]["snippet"]

    print(f"The longest text token number in conala-mined dataset is {max_text_length}")
    print(f"The longest code token number in conala-mined dataset is {max_code_length}")
    print(f"The longest text is {longest_text}")
    print(f"The longest code is {longest_code}")


def get_longest_sequence_length_api():
    data = pd.read_csv('data/pythonapi/api-mined.csv')
    max_text_length = 0
    max_code_length = 0
    longest_text = ''
    longest_code = ''
    tokenizer = RobertaTokenizer.from_pretrained("model/code-T5-base")
    for i in range(len(data)):
        text = tokenizer.tokenize(data.iloc[i]["intent"])
        code = tokenizer.tokenize(data.iloc[i]["snippet"])
        if len(text) > max_text_length:
            max_text_length = len(text)
            longest_text = data.iloc[i]["intent"]
        if len(code) > max_code_length:
            max_code_length = len(code)
            longest_code = data.iloc[i]["snippet"]

    print(f"The longest text in pythonapi-mined dataset is {max_text_length}")
    print(f"The longest code in pythonapi-mined dataset is {max_code_length}")
    print(f"The longest text is {longest_text}")
    print(f"The longest code is {longest_code}")


def get_longest_sequence_length_mbpp():
    data = pd.read_csv('data/mbpp/mbpp.csv')
    max_text_length = 0
    max_code_length = 0
    longest_text = ''
    longest_code = ''
    tokenizer = RobertaTokenizer.from_pretrained("model/code-T5-base")
    for i in range(len(data)):
        text = tokenizer.tokenize(data.iloc[i]["text"])
        code = tokenizer.tokenize(data.iloc[i]["code"])
        if len(text) > max_text_length:
            max_text_length = len(text)
            longest_text = data.iloc[i]["text"]
        if len(code) > max_code_length:
            max_code_length = len(code)
            longest_code = data.iloc[i]["code"]

    print(f"The longest text in mbpp dataset is {max_text_length}")
    print(f"The longest code in mbpp dataset is {max_code_length}")
    print(f"The longest text is {longest_text}")
    print(f"The longest code is {longest_code}")


class Conala_Dataset(Dataset):
    def __init__(self, data, text_length, code_length):
        path = 'data/conala/conala-mined/'
        self.data = pd.read_csv(path + data)
        self.tokenizer = RobertaTokenizer.from_pretrained("model/code-T5-base")
        self.text_length = text_length
        self.code_length = code_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        nl = item['intent']
        code = item['snippet']
        # Encode the nl and code
        task_prefix = "Generate Python code from natural language: "
        encoding = self.tokenizer(
            task_prefix + nl,
            padding='max_length',
            max_length=self.text_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        target_encoding = self.tokenizer(
            code,
            padding='max_length',
            max_length=self.code_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encoding.input_ids
        # replace padding token id's of the labels by -100, so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        return input_ids, attention_mask, labels


class Api_Dataset(Dataset):
    def __init__(self, data, text_length, code_length):
        path = 'data/pythonapi/'
        self.data = pd.read_csv(path + data)
        self.tokenizer = RobertaTokenizer.from_pretrained("model/code-T5-base")
        self.text_length = text_length
        self.code_length = code_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        nl = item['intent']
        code = item['snippet']

        # Encode the nl and code
        task_prefix = "Generate Python code from natural language: (from Pythonapi)"
        encoding = self.tokenizer(
            task_prefix + nl,
            padding='max_length',
            max_length=self.text_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        target_encoding = self.tokenizer(
            code,
            padding='max_length',
            max_length=self.code_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encoding.input_ids
        # replace padding token id's of the labels by -100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        return input_ids, attention_mask, labels


def download_file(url, save_path):  # Download mbpp dataset from the url
    try:
        urllib.request.urlretrieve(url, save_path)
        print("Download completed!")
    except Exception as e:
        print(f"An error occurred while downloading the file: {e}")


class Mbpp_Dataset(Dataset):
    def __init__(self, data, text_length, code_length):
        path = 'data/mbpp/'
        self.data = pd.read_csv(path + data)
        self.tokenizer = RobertaTokenizer.from_pretrained("model/code-T5-base")
        self.text_length = text_length
        self.code_length = code_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        nl = item['text']
        code = item['code']

        # Encode the nl and code
        task_prefix = "Generate Python code from natural language: (from Mbpp)"
        encoding = self.tokenizer(
            task_prefix + nl,
            padding='max_length',
            max_length=self.text_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        target_encoding = self.tokenizer(
            code,
            padding='max_length',
            max_length=self.code_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encoding.input_ids
        # replace padding token id's of the labels by -100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        return input_ids, attention_mask, labels


def test_csv_processing(input_file_path, output_file_path):  # Process test_csv
    dataset = pd.read_csv(input_file_path)
    dataset = dataset.rename(columns={'snippet': 'code', 'intent': 'text'})
    dataset = dataset[['text', 'code']]
    dataset.to_csv(output_file_path, index=False)


class Test_Dataset(Dataset):
    def __init__(self, data, task_prefix):
        self.data = pd.read_csv(data)
        self.task_prefix = task_prefix
        self.tokenizer = RobertaTokenizer.from_pretrained("model/code-T5-base")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        nl = item['text']
        input = self.tokenizer(
            [self.task_prefix + nl],
            return_tensors="pt",
            padding='max_length',
            max_length=128,
            truncation=True
        )
        return input


if __name__ == '__main__':
    # # Download mbpp.jasonl
    # download_file(url="https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl",
    #               save_path="data/mbpp/mbpp.jsonl")
    get_longest_sequence_length_conala()
    get_longest_sequence_length_api()
    get_longest_sequence_length_mbpp()
    # # Test Conala_Dataset
    # valid_dataset = Conala_Dataset('conala-mined_valid.csv')
    # print(valid_dataset.__getitem__(2))
    # # Test Api_Dataset
    # valid_dataset = Api_Dataset('api-mined_train.csv')
    # print(valid_dataset.__getitem__(3))
    # # Test Mbpp_Dataset
    # valid_dataset = Mbpp_Dataset('mbpp_valid.csv')
    # print(valid_dataset.__getitem__(4))
