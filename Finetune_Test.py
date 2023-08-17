import torch
from Datasets import Test_Dataset
from torch.utils.data import DataLoader
import pandas as pd
from Logger import Logger
import sys
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import metric.code_bert_score as code_bert_score


def generation(model, tokenizer, task_prefix, test_file_path, output_file_path, save_log_path):
    test_dataset = Test_Dataset(data=test_file_path, task_prefix=task_prefix)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    generation = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    sys.stdout = Logger(save_log_path, sys.stdout)
    for i, batch in enumerate(test_dataloader):
        input_ids = batch['input_ids'].squeeze(1)
        input_ids = input_ids.to(device)
        attention_mask = batch["attention_mask"].squeeze(1)
        attention_mask = attention_mask.to(device)
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=128
        )
        generated_code = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generation += generated_code
        print(f'Batch: {i+1}/{len(test_dataloader)}. First generated code of this batch: {[generated_code[0]]}')

    test_dataset_df = pd.read_csv(test_file_path)
    test_dataset_df['generation'] = None
    test_dataset_df['generation'] = generation
    test_dataset_df.to_csv(output_file_path, index=False)


def evaluation_bleu(generation_file_path, output_file_path, save_log_path):
    bleu = []
    onegram_bleu = []
    twogram_bleu = []
    threegram_bleu = []
    fourgram_bleu = []

    sys.stdout = Logger(save_log_path, sys.stdout)
    # Define smoothing function
    smooth_func = SmoothingFunction().method1
    test_dataset_df = pd.read_csv(generation_file_path)
    for i in range(len(test_dataset_df)):
        reference = test_dataset_df.iloc[i]['code']
        generation = test_dataset_df.iloc[i]['generation']
        code_tokenized = [reference.split()]
        generation_tokenized = generation.split()

        # Calculate BLEU score with smoothing
        bleu_score = sentence_bleu(code_tokenized, generation_tokenized,
                                   weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_func)

        onegram_bleu_score = sentence_bleu(code_tokenized, generation_tokenized,
                                            weights=(1, 0, 0, 0), smoothing_function=smooth_func)
        twogram_bleu_score = sentence_bleu(code_tokenized, generation_tokenized,
                                           weights=(0, 1, 0, 0), smoothing_function=smooth_func)
        threegram_bleu_score = sentence_bleu(code_tokenized, generation_tokenized,
                                           weights=(0, 0, 1, 0), smoothing_function=smooth_func)
        fourgram_bleu_score = sentence_bleu(code_tokenized, generation_tokenized,
                                           weights=(0, 0, 0, 1), smoothing_function=smooth_func)

        bleu.append(bleu_score)
        onegram_bleu.append(onegram_bleu_score)
        twogram_bleu.append(twogram_bleu_score)
        threegram_bleu.append(threegram_bleu_score)
        fourgram_bleu.append(fourgram_bleu_score)

    # Calculate the average BLEU score
    average_bleu = sum(bleu) / (len(bleu))
    print(f'The average bleu score is: {average_bleu}')

    average_onegram_bleu = sum(onegram_bleu) / (len(onegram_bleu))
    print(f'The average 1-gram bleu score is: {average_onegram_bleu}')

    average_twogram_bleu = sum(twogram_bleu) / (len(twogram_bleu))
    print(f'The average 2-gram bleu score is: {average_twogram_bleu}')

    average_threegram_bleu = sum(threegram_bleu) / (len(threegram_bleu))
    print(f'The average 3-gram bleu score is: {average_threegram_bleu}')

    average_fourgram_bleu = sum(fourgram_bleu) / (len(fourgram_bleu))
    print(f'The average 4-gram bleu score is: {average_fourgram_bleu}')

    # Add to the csv
    test_dataset_df['bleu'] = bleu
    test_dataset_df['1-gram bleu'] = onegram_bleu
    test_dataset_df['2-gram bleu'] = twogram_bleu
    test_dataset_df['3-gram bleu'] = threegram_bleu
    test_dataset_df['4-gram bleu'] = fourgram_bleu

    test_dataset_df.to_csv(output_file_path, index=False)


def evaluation_codebertscore(generation_file_path, output_file_path, save_log_path):

    sys.stdout = Logger(save_log_path, sys.stdout)
    test_dataset_df = pd.read_csv(generation_file_path)

    references = test_dataset_df['code'].tolist()
    generations = test_dataset_df['generation'].tolist()
    text = test_dataset_df['text'].tolist()
    precision, recall, F1, F3 = code_bert_score.score(cands=generations, refs=references, lang='python', sources=text)
    precision = precision.tolist()
    recall = recall.tolist()
    F1 = F1.tolist()
    F3 = F3.tolist()

    average_precision = sum(precision)/len(precision)
    print(f'The average precision is: {average_precision}')
    average_recall = sum(recall)/len(recall)
    print(f'The average recall is: {average_recall}')
    average_f1 = sum(F1)/len(F1)
    print(f'The average F1 is: {average_f1}')
    average_f3 = sum(F3)/len(F3)
    print(f'The average F3 is: {average_f3}')

    # Add to the csv
    test_dataset_df['precision'] = precision
    test_dataset_df['recall'] = recall
    test_dataset_df['F1'] = F1
    test_dataset_df['F3'] = F3

    test_dataset_df.to_csv(output_file_path)
