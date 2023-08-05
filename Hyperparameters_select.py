from transformers import RobertaTokenizer, T5ForConditionalGeneration
from Datasets import test_csv_processing
from Finetune_Test import generation, evaluation_bleu, evaluation_codebertscore

if __name__ == '__main__':
    test_csv_processing(input_file_path='data/conala/conala-mined/conala-mined_test.csv',
                        output_file_path='model/conala_fine-tune/test_processing.csv')

    tokenizer = RobertaTokenizer.from_pretrained('model/conala_fine-tune/learning_rate_1e-4/checkpoint_epoch_30')
    model = T5ForConditionalGeneration.from_pretrained('model/conala_fine-tune/learning_rate_1e-4/checkpoint_epoch_30')

    generation(
        tokenizer=tokenizer,
        model=model,
        task_prefix='Generate code from natural language: ',
        test_file_path='model/conala_fine-tune/test_processing.csv',
        output_file_path='model/conala_fine-tune/test_generation_1e-4.csv',
        save_log_path='model/conala_fine-tune/test_log/test_1e-4.log'
    )

    tokenizer = RobertaTokenizer.from_pretrained('model/conala_fine-tune/learning_rate_3e-4/checkpoint_epoch_30')
    model = T5ForConditionalGeneration.from_pretrained('model/conala_fine-tune/learning_rate_3e-4/checkpoint_epoch_30')

    generation(
        tokenizer=tokenizer,
        model=model,
        task_prefix='Generate code from natural language: ',
        test_file_path='model/conala_fine-tune/test_processing.csv',
        output_file_path='model/conala_fine-tune/test_generation_3e-4.csv',
        save_log_path='model/conala_fine-tune/test_log/test_3e-4.log'
    )

    evaluation_bleu(
        generation_file_path='model/conala_fine-tune/test_generation_1e-4.csv',
        output_file_path='model/conala_fine-tune/test_evaluation_1e-4.csv',
        save_log_path='model/conala_fine-tune/test_log/test_1e-4.log'
    )

    evaluation_bleu(
        generation_file_path='model/conala_fine-tune/test_generation_3e-4.csv',
        output_file_path='model/conala_fine-tune/test_evaluation_3e-4.csv',
        save_log_path='model/conala_fine-tune/test_log/test_3e-4.log'
    )

    evaluation_codebertscore(
        generation_file_path='model/conala_fine-tune/test_evaluation_1e-4.csv',
        output_file_path='model/conala_fine-tune/test_evaluation_1e-4.csv',
        save_log_path='model/conala_fine-tune/test_log/test_1e-4.log'
    )

    evaluation_codebertscore(
        generation_file_path='model/conala_fine-tune/test_evaluation_3e-4.csv',
        output_file_path='model/conala_fine-tune/test_evaluation_3e-4.csv',
        save_log_path='model/conala_fine-tune/test_log/test_3e-4.log'
    )