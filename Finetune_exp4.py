from transformers import T5ForConditionalGeneration, RobertaTokenizer
from Finetune_Train import fine_tune
from Datasets import Api_Dataset, Mbpp_Dataset
from Finetune_Test import generation, evaluation_bleu, evaluation_codebertscore

if __name__ == '__main__':
    Api_train_dataset = Api_Dataset('api-mined_train.csv', text_length=256, code_length=256)
    Api_valid_dataset = Api_Dataset('api-mined_valid.csv', text_length=256, code_length=256)
    fine_tune(start_epoch=1,
              end_epoch=2,
              learning_rate=1e-4,
              train_dataset=Api_train_dataset,
              valid_dataset=Api_valid_dataset,
              batch_size=16,
              pretrained_model_path='model/conala_fine-tune/learning_rate_1e-4/checkpoint_epoch_30',
              save_model_path='model/experiment_4/pythonapi_fine-tune',
              save_strategy=100,
              save_log_path='model/experiment_4/pythonapi_fine-tune/train_log/train.log'
              )

    Mbpp_train_dataset = Mbpp_Dataset('mbpp_train.csv', text_length=128, code_length=128)
    Mbpp_valid_dataset = Mbpp_Dataset('mbpp_valid.csv', text_length=128, code_length=128)
    fine_tune(start_epoch=1,
              end_epoch=20,
              learning_rate=1e-4,
              train_dataset=Mbpp_train_dataset,
              valid_dataset=Mbpp_valid_dataset,
              batch_size=8,
              pretrained_model_path='model/experiment_4/pythonapi_fine-tune/checkpoint_epoch_2',
              save_model_path='model/experiment_4/mbpp_fine-tune',
              save_strategy=20,
              save_log_path='model/experiment_4/mbpp_fine-tune/train_log/train.log')

    tokenizer = RobertaTokenizer.from_pretrained('model/experiment_4/final_model')
    model = T5ForConditionalGeneration.from_pretrained('model/experiment_4/final_model')

    generation(
        tokenizer=tokenizer,
        model=model,
        task_prefix='Generate code from natural language: ',
        test_file_path='data/conala/conala-mined/test_processing.csv',
        output_file_path='data/conala/conala-mined/experiment_4/test_generation.csv',
        save_log_path='data/conala/conala-mined/experiment_4/test_log/test.log'
    )

    generation(
        tokenizer=tokenizer,
        model=model,
        task_prefix="Generate code from natural language: (from Pythonapi)",
        test_file_path='data/pythonapi/test_processing.csv',
        output_file_path='data/pythonapi/experiment_4/test_generation.csv',
        save_log_path='data/pythonapi/experiment_4/test_log/test.log'
    )

    generation(
        tokenizer=tokenizer,
        model=model,
        task_prefix="Generate code from natural language: (from Mbpp)",
        test_file_path='data/mbpp/mbpp_test.csv',
        output_file_path='data/mbpp/experiment_4/test_generation.csv',
        save_log_path='data/mbpp/experiment_4/test_log/test.log'
    )

    evaluation_bleu(generation_file_path='data/conala/conala-mined/experiment_4/test_generation.csv',
                             output_file_path='data/conala/conala-mined/experiment_4/test_evaluation.csv',
                             save_log_path='data/conala/conala-mined/experiment_4/test_log/test.log')

    evaluation_bleu(generation_file_path='data/pythonapi/experiment_4/test_generation.csv',
                    output_file_path='data/pythonapi/experiment_4/test_evaluation.csv',
                    save_log_path='data/pythonapi/experiment_4/test_log/test.log')

    evaluation_bleu(generation_file_path='data/mbpp/experiment_4/test_generation.csv',
                    output_file_path='data/mbpp/experiment_4/test_evaluation.csv',
                    save_log_path='data/mbpp/experiment_4/test_log/test.log')

    evaluation_codebertscore(generation_file_path='data/conala/conala-mined/experiment_4/test_evaluation.csv',
                             output_file_path='data/conala/conala-mined/experiment_4/test_evaluation.csv',
                             save_log_path='data/conala/conala-mined/experiment_4/test_log/test.log')

    evaluation_codebertscore(generation_file_path='data/pythonapi/experiment_4/test_evaluation.csv',
                             output_file_path='data/pythonapi/experiment_4/test_evaluation.csv',
                             save_log_path='data/pythonapi/experiment_4/test_log/test.log')

    evaluation_codebertscore(generation_file_path='data/mbpp/experiment_4/test_evaluation.csv',
                             output_file_path='data/mbpp/experiment_4/test_evaluation.csv',
                             save_log_path='data/mbpp/experiment_4/test_log/test.log')