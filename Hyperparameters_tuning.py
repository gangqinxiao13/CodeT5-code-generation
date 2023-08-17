import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from Datasets import test_csv_processing
from Finetune_Test import generation, evaluation_bleu, evaluation_codebertscore


def extract_losses_from_log(log_file_path):
    train_loss_list = []
    valid_loss_list = []

    with open(log_file_path, "r") as log_file:
        for line in log_file:
            if "Train: batch" in line:
                train_loss = float(line.split("Average loss:")[1].split(",")[0].strip())
                train_loss_list.append(train_loss)
            elif "Valid: batch" in line:
                valid_loss = float(line.split("Average loss:")[1].split(",")[0].strip())
                valid_loss_list.append(valid_loss)

    return train_loss_list, valid_loss_list


def plot_conala_loss_curves(train_loss_values1, train_loss_values2, valid_loss_values1, valid_loss_values2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    train_batches = range(1, len(train_loss_values1) + 1)
    valid_batches = range(1, len(valid_loss_values1) + 1)

    ax1.plot(train_batches, train_loss_values1, label="Train Loss 1e-4")
    ax1.plot(train_batches, train_loss_values2, label="Train Loss 3e-4", color='red', linestyle='dashed')
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss Curve")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(valid_batches, valid_loss_values1, label="Valid Loss 1e-4")
    ax2.plot(valid_batches, valid_loss_values2, label="Valid Loss 3e-4", color='red', linestyle='dashed')
    ax2.set_xlabel("Batch")
    ax2.set_title("Validation Loss Curve")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_conala_main():
    train_loss_values1, valid_loss_values1 = extract_losses_from_log("model/conala_fine-tune/train_log/train_1e-4.log")
    train_loss_values2, valid_loss_values2 = extract_losses_from_log("model/conala_fine-tune/train_log/train_3e-4.log")
    plot_conala_loss_curves(train_loss_values1, train_loss_values2, valid_loss_values1, valid_loss_values2)


def plot_exp_loss_curves(train_loss_values1, train_loss_values2, train_loss_values3, train_loss_values4, train_loss_values5,
                        valid_loss_values1, valid_loss_values2, valid_loss_values3, valid_loss_values4, valid_loss_values5):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    train_batches = range(1, len(train_loss_values1) + 1)
    valid_batches = range(1, len(valid_loss_values1) + 1)

    ax1.plot(train_batches, train_loss_values1, label="Train Loss Exp1", linestyle='dashed')
    ax1.plot(train_batches, train_loss_values2, label="Train Loss Exp2", color='red', linestyle='dashed')
    ax1.plot(train_batches, train_loss_values3, label="Train Loss Exp3", color='yellow', linestyle='dashed')
    ax1.plot(train_batches, train_loss_values4, label="Train Loss Exp4", color='green', linestyle='dashed')
    ax1.plot(train_batches, train_loss_values5, label="Train Loss Exp5", color='pink', linestyle='dashed')
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss Curve")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(valid_batches, valid_loss_values1, label="Valid Loss Exp1", linestyle='dashed')
    ax2.plot(valid_batches, valid_loss_values2, label="Valid Loss Exp2", color='red', linestyle='dashed')
    ax2.plot(valid_batches, valid_loss_values3, label="Valid Loss Exp3", color='yellow', linestyle='dashed')
    ax2.plot(valid_batches, valid_loss_values4, label="Valid Loss Exp4", color='green', linestyle='dashed')
    ax2.plot(valid_batches, valid_loss_values5, label="Valid Loss Exp5", color='pink', linestyle='dashed')
    ax2.set_xlabel("Batch")
    ax2.set_title("Validation Loss Curve")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_pythonapi_main():
    train_loss_values1, valid_loss_values1 = extract_losses_from_log(
        'model/experiment_1/pythonapi_fine-tune/train_log/train.log')
    train_loss_values2, valid_loss_values2 = extract_losses_from_log(
        'model/experiment_2/pythonapi_fine-tune/train_log/train.log')
    train_loss_values3, valid_loss_values3 = extract_losses_from_log(
        'model/experiment_3/pythonapi_fine-tune/train_log/train.log')
    train_loss_values4, valid_loss_values4 = extract_losses_from_log(
        'model/experiment_4/pythonapi_fine-tune/train_log/train.log')
    train_loss_values5, valid_loss_values5 = extract_losses_from_log(
        'model/experiment_5/pythonapi_fine-tune/train_log/train.log')
    plot_exp_loss_curves(train_loss_values1,train_loss_values2,train_loss_values3,train_loss_values4,train_loss_values5,
                         valid_loss_values1,valid_loss_values2,valid_loss_values3,valid_loss_values4,valid_loss_values5)


def plot_mbpp_main():
    train_loss_values1, valid_loss_values1 = extract_losses_from_log(
        'model/experiment_1/mbpp_fine-tune/train_log/train.log')
    train_loss_values2, valid_loss_values2 = extract_losses_from_log(
        'model/experiment_2/mbpp_fine-tune/train_log/train.log')
    train_loss_values3, valid_loss_values3 = extract_losses_from_log(
        'model/experiment_3/mbpp_fine-tune/train_log/train.log')
    train_loss_values4, valid_loss_values4 = extract_losses_from_log(
        'model/experiment_4/mbpp_fine-tune/train_log/train.log')
    train_loss_values5, valid_loss_values5 = extract_losses_from_log(
        'model/experiment_5/mbpp_fine-tune/train_log/train.log')
    plot_exp_loss_curves(train_loss_values1, train_loss_values2, train_loss_values3, train_loss_values4,
                         train_loss_values5,
                         valid_loss_values1, valid_loss_values2, valid_loss_values3, valid_loss_values4,
                         valid_loss_values5)

if __name__ == '__main__':
    plot_conala_main()
    plot_pythonapi_main()
    plot_mbpp_main()
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