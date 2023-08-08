import argparse
import matplotlib.pyplot as plt


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


def plot_loss_curves(train_loss_values, valid_loss_values):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    train_batches = range(1, len(train_loss_values) + 1)
    valid_batches = range(1, len(valid_loss_values) + 1)

    ax1.plot(train_batches, train_loss_values, label="Train Loss")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss Curve")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(valid_batches, valid_loss_values, label="Validation Loss", color='orange')
    ax2.set_xlabel("Batch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Validation Loss Curve")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    default_log_file_path = "model/conala_fine-tune/train_log/train_1e-4.log"
    parser = argparse.ArgumentParser(description="Extract train and valid loss values from log file")
    parser.add_argument("--log_file", default=default_log_file_path, help="Path to the log file")
    args = parser.parse_args()

    train_loss_values, valid_loss_values = extract_losses_from_log(args.log_file)
    plot_loss_curves(train_loss_values, valid_loss_values)


if __name__ == "__main__":
    main()

