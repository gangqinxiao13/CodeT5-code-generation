from transformers import T5ForConditionalGeneration
from torch.utils.data import DataLoader
import torch
import sys
from Logger import Logger
from Datasets import Conala_Dataset


def fine_tune(start_epoch, end_epoch, learning_rate,
              train_dataset, valid_dataset, batch_size,
              pretrained_model_path, save_model_path,
              save_strategy, save_log_path):
    # Create the dataset and dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the pre-trained T5 model
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
    model = model.to(device)
    # Set up the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    sys.stdout = Logger(save_log_path, sys.stdout)
    # Training loop
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        print(f"Epoch: {epoch}/{end_epoch}")
        # Training
        total_loss = 0
        batch_num = 1
        for batch in train_dataloader:
            # Load input_ids, attention_mask and labels from batch
            input_ids, attention_mask, labels = batch
            # Resize the tensor dimension for adapting (batch_size, sequence_length)
            input_ids = input_ids.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
            labels = labels.squeeze(1)
            # Pass the data into the device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            # Initialize the gradient to zero,and the derivative of loss with respect to weight becomes zero
            optimizer.zero_grad()
            # Call forward function: pass the data into the model,and propagate forward to calculate the predicted value
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # Calculate loss
            loss = outputs.loss
            # Backpropagation for gradient
            loss.backward()
            # Update all parameters of the model
            optimizer.step()

            total_loss += loss.item()

            if (batch_num % 10 == 0) or (batch_num == len(train_dataloader)):
                print(f'Train: batch: {batch_num}/{len(train_dataloader)}, Average loss:{total_loss/batch_num}, Current loss:{loss.item()}')

            # Every 'save_strategy' times, model is saved once.
            if (batch_num % save_strategy == 0) or (batch_num == len(train_dataloader)):
                print("Model is saving.....")
                model.save_pretrained(f"{save_model_path}/checkpoint_epoch_{epoch}")
                print(f"Saved as '{save_model_path}/checkpoint_epoch_{epoch}'")
            batch_num += 1
        train_loss = total_loss / len(train_dataloader)
        print(f"Train Loss: {train_loss}")

        # Validation
        model.eval()
        total_loss = 0
        batch_num = 1
        for batch in valid_dataloader:
            input_ids, attention_mask, labels = batch

            input_ids = input_ids.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
            labels = labels.squeeze(1)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

            if batch_num % 10 == 0 :
                print(f'Valid: batch: {batch_num}/{len(valid_dataloader)}, Average loss:{total_loss/batch_num}, Current Loss:{loss.item()}')
            batch_num += 1
        valid_loss = total_loss / len(valid_dataloader)
        print(f"Valid Loss: {valid_loss}")

        # Save the current model after each epoch
        print("Model is saving.....")
        model.save_pretrained(f"{save_model_path}/checkpoint_epoch_{epoch}")
        print(f"Saved as '{save_model_path}/checkpoint_epoch_{epoch}'")
        # Clear the GPU cache
        torch.cuda.empty_cache()


if __name__ == '__main__':
    Conala_train_dataset = Conala_Dataset('conala-mined_train.csv', text_length=128, code_length=128)
    Conala_valid_dataset = Conala_Dataset('conala-mined_valid.csv', text_length=128, code_length=128)
    fine_tune(start_epoch=1,
              end_epoch=30,
              learning_rate=1e-4,
              train_dataset=Conala_train_dataset,
              valid_dataset=Conala_valid_dataset,
              batch_size=32,
              pretrained_model_path='model/code-T5-base',
              save_model_path='model/conala_fine-tune/learning_rate_1e-4',
              save_strategy=500,
              save_log_path='model/conala_fine-tune/train_log/train_1e-4.log'
              )

    Conala_train_dataset = Conala_Dataset('conala-mined_train.csv', text_length=128, code_length=128)
    Conala_valid_dataset = Conala_Dataset('conala-mined_valid.csv', text_length=128, code_length=128)
    fine_tune(start_epoch=1,
              end_epoch=30,
              learning_rate=3e-4,
              train_dataset=Conala_train_dataset,
              valid_dataset=Conala_valid_dataset,
              batch_size=32,
              pretrained_model_path='model/code-T5-base',
              save_model_path='model/conala_fine-tune/learning_rate_3e-4',
              save_strategy=500,
              save_log_path='model/conala_fine-tune/train_log/train_3e-4.log'
              )
