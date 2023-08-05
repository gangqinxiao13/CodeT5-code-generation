import pandas as pd
import json

def conalamined_jsonl_to_dataframe(filename):
    data = []
    with open(filename, 'r') as file:
        # Read JSONL file line by line
        for line in file:
            # Parse each line as JSON
            json_data = json.loads(line)
            data.append(json_data)

    # Create a DataFrame from the list of JSON objects
    df = pd.DataFrame(data)

    # Delete some columns
    columns_drop = ["parent_answer_post_id", "prob", "id", "question_id"]
    df = df.drop(columns_drop, axis=1)
    # Save the dataframe as csv file
    df.to_csv('data/conala/conala-mined/conala-mined.csv', index=False)

    # Detect if there is any 'None' value in df. If there is, delete the data_processing.
    df = pd.read_csv('data/conala/conala-mined/conala-mined.csv')
    df.dropna(axis=0, inplace=True)
    # Save the dataframe as csv file
    df.to_csv('data/conala/conala-mined/conala-mined.csv', index=False)
    # df=pd.read_csv('conala/conala-mined/conala-mined.csv')
    # print(df.isnull().value_counts())

def apimined_jsonl_to_dataframe(filename):
    data = []
    with open(filename, 'r') as file:
        # Read JSONL file line by line
        for line in file:
            # Parse each line as JSON
            json_data = json.loads(line)
            data.append(json_data)

    # Create a DataFrame from the list of JSON objects
    df = pd.DataFrame(data)

    # Delete some columns
    columns_drop = ["question_id"]
    df = df.drop(columns_drop, axis=1)
    # Save the dataframe as csv file
    df.to_csv('data/pythonapi/api-mined.csv', index=False)

    # Detect if there is any 'None' value in df. If there is, delete the data_processing.
    df = pd.read_csv('data/pythonapi/api-mined.csv')
    df.dropna(axis=0, inplace=True)
    # Save the dataframe as csv file
    df.to_csv('data/pythonapi/api-mined.csv', index=False)

def mbpp_jsonl_to_dataframe(filename):
    data = []
    with open(filename, 'r') as file:
        # Read JSONL file line by line
        for line in file:
            # Parse each line as JSON
            json_data = json.loads(line)
            data.append(json_data)

    # Create a DataFrame from the list of JSON objects
    df = pd.DataFrame(data)

    # Delete some columns
    columns_drop = ["task_id", "test_setup_code", "test_list", "challenge_test_list"]
    df = df.drop(columns_drop, axis=1)
    # Save the dataframe as csv file
    df.to_csv('data/mbpp/mbpp.csv', index=False)

    # Detect if there is any 'None' value in df. If there is, delete the data_processing.
    df = pd.read_csv('data/mbpp/mbpp.csv')
    df.dropna(axis=0, inplace=True)
    # Save the dataframe as csv file
    df.to_csv('data/mbpp/mbpp.csv', index=False)

def split_dataset(file_path, train_csv, valid_csv, test_csv, train_ratio=0.8, valid_ratio=0.1):
    # Read the input CSV file
    df = pd.read_csv(file_path)

    # Shuffle the DataFrame randomly
    df = df.sample(frac=1, random_state=42)

    # Calculate the number of rows for each dataset
    num_rows = len(df)
    train_rows = int(num_rows * train_ratio)
    valid_rows = int(num_rows * valid_ratio)

    # Split the DataFrame into train, validation, and test datasets
    train_df = df[:train_rows]
    valid_df = df[train_rows:train_rows + valid_rows]
    test_df = df[train_rows + valid_rows:]

    # Save the datasets as separate CSV files
    train_df.to_csv(train_csv, index=False)
    valid_df.to_csv(valid_csv, index=False)
    test_df.to_csv(test_csv, index=False)

if __name__  == '__main__':
    # Conala dataset processing
    conalamined_jsonl_to_dataframe('data/conala/conala-mined/conala-mined.jsonl')
    apimined_jsonl_to_dataframe('data/pythonapi/api-mined.jsonl')
    mbpp_jsonl_to_dataframe('data/mbpp/mbpp.jsonl')
    conala_data_file_path = 'data/conala/conala-mined/conala-mined.csv'
    conala_train_data_path = 'data/conala/conala-mined/conala-mined_train.csv'
    conala_valid_data_path = 'data/conala/conala-mined/conala-mined_valid.csv'
    conala_test_data_path = 'data/conala/conala-mined/conala-mined_test.csv'
    split_dataset(file_path=conala_data_file_path, train_csv=conala_train_data_path,
                  valid_csv=conala_valid_data_path, test_csv=conala_test_data_path)
    # Api dataset processing
    api_data_file_path = 'data/pythonapi/api-mined.csv'
    api_train_data_path = 'data/pythonapi/api-mined_train.csv'
    api_valid_data_path = 'data/pythonapi/api-mined_valid.csv'
    api_test_data_path = 'data/pythonapi/api-mined_test.csv'
    split_dataset(file_path=api_data_file_path, train_csv=api_train_data_path,
                  valid_csv=api_valid_data_path, test_csv=api_test_data_path)
    # Mbpp dataset processing
    mbpp_data_file_path = 'data/mbpp/mbpp.csv'
    mbpp_train_data_path = 'data/mbpp/mbpp_train.csv'
    mbpp_valid_data_path = 'data/mbpp/mbpp_valid.csv'
    mbpp_test_data_path = 'data/mbpp/mbpp_test.csv'
    split_dataset(file_path=mbpp_data_file_path, train_csv=mbpp_train_data_path,
                  valid_csv=mbpp_valid_data_path, test_csv=mbpp_test_data_path)