import sys


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()
    def close(self):
        self.log.close()

    def flush(self):
        pass


def remove_blank_lines(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        for line in lines:
            if line.strip():  # Check if line is not blank
                file.write(line)

    print("Blank lines removed successfully.")


if __name__ == '__main__':
    # sys.stdout = Logger('output.log', sys.stdout)
    # sys.stderr = Logger('output.log', sys.stderr)		# redirect std err, if necessary

    # Delete the blank lines in the .log files.
    remove_blank_lines(input_file='model/conala_fine-tune/train_log/train.log',
                       output_file='model/conala_fine-tune/train_log/train_clean.log')
    remove_blank_lines(input_file='model/experiment_1/pythonapi_fine-tune/train_log/train.log',
                       output_file='model/experiment_1/pythonapi_fine-tune/train_log/train_clean.log')
    remove_blank_lines(input_file='model/experiment_1/mbpp_fine-tune/train_log/train.log',
                       output_file='model/experiment_1/mbpp_fine-tune/train_log/train_clean.log')