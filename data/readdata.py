import jsonlines
import json

def readjsonl(filename):
    length = 0
    with open(filename, 'r+') as file:
        for item in jsonlines.Reader(file):
            print(item)
            length += 1
    print(f'Lengths of the dataset: {length}')

def readjson(filename):
    length = 0
    with open(filename, 'r', encoding='utf-8') as file:
        for item in json.load((file)):
            print(item)
            length += 1
        print(f'Lengths of the dataset: {length}')

if __name__ == '__main__':
    #readjsonl('conala/conala-mined/conala-mined.jsonl')
    #readjson('conala/conala/conala-train.json')
    #readjson('conala/conala/conala-test.json')
    readjsonl('pythonapi/python-docs.jsonl')


