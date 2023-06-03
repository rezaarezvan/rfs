import os
import torch
from transformers import BertTokenizer


# .txt files is under text/XX/wiki_XX.txt
# Where XX is for example: AA, AB ... AZ, BA, BB ... BZ, CA, CB ... CZ, DA, DB ... DZ
# And wiki_XX.txt are for example wiki_00.txt, wiki_01.txt ... wiki_99.txt

def dataset_generator(tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'), directory="./text", batch_size=32):
    for subdir in os.listdir(directory):
        for filename in os.listdir(os.join(directory, subdir)):
            with open(os.join(directory, subdir, filename), 'r') as f:

                # Read the whole file
                text = f.read()

                # Tokenize the text
                tokens = tokenizer.tokenize(text)

                # Numeericalize the tokens
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # Yield batches
                for i in range(0, len(input_ids), batch_size):
                    yield torch.tensor(input_ids[i:i+batch_size], dtype=torch.long)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
batch_size = 32
data = dataset_generator(
    tokenizer=tokenizer, directory="./text/", batch_size=batch_size)
