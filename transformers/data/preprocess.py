import os
import torch
from transformers import BertTokenizer
import tqdm


# .txt files is under text/XX/wiki_XX.txt
# Where XX is for example: AA, AB ... AZ, BA, BB ... BZ, CA, CB ... CZ, DA, DB ... DZ
# And wiki_XX.txt are for example wiki_00.txt, wiki_01.txt ... wiki_99.txt

def dataset_generator(tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'), directory="./text", batch_size=32):
    # Count total number of files
    total_files = sum([len(files) for r, d, files in os.walk(directory)])
    progress_bar = tqdm.tqdm(total=total_files, desc="Processing files")
    k = 0

    for subdir in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, subdir)):
            with open(os.path.join(directory, subdir, filename), 'r') as f:

                # Read the whole file
                text = f.read()

                # Tokenize the text
                tokens = tokenizer.tokenize(text)

                # Numeericalize the tokens
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # Yield batches
                for i in range(0, len(input_ids) - batch_size + 1, batch_size):
                    yield torch.tensor(input_ids[i:i+batch_size], dtype=torch.long)

            # Update progress bar
            progress_bar.update(1)
            k += 1
            print(k)
            if k == 5:
                print("break")
                break

        progress_bar.close()


def get_dataset(directory="./text", batch_size=32):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return dataset_generator(tokenizer=tokenizer, directory=directory, batch_size=batch_size)
