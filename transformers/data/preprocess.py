import os
import torch
import tqdm

from transformers import BertTokenizer


# .txt files is under text/XX/wiki_XX.txt
# Where XX is for example: AA, AB ... AZ, BA, BB ... BZ, CA, CB ... CZ, DA, DB ... DZ
# And wiki_XX.txt are for example wiki_00.txt, wiki_01.txt ... wiki_99.txt

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, directory="./text", batch_size=32):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.directory = directory
        self.batch_size = batch_size
        self.data = list(self.dataset_generator())

    def dataset_generator(self):
        total_files = sum([len(files)
                          for r, d, files in os.walk(self.directory)])
        progress_bar = tqdm.tqdm(total=total_files, desc="Processing files")
        k = 0

        for subdir in os.listdir(self.directory):
            for filename in os.listdir(os.path.join(self.directory, subdir)):
                with open(os.path.join(self.directory, subdir, filename), 'r') as f:

                    text = f.read()
                    tokens = self.tokenizer.tokenize(text)
                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                    for i in range(0, len(input_ids) - self.batch_size + 1, self.batch_size):
                        inputs = torch.tensor(
                            input_ids[i:i + self.batch_size], dtype=torch.long)
                        targets = torch.tensor(
                            input_ids[i + 1:i + 1 + self.batch_size], dtype=torch.long)
                        yield inputs, targets

                progress_bar.update(1)
                k += 1
                print(k)
                if k == 5:
                    print("break")
                    break

            progress_bar.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataset(directory="./data/text", batch_size=32):
    return TextDataset(directory=directory, batch_size=batch_size)
