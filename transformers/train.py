import torch
from transformers import BertTokenizer
from torch import nn, optim
from torch.utils.data import DataLoader
from models.transformer import Transformer
from data.preprocess import get_dataset
import tqdm


class WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, generator):
        self.data = list(generator)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train(transformer_model, train_data, num_epochs=5, learning_rate=1e-3, batch_size=64):
    device = torch.device("cpu")
    print(f'Using device {device}')

    data_loader = DataLoader(train_data, batch_size=batch_size)
    optimizer = optim.Adam(transformer_model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    transformer_model.to(device)
    criterion.to(device)

    for epoch in range(num_epochs):
        progress_bar = tqdm.tqdm(
            data_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for _, batch in enumerate(data_loader):
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            # Make targets be of shape [1, 10000]
            targets = targets.unsqueeze(0)
            targets = targets.permute(1, 0)
            targets = targets.squeeze(0)
            targets = targets.long()

            # Forward pass
            outputs = transformer_model.forward(inputs)
            print(outputs.shape)
            print(targets.shape)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

    print('Finished Training')


# Assume train_generator and test_generator are your preprocessed datasets
train_generator = get_dataset('./data/text/')
# test_generator = dataset_generator()

train_data = WikiTextDataset(train_generator)
# test_data = WikiTextDataset(test_generator)


# Init model
tokens = 32
heads = 8
depth = 6
seq_length = 32
num_tokens = BertTokenizer.from_pretrained('bert-base-uncased').vocab_size
num_classes = 10_000

transformer_model = Transformer(
    tokens, heads, depth, seq_length, num_tokens, num_classes)

train(transformer_model, train_data)
