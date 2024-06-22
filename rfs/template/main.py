import torch.optim

from utils import get_config
from data_loader import mnist
from model import SimpleNN
from trainer import train

def main():
    config = get_config()
    '''
    # template `config.yaml`
    name: "project_name"

    device: "cuda"
    batch_size: 32
    epochs: 40
    learning_rate: 0.001
    data_path: "data/"
    model_save_path: "models/"
    log_dir: "logs/"
    '''
    bs = config['batch_size']
    epochs = config['epochs']
    lr = config['learning_rate']
    data_path = config['data_path']

    train_loader, test_loader = mnist(data_path, bs)
    num_inputs, num_outputs = 28*28, 10

    model = SimpleNN(num_inputs, num_outputs)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.CrossEntropyLoss()

    train(model, train_loader, test_loader, optim, epochs=epochs, lossfn=lossfn, writer=None)

if __name__ == '__main__':
    main()
