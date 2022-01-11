
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from src.models.model import CNN_MNIST, CNN_MNIST_DILATION, CNN_MNIST_STRIDE


class DatasetMNIST(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, all_images, all_labels):
        super().__init__()

        self.all_images = all_images
        self.all_labels = all_labels


    def __len__(self):
        'Denotes the total number of samples'
        return self.all_images.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'

        image = self.all_images[index, :, :]
        label = self.all_labels[index]

        return image, label


def load_mnist_dataset(LOAD_DATA_DIR, train_or_test):
    
    mnist_data_paths = sorted(
        glob.glob(LOAD_DATA_DIR + '*')
    )
    
    all_images = np.array([], dtype=float).reshape(0, 28, 28)
    all_labels = np.array([], dtype=float)

    for f in mnist_data_paths:
        data = np.load(f)
        all_images = np.vstack([all_images, data["images"]])
        all_labels = np.append(all_labels, data["labels"])
        print(f'finished processing {f}')

    return DatasetMNIST(all_images, all_labels)


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self, command, LOAD_DATA_DIR, MODEL_SAVE_DIR, PLOT_SAVE_DIR, lr, model_name):

        if not hasattr(self, command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)

        self.LOAD_DATA_DIR = LOAD_DATA_DIR
        self.MODEL_SAVE_DIR = MODEL_SAVE_DIR
        self.PLOT_SAVE_DIR = PLOT_SAVE_DIR
        self.lr = lr
        self.model_name = model_name

        # use dispatch pattern to invoke method with same name
        getattr(self, command)()


    def train(self):
        print("Training day and night")
        # add any additional argument that you want
        
        if self.model_name == 'CNN_MNIST':
            model = CNN_MNIST().float()
        elif self.model_name == 'CNN_MNIST_STRIDE':
            model = CNN_MNIST_STRIDE().float()
        elif self.model_name == 'CNN_MNIST_DILATION':
            model = CNN_MNIST_DILATION().float()

        train_set = load_mnist_dataset(self.LOAD_DATA_DIR, "train")
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        epochs = 30
        self.df_train_loss = pd.DataFrame([], columns=["Train loss"])
        for epoch in range(epochs):
            running_loss = 0
            for i, (images, labels) in enumerate(trainloader):
                
                optimizer.zero_grad()
                log_ps = model(images.float())
                loss = criterion(log_ps, labels.long())
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            print("Epoch " + str(epoch) + "  Loss=" + str(round(running_loss / i, 2)))
            #print("log_ps"); print(log_ps)
            #print("Gradients:")
            #print(model.hidden_layers[-1].weight.data)
            # Save plot and model checkpoint
            self.save_to_plot(running_loss / i, epoch)
            self.save_checkpoint(model)


    def save_checkpoint(self, model):
        checkpoint = model.state_dict()
        torch.save(checkpoint, os.path.join(self.MODEL_SAVE_DIR, 'checkpoint.pth'))
        return None


    def save_to_plot(self, mean_loss, epoch):
        self.df_train_loss.loc[epoch, "Train loss"] = mean_loss
        fig = self.df_train_loss.plot(xlabel="epoch", \
            ylabel="Loss", figsize=(14, 12), fontsize=15).get_figure()
        fig.legend("Train loss")
        fig.savefig(os.path.join(self.PLOT_SAVE_DIR, 'train_loss.pdf'))
        return None


    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = torch.load(args.load_model_from)
        test_set = mnist(LOAD_DATA_DIR = self.LOAD_DATA_DIR, train_or_test="test")

        ps = torch.exp(model(images))
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))

        print(f'Accuracy: {accuracy.item()*100}%')


if __name__ == '__main__':
    """
        Train on data from processed and save into models
        or load from models and evaluate
    """

    parser = argparse.ArgumentParser(
        description="Script for either training or evaluating",
        usage="python3 src/models/train_model.py <command> <LOAD_DATA_DIR> <MODEL_SAVE_DIR>"
    )
    parser.add_argument("--command", help="Subcommand to run")
    parser.add_argument("--LOAD_DATA_DIR", help="Path to processed data")
    parser.add_argument("--MODEL_SAVE_DIR", help="Path to model checkpoint")
    parser.add_argument("--PLOT_SAVE_DIR", help="Path to visualizations")
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument("--model_name", help="Name of neural network model")


    args = parser.parse_args()
    config = vars(args)

    script_dir = os.path.dirname(__file__)
    LOAD_DATA_DIR = os.path.join(script_dir, "../../", config["LOAD_DATA_DIR"])
    MODEL_SAVE_DIR = os.path.join(script_dir, "../../", config["MODEL_SAVE_DIR"], config["model_name"])
    PLOT_SAVE_DIR = os.path.join(script_dir, "../../", config["PLOT_SAVE_DIR"], config["model_name"])
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

    TrainOREvaluate(config["command"], LOAD_DATA_DIR, MODEL_SAVE_DIR, PLOT_SAVE_DIR, config["lr"], config["model_name"])