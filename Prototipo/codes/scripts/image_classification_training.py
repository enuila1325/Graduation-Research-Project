import sys
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm

from auxiliar_functions import save_model, save_plots
from image_classificator_nn import image_classificator
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from os import listdir


# argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=20,
    help="number of epochs to train our network for",
)
parser.add_argument(
    "-tr",
    "--training",
    type=str,
    default="./training",
    help="default route images for training",
)
parser.add_argument(
    "-v",
    "--validation",
    type=str,
    default="./validation",
    help="default route images for valdation",
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="./modelo.pth",
    help="default route to save trained model",
)
args = vars(parser.parse_args())
BATCH_SIZE = 32


def main():
    training_images_route = args["training"]
    validation_images_route = args["validation"]
    model_route = args["model"]

    # Standarizing images size to 224x224, converting the to tensors and normalizing the data
    images_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            # transforms.RandomRotation(degrees=(30, 70)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    training_dataset = ImageFolder(
        root=training_images_route, transform=images_transform
    )
    validation_dataset = ImageFolder(
        root=validation_images_route, transform=valid_transform
    )
    training_dataset_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    validation_dataset_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = True if torch.cuda.is_available() else False
   

    # defining learning values
    learning_rate = 0.025
    epochs = args["epochs"]
    print(f"Training and validating using: {device}")
    print(f"Training classes: {training_dataset.classes}")
    outs = len(training_dataset.classes)
    print(f"CNN having {outs} outputs")
    model = image_classificator(out=outs)
    if use_cuda:
        model = model.to(device)
        print("Model sent to CUDA")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate)
    minimun_valid_loss = np.inf

    # total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"{total_trainable_params:,} training parameters.")
    
    #declaring conditions for early stopping to prevent overfitting
    current_loss = 100.0
    last_loss = 0.0
    tolerance = 20
    trigger = 0
    final_epoch = 0
    
    # Training the network through all the epochs
    train_loss, train_acc = [], []
    validation_loss, validation_acc = [], []
    for epoch in range(epochs):
        print(f"INFO: Epoch [{epoch+1}] of [{epochs}]")
        epoch_loss_training, epoch_accurracy_training = training(
            model, training_dataset_loader, device, criterion, optimizer
        )
        epoch_loss_validation, epoch_accurracy_validation = validate(
            model, validation_dataset_loader, criterion, device
        )
        # storing loss and accurracy values for plotting them
        train_loss.append(epoch_loss_training)
        train_acc.append(epoch_accurracy_training)
        validation_loss.append(epoch_loss_validation)
        validation_acc.append(epoch_accurracy_validation)

        print(
            f"Training loss: {epoch_loss_training:.3f}, training acc: {epoch_accurracy_training:.3f}"
        )
        print(
            f"Validation loss: {epoch_loss_validation:.3f}, validation acc: {epoch_accurracy_validation:.3f}"
        )
        print("-" * 200)
        last_loss = epoch_loss_validation
        if last_loss < current_loss:
            print(f" Validation loss decreased from {current_loss:.3f} to {last_loss:.3f}, saving current model")
            current_loss = last_loss
            final_epoch = epoch
            final_epoch += 1
            # save the trained model weights
            save_model(epochs, model, optimizer, criterion, model_route)
            trigger = 0
        else:
            print(f" Current validation didn't decreased from las saved value nor improved last saved model.")
            trigger+=1
        
        print("-" * 200)
        time.sleep(2)
        if trigger>=tolerance:
            print(f"Validation hasn't decreased in a while. Ending training and validation proccess.\nKeeping saved model at epoch {final_epoch}")
            break
    
    # save the loss and accuracy plots
    save_plots(train_acc, validation_acc, train_loss, validation_loss)
    print("TRAINING COMPLETE")
    print("Succesfull run")


def training(model, trainloader, device, criterion, optimizer):
    # begin training
    model.train()
    print("TRAINING")
    training_running_loss = 0.0
    training_running_correct = 0
    cont = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        cont = cont + 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, labels)
        training_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(output.data, 1)
        training_running_correct += (preds == labels).sum().item()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
    epoch_loss = training_running_loss / cont
    epoch_acurracy = 100.0 * (training_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acurracy


def validate(model, validation_loader, criterion, device):
    model.eval()
    print("VALIDATING")
    validation_running_loss = 0.0
    validation_running_correct = 0
    cont = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(validation_loader), total=len(validation_loader)):
            cont += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            validation_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            validation_running_correct += (preds == labels).sum().item()

    epoch_loss = validation_running_loss / cont
    epoch_accurracy = 100.0 * (
        validation_running_correct / len(validation_loader.dataset)
    )
    return epoch_loss, epoch_accurracy


if __name__ == "__main__":
    main()
