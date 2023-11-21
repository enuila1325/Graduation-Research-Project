import torch
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def save_model(epochs, model, optimizer, criterion, model_route):
    """
    Function to save the trained model to disk.
    """
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": criterion,
        },
        model_route,
    )


def save_plots(train_acc, valid_acc, train_loss, valid_loss, recall, f1, classes):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color="green", linestyle="-", label="train accuracy")
    plt.plot(valid_acc, color="blue", linestyle="-", label="validataion accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("./codes/outputs/accuracy.png")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", linestyle="-", label="train loss")
    plt.plot(valid_loss, color="red", linestyle="-", label="validataion loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./codes/outputs/loss.png")

    # recall plots
    class_1, class_2, class_3, class_4 = [], [], [], []
    for i, value in enumerate(recall):
        if(i%4==0):
            class_1.append(value)
        elif (i+1)%4==0:
            class_4.append(value)
        elif i%2==0:
            class_3.append(value)
        else:
            class_2.append(value)

    plt.figure(figsize=(10, 7))
    plt.plot(class_1, color="cyan", linestyle="-", label=classes[0])
    plt.plot(class_2, color="pink", linestyle="-", label=classes[1])
    plt.plot(class_3, color="blue", linestyle="-", label=classes[2])
    plt.plot(class_4, color="purple", linestyle="-", label=classes[3])
    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.legend()
    plt.savefig("./codes/outputs/recall.png")

    # f-1 plots
    class_1, class_2, class_3, class_4 = [], [], [], []
    for i, value in enumerate(f1):
        if(i%4==0):
            class_1.append(value)
        elif (i+1)%4==0:
            class_4.append(value)
        elif i%2==0:
            class_3.append(value)
        else:
            class_2.append(value)
    plt.figure(figsize=(10, 7))
    plt.plot(class_1, color="salmon", linestyle="-", label=classes[0])
    plt.plot(class_2, color="darkorange", linestyle="-", label=classes[1])
    plt.plot(class_3, color="lime", linestyle="-", label=classes[2])
    plt.plot(class_4, color="indigo", linestyle="-", label=classes[3])
    plt.xlabel("Epochs")
    plt.ylabel("F1-Score")
    plt.legend()
    plt.savefig("./codes/outputs/f1-score.png")
