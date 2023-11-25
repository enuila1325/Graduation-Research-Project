from image_classificator_nn import image_classificator
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from os import listdir
import torch
import time
import cv2
import sys

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def main():
    image_route = sys.argv[1]
    images_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    device = "cuda"
    model = torch.load("./tuberculosis-pneumonia-model.pth", map_location=device)
    red = image_classificator(3).to("cuda")
    red.load_state_dict(model["model_state_dict"])
    print("Model loaded and sent to CUDA")
    classes = ["normal", "pneumonia", "tuberculosis"]
    validation_dataset = ImageFolder(root=image_route, transform=images_transform)
    validation_dataset_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        drop_last=True,
    )
    y_values, predictions = [], []
    with torch.no_grad():
        for i, data in tqdm(
            enumerate(validation_dataset_loader), total=len(validation_dataset_loader)
        ):
            cont = 0
            cont += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = red(image)
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            y_values.append(labels.cpu().numpy())
            predictions.append(preds.cpu().numpy())
        # time.sleep(0.2)

    y_values = [i[0] for i in y_values]
    predictions = [i[0] for i in predictions]
    print(
        classification_report(
            y_values, predictions, zero_division=0, target_names=classes
        )
    )

    # confussion matrix
    cm = confusion_matrix(y_values, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.show()


if __name__ == "__main__":
    main()
