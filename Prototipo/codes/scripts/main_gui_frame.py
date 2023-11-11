import tkinter as tk
import tkinter.ttk as ttk
import torch
import cv2
from tkinter.filedialog import askopenfilename
from image_classificator_nn import image_classificator as alzheimer_classificator
from brain_tumor_classification_nn import (
    image_classificator as brain_tumor_classificator,
)
from tuberculosis_classification_nn import (
    image_classificator as tuberculosis_classificator,
)

from torchvision.transforms import transforms


class App(tk.Tk):
    def __init__(self):
        super().__init__()


def specific_prognosis_frame():
    app = App()
    app.geometry("720x240")
    brain_tumor_detection = tk.Button(
        app,
        text="Detección y clasificación de tumores cerebrales",
        command=lambda: open_file_choser(1),
    )
    tuberculosis_pneumonia_detection = tk.Button(
        app,
        text="Detección y clasificación de tuberculosis y neumonía",
        command=lambda: open_file_choser(2),
    )
    alzheimer_detection = tk.Button(
        app,
        text="Detección y clasificación de Alzheimer",
        command=lambda: open_file_choser(3),
    )
    brain_tumor_detection.place(x=220, y=50)
    tuberculosis_pneumonia_detection.place(x=200, y=100)
    alzheimer_detection.place(x=230, y=150)
    app.mainloop()


def open_file_choser(disease):
    filename = askopenfilename()
    app_aux = App()
    app_aux.geometry("720x250")
    frame = ttk.Frame(app_aux)
    label = ttk.Label(
        frame,
        text="Model loaded and sent to CUDA\nImage opened.\nTransforming...\nImage transformed",
    )
    label.pack(padx=5)
    frame.pack(padx=10, pady=50, expand=True, fill=tk.BOTH)
    predict_class = ""
    if disease == 1:
        predict_class = brain_tumor_detection(filename)
    elif disease == 2:
        predict_class = tuberculosis_detection(filename)
    elif disease == 3:
        predict_class = alzheimer_detection(filename)
    button = tk.Button(
        app_aux,
        text="ANALIZAR IMAGEN CARGADA",
        command=lambda: diagnosis_frame(predict_class),
    )
    button.place(x=250, y=200)
    app_aux.mainloop()
    predict_class = ""


def diagnosis_frame(predicted_class):
    app = App()
    app.geometry("400x200")
    frame = ttk.Frame(app)
    label = ttk.Label(frame, text=predicted_class)
    label.pack(padx=5)
    frame.pack(padx=10, pady=50, expand=True, fill=tk.BOTH)
    app.mainloop()


def alzheimer_detection(image_route):
    images_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    model = torch.load(
        "Prototipo/codes/models/model_alzheimer-47_epochs-93Validation.pth",
        map_location="cuda",
    )
    red = alzheimer_classificator(4).to("cuda")
    red.load_state_dict(model["model_state_dict"])
    classes = [
        "Mild Demented",
        "Moderated Demented",
        "Non Demented",
        "Very Mild Demented",
    ]
    images = cv2.imread(image_route)
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    image_to_predict = images_transform(images)
    image_to_predict = torch.unsqueeze(image_to_predict, 0)
    red.eval()
    with torch.no_grad():
        output = red(image_to_predict.to("cuda"))
    output_label = torch.topk(output, 1)
    predicted_class = classes[int(output_label.indices)]
    return predicted_class


def brain_tumor_detection(image_route):
    images_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    model = torch.load(
        "Prototipo/codes/models/model_tumor.pth",
        map_location="cuda",
    )
    red = brain_tumor_classificator(4).to("cuda")
    classes = ["glioma", "meningioma", "no_tumor", "pituitary"]
    images = cv2.imread(image_route)
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    image_to_predict = images_transform(images)
    image_to_predict = torch.unsqueeze(image_to_predict, 0)
    red.eval()
    with torch.no_grad():
        output = red(image_to_predict.to("cuda"))
    output_label = torch.topk(output, 1)
    predicted_class = classes[int(output_label.indices)]
    return predicted_class


def tuberculosis_detection(image_route):
    images_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    device = "cuda"
    model = torch.load(
        "Prototipo/codes/models/model_tuberculosis.pth", map_location="cuda"
    )
    red = tuberculosis_classificator(2).to("cuda")
    red.load_state_dict(model["model_state_dict"])
    classes = ["healthy lungs", "tuberculosis"]
    images = cv2.imread(image_route)
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    image_to_predict = images_transform(images)
    image_to_predict = torch.unsqueeze(image_to_predict, 0)
    red.eval()
    with torch.no_grad():
        output = red(image_to_predict.to("cuda"))
    output_label = torch.topk(output, 1)
    predicted_class = classes[int(output_label.indices)]
    return predicted_class


def main():
    app = App()
    app.geometry("720x320")
    diverse_classification_button = tk.Button(
        app, text="Clasificar y diagnosticar una imágen"
    )
    specific_classification_button = tk.Button(
        app, text="Hacer un diagnóstico en especifico", command=specific_prognosis_frame
    )
    diverse_classification_button.place(x=240, y=100)
    specific_classification_button.place(x=240, y=200)
    app.mainloop()


if __name__ == "__main__":
    main()
