from image_classificator_nn import image_classificator
from torchvision.transforms import transforms
import torch
import cv2
import sys

def main():
    image_route = sys.argv[1]
    images_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    device = 'cuda'
    model = torch.load('./model_alzheimer-47_epochs-93Validation.pth', map_location=device)
    red = image_classificator(4).to(device=device)
    red.load_state_dict(model["model_state_dict"])
    print("Model loaded and sent to CUDA")
    classes = ['Mild Demented','Moderated Demented', 'Non Demented', 'Very Mild Demented']
    images = cv2.imread(image_route)
    print("Image opened.\nTransforming...")
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    image_to_predict = images_transform(images)
    image_to_predict = torch.unsqueeze(image_to_predict, 0)
    print("Image transformed\nPredicting...")
    red.eval()
    with torch.no_grad():
        output = red(image_to_predict.to(device=device))
    output_label = torch.topk(output, 1)
    predicted_class = classes[int(output_label.indices)]
    print('This image represents a brain MRI that has', predicted_class)
    print('Prediction finished\nFinalizing program')
    exit(0)
    
if __name__ == '__main__':
    main()