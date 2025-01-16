import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import warnings
import os
from sklearn import metrics

warnings.filterwarnings("ignore")
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
sys.stdout.flush()
dinov2_vits14.to(device)
transform_image = T.Compose(
    [
        T.ToTensor(),
        T.Resize(244),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_image(img: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    img = Image.open(img)

    transformed_img = transform_image(img)[:3].unsqueeze(0)

    return transformed_img


def get_files(ROOT_DIR):
    labels = {}

    for folder in os.listdir(ROOT_DIR):
        for file in os.listdir(os.path.join(ROOT_DIR, folder)):
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                full_name = os.path.join(ROOT_DIR, folder, file)
                labels[full_name] = folder

    files = labels.keys()
    return files, labels


model_names = ["SVM"]  # "LightGBM", "CatBoost", "RandomForest", "KNN",
for model_name in model_names:
    print(f"Model: {model_name}")
    sys.stdout.flush()
    clf = joblib.load(f"../checkpoints/{model_name}.pkl")
    images, targets = get_files("../../../dataset/3_class_test")
    targets = np.array(list(targets.values()))
    print(f"Number of images: {len(images)}")
    print(f"Number of targets: {len(targets)}")
    print(
        "=============================================================================================="
    )
    sys.stdout.flush()
    predictions = []
    for img in images:
        new_image = load_image(img)
        with torch.no_grad():
            embedding = dinov2_vits14(new_image.to(device))
            prediction = clf.predict(np.array(embedding[0].cpu()).reshape(1, -1))
            predictions.append(prediction[0])
    predictions = np.array(predictions, dtype=np.int32).reshape(-1)
    targets = np.array(targets).reshape(-1)
    print(accuracy_score(targets, predictions))
    print(classification_report(targets, predictions))
    cm = confusion_matrix(targets, predictions)
    print(cm)
    print(
        "=============================================================================================="
    )
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    precision = np.mean([cm[i, i] / np.sum(cm[:, i]) for i in range(3)])
    recall = np.mean([cm[i, i] / np.sum(cm[i, :]) for i in range(3)])
    recall = metrics.recall_score(targets, predictions)
    precision = metrics.precision_score(targets, predictions)
    f1_score = metrics.f1_score(targets, predictions)
    print("Recall for {}: {:.3f}".format(model_name, recall))
    print("Precision for {}: {:.3f}".format(model_name, precision))
    print("F1 Score for {}: {:.3f}".format(model_name, f1_score))
    print(
        "=============================================================================================="
    )
    with open(f"../metrics/{model_name}_Test.txt", "w") as f:
        f.write(
            f"Validation Accuracy: {accuracy_score(targets, predictions)}\n"
            f"Confusion Matrix: {confusion_matrix(targets, predictions)}\n"
            f"Recall: {recall}\n"
            f"Precision: {precision}\n"
            f"F1 Score: {f1_score}\n"
            f"Classification Report: {classification_report(targets, predictions)}"
        )
    sys.stdout.flush()
