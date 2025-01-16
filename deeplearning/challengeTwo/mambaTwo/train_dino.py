import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
import json
from tqdm.notebook import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib
import warnings
import logging
from sklearn import metrics
import sys

warnings.filterwarnings("ignore")
print(torch.__version__)

dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_vits14.to(device)
print(f"Device: {device}")
sys.stdout.flush()
logging.info(f"Device: {device}")

transform_image = T.Compose(
    [
        T.ToTensor(),
        T.Resize(244),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_files(ROOT_DIR):
    labels = {}

    for folder in os.listdir(ROOT_DIR):
        for file in os.listdir(os.path.join(ROOT_DIR, folder)):
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                full_name = os.path.join(ROOT_DIR, folder, file)
                labels[full_name] = folder

    files = labels.keys()
    return files, labels


def load_image(img: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    img = Image.open(img)

    transformed_img = transform_image(img)[:3].unsqueeze(0)

    return transformed_img


def compute_embeddings(files: list, name: str) -> dict:
    """
    Create an index that contains all of the images in the specified list of files.
    """
    all_embeddings = {}

    with torch.no_grad():
        for i, file in enumerate(tqdm(files)):
            embeddings = dinov2_vits14(load_image(file).to(device))

            all_embeddings[file] = (
                np.array(embeddings[0].cpu().numpy()).reshape(1, -1).tolist()
            )

    with open(f"../checkpoints/{name}_embeddings.json", "w") as f:
        f.write(json.dumps(all_embeddings))

    return all_embeddings


def main():
    files, labels = get_files("../../../dataset/3_class_train")
    print("Computing train embeddings...")
    embeddings = compute_embeddings(files, "3train")
    # load the embeddings from the file
    # with open(
    #     "../checkpoints/3train_embeddings.json"
    # ) as f:
    #     embeddings = json.load(f)
    print("Train Embeddings computed successfully!")
    y = [labels[file] for file in files]
    embedding_list = list(embeddings.values())
    y = np.array(y)
    X = np.array(embedding_list).reshape(-1, 384)

    print("Computing Val Embeddings....")
    val_files, val_labels = get_files("../../../dataset/3_class_val")
    val_embeddings = compute_embeddings(val_files, "3val")
    # with open(
    #     "/media/nfs/data/kamuhammad/paper_project/checkpoints/3val_embeddings.json"
    # ) as f:
    #     val_embeddings = json.load(f)
    print("Validation Embeddings computed successfully!")
    y_val = [val_labels[val_file] for val_file in val_files]
    val_embedding_list = list(val_embeddings.values())
    y_val = np.array(y_val)
    X_val = np.array(val_embedding_list).reshape(-1, 384)
    print("Train embeddings shape: ", X.shape)
    print("Validation embeddings shape: ", X_val.shape)
    print("Train labels shape: ", y.shape)
    print("Validation labels shape: ", y_val.shape)
    print("Training the model...")
    sys.stdout.flush()

    ############################################## sklearn  model ###############################################################################
    model_list = [
        SVC(random_state=42, probability=True, C=10, gamma=0.001, kernel="rbf"),
        XGBClassifier(random_state=42, n_estimators=100, max_depth=3),
        RandomForestClassifier(random_state=42, n_estimators=100, max_depth=3),
        KNeighborsClassifier(n_neighbors=3),
        lgb.LGBMClassifier(n_estimators=100, max_depth=3),
        CatBoostClassifier(
            iterations=100, depth=3, learning_rate=0.1, loss_function="Logloss"
        ),
    ]
    model_names = ["SVM", "XGBoost", "RandomForest", "KNN", "LightGBM", "CatBoost"]
    for model, model_name in zip(model_list, model_names):
        print(f"Model: {model_name}")
        print(
            "=============================================================================================="
        )
        model.fit(X, y)
        # Save the model
        joblib.dump(
            model,
            "../checkpoints/{}.pkl".format(model_name),
        )
        print(f"{model_name} model saved successfully!")
        print(
            "=============================================================================================="
        )
        sys.stdout.flush()

        y_val_pred = model.predict(X_val)
        val_accuracy = metrics.accuracy_score(y_val, y_val_pred)
        print("Validation Accuracy for {}: {:.3f}".format(model_name, val_accuracy))
        cm = metrics.confusion_matrix(y_val, y_val_pred)
        print("Confusion Matrix for {}: \n{}".format(model_name, cm))
        print(
            f"Classification Report for {model_name}: \n{metrics.classification_report(y_val, y_val_pred)}"
        )
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        precision = np.mean([cm[i, i] / np.sum(cm[:, i]) for i in range(3)])
        recall = np.mean([cm[i, i] / np.sum(cm[i, :]) for i in range(3)])
        f1_score = 2 * (precision * recall) / (precision + recall)
        print("Accuracy for {}: {:.3f}".format(model_name, accuracy))
        print("Sensitivity for {}: {:.3f}".format(model_name, recall))
        recall = metrics.recall_score(y_val, y_val_pred)
        precision = metrics.precision_score(y_val, y_val_pred)
        f1_score = metrics.f1_score(y_val, y_val_pred)
        print("Recall for {}: {:.3f}".format(model_name, recall))
        print("Precision for {}: {:.3f}".format(model_name, precision))
        print("F1 Score for {}: {:.3f}".format(model_name, f1_score))
        print(
            "=============================================================================================="
        )

        with open("../metrics/{}_Val.txt".format(model_name), "w") as f:
            f.write(
                f"Validation Accuracy: {val_accuracy}\n"
                f"Confusion Matrix: {cm}\n"
                f"Recall: {recall}\n"
                f"Precision: {precision}\n"
                f"F1 Score: {f1_score}\n"
                f"Classification Report: {metrics.classification_report(y_val, y_val_pred)}"
            )

        print(
            "=============================================================================================="
        )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
