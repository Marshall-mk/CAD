import os
import sys
import json
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from utils import focal_loss, plot_confusion_matrix
import matplotlib.pyplot as plt
import logging
from MedMamba import VSSM as medmamba
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix, recall_score, f1_score, precision_score
import warnings
warnings.filterwarnings('ignore')


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomRotation(20),
                                     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.6679, 0.5354, 0.5194), (0.2224, 0.2053, 0.2171))]),
        
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.6679, 0.5354, 0.5194), (0.2224, 0.2053, 0.2171))])}

    train_dataset = datasets.ImageFolder(root="../../../dataset/train/",
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    class_weights = [0.4916, 0.5084]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root="../../../dataset/val/",
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("Using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))


    net = medmamba(num_classes=2)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # loss_function = focal_loss(alpha=class_weights, gamma=2.0, reduction="mean")
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 60
    best_acc = 0.0
    best_loss = 100.0
    save_path = './{}Net.pth'.format("mamba_skinBinaryCELoss_Adam")
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # calculate accuracy
            _, predicts = torch.max(outputs, 1)
            corrects = torch.sum(predicts == labels.data.to(device))
            accuracy = corrects.double() / batch_size

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} accuracy:{:.3f}".format(epoch + 1,
                                                                                      epochs,
                                                                                      loss.item(),
                                                                                        accuracy)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                val_loss += loss_function(outputs, val_labels.to(device)).item()
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        # Calculate additional metrics
        val_labels = torch.cat([val_labels.to(device) for _, val_labels in validate_loader])
        val_preds = torch.cat([torch.max(net(val_images.to(device)), dim=1)[1] for val_images, _ in validate_loader])
        val_loss /= val_num
        
        # Confusion Matrix
        vl = val_labels.cpu().numpy()
        vp = val_preds.cpu().numpy()
        conf_matrix = confusion_matrix(vl, vp)
        # Extract TP, TN, FP, FN from confusion matrix
        tn, fp, fn, tp = conf_matrix.ravel()
        val_kappa = cohen_kappa_score(vl, vp)
        # F1 Score, Recall, and Precision
        f1 = f1_score(vl, vp, average='weighted')
        recall = recall_score(vl, vp, average='weighted')
        precision = precision_score(vl, vp, average='weighted')
        class_report = classification_report(vl, vp)
        val_accurate = acc / val_num
        print(f'Confusion Matrix:\n{conf_matrix}')
        print(f'Classification Report:\n{class_report}')
        if not os.path.exists(
                "./mamba_binary_CELoss_Adam_metrics.csv"
            ):
            pd.DataFrame(
                columns=[
                    "Epoch",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1-Score",
                    "Loss",
                    "TP",
                    "TN",
                    "FP",
                    "FN",
                    "Kappa",
                ]
            ).to_csv(
                "./mamba_binary_CELoss_Adam_metrics.csv",
                index=False,
            )
        pd.DataFrame(
            [
                [
                    epoch + 1,
                    val_accurate,
                    precision,
                    recall,
                    f1,
                    val_loss,
                    tp,
                    tn,
                    fp,
                    fn,
                    val_kappa,
                ]
            ],
            columns=[
                "Epoch",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "Loss",
                "TP",
                "TN",
                "FP",
                "FN",
                "Kappa",
            ],
        ).to_csv(
            "./mamba_binary_CELoss_Adam_metrics.csv",
            mode="a",
            header=False,
            index=False,
        )
        try:
            plot_confusion_matrix(
                conf_matrix,
                classes=["Nevus", "Others"],
                title="Val Confusion Matrix",
            )
            plt.savefig(
                f"./mamba_binary_CELoss_Adam_confusion_matrix_{epoch + 1}.png"
            )
        except Exception as e:
            logging.error(f"Error while plotting confusion matrix: {e}")
        
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            print("Save model")
        if val_loss < best_loss:
            best_loss = val_loss
    print(f'Best Accuracy:{best_acc}')
    print(f'Best Loss:{best_loss}')
    print('Finished Training')

if __name__ == '__main__':
    main()
