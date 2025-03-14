import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from model import NumberClassification
from dataset import MyMNISTDataset
from torch.optim import SGD, Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description="Number Classifier")
    parser.add_argument("--num-epochs", "-e", type=int, default=100, help="number of epoch")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="batch size of training progress")
    parser.add_argument("--num-workers", "-n", type=int, default=0, help="ammount of num workers")
    parser.add_argument("--learning-rate", "-l", type=float, default=1e-3, help="learning rate of optimizer")
    parser.add_argument("--momentum", "-m", type=float, default=0.9, help="momentum of optimizer")
    parser.add_argument("--image-size", "-i", type=int, default=32, help="size of images")
    parser.add_argument("--log-path", "-o", type=str, default="my_tensorboard", help="folder of model visualization")
    parser.add_argument("--checkpoint-path", "-c", type=str, default="my_checkpoint", help="folder of model saving")
    parser.add_argument("--data-path", "-d", type=str, default=r"D:\VSC saves\ML4CV\MNIST", help="path to MNIST folder")
    parser.add_argument("--resume-training", "-r", type=bool, default=False, help="continue training or not")
    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="Wistia")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def TrainModel(args):
    transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor()
    ])
    # chạy trên GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # lấy data bộ train
    train_dataset = MyMNISTDataset(root=args.data_path, train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    num_iters = len(train_dataloader)
    # lấy data bộ validate
    valid_dataset = MyMNISTDataset(root=args.data_path, train=False, transform=transform)
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    # khởi tạo model
    model = NumberClassification(num_classes=len(train_dataset.categories)).to(device)
    # loss function
    criterion  = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if not os.path.isdir(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    writer = SummaryWriter(args.log_path)
    if args.resume_training:
        checkpoint = os.path.join(args.checkpoint_path, "last.pt")
        saved_data = torch.load(checkpoint)
        model.load_state_dict(saved_data["model"])
        optimizer.load_state_dict(saved_data["optimizer"])
        start_epoch = saved_data["epoch"]
        best_accuracy = saved_data["best_accuracy"]
    else:
        start_epoch = 0
        best_accuracy = -1
    for epoch in range(start_epoch, args.num_epochs):
        # TRAINING MODE
        progress_bar = tqdm(train_dataloader, colour="cyan")
        model.train()
        total_losses = []
        for iter, (images, labels) in enumerate(progress_bar):
            # foward
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            total_losses.append(loss.item())
            avg_loss = np.mean(total_losses)
            progress_bar.set_description("epoch: {}/{} loss: {:0.4f}".format(epoch+1, args.num_epochs, loss))
            writer.add_scalar("Train/Loss", avg_loss, global_step=epoch*num_iters+iter)
            # backward
            optimizer.zero_grad()   # làm sạch buffer
            loss.backward() # update parameter của model
            optimizer.step()    # cập nhật weight

        # VALIDATION MODE
        progress_bar = tqdm(valid_dataloader, colour="magenta")
        model.eval()
        total_losses = []
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for iter, (images, labels) in enumerate(progress_bar):
                # foward
                images = images.to(device)
                all_labels.extend(labels)   # nên extend trước khi đưa label vào GPU để tránh tích lũy memory
                labels = labels.to(device)
                output = model(images)
                prediction = torch.argmax(output, dim=1)
                all_predictions.extend(prediction.tolist())
                loss = criterion(output, labels)
                total_losses.append(loss.item())

        avg_loss = np.mean(total_losses)
        accuracy = accuracy_score(all_labels, all_predictions)
        writer.add_scalar("Valid/Loss", avg_loss, global_step=epoch)
        writer.add_scalar("Valid/Accuracy", accuracy, global_step=epoch)
        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions), train_dataset.categories, epoch)
        print("Epoch: {}. Average Loss: {:0.4f}. Accuracy: {:0.4f}%.".format(epoch+1, avg_loss, accuracy*100))

        # save model
        saved_data = {
            "model": model.state_dict(),    # model parameter
            "optimizer": optimizer.state_dict(),
            "epoch": epoch+1,   # next training epoch
            "best_accuracy": best_accuracy,
        }
        
        checkpoint = os.path.join(args.checkpoint_path, "last.pt")
        torch.save(saved_data, checkpoint)       
        if accuracy > best_accuracy:
            checkpoint = os.path.join(args.checkpoint_path, "best.pt")
            torch.save(saved_data, checkpoint)
            best_accuracy = accuracy

if __name__ == "__main__":
    args = get_args()
    TrainModel(args)
    
