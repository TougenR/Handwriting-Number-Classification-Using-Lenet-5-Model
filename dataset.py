from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import cv2
from torchvision.transforms import ToTensor, Resize, Compose

class MyMNISTDataset(Dataset):
    def __init__(self, root, train, transform=None):
        self.image_path = []
        self.labels = []
        self.categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        # Đường dẫn đến file dataset Mnist
        data_path = os.path.join(root, "MNIST_dataset")
        # chia đường dẫn bộ train và test
        if train:
            data_path = os.path.join(data_path, "training")
        else:
            data_path = os.path.join(data_path, "testing")
        # Tìm đường dẫn cho 1 bức ảnh và đánh dấu label
        for i, category in enumerate(self.categories):
            file_path = os.path.join(data_path, category)
            for item in os.listdir(file_path):
                path = os.path.join(file_path, item)
                self.image_path.append(path)
                self.labels.append(i)
       
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = self.image_path[index]
        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label
    
