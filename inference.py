import torch
import torch.nn as nn
from model import NumberClassification
import os
import numpy as np
import argparse
import cv2
import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description="Number Classifier")
    parser.add_argument("--image-size", "-i", type=int, default=32, help="size of images")
    parser.add_argument("--checkpoint-path", "-c", type=str, default="my_checkpoint", help="folder of model saving")
    parser.add_argument("--image-path", "-p", type=str, default="test_images/9.png", help="path to test image")
    args = parser.parse_args()
    return args

def Inference(args):
    categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    # chạy trên GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # khởi tạo model
    model = NumberClassification(num_classes=len(categories)).to(device)
    checkpoint = os.path.join(args.checkpoint_path, "best.pt")
    saved_data = torch.load(checkpoint)
    model.load_state_dict(saved_data["model"])
    model.eval()

    # đọc ảnh, chuyển ảnh về GRAYSCALE, resize ảnh thành 32x32
    original_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (args.image_size, args.image_size))
    # scale các giá trị trong ảnh thuộc [0,1] và chuyển vị kênh từ (W, H, C) -> (C, W, H)
    image = image/255
    # thêm Batch size dim + Channel dim vì model yêu cầu tensor 4 chiều
    # cách 1
    # image = np.expand_dims(image, axis=0) # (1, W, H)
    # image = np.expand_dims(image, axis=0) # (1, 1, W, H) 
    # cách 2
    image = image[None, None, :, :] # (1, 1, W, H)
    image = torch.from_numpy(image).float().to(device) # chuyển từ numpy array thành tensor và chuyển data type về float để khớp với parameter trong mô hình
    softmax = nn.Softmax() # thêm softmax để tính accuracy
    with torch.no_grad():
        output = model(image)[0]
        predict_class = categories[torch.argmax(output)] # dự đoán
        prob = softmax(output)[torch.argmax(output)]
        print("Number is {}".format(predict_class))
        print("Probabillity {:0.2f}%".format(prob*100))
        cv2.imshow("{} ({:0.2f}%)".format(predict_class, prob*100), original_image)
        cv2.waitKey(0)

if __name__ == "__main__":
    args = get_args()
    Inference(args)
