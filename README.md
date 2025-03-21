- Dataset: MNIST  
- Abstract: Classify handwriting number from 0-9, input is an image with 1x32x32 and output is a prediction which number is it.  
- Currently running project on python 3.12 and CUDA 12.6.  
- installed libraries:
    + torch
    + torchvision 
    + PILLOW
    + os
    + tqdm
    + tensorboard
    + numpy
    + argparse
    + matplotlib
    + cv2
    + sklearn  

- Base model: Lenet-5
    * Adding BatchNorm Layer, Dropout Layer and using ReLU activation instead of Tanh for avoiding vanishing gradient and overfitting.  

- Performance:  
![Screenshot 2025-03-15 163030](https://github.com/user-attachments/assets/c28704a1-6b23-449d-96d4-8cafce975c36)  
Best accuracy reach: 99.48%  

Confusion Matrix:  
![Screenshot 2025-03-15 163108](https://github.com/user-attachments/assets/34d097cd-c64d-47da-b1ca-24610153ffae)  

- Demo:
  step 1: draw a random number.
  ![Screenshot 2025-03-21 134629](https://github.com/user-attachments/assets/589af5de-0cb5-4100-b51c-11f8f0629a34)
  step 2: copy image to test_image folder.
  ![Screenshot 2025-03-21 134718](https://github.com/user-attachments/assets/b9ddaad6-4cb1-413e-ba76-95e2a428d6f5)
  step 3: change image name in inference.py  
  ![Screenshot 2025-03-21 134740](https://github.com/user-attachments/assets/7c5f1cba-edbe-4c32-bf71-20654ee3e474)
  step 4: run inference.py.
    Result
      ![Screenshot 2025-03-21 134940](https://github.com/user-attachments/assets/e7543582-1934-4627-9a4a-d57ad58ae452)  

  





