from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import os
import cv2


# This file contains some function to build image dataloader for pytorch, image preprocessing and data augmentation

def build_dataset():
    # This function build dataset from Subset For Assignment SFEW new
    width = 231
    height = 288
    data_transforms = T.Compose([
        # T.ToPILImage(),
        T.Resize(size=(width, height)),  # resize the image
        T.Grayscale(),
        # T.RandomRotation(degrees=5), # Uncomment this line to add image random rotation
        # T.RandomHorizontalFlip(p=0.5), # Uncomment this line to add image horizontal flipping
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if os.path.exists("Subset For Assignment SFEW new"):
        dataset = ImageFolder('Subset For Assignment SFEW new/', transform=data_transforms)
    else:
        dataset = ImageFolder('Subset For Assignment SFEW/', transform=data_transforms)

    return dataset


def create_path():
    # This function create path for storing Equalizes the histogram
    path = "Subset For Assignment SFEW new"
    if not os.path.exists(path):
        class_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        os.makedirs(path)
        for emotion in class_list:
            os.makedirs(path + '/' + emotion)
        equalize_hist()


def equalize_hist():
    # This function do the histogram equalization for all images and write them to a new file directory
    path = 'Subset For Assignment SFEW/'
    new_path = 'Subset For Assignment SFEW new/'
    class_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    for emotion in class_list:
        for root, dir, files in os.walk(path + emotion):
            for file in files:
                srcImage = cv2.imread(path + emotion + '/' + str(file))
                # srcImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
                roiImage = hisEqulColor(srcImage)
                roiImage = cv2.medianBlur(roiImage, 3) # Smooth images, reduce noises
                cv2.imwrite(new_path + emotion + '/' + str(file), roiImage)  # write new histogram equalization images


def hisEqulColor(img):
    # This function apply histogram equalization to a color image
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img
