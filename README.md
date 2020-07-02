# CNN_system
Keywords: CNN, Fully connect neural network, SFEW dataset, Image Preprocessing, Data Augmentation, Leakey ReLU, k-fold cross validation, Casper. In this project, I build my own CNN system with Image Preprocessing and Data Augmentation which are based on the computation ability and characteristic of used Dataset. This project implemented with Pytorch.

Please add dataset file to root path. If there is a problem with dataset load, see function preprocessing.py to find where to add dataset.

If there is a problem while import torchvision, please do the following code in terminal
pip3 install torchvision==0.2.0

This project needs opencv library. Opencv was used to do histogram equalization for images
Please do the following code in terminal if not installed cv2
pip install cv2

Run KfoldMain.py to Evaluate designed CNN network system.
