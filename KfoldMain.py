from CNNnet import CNNnet
from DataLoader import *
import torch
from torch import nn

from utils import plot_confusion

# Use GPU if there is. Or just use CPU. CPU will be very slow.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 7
num_epochs = 30  # Add num_epochs if using data augmentation
batch_size = 64  # change batch_size if memory cost over the limitation. Default for this project, batch = 64
k = 10  # Here can change the number of k in k-fold cross validation

net_list = list()
accuracy_list = list()
for n in range(k):
    # Load data for current fold
    train_loader, validation_loader = load_data_kfold(batch_size, k, n)
    print("No.", n, "Fold")
    net = CNNnet()  # Load empty cnn model
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # After several testing, Adadelta is faster than other optimizers.
    optimizer = torch.optim.Adadelta(net.parameters(), lr=0.01, weight_decay=0.00001)

    total_step = len(train_loader)
    # Start training for current fold
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            net = net.to(device)
            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)  # Get predicted result
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # Show current loss and accuracy in train set
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, num_epochs, loss.item(),
                                                                     100 * correct / total), '%')
    path = "pretrained_model/model{}".format(n)
    torch.save(net, path)

    # Test the model. The logic is similar to train model. But no grad and no backward.
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    val_data = [torch.Tensor(), torch.Tensor()]
    for i, (inputs, labels) in enumerate(validation_loader):
        val_data[0] = torch.cat((val_data[0], inputs), 0)
        val_data[1] = torch.cat((val_data[1].float(), labels.float()))
    with torch.no_grad():
        correct = 0
        total = 0
        inputs = val_data[0].to(device)
        labels = val_data[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # show Confusion matrix. Then it can be used to calculate recall, precision, ...
        print('Confusion matrix for testing:')
        print(plot_confusion(inputs.shape[0], num_classes, predicted.long().data, labels.long().data))
        print("No.", n, "Fold test Accuracy of the model on the test images: {} %".format(100 * correct / total))
        accuracy_list.append(100 * correct / total)

# Show results
for n in range(len(accuracy_list)):  # show results for each fold
    print("No.", n, "Fold test Accuracy of the model on the test images: {} %".format(accuracy_list[n]))
print("Std in 10-Fold is", np.std(accuracy_list))
print("Med in 10-Fold is", np.median(accuracy_list), "%")
print("Average accuracy in 10-Fold is", np.average(accuracy_list), "%")
