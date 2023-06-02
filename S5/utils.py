from torchvision import transforms
import matplotlib.pyplot as plt

"""CODE BLOCK: 3"""
def data_transforms():

    # Train data transformations
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])

    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1407,), (0.4081,))
        ])
    
    return train_transforms, test_transforms

"""CODE BLOCK: 6"""
def plot_data(batch_data, batch_label, num_plots, row, col): 

# batch_data, batch_label = next(iter(train_loader)) 

    fig = plt.figure()

    for i in range(num_plots):
        plt.subplot(row,col,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

"""CODE BLOCK: 11"""
def plot_loss_accuracy(train_losses, test_losses, train_acc, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")