# Standard library imports
import os
import xml.etree.ElementTree as ET  # For parsing XML files
from datetime import datetime

# Third-party library imports
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # For logging in TensorBoard
from torchvision.transforms import v2  # Image transformations

# Local module imports
from ori_model import PoseModel  # Importing the PoseModel from local file

label_list = []  # List to store labels
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initializing TensorBoard writer for logging
writer = SummaryWriter("runs/pose_model_{}".format(datetime.now().strftime("%Y%m%d-%H%M%S")))


# Custom dataset class for handling image data
class MyDataset(Dataset):
    def __init__(self, image_list, labels, transform=None):
        # Initialize dataset with images, labels, and optional transform
        self.image_list = image_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # Return the total number of images in the dataset
        return len(self.image_list)

    def __getitem__(self, idx):
        # Retrieve an image and its label by index
        image = self.image_list[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)  # Apply transformation

        return image, label


# Training loop for the model
def training_loop(model, train_loader, test_loader, num_epochs, loss_fn, optimizer):
    i = -1

    for epoch in range(num_epochs):
        for idx, (inputs, labels) in enumerate(train_loader):
            model.train()
            i += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Memory management
            del inputs, labels, outputs

            # Logging for TensorBoard
            if (idx + 1) % 1 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{idx + 1}], Loss: {loss.item():.6f}')
                writer.add_scalar('Loss/train', loss, i)

            # Validation phase
            if (idx + 1) % 50 == 0:
                model.eval()  # Set model to evaluation mode
                with torch.no_grad():  # Disable gradient calculation
                    correct = 0
                    total = 0
                    for _, (inputs, labels) in enumerate(test_loader):
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                    accuracy = (correct / total) * 100
                    # Memory management
                    del inputs, labels, outputs, predicted

                    writer.add_scalar('Accuracy/test', accuracy, i)
                    print(f'Accuracy of the network on the test images: {accuracy} %')

        # Save the model at the end of each epoch
        torch.save(model.state_dict(), "model/PoseModel_epoch{}.pt".format(epoch))
    # Save the final model
    torch.save(model, "model/PoseModel_whole.pt")


# Function to read XML and extract labels
def read_xml(path, id_to_search):
    mytree = ET.parse(path)  # Parse the XML file
    myroot = mytree.getroot()  # Get the root of the XML
    # Iterate over XML elements to find and append labels
    for x in myroot.findall('behaviours'):
        for y in x.findall('behaviour'):
            if y.attrib['id'] == id_to_search:
                for category in y.findall('category'):
                    label_list.append(category.text)


# Function to prepare data for training and testing
def prepare_data():
    vid_folder_path = './data/out/'
    label_folder_path = './data/label/'

    vid_filename_list = os.listdir(vid_folder_path)
    vid_path_list = [os.path.join(vid_folder_path, vid_filename) for vid_filename in vid_filename_list]

    # Read labels from XML files
    for vid_name in vid_filename_list:
        strr = vid_name.split('.')[0]
        b_id = strr.split('_')[3] + '_' + strr.split('_')[4]
        filename = strr[:-5]
        read_xml(os.path.join(label_folder_path, filename + '.xml'), b_id)

    # Define image transformations
    transform = v2.Compose([
        v2.ToPILImage(),
        v2.Resize((224, 224)),
        v2.Grayscale(num_output_channels=3),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    frames = []
    labels = []
    i = 0
    # Read frames from each video and associate them with labels
    for idx, path in enumerate(vid_path_list):
        cap = cv2.VideoCapture(path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Break the loop if there are no frames left
            frames.append(frame)  # Append the frame to the list
            labels.append(label_list[idx])  # Append the corresponding label

            i += 1
        cap.release()
        cv2.destroyAllWindows()

    digi_label = []  # List to store digital labels

    # Define label names
    handclapping_label = 'handclapping'
    armflapping_label = 'armflapping'
    fingerrubbing_label = 'fingerrubbing'

    # Convert label names to numeric labels
    for l in labels:
        if handclapping_label in l:
            digi_label.append(0)
        elif armflapping_label in l:
            digi_label.append(1)
        elif fingerrubbing_label in l:
            digi_label.append(2)

    # Create a dataset with the frames and digital labels
    train_data = MyDataset(image_list=frames, labels=digi_label, transform=transform)

    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(train_data))
    test_size = len(train_data) - train_size
    train_data, test_data = torch.utils.data.random_split(train_data, [train_size, test_size])

    # Create data loaders for training and testing
    train_loader = DataLoader(dataset=train_data, batch_size=512, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=512, shuffle=True)

    return train_loader, test_loader  # Return the data loaders


# Main function to execute the training process
def main():
    num_epochs = 10  # Number of training epochs
    learning_rate = 1e-4  # Learning rate for the optimizer

    # Initialize the model and move it to the appropriate device
    model = PoseModel().to(device)

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Prepare data for training and testing
    train_loader, test_loader = prepare_data()

    # Execute the training loop
    training_loop(model, train_loader, test_loader, num_epochs, loss_fn, optimizer)

    writer.close()


if __name__ == '__main__':
    main()
