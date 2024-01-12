# # Standard library imports
# import os
# import xml.etree.ElementTree as ET
# from datetime import datetime
#
# # Third-party library imports
# import cv2
# import torch
# from torch import nn
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torchvision.transforms import v2
#
# # Local module imports
# from potato_model import PotatoNet
#
#
# label_list = []
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# writer = SummaryWriter("runs/pose_model_{}".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
#
#
# class MyDataset(Dataset):
#     def __init__(self, image_list, labels):
#         self.image_list = image_list
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.image_list)
#
#     def __getitem__(self, idx):
#         image = self.image_list[idx]
#
#         label = self.labels[idx]
#
#         # if self.transform:
#
#         return image, label
#
#
# def video_to_frames(video_path):
#     frames = []
#     cap = cv2.VideoCapture(video_path)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#     return frames
#
#
# def create_time_windows(frames, window_size):
#     windows = []
#     for i in range(len(frames) - window_size + 1):
#         window = frames[i:i + window_size]
#         windows.append(window)
#
#     return windows
#
#
# def training_loop(model, train_loader, test_loader, num_epochs, loss_fn, optimizer):
#     i = -1
#     for epoch in range(num_epochs):
#         for idx, (inputs, labels) in enumerate(train_loader):
#             model.train()
#             i += 1
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = loss_fn(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             del inputs, labels, outputs
#
#             if (idx + 1) % 1 == 0:
#                 print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{idx + 1}], Loss: {loss.item():.6f}')
#                 writer.add_scalar('Loss/train', loss, i)
#             if (idx + 1) % 10 == 0:
#                 model.eval()
#                 with torch.no_grad():
#                     correct = 0
#                     total = 0
#                     for _, (inputs, labels) in enumerate(test_loader):
#                         inputs = inputs.to(device)
#                         labels = labels.to(device)
#                         outputs = model(inputs)
#                         _, predicted = torch.max(outputs.data, 1)
#                         print(inputs.size)
#                         print(labels.size)
#                         print(predicted.size)
#                         total += labels.size(0)
#                         correct += (predicted == labels).sum().item()
#
#                         print(total)
#                         print(correct)
#
#                         accuracy = (correct / total) * 100
#                         del inputs, labels, outputs, predicted
#
#                     writer.add_scalar('Accuracy/test', accuracy, i)
#                     print(f'Accuracy of the network on the test images: {accuracy} %')
#
#     torch.save(model.state_dict(), "model/PoseModel_epoch{}.pt".format(epoch))
#     torch.save(model, "model/PoseModel_whole.pt")
#
#
# def read_xml(path, id_to_search):
#     mytree = ET.parse(path)
#     myroot = mytree.getroot()
#     for x in myroot.findall('behaviours'):
#         for y in x.findall('behaviour'):
#             if y.attrib['id'] == id_to_search:
#                 for category in y.findall('category'):
#                     label_list.append(category.text)
#
#
# def main():
#     num_epochs = 1000
#     learning_rate = 1e-4
#     timesteps = 10
#
#     # model = PoseModel().to(device)
#     model = PotatoNet(num_classes=3, hidden_size=128, num_layers=2).to(device)
#
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#
#     vid_folder_path = './data/out/'
#     label_folder_path = './data/label/'
#
#     vid_filename_list = os.listdir(vid_folder_path)
#     vid_path_list = [os.path.join(vid_folder_path, vid_filename) for vid_filename in vid_filename_list]
#
#     for vid_name in vid_filename_list:
#         strr = vid_name.split('.')[0]
#         b_id = strr.split('_')[3] + '_' + strr.split('_')[4]
#         filename = strr[:-5]
#         read_xml(os.path.join(label_folder_path, filename + '.xml'), b_id)
#
#     transform = v2.Compose([
#         v2.ToPILImage(),
#         v2.Resize((64, 64)),
#         v2.ToImage(),
#         v2.ToDtype(torch.float32, scale=True)
#         # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     frames = []
#     labels = []
#     windows = []
#
#     for idx, path in enumerate(vid_path_list):
#         cap = cv2.VideoCapture(path)
#
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             transformed_frame = transform(frame)
#             # transformed_frame = transformed_frame.permute(2, 0, 1)
#             frames.append(transformed_frame)
#             if len(frames) == timesteps:
#                 window = torch.stack(frames)
#                 windows.append(window)
#                 labels.append(label_list[idx])
#                 if label_list[idx] == 'handclapping' or label_list[idx] == 'fingerrubbing':
#                     windows.append(window)
#                     labels.append(label_list[idx])
#
#                 frames.clear()
#                 del window
#
#         frames.clear()
#
#         cap.release()
#         cv2.destroyAllWindows()
#
#     digi_label = []
#
#     a = 'handclapping'
#     b = 'armflapping'
#     c = 'fingerrubbing'
#
#     for l in labels:
#         if a in l:
#             digi_label.append(0)
#         elif b in l:
#             digi_label.append(1)
#         elif c in l:
#             digi_label.append(2)
#
#     digi_label_tensor = torch.tensor(digi_label)
#     windows = torch.stack(windows)
#
#     train_data = MyDataset(image_list=windows, labels=digi_label_tensor)
#
#     train_size = int(0.8 * len(train_data))
#     test_size = len(train_data) - train_size
#     train_data, test_data = torch.utils.data.random_split(train_data, [train_size, test_size])
#
#     train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True)
#     test_loader = DataLoader(dataset=test_data, batch_size=256, shuffle=True)
#
#     training_loop(model, train_loader, test_loader, num_epochs, loss_fn, optimizer)
#
#     writer.close()
#
#
# if __name__ == '__main__':
#     main()
# Standard library imports
import os
import xml.etree.ElementTree as ET
from datetime import datetime

# Third-party library imports
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2

# Local module imports
from potato_model import PotatoNet  # Importing a custom neural network module

label_list = []  # Initialize an empty list to store labels
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Determine the device (GPU or CPU) to use

# Create a SummaryWriter for TensorBoard logs
writer = SummaryWriter("runs/pose_model_{}".format(datetime.now().strftime("%Y%m%d-%H%M%S")))


# Define a custom Dataset class for handling the data
class MyDataset(Dataset):
    def __init__(self, image_list, labels):
        self.image_list = image_list
        self.labels = labels

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        label = self.labels[idx]
        return image, label


# Function to create time windows from a list of frames
def create_time_windows(frames, window_size):
    windows = []
    for i in range(len(frames) - window_size + 1):
        window = frames[i:i + window_size]
        windows.append(window)
    return windows


# Function to train the model
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

            if (idx + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
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
                        del inputs, labels, outputs, predicted

                    writer.add_scalar('Accuracy/test', accuracy, i)
                    print(f'Accuracy of the network on the test images: {accuracy} %')

        torch.save(model.state_dict(), "model/PoseModel_epoch{}.pt".format(epoch))
    torch.save(model, "model/PoseModel_whole.pt")


# Function to read XML files and extract labels
def read_xml(path, id_to_search):
    mytree = ET.parse(path)
    myroot = mytree.getroot()
    for x in myroot.findall('behaviours'):
        for y in x.findall('behaviour'):
            if y.attrib['id'] == id_to_search:
                for category in y.findall('category'):
                    label_list.append(category.text)


# Function to prepare data for training and testing
def prepare_data():
    # Define paths for video and label folders
    vid_folder_path = './data/out/'
    label_folder_path = './data/label/'

    timesteps = 10

    vid_filename_list = os.listdir(vid_folder_path)
    vid_path_list = [os.path.join(vid_folder_path, vid_filename) for vid_filename in vid_filename_list]

    # Read XML labels for each video
    for vid_name in vid_filename_list:
        strr = vid_name.split('.')[0]
        b_id = strr.split('_')[3] + '_' + strr.split('_')[4]
        filename = strr[:-5]
        read_xml(os.path.join(label_folder_path, filename + '.xml'), b_id)

    # Define a data transformation pipeline
    transform = v2.Compose([
        v2.ToPILImage(),
        v2.Resize((64, 64)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    frames = []
    labels = []
    windows = []

    # Process video frames and labels
    for idx, path in enumerate(vid_path_list):
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            transformed_frame = transform(frame)
            frames.append(transformed_frame)
            if len(frames) == timesteps:
                window = torch.stack(frames)
                windows.append(window)
                labels.append(label_list[idx])
                if label_list[idx] == 'handclapping' or label_list[idx] == 'fingerrubbing':
                    windows.append(window)
                    labels.append(label_list[idx])
                frames.clear()
                del window

        frames.clear()
        cap.release()
        cv2.destroyAllWindows()

    digi_label = []

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

    digi_label_tensor = torch.tensor(digi_label)
    windows = torch.stack(windows)

    train_data = MyDataset(image_list=windows, labels=digi_label_tensor)

    train_size = int(0.8 * len(train_data))
    test_size = len(train_data) - train_size
    train_data, test_data = torch.utils.data.random_split(train_data, [train_size, test_size])

    train_loader = DataLoader(dataset=train_data, batch_size=512, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=512, shuffle=True)

    return train_loader, test_loader  # Return the data loaders


# Main function
def main():
    num_epochs = 100
    learning_rate = 1e-4

    # Create an instance of the custom neural network model (PotatoNet)
    model = PotatoNet(num_classes=3, hidden_size=128, num_layers=2).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Prepare data for training and testing
    train_loader, test_loader = prepare_data()

    # Train the model using the training loop
    training_loop(model, train_loader, test_loader, num_epochs, loss_fn, optimizer)

    # Close the TensorBoard writer
    writer.close()


if __name__ == '__main__':
    main()
