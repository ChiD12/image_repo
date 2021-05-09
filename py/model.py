import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import Adam
import torchvision.transforms as transforms
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
import math
import re

imageDir = "./images"

#Dataset object for pytorch to use
class ImagesDataset(Dataset):
    
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)
        self.all_imgs.sort(key=sortFunc)
        print(self.all_imgs)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")

        if self.transform is not None:
            tensor_image = self.transform(image)
            tensor_image = nomalizeImgShape(tensor_image)

        return tensor_image, tensor_image

def sortFunc(e):
    indexOfFirstAlpha = re.search(r"[a-zA-Z]", e).start()
    return int(e[0:indexOfFirstAlpha])

def nomalizeImgShape(img):
    desiredW = 256
    desiredH = 256

    imgH = img.shape[1]
    imgW = img.shape[2]

    if imgH > imgW:
        dif = imgH - imgW
        split = int(dif/2)
        if dif %2 == 0:
            img = img[:, split: imgH-(split), :]
        elif dif %2 != 0:
            img = img[:, split +1: imgH-(split), :]
    elif imgH < imgW:
        dif = imgW - imgH
        split = int(dif/2)
        if dif %2 == 0:
            img = img[:, :, split: imgW-(split)]
        elif dif %2 != 0:
            img = img[:, :, split +1: imgW-(split)]

    downsampleTimes = int(min(img.shape[1]/(desiredH *2), img.shape[2]/(desiredW*2)))

    #if we cant downsample by atleast 2
    if downsampleTimes == 0:
        #how much we need to cut off from all sides
        hToCut =  (img.shape[1]- desiredH)/2
        wToCut =  (img.shape[2]- desiredW)/2
        
        #if the image has an odd resolutions, use ceil for lower bound 
        img3 = img[:: , math.ceil(hToCut)  :img.shape[1] - math.floor(hToCut), math.ceil(wToCut):img.shape[2] - math.floor(wToCut)]
        
    else:
        #downsample
        downsampleTimes = 2* downsampleTimes
        img2 = img[:: , :: downsampleTimes, ::downsampleTimes]

        #cut off extra
        hToCut =  (img2.shape[1]- desiredH)/2
        wToCut =  (img2.shape[2]- desiredW)/2
        img3 = img2[:: , math.ceil(hToCut)  :img2.shape[1] - math.floor(hToCut), math.ceil(wToCut):img2.shape[2] - math.floor(wToCut)]
    

    return img3

class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.maxpool3 = nn.MaxPool2d((2, 2))

        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.maxpool4 = nn.MaxPool2d((2, 2))

        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1))
        self.maxpool5 = nn.MaxPool2d((2, 2))

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        torch.nn.init.xavier_uniform_(self.conv5.weight)

    def forward(self, x):
        # Downscale the image with conv maxpool etc.
        relu = torch.nn.ReLU()

        x = relu(self.conv1(x))
        x = self.maxpool1(x)

        x = relu(self.conv2(x))
        x = self.maxpool2(x)

        x = relu(self.conv3(x))
        x = self.maxpool3(x)

        x = relu(self.conv4(x))
        x = self.maxpool4(x)

        x = relu(self.conv5(x))
        x = self.maxpool5(x)
        
        return x

class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2))
        self.deconv2 = nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2))
        self.deconv3 = nn.ConvTranspose2d(64, 32, (2, 2), stride=(2, 2))
        self.deconv4 = nn.ConvTranspose2d(32, 16, (2, 2), stride=(2, 2))
        self.deconv5 = nn.ConvTranspose2d(16, 3, (2, 2), stride=(2, 2))

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.deconv1.weight)
        torch.nn.init.xavier_uniform_(self.deconv2.weight)
        torch.nn.init.xavier_uniform_(self.deconv3.weight)
        torch.nn.init.xavier_uniform_(self.deconv4.weight)
        torch.nn.init.xavier_uniform_(self.deconv5.weight)

    def forward(self, x):
         # Upscale the image with convtranspose etc.
        relu = torch.nn.ReLU()

        x = relu(self.deconv1(x))
        x = relu(self.deconv2(x))
        x = relu(self.deconv3(x))
        x = relu(self.deconv4(x))
        x = relu(self.deconv5(x))

        return x

def trainModels():
    transform = transforms.Compose([
        transforms.ToTensor()    
    ])

    ImagesDS = ImagesDataset("./images", transform)
    print(ImagesDS.all_imgs)
    
    #split datasaet into training and validation
    val_percent = 0.3
    val_size = int(len(ImagesDS)*val_percent)
    train_size = len(ImagesDS) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(ImagesDS, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    # Training hyper parameters
    epochs = 150
    learning_rate = 0.006

    device = "cpu"
    print(f'device: {device}')

    encoder = ConvEncoder()
    decoder = ConvDecoder()

    encoder.to(torch.device(device))
    decoder.to(torch.device(device))

    autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = Adam(autoencoder_params, lr=learning_rate) # Adam Optimizer
    loss_fn = nn.MSELoss()
          

    for epoch in range(epochs):
        train_loss = train_step(encoder, decoder, train_loader, loss_fn, optimizer, device=device)
        val_loss = train_step(encoder, decoder, val_loader, loss_fn, optimizer, device=device, val = True)

        print(f"Epochs = {epoch}, Training Loss : {train_loss}")
        print(f"Epochs = {epoch}, Validation Loss : {val_loss}")

        #save best models
        min_loss = 1
        if val_loss < min_loss:
            print("Validation Loss decreased, saving new best model")
            torch.save(encoder.state_dict(), "./models/encoder_model.pt")
            torch.save(decoder.state_dict(), "./models/decoder_model.pt")
            min_loss = val_loss
            

    EMBEDDING_SHAPE = (1, 256, 8, 8)
    Images_loader = torch.utils.data.DataLoader(ImagesDS, batch_size=32)
    embedding = create_embedding(encoder, Images_loader, EMBEDDING_SHAPE, device)

    #save embedings
    numpy_embedding = embedding.cpu().detach().numpy()
    num_images = numpy_embedding.shape[0]
    flattened_embedding = numpy_embedding.reshape((num_images, -1))
    np.save("./models/data_embedding.npy", flattened_embedding)


def train_step(encoder, decoder, train_loader, loss_fn, optimizer, device, val = False):
    #  Set networks to train mode.
    encoder.train()
    decoder.train()

    if not val:
        for i, (train_img, target_img) in enumerate(train_loader):
            # Move images to device
            train_img = train_img.to(device)
            target_img = target_img.to(device)
            
            optimizer.zero_grad()

            enc_output = encoder(train_img)
            dec_output = decoder(enc_output)
            loss = loss_fn(dec_output, target_img)

            loss.backward()
            optimizer.step()
    if val:
        with torch.no_grad():
            for i, (train_img, target_img) in enumerate(train_loader):
                # Move images to device
                train_img = train_img.to(device)
                target_img = target_img.to(device)

                enc_output = encoder(train_img)
                dec_output = decoder(enc_output)
                loss = loss_fn(dec_output, target_img)
    
    return loss.item()


def create_embedding(encoder, full_loader, embedding_dim, device):
    encoder.eval()
    embedding = torch.randn(embedding_dim)
    
    with torch.no_grad():
        for i, (train_img, target_img) in enumerate(full_loader):
            train_img = train_img.to(device)
            
            enc_output = encoder(train_img).cpu()
            embedding = torch.cat((embedding, enc_output), 0)
    
    return embedding

#compare an image to all the image imbeddings 
def compute_similar_images(image, num_images, embedding= None, device = None):
    
    image_tensor = transforms.ToTensor()(image)
    # image_tensor = nomalizeImgShape(image_tensor)
    image_tensor = image_tensor.unsqueeze(0)
    
    encoder = ConvEncoder()
    encoder.load_state_dict(torch.load("./models/encoder_model.pt"))
    encoder.eval()

    embedding = np.load("./models/data_embedding.npy")

    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()
        
    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))

    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)

    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.flatten().tolist()
    print(indices)
    return indices_list