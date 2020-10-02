from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def show_image(path):
    img = Image.open(path)
    img_arr = np.array(img)
    plt.figure(figsize=(5,5))
    plt.imshow(np.transpose(img_arr, (0, 1, 2)))

import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transformations = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

total_dataset = datasets.ImageFolder("flowers", transform = transformations)
dataset_loader = DataLoader(dataset = total_dataset, batch_size = 100)
items = iter(dataset_loader)
image, label = items.next()


num_classes = len(total_dataset.classes)
num_classes

def show_transformed_image(image):
    np_image = image.numpy()
    plt.figure(figsize=(20,20))
    plt.imshow(np.transpose(np_image, (1, 2, 0)))
    
from torch.utils.data import random_split

import torch.nn as nn

class FlowerClassifierCNNModel(nn.Module):
    
    def __init__(self, num_classes=5):
        super(FlowerClassifierCNNModel,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3,stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
        self.lf = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes)
    
    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)
        
        output = self.maxpool1(output)
        
        output = self.conv2(output)
        output = self.relu2(output)
        
        output = output.view(-1, 32 * 32 * 24)

        output = self.lf(output)

        return output
    
cnn_model = FlowerClassifierCNNModel()
cnn_model.load_state_dict(torch.load('flowermodel'))
cnn_model.to(device)


def getImageClassName(sdfds):
    test_image = Image.open(sdfds)
    test_image_tensor = transformations(test_image).float()
    test_image_tensor = test_image_tensor.unsqueeze_(0)
    test_image_tensor = test_image_tensor.to(device)
    output = cnn_model(test_image_tensor)
    output.data.cpu().numpy().argmax()
    return total_dataset.classes[output.data.cpu().numpy().argmax()].capitalize()

        
        
def get_vector(image_name):
    # 1. Load the image with Pillow library
    
    test_image = Image.open(image_name)
    test_image_tensor = transformations(test_image).float()
    test_image_tensor = test_image_tensor.unsqueeze_(0)
    test_image_tensor = test_image_tensor.to(device)

    t_img = test_image_tensor
    
    df = cnn_model(t_img)

    return df



def getCosSimi(pic_one_vector,pic_two_vector):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim = cos(pic_one_vector,
                  pic_two_vector)
    return cos_sim.tolist()[0]



def getListOfSlrImges(chooseImage):
    pic_one_vector = get_vector(chooseImage)

    listOfImages = []

    for folder in os.listdir('flowers'):
        for filename in os.listdir(os.path.join('flowers', folder)):
            pic_two_vector = get_vector(os.path.join('flowers', folder,filename) )
            #print(os.path.join('flowers', folder,filename))
            if getCosSimi(pic_one_vector,pic_two_vector) > 0.98:
                listOfImages.append((os.path.join('flowers', folder,filename)))
    return listOfImages




 
net=torch.load('newmodel.pth') 
net.eval()

# print(net)


#for getting top images 
from PIL import Image
from torch.autograd import Variable



cnn_model.eval()
cnn_model.to(device)



def get_vector(image_name):
    # 1. Load the image with Pillow library
    
    test_image = Image.open(image_name)
    test_image_tensor = transformations(test_image).float()
    test_image_tensor = test_image_tensor.unsqueeze_(0)
    test_image_tensor = test_image_tensor.to(device)

    t_img = test_image_tensor
    
    df = cnn_model(t_img)

    return df


def getCosSimi(pic_one_vector,pic_two_vector):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim = cos(pic_one_vector,
                  pic_two_vector)
    return cos_sim.tolist()[0]

from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *
import tkinter.messagebox as messagebox
from PIL import ImageTk, Image
import os

#from tkinter.filedialog import askopenfilename,ask


window = Tk()
window.title("Welcome to Neural Style Transfer")


def selectContentFile():
    filename = filedialog.askopenfilename(initialdir='C:/Users/ashish/Machine Learning Noida')
    if filename != "":
        if filename.endswith('jpg') or filename.endswith('jpeg') or filename.endswith('png'):
            txt.delete(0,END)
            txt.insert(0,filename)   
            
            
            content_image = Image.open(txt.get())
            resized = content_image.resize(( int(content_image.width*0.75), int(content_image.height*0.75) ), Image.ANTIALIAS)
            content_photo = ImageTk.PhotoImage(resized)
            content_label.configure(image=content_photo)
#          content_label = Label(image=content_photo)
            content_label.image = content_photo # keep a reference!
            content_label.grid(row =1, column =1,pady=(20,20))
            

            


    
def applyStyle():
    
    
    if(txt.get()=='' ):
        messagebox.showerror("Error", "Please Fill All the Boxes")
    else:
        sdf = getImageClassName(txt.get())
        sdf = 'It is a '+sdf
        lbl2.config(text=sdf)
        lis = getListOfSlrImges(txt.get())
        lis = lis[:6]
        COLUMNS = 3
        image_count = 0
        for infile in lis:
            infile.replace("\\","/")
            print(infile)
            if infile in txt.get():
                continue
            image_count += 1
            r, c = divmod(image_count-1, COLUMNS)
            im = Image.open(infile )
            im = im.resize((150,150))
            cp = ImageTk.PhotoImage(im)
            cl = Label(image=cp)
            cl.image = cp # keep a reference!
            cl.grid(row=r+5, column=c)
            r=r+1
        
    
#Selecting the content image
            
lbl = Label(window, text=" Your Content Image: ")
lbl.grid(column=0, row=0,pady=(5,6))

txt = Entry(window,width=75)
txt.grid(column=1, row=0,pady=(5,6))


btn = Button(window, text="Select File", command=selectContentFile)
btn.grid(column=2, row=0, padx=(10,10))

content_image = Image.open("rose.jpg")
resized = content_image.resize(( int(content_image.width*0.75), int(content_image.height*0.75) ), Image.ANTIALIAS)

content_photo = ImageTk.PhotoImage(resized)
content_label = Label(image=content_photo)
content_label.image = content_photo # keep a reference!
content_label.grid(row =1, column =1,pady=(20,20))
        
    
btn2 = Button(window, text="Apply", command=applyStyle)
btn2.grid(column=1, row=2)

lbl2 = Label(window, text="")
lbl2.grid(column=1, row=3,pady=(20,20))


window.mainloop()



