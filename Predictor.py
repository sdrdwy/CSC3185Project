import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,4,kernel_size=5)
        self.conv2=nn.Conv2d(4,8,kernel_size=5)
        self.conv3=nn.Conv2d(8,16,kernel_size=5)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=2)
        self.fc1=nn.Linear(11424,324)
        self.fc2=nn.Linear(324,46)
        self.dropout=nn.Dropout(0.5)
    def forward(self,x):
        x0=x.size(0)
        x=self.maxpool(self.relu(self.conv1(x)))
        x=self.maxpool(self.relu(self.conv2(x)))
        x=self.maxpool(self.relu(self.conv3(x)))
        x=x.view(x0,-1)
        x=self.dropout(x)
        x=self.relu(self.fc1(x))
        x=self.dropout(x)
        x=F.log_softmax(self.fc2(x),dim=1)
        #x=self.relu(self.fc2(x))
        return x
class Predictor:
    def __init__(self,model="model_cnn.pt",labels=[],device="gpu") -> None:
        self.device=device
        if device=="gpu":
            self.model = CNN().cuda()
            self.model.load_state_dict(torch.load(model))
        else:
            self.model = CNN()
            self.model.load_state_dict(torch.load(model,map_location=torch.device("cpu")))
        self.labels=labels
        self.transform = transforms.Compose([transforms.ToTensor()])
    def predict(self,imgpath):
        self.model.eval()
        with torch.no_grad():
            self.image=Image.open(imgpath)
            if self.device=="gpu":
                self.tensor_image=self.transform(self.image).cuda()
            else:
                self.tensor_image=self.transform(self.image)
            self.expanded_tesnor=self.tensor_image.unsqueeze(0)
            self.outputs=self.model(self.expanded_tesnor)
            self.ret=torch.argmax(self.outputs).item()
        return self.labels[self.ret]
    def load_label(self,csv,drops):
        self.data=pd.read_csv(csv)
        self.labels=self.data.drop(labels=drops,axis=1)
        
        self.labels=self.labels.columns.values.tolist()
        self.labels[0]="Normal"
if __name__=="__main__":
    pred=Predictor(model="model_cnn_best.pt")
    pred.load_label("Training_Set/RFMiD_Training_Labels.csv",["ID"])
    for i in range(1,800):
        ret=pred.predict("Training_Set/Training/"+str(i)+".png")
        print("\t"+str(i)+" ",end="")
        print(ret,end="\t")
   
