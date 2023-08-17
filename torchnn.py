from torch import nn, save, load # neural nets
from torch.optim import Adam #optimizer
from torch.utils.data import DataLoader
from torchvision import datasets # get images
from torchvision.transforms import ToTensor #import images to Tensors
import torch
from PIL import Image


#use image classiication datasets. Classes are 0-9
# Getting data below
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32) # batches of 32 images
#1,28,28, classes - 0-9

# create NN class Image Classifier class
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            #use CNN layers
            # input channel is 1, since images are black & white
            # 32 filters/kernels of shape 3x3
            nn.Conv2d(1, 32, (3,3)),  # first layer
            #activatoins for non-linearity
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),  # second layer -> takes in 32 inputs 64
            #activatoins for non-linearity
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),  # 3rd layer -> takes in 64 inputs 64
            #activatoins for non-linearity
            nn.ReLU(),
            ## We are taking in 2 pixels and shaving off  height and width of the image
            ## We are going to adjust in o/p when we apply linear layer. 
            nn.Flatten(),
            # input layer 64 passed into CNN --> 64 since final o/p from CNN layer
            # image from MNIS are 1, 28, 28. For 3 CNN layers we are shaving off 2 each. So substract 6
            # o/p shapes is no. of classes (0-9) so 10 o/ps
            nn.Linear(64 * (28-6) * (28-6), 10) 
        )

    #akin to call() method in Tensorflow
    def forward(self, x): 
        return self.model(x)
    
    #Instance of NN, Loss, optimizer
clf = ImageClassifier().to('cpu') #gpu edition is cuda equivalent. Since no "GPU" I set this to CPU. 
opt = Adam(clf.parameters(), lr=1e-3) #optimizer
loss_fn = nn.CrossEntropyLoss() #loss function

    #Training flow (Run this first)
# if __name__ == "__main__":
#         #run DL for a number of 10 epochs
#         for epoch in range(10):
#             for batch in dataset:
#                 X,y = batch
#                 X, y = X.to('cpu'), y.to('cpu')
#                 yhat = clf(X) #generate prediction
#                 loss = loss_fn(yhat, y)

#                 #apply backprop
#                 opt.zero_grad() #zreo out existing gradients
#                 loss.backward() #calculate gradients
#                 opt.step() # go and take a step and apply gradient descent
#             print(f"Epoch:{epoch} loss is {loss.item()}")
    
# with open('model_state.pt','wb') as f:
#         save(clf.state_dict(), f)
if __name__ == "__main__":
    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f))
    # make predictions
    #img = Image.open('img_1.jpg') ## This predicts image 1
    img = Image.open('img_2.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')
    print(torch.argmax(clf(img_tensor)))





        




