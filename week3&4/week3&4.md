# Week  3 and Week 4 : Still learning Pytorch
##  Using Tensor
Tensor can represent high latitude data structures and be supported by GPU acceleration
##  Containers
``` forward()```

###  Convolution Layers
![grad](./images/Conv2D.png)

```Conv2D()```
Like fliter(IIP)

###  Maxed Pooling Layer
``` MaxPool2D()```
feature extract extraction   (1080p -> 360p)
###  Non-linear Activation

##  Loss Function 
A function that measures the difference between the predicted value```outputs``` and the true value```targets```
  ```L1Loss```  Mean Absolute Error
 ``` MSELoss```Mean Squared Error, or regression tasks
 ```CrossEntropyLoss```  For classification tasks

### Backpropagation
It updates the model parameters by calculating the gradient of the loss function with respect to each parameter. (Like heuristic algorithm in AIM)
![grad](./images/grd.png)

##  Optimizers

```python
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

**SGD (Stochastic Gradient Descent)**:
- Basic stochastic gradient descent optimizer.
- Optionally use momentum to speed up convergence.

##  Try VGG16 architecture
![vgg16_image](/images/VGG16.jpg)

### Load the CIFAR-10 dataset 
```python
train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform = torchvision.transforms.ToTensor(), download=True)  
test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform = torchvision.transforms.ToTensor(), download=True)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)  
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

`root`: Indicates the storage location of the dataset.
`train`: A Boolean value indicating whether to load the training set or the test set
`transform`: One for preprocessing the data (converting images into PyTorch tensors)
`download`: Download or not 

###  Build VGG16 architecture
```python
#创建神经网络  
class Tudui(nn.Module):  
    def __init__(self):  
        super(Tudui, self).__init__()  
        self.model = nn.Sequential(  
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),  
            nn.MaxPool2d(2),  
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),  
            nn.MaxPool2d(2),  
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  
            nn.MaxPool2d(2),  
            nn.Flatten(),  
            nn.Linear(64 * 4 * 4, 64),  
            nn.Linear(64,10),  
        )  
  
    def forward(self, x):  
        x = self.model(x)  
        return x
```






###  Call gpu
 ``` python
device = torch,device("cuda")
·
·
·
for data in dataloader:
	images, targets = data
	images, targets = images.to(device), targets.to(device) 
·
·
· 
```

###  Define loss function
``` python
loss_fn = nn.CrossEntropyLoss()  
loss_fn = loss_fn.cuda()
```
CIFAR-10 is a dataset for image classification, so we choose CrossEntropyLoss

### Define optimizer
```
learning_rate = 0.01  
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)
```


###  Start train
Define 100 epoches

```python
for i in range(epoch):  
    print('-----------Epoch {}/{}----------'.format(i+1, epoch))  
    tudui.train()  
    for data in train_dataloader:  
        imgs, targets = data  
        imgs, targets = imgs.cuda(), targets.cuda()  
        outputs = tudui(imgs)  
        loss = loss_fn(outputs, targets)  
  
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
  
        total_train_step += 1  
  if total_train_step % 100 == 0:  
            end_time = time.time()  
            print(end_time - start_time)  
            print(f"训练次数{total_train_step}, Loss{loss.item()}")  
            writer.add_scalar('Train_loss', loss.item(), total_train_step)  
  
    #测试  
  tudui.eval()  
    total_test_loss = 0  
  total_accuracy = 0  
  with torch.no_grad():  
        for data in test_dataloader:  
            imgs, targets = data  
            imgs, targets = imgs.cuda(), targets.cuda()  
            outputs = tudui(imgs)  
            loss = loss_fn(outputs, targets)  
            total_test_loss = total_test_loss + loss.item()  
            accuracy = (outputs.argmax(1) == targets).sum()  
            total_accuracy = total_accuracy + accuracy  
  
    print("整体测试集Loss{}".format(total_test_loss))  
    print("整体测试集Accuracy{}".format(total_accuracy/test_data_size))  
    writer.add_scalar('Test_loss', total_test_loss, total_test_step)  
    writer.add_scalar('Test_accuracy', total_accuracy/test_data_size, total_test_step)  
    total_test_step += 1
```



###  Save trained model
``` python
if total_test_step == 100:  
    torch.save(tudui, "tudui_{}.pth".format(total_test_step))  
    print("model saved")
```

### Load trained model
Load the image 
``` python
image_path = "dog.png"  
image = Image.open(image_path)  
print(image)  
image = image.convert('RGB')
``` 
Without `image = image.convert('RGB')` may report an error. PNG images may have four channels。


```python
import torchvision.transforms  
from PIL import Image  
from torch import nn  
from torchvision import transforms  
import torch  
  
image_path = "dog.png"  
image = Image.open(image_path)  
print(image)  
image = image.convert('RGB')  
transform = transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()] )  
  
image = transform(image)  
print(image.shape)
model = torch.load("tudui_100.pth", map_location=torch.device('cpu'))  
print(model)  
  
image = torch.reshape(image, (1, 3, 32, 32))  
model.eval()  
with torch.no_grad():  
    output = model(image)  
print(output)  
  
print(output.argmax(1))
```


Run successfully







> Written with [StackEdit](https://stackedit.io/).
