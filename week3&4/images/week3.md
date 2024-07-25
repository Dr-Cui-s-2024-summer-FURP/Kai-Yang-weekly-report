# Week  3 and Week 4 : Still learning Pytorch
##  Using Tensor
Tensor can represent high latitude data structures and be supported by GPU acceleration
##  Containers
``` forward()```

###  Convolution Layers
```Conv2D()```
fliter(IIP)
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


##  Try VGG16 architecture
Load the CIFAR-10 data set and train

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

###  Save trained model
