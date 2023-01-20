# Comparing-Different-Model-Architectures
I studied many neural network architectures in this research that were modeled after well-known architectures like ResNet, VGG, and LeNet. On a dataset with hundreds of 32 \* 32 pixel images, I have compared how well they performed.

## Results
VGG-19 gives the best overall performance, closely followed by ResNet-34. LeNet's performance was well behind that of either of the two earlier models. This may be as a result of LeNet's architecture being so straightforward. It was unexpected to see VGG outperform ResNet. I believe this is as a result of the best CIFAR-10 performance being achieved by many convolutional layers followed by pooling.

## Accuracy scores
VGG: 96% accuracy
ResNet: 95% accuracy
LeNet: 64% accuracy
