# Comparing-Different-Model-Architectures
I studied many neural network architectures in this research that were modeled after well-known architectures like ResNet, VGG, and LeNet. On a dataset with hundreds of 32*32 pixel images, I have compared how well they performed.

## Results
VGG-19 performs the best overall, with ResNet-34 following close behind. LeNet did not come close to the performance of either of the two previous models. This might be due to the simplicity in LeNet's architecture. It was surprising to see VGG perform better than ResNet. I think this is due to the fact that multiple convolutional layers followed with pooling perform the best on CIFAR-10. 

## Accuracy scores
VGG: 96% accuracy
ResNet: 95% accuracy
LeNet: 64% accuracy
