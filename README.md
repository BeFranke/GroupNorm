# DeepVision - GroupNormalization Project
 by Benedikt Franke

## Project Structure
| File / Folder name | Description  |
| ------------------ |----------- |
| logs/              | tensorboard logs of the training |
| models/            | trained models in keras format   |
| layer.py           | keras implementation of GroupNorm layer |
| plot.py            | script to produce plot.png from results.csv |
| plot.png           | reproduction of Figure 1 from [1] on CIFAR-10 with smaller network |
| README.md          | this file |
| reproduce.py       | writes new results.csv by evaluating trained models |
| results.csv        | results of the evaluation                           |
| train.py           | training script, contains importable model definition |
| train.sh           | slurm-job for train.py |

## Motivation of Methodology
The basic methodology of [1] was to take a working setup to train ResNet50 on ImageNet, 
then replace BatchNorm with GroupNorm and observe the differences.
For this, they used the ImageNet-training procedure described in [3].
They did not optimize any parameters separately.

To keep close to the general procedure, while using another network and another dataset, 
I decided to follow the CIFAR-10 training procedure and ResNet-architecture described in [2], Section 4.2. 
I selected the ResNet20-architecture from the described networks with the adaptions for CIFAR-10, 
which mostly consist of a lower amount of feature maps, no intermediate pooling and longer network 
paths without change in dimensions compared to ResNet for ImageNet.
My only deviation from the described architecture consists in the residual connections. 
While in [2], only identity shortcuts where used to keep the number of parameters equal to their 
non-residual baseline, I used projection shortcuts when dimensions change between layers. 
The experiments in [2] showed projection shortcuts to be beneficial, and I do not have a baseline 
I need to match in parameters.

For the parameter G of the GroupNorm layer, [1] used 32 in all cases. As the CIFAR-ResNet has layers with
only 16 feature maps, I set G = 8 to keep to the relationship that G is equal to half of the minimum amount of 
feature maps in the network.

I also deviated from the training procedure of [2] slightly, as I train for 100 epochs 
(with variable batch size for the experiments) instead of 64k iterations with a fixed batch size of 128.
The number of epochs, and the learning rate schedule were again taken from [1], 
which is similar enough to [2] (dividing the learning rate by 10 at evenly spaced points of the training).
I used nesterov momentum like recommended in [3].

While [1] evaluated their models on center 224 x 224 crops of ImageNet and the median 
of the final 5 epochs' validation error were taken, 
the evaluation for my experiments was done on the plain, unmodified test set of CIFAR-10 like in [2].
Also like in [2], only the final test score is reported.

## Reproducing the Results

## References
[1] Wu & He (2018), Group Normalization, https://arxiv.org/abs/1803.08494

[2] He et al. (2015), Deep Residual Learning for Image Recognition, https://arxiv.org/abs/1512.03385

[3] Gross & Wilber (2016), Training and investigating residual nets, https://github.com/facebook/fb.resnet.torch
