# DeepVision - Group Normalization Project
 by Benedikt Franke

This document contains an explanation of the project, 
motivation of the methodology used, answers to the theoretical questions 
(Part "Reading and Theory" of the assignment), and instructions on how to reproduce 
my results.

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
| train.sh           | slurm-job for train.py (irrelevant for the submission) |

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

For the parameter G of the GroupNorm layer, [1] used G = 32 in all cases. As the CIFAR-ResNet has layers with
only 16 feature maps, I set G = 8 to keep to the relationship that G is equal to half of the minimum amount of 
feature maps in the network.

I also deviated from the training procedure of [2] slightly, as I train for 100 epochs 
(with variable batch size for the experiments) instead of 64k iterations with a fixed batch size of 128.
Fixing iterations would introduce an error, as smaller batch sizes would get less training than larger batch sizes.
The number of epochs, and the learning rate schedule were again taken from [1], 
which is similar enough to [2] (dividing the learning rate by 10 at evenly spaced points of the training).

While [1] evaluated their models on center 224 x 224 crops of ImageNet and the median 
of the final 5 epochs' validation error were taken, 
the evaluation for my experiments was done on the plain, unmodified test set of CIFAR-10 like in [2].

## Theoretical Questions
### What are the key points of the paper?
Wu & He propose a novel normalization layer that divides activations / feature maps into groups and normalizes 
them group-wise.
This is motivated by the (empirically observed and hereby reproduced) fact that the state-of-the-art Batch Normalization
layer gets less powerful the smaller the batch size gets. 
This poses a challenge for very large models that can only be trained with small batches due to hardware constraints.


### Which of the lecture topics does the paper relate to?
This paper relates strongly to Part 4 "Training Deep Network Architectures". 
It describes a Normalization technique, and was covered in this chapter.
It also addresses the instability problem when using small batch sizes while training deep convolutional networks,
which is also covered in this chapter. Group Norma can also be seen as a generalization of Instance Norm and Layer Norm,
which were discussed in this part of the lecture.

As the experiments were done in the context of ResNet on ImageNet, this paper also relates strongly to Part 5 
"Deep Network Architectures and Vision Applications" and Part 3 "DCNN - Concepts and Components".

TODO what about the other experiments that I did not read?

TODO it also relates to generative and recurrent networks, as it is a generalization of GN and LN

### Based on the knowledge that you acquired in the lecture, what is the position of the paper in the literature?
As a work orthogonal to Batch Normalization (in the actual spatial sense if you look at [1], Figure 2), 
it naturally competes with the works on Instance Norm and Layer Norm. 
However, according to the experiments in [1], GN beats IN and LN while also being a flexible generalization of 
those two.

GN's advantage over BN on smaller batch sizes could mean that it will get more important as models get bigger and
bigger. 
This advantage could, however, be offset by the findings (see [1], Section 5) that GN's performance degrades in 
heavily parallel settings.

TODO should I mention Transformers (Layer Norm)?

### Which issues remain open and should be addressed in future work?
The paper did not investigate thoroughly the optimal choice of the hyperparameter G. 
As extreme values for G transform GN into IN or LN respectively, a study about the effects of the group size
could lead valuable insights also in regard to layer- and instance norm.

Furthermore, the paper did not - openly stated in Section 6 - study the potential of Group Norm models when 
the architecture is designed from the ground up with GN in mind.
In the current experiments, all models were built with BN and only adapted to use GN.
Therefore, the question about the maximum possible performance of GN-based models remains unanswered.

## Reproducing the Results

## References
[1] Wu & He (2018), Group Normalization, https://arxiv.org/abs/1803.08494

[2] He et al. (2015), Deep Residual Learning for Image Recognition, https://arxiv.org/abs/1512.03385

[3] Gross & Wilber (2016), Training and investigating residual nets, https://github.com/facebook/fb.resnet.torch
