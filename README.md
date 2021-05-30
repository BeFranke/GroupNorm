# DeepVision - Group Normalization Project
 by Benedikt Franke

This document contains an explanation of the project and instructions on how to reproduce 
my results.

For the answers to the theoretical questions and an explanation of my methodology, see "questions.pdf".

## Requirements
### Automatic installation
run `pip install -r requirements.txt` (tested for Python 3.8 on Linux)


### Manual installation

- tensorflow
- tensorflow-datasets
- pandas
- matplotlib (for plot.py & reproduce.py)
- seaborn (for plot.py & reproduce.py)

## Project Structure
| File / Folder name | Description  |
| ------------------ |----------- | 
| models/            | trained models in keras format   |
| hp.json            | results of the search for the best value of G |
| hpopt.py           | script used to search for the best value of G |
| layer.py           | keras implementation of GroupNorm layer |
| plot.py            | script to produce plot.png from results.csv |
| plot.png           | reproduction of Figure 1 from Wu & He Paper on CIFAR-10 with smaller network |
| README.md          | this file |
| questions.pdf      | PDF containing the answers to the theoretical questions and explanation of my methodology |
| reproduce.py       | writes new results.csv by evaluating trained models |
| requirements.txt   | requirements file for pip                           |
| results.csv        | results of the evaluation                           |
| train.py           | training script, contains importable model definition |


## Reproducing the Results
### Reproducing only  the Evaluation

run `python reproduce.py`

### Reproducing the Entire Training Procedure
This will overwrite the submitted results.csv!

run `python train.py --restart --seeds 1 2 3 4 5`

The complete training will take about 10 days on a single GTX 1070.
