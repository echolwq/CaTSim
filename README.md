# CaTSim
Solving the Graph Edit Distance problem using Deep Learning models
Implementation of CaTSim: Cross Contrast and Top Nodes Similarity Learning and Computation

### Requirements
The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
networkx          2.4
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
scipy             1.1.0
argparse          1.1.0
torch             1.1.0
torch-scatter     1.4.0
torch-sparse      0.4.3
torch-cluster     1.4.5
torch-geometric   1.3.2
torchvision       0.3.0
scikit-learn      0.20.0
```
### Datasets
The datasets for this project is extracted from torch_geometric.GEDDatasets
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.GEDDataset.html


### Options
<p align="justify">
Training a CaTSim model is handled by the `src/main.py` script which provides the following command line arguments.</p>

#### Model options
```
  --epochs                INT         Number of training epochs.               Default is 5.
  --filters-1             INT         Number of filter in 1st GCN layer.       Default is 128.
  --filters-2             INT         Number of filter in 2nd GCN layer.       Default is 64. 
  --filters-3             INT         Number of filter in 3rd GCN layer.       Default is 32.
  --tensor-neurons        INT         Neurons in tensor network layer.         Default is 16.
  --bottle-neck-neurons   INT         Bottle neck layer neurons.               Default is 16.
  --bins                  INT         Number of histogram bins.                Default is 16.
  --batch-size            INT         Number of pairs processed per batch.     Default is 128. 
  --dropout               FLOAT       Dropout rate.                            Default is 0.5.
  --learning-rate         FLOAT       Learning rate.                           Default is 0.001.
  --weight-decay          FLOAT       Weight decay.                            Default is 10^-5.
  --model                 INT         The model specification                  Default is 0.
  --dataset               STR         Dataset used to train                    Default is AIDS700nef.
```
### Examples
<p align="justify">
The following commands learn a neural network and score on the test set. Training a CaTSim model on the default dataset.</p>

```
python src/main.py
```

Training CaTSim model for a 100 epochs with a batch size of 512.
```
python src/main.py --epochs 100 --batch-size 512
```
Increasing the learning rate and the dropout.
```
python src/main.py --learning-rate 0.01 --dropout 0.9
```
Then you can load a pretrained model using the `--load` parameter; **note that the model will be used as-is, no training will be performed**.
```
python src/main.py --load /model_saved/model-name
```

### Reference

> SimGNN: A Neural Network Approach to Fast Graph Similarity Computation.
> Yunsheng Bai, Hao Ding, Song Bian, Ting Chen, Yizhou Sun, Wei Wang.
> WSDM, 2019.
> [[Paper]](http://web.cs.ucla.edu/~yzsun/papers/2019_WSDM_SimGNN.pdf) --
----------------------------------------------------------------------

