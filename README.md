
# On Robustness of Neural Differential Equations

#### I have tried to reproduce the code for this paper.
#### If you recognize that the code wrong in somewhere. Please notice me. Thanks

Paper: [On Robustness of Neural Differential Equations](https://arxiv.org/abs/1910.05513)


```
usage: run.py [-h] [-d DEVICE] [-bs BATCH_SIZE] [-ep EPOCHS] [-f FOLDER]
              [-r RESULT] [-tr TRAIN] [-vl VALID] [-lr LEARNING_RATE]

optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        Device which the PyTorch run on
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size of 1 iteration
  -ep EPOCHS, --epochs EPOCHS
                        Numbers of epoch
  -f FOLDER, --folder FOLDER
                        Folder /path/to/mnist/dataset
  -r RESULT, --result RESULT
                        Folder where the result going in
  -tr TRAIN, --train TRAIN
                        Number of train images
  -vl VALID, --valid VALID
                        Number of validation images
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate in optimizer
```
