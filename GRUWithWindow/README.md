# Recurrent network with GRU and sliding window input

This is a 'lightweight' recurrent neural network. Inspired by the RNN in NeuralNILM, this one has a similar architecture. However, it uses layers with less neurons. Also the LSTM neurons were replaced with GRU. This makes for less parameters and thus faster training.

## To set up the project
Run
```bash
python gen.py
```

This will create the trainsets and download the Neural NILM test set. The trainset comes from the data used in Neural NILM. This may take some time.

## To train and test the network
Run
```bash
python experiment.py <device>
```
Where device can be
* ```dishwasher```
* ```fridge```
* ```kettle```
* ```microwave```
* ```washing_machine```

__OR__

import the experiment function and use it in your code
```python
from experiment import experiment
experiment('kettle', 0, 120)
```
This will train the network for 120 epochs. You can resume training lie this:
```python
experiment('kettle', 120, 130)
```
