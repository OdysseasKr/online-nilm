# ShortSeq2Point
A modified version of the Sequence-to-point network.

Improvements:

- Smaller window lengths (5-10 mins) to make it suitable for real-time applications.
- Added dropout

## To set up the project
Run
```bash
python gen.py <path to your UKDALE h5>
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
This will train the network for 120 epochs. You can resume training like this:
```python
experiment('kettle', 120, 130)
```
