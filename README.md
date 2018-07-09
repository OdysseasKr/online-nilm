# Online NILM
Here you can find the code for the two proposed Neural Network architectures from 

> Sliding Window Approach for Online Energy Disaggregation Using Artificial Neural Networks, O. Krystalakos, C. Nalmpantis and D. Vrakas, SETN, 2018

Full Paper: [3200947.3201011](https://dl.acm.org/citation.cfm?doid=3200947.3201011)

All code is written using Keras and Tensorflow.

**Requires NILMTK to run. You can find it here:** https://github.com/nilmtk/nilmtk.

## The networks
In each folder you can find a README with instructions on how to run the experiments. For every network you will find 4 Python files:

- experiment.py: Code for running the experiment. Each experiment includes training and evaluating the network.
- gen.py: Downloads all necessary resources and generates a trainset and a testset.
- model.py: The network architecture.
- metrics.py: The metrics used to evaluate the network.

The experiments are available for the following appliances from the UKDALE dataset:
- Dishwasher
- Fridge
- Kettle
- Microwave
- Washing Machine

## Related works
Neural NILM: https://arxiv.org/pdf/1507.06594.pdf  
Original Sequence-to-point: https://arxiv.org/pdf/1612.09106v3.pdf.
