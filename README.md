Machine Learning with SpiNNaker
===============================

This repository contains code and resources for developing a neural network to
learn about trajectories from AER data (eventually directly from an eDVS) in
order to support various roles e.g. build a robotic fielder or goalkeeper.

Resources: https://spinnakermanchester.github.io/

Hanyi's project repository can be found here: https://github.com/hh2712/hh2712-Machine-Learning-On-The-SpiNNaker-Platform-For-The-Visual-Input-Processing
This contains the data files as well as the code, however binary files should not be included in this repository. 

Other useful code for building suitable neural networks may be found here: https://github.com/bio-modelling/SpinBot. 
This repository also contains code for controlling a robot with an Arduino microcontroller. 

The Python module for reading AER files may be found here: https://github.com/bio-modelling/py-aer. 


Installation
------------

SpiNNaker with PyNN is best used in an anaconda virtual environment.
Adapt the instructions here: https://spinnakermanchester.github.io/spynnaker/3.0.0/PyNNOnSpinnakerInstall.html


Tutorials
---------

PyNN on SpiNNaker Tutorial: https://spinnakermanchester.github.io/spynnaker/3.0.0/index.html

Brian Tutorial (general neural network concepts): http://brian2.readthedocs.io/en/stable/resources/tutorials/


TODO
----

- [ ] Train and test Hanyi’s network with new data with more events for each ball movement
- [ ] Generalise the wrapper.py script for constructing the SpikeSourceArray input data to facilitate work with any network in the future relying on recorded/generated data
- [ ] Extract useful code from SpinBot (Wenbo's repository)


Useful papers
-------------
Bichler, Querlioz, Thorpe, Bourgoin, Gamrat, 2012. Extraction of temporally correlated features from dynamic vision sensors with spike-timing-dependent plasticity. Neural Networks, 32, p339--348. http://dx.doi.org/10.1016/j.neunet.2012.02.022
