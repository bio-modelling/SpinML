__author__ = "hanyi"
from spynnaker.pyNN import *
import pylab
import numpy as np
import random
import matplotlib.pyplot as plt
from pyNN.random import NumpyRNG, RandomDistribution
from types import FunctionType
import scipy.io


#
# Class defines a two layer network topology, with all-to-all connectivity between layers,
# and all-to-all inhibition within output layer
#
# Runs simulation in the chosen mode of operation:
#   - stdp_mode: true  - initialises plastic synapses, to be used for training the network
#                false - initialises static synapses, to be used for testing the network
#   - allsameweight: uses the same initial weight for each synapse
#

class two_layer_topology:
    
    # Initialises all parameters for simulation
    def __init__(self, pre_pop_size=None, post_pop_size=None, e_syn_weight=None, i_syn_weight=None,
                 cell_params_lif=None, setup_cond=None, stdp_param=None, stdp_mode=True,
                 inhibitory_spike_mode=False, allsameweight=False):
        
        self.pre_pop_size = pre_pop_size
        self.post_pop_size = post_pop_size
        self.e_syn_weight = e_syn_weight
        self.i_syn_weight = i_syn_weight
        self.cell_params_lif = cell_params_lif
        self.setup_cond = setup_cond
        self.stdp_param = stdp_param
        self.stdp_mode = stdp_mode
        self.inhibitory_spike_mode = inhibitory_spike_mode
        self.allsameweight = allsameweight
    
    # Receives training data
    def input_spike(self,in_spike):
        self.in_spike = in_spike
        #print self.in_spike
    
    # Imports synapse list or creates one for FromListConnector()
    # with random weights from a uniform distribution
    def connection_list_converter_uniform(self, self_import, conn_list, min, max, delay):
        print "------------------------------------------------------------------"
        print "uniform distribution used"
        print "------------------------------------------------------------------"
        
        if(self_import):
            self.conn_list = conn_list
        else:
            conn_list = []
            for i in range(self.post_pop_size):
                for j in range(self.pre_pop_size):
                    rand_num = np.random.uniform(min, max, 1)
                    if(rand_num[0] < 0.05):
                        rand_num[0] = 0.05
                    conn_list.append((j, i, rand_num[0], delay))
            self.conn_list = conn_list

    # Defines populations and synapses, then runs the simulation on spiNNaker with the user defined modes:
    # stdp_mode: plastic synapses for training
    # allsameweight: uses the same initial weight for each synapse
    def start_sim(self, sim_time):
    
        # Simulation setup
        self.simtime = sim_time
        setup(timestep=self.setup_cond["timestep"],
              min_delay=self.setup_cond["min_delay"],
              max_delay=self.setup_cond["max_delay"])
            
        # Initializes the neural populations
        spikeArrayOn = {'spike_times': self.in_spike}

        pre_pop = Population(self.pre_pop_size, SpikeSourceArray, spikeArrayOn, label='inputSpikes_On')
              
        post_pop = Population(self.post_pop_size, IF_curr_exp, self.cell_params_lif, label='post_1')
              
        # Defines weight and timing dependence of STDP learning rule
        t_dependence = SpikePairRule(tau_plus=self.stdp_param["tau_plus"],
                                     tau_minus=self.stdp_param["tau_minus"],
                                     nearest=True)
                                                
        w_dependence = MultiplicativeWeightDependence(w_min=self.stdp_param["w_min"],
                                                      w_max=self.stdp_param["w_max"],
                                                      A_plus=self.stdp_param["A_plus"],
                                                      A_minus=self.stdp_param["A_minus"])
    
        stdp_model = STDPMechanism(timing_dependence=t_dependence,
                                   weight_dependence=w_dependence)

        # Initializes synapses for chosen mode of operation
        if(self.inhibitory_spike_mode):
            connectionsOn = Projection(pre_pop, post_pop,
                                       AllToAllConnector(weights = self.i_syn_weight, delays=1,
                                                         allow_self_connections=False),
                                       target='inhibitory')
        else: 
            if(self.stdp_mode):
                if(self.allsameweight):
                    connectionsOn = Projection(pre_pop, post_pop,
                                               AllToAllConnector(weights = self.e_syn_weight, delays=1),
                                               synapse_dynamics=SynapseDynamics(slow=stdp_model),
                                               target='excitatory')
                else:
                    connectionsOn = Projection(pre_pop, post_pop,
                                               FromListConnector(self.conn_list),
                                               synapse_dynamics=SynapseDynamics(slow=stdp_model),
                                               target='excitatory')
            else:
                if(self.allsameweight):
                    connectionsOn = Projection(pre_pop, post_pop,
                                               AllToAllConnector(weights = self.e_syn_weight, delays=1),
                                               target='excitatory')
                else:
                    connectionsOn = Projection(pre_pop, post_pop,
                                               FromListConnector(self.conn_list),
                                               target='excitatory')
        
        # Inhibitory connections between neurons of the post-synaptic neuron population
        connection_I = Projection(post_pop, post_pop,
                                  AllToAllConnector(weights=self.i_syn_weight, delays=1,
                                                    allow_self_connections=False),
                                  target='inhibitory')
    
        # Sets up recording of populations
        pre_pop.record()
        post_pop.record()
        post_pop.record_v()
        
        # Runs the simulation for 'simtime' milliseconds
        run(self.simtime)
        
        # Gets recorded data
        self.pre_spikes = pre_pop.getSpikes(compatible_output=True)
        self.post_spikes = post_pop.getSpikes(compatible_output=True)
        self.post_spikes_v = post_pop.get_v(compatible_output=True)
        self.trained_weights = connectionsOn.getWeights(format='array')
        
        # End spiNNaker session simulation
        end()
        
        # Saves initial weights
        '''scipy.io.savemat('initial_weight.mat', {'initial_weight': self.init_weights})'''
        # Saves final weights
        scipy.io.savemat('trained_weight.mat', {'trained_weight': self.trained_weights})
    
    #
    # Functions for displaying results
    #
    
    # Prints out weight matrix
    def display_weight(self):
        for i in range(self.post_pop_size):
            print ([x[i] for x in self.trained_weights])

    # Raster plot of spike events
    def plot_spikes(self, spike_type, size, title):
        if (spike_type == "input"):
            spikes = self.pre_spikes
        if (spike_type == "output"):
            spikes = self.post_spikes
        #print spikes
        if spikes is not None:
            pylab.figure()
            ax = plt.subplot(111, xlabel='Time/ms', ylabel='Neruons #', title=title)
            pylab.xlim((0, self.simtime))
            pylab.ylim((-0.5, size - 0.5))
            lines = pylab.plot([i[1] for i in spikes], [i[0] for i in spikes], ".")
            pylab.axvline(32500, 0, 1, linewidth=4, color='c', alpha=0.75, linestyle='dashed')
            pylab.axvline(64500, 0, 1, linewidth=4, color='c', alpha=0.75, linestyle='dashed')
            pylab.axvline(96500, 0, 1, linewidth=4, color='c', alpha=0.75, linestyle='dashed')
            pylab.setp(lines, markersize=10, color='r')
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                          ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(20)
            pylab.show()
        else:
            print "No spikes received"
    
    # Heat-map plot of weight matrix for each output neuron
    def Plot_WeightDistribution(self, bin_num, title):
        hist, bins = np.histogram(self.trained_weights, bins=bin_num)
        center = (bins[:-1] + bins[1:]) / 2
        width = (bins[1] - bins[0]) * 0.7
        ax = pylab.subplot(111, xlabel='Weight', title=title)
        plt.bar(center, hist, align='center', width=width)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                      ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(15)
        plt.show()

    #
    def display_membrane_potential(self, title, xmin=0, xmax=50, ymin=-70, ymax=-63):
        post_spikes_v = self.post_spikes_v
        #print post_spikes_v
        if post_spikes_v is not None:
            pylab.figure()
            ax = plt.subplot(111, xlabel='Time/ms', ylabel=' Membrane Potential/V', title=title)
            pylab.xlim((xmin, xmax))
            pylab.ylim((ymin, ymax))
            pylab.plot([i[1] for i in post_spikes_v], [i[2] for i in post_spikes_v])
            #pylab.setp(lines, markersize=10, color='r')
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                          ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(20)
            pylab.show()
        else:
            print "No spikes received"

