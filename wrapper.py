__author__ = "hanyi"
from TwoLayerTopology import TwoLayerTopology
import numpy as np
from types import FunctionType
import random
import scipy.io
import pylab
import matplotlib.pyplot as plt

                            
#--------------------------------------------------#
# Generating the training E-Pockets from recorded data
#--------------------------------------------------# 
print "---------------------------------------------"
print "Generating Trajectory E-Pockets: Need Processed Data"

last_episodes = np.repeat(np.array([0, 9, 18, 27]), 1)
order = []
num_training = 80

# This function selects the E-pocket randomly according to the num_training value
def create_order(num_copy):
    order = np.array([1])
    order_temp = []
    for j in range(num_copy):
        order_temp.append(random.randrange(0, 9))
    for j in range(num_copy):
        order_temp.append(random.randrange(9, 18))
    for j in range(num_copy):
        order_temp.append(random.randrange(18, 27))
    for j in range(num_copy):
        order_temp.append(random.randrange(27, 36))
    
    order = np.concatenate((order, order_temp))
    return order

order = create_order(num_training)
# order = np.concatenate((order, last_episodes))

print "---------------------------------------------"
print "Importing pre-recorded data"
#--------------------------------------------------#
# Importing pre-recorded data
#--------------------------------------------------# 
argw = []


def get_data(filename):
    dvs_data = scipy.io.loadmat(filename)
    ts = dvs_data['ts'][0]
    ts = (ts - ts[0])  # from us to ms
    x = dvs_data['X'][0]
    y = dvs_data['Y'][0]
    p = dvs_data['t'][0]
    return x, y, p, ts


def data_set(prefix):
    postfix = '.mat'
    for i in range(1, 10):
        argw.append(get_data(prefix + str(i) + postfix))


data_set('l_to_r')
data_set('r_to_l')
data_set('t_to_b')
data_set('b_to_t')
   
print "---------------------------------------------"   

#--------------------------------------------------#
# Raster plot of input recording
#--------------------------------------------------#
print "---------------------------------------------"
print "Raster plot of input recording"


# This function creates a raster plot of 4 input recordings in 4 direction
def raster_plot_4_dir():
    pylab.figure()
    pylab.xlabel('Time/ms')
    pylab.ylabel('spikes')
    for ii in range(0, 4):
        pylab.plot(argw[9*ii][3]+200*(2*ii+1), argw[9*ii][0]+argw[9*ii][1]*16, ".")
    pylab.title('Raster Plot of Virtual Retina Neuron Population in 4 Direction')
    pylab.xlim((0, 1800))
    pylab.ylim((0, 270))
    pylab.show()
    

# This function creates a raster plot of 36 input recordings in 4 direction
def raster_plot():
    pylab.figure()
    pylab.xlabel('Time/ms')
    pylab.ylabel('spikes')
    for ii in range(0, len(argw)):
        pylab.plot(argw[ii][3]+200*(ii+ii/9), argw[ii][0]+argw[ii][1]*16, ".")
    pylab.axvline(1900, 0, 1, linewidth=4, color='r', alpha=0.75, linestyle='dashed')
    pylab.axvline(3900, 0, 1, linewidth=4, color='r', alpha=0.75, linestyle='dashed')
    pylab.axvline(5900, 0, 1, linewidth=4, color='r', alpha=0.75, linestyle='dashed')
    pylab.title('Raster Plot of 36 Neuron Population Training Sets')
    pylab.ylim((0, 270))
    pylab.show()

raster_plot()



#--------------------------------------------------#
# Excitatory Mode: Observe membrane potential change of one pre-synaptic neuron
# Inhibitory Mode: Observe membrane potential change of one pre-synaptic neuron
# Training Mode:   Used for training the system 
#--------------------------------------------------# 
excitatory_mode = False
inhibitory_mode = False
training_mode = True

print "---------------------------------------------"
print "Setting up neural network parameters"

#--------------------------------------------------#
# Defining Network Parameters
#--------------------------------------------------#

# STDP learning rule parameters
stdp_param = {'tau_plus': 50.0,
              'tau_minus': 60.0,
              'w_min': 0,
              'w_max': 1,
              'A_plus': 0.05,
              'A_minus': 0.05}

# Leaky integrate and fire model parameters
cell_params_lif = {'cm': 12,
                   'i_offset': 0.0,
                   'tau_m': 110,
                   'tau_refrac': 40.0,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 10.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -61.0}

# Synaptic delay
delay = 1

# Imports the trained weights to see the evolution of the weight
NetworkInfo = scipy.io.loadmat('trained_weight4dir.mat')
weights_import = NetworkInfo['trained_weight']


# This function converts the weights from a MatLab file to a list
def convert_weights_to_list(matrix, delay):
    def build_list(indices):
        # Extract weights from matrix using indices
        weights = matrix[indices]
        # Build np array of delays
        delays = np.repeat(delay, len(weights))
        # Zip x-y coordinates of non-zero weights with weights and delays
        return zip(indices[0], indices[1], weights, delays)

    # Get indices of non-nan i.e. connected weights
    connected_indices = np.where(~np.isnan(matrix))
    # Return connection lists
    return build_list(connected_indices)


# This function build the input spike events from the trajectory E-Pockets
def BuildTrainingSpike_with_noise(order, ONOFF, noise_spikes):
    noise_nid = []
    for i in range(0, len(order)):
        noisetemp = np.random.randint(0, 256, noise_spikes)
        noisetemp.sort()
        noisetemp = noisetemp.tolist()
        noise_nid.append(noisetemp)
    # print len(noise_nid)
    complete_Time = []
    for nid in range(0, pre_pop_size):
        SpikeTimes = []
        for tid in range(0, len(order)):
            # print dead_zone_cnt
            temp = []
            loc = order[tid]
            j = np.repeat(nid, len(argw[loc][1]))
            p = np.repeat(ONOFF, len(argw[loc][1]))
            temp = 200*(2*tid+1)+argw[loc][3][(j % 16 == argw[loc][0]) &
                                              (j / 16 == argw[loc][1]) &
                                              (p == argw[loc][2])]
            if(nid in noise_nid[tid]):
                t_noise = 200*(2*tid+1) + np.random.uniform(0, 200, 1)
                temp = np.concatenate((temp, t_noise))
                temp.sort()
            if temp.size > 0:
                SpikeTimes = np.concatenate((SpikeTimes, temp))
        if type(SpikeTimes) is not list:
           complete_Time.append(SpikeTimes.tolist())
        else:
           complete_Time.append([])
    return complete_Time


# period of trajectory time
animation_time = 200


# on off event selection: ON 1, OFF 0
# CANNOT USE OFF EVENT, FILTERED BY INPUT_NORMALISER.PY
ON_OFF = 1
num_noise = 11


if(training_mode):
    pre_pop_size = 256
    post_pop_size = 4
    test_STDP = False
    stdp_mode = True
    inhibitory_spike_mode = False
    allsameweight = False
    self_import = False
    in_spike = BuildTrainingSpike_with_noise(order, ON_OFF, num_noise)
    print(in_spike)
    sim_time = (2 * len(order) + 1) * animation_time
else:
    test_STDP = True
    sim_time = 4000
    if(inhibitory_mode):
        pre_pop_size = 4
        post_pop_size = 1
        stdp_mode = False
        inhibitory_spike_mode = True
        allsameweight = True
        self_import = False
        in_spike = [[10], [], [], []]
    if(excitatory_mode): 
        pre_pop_size = 256
        post_pop_size = 1
        stdp_mode = True
        inhibitory_spike_mode = False
        allsameweight = False
        self_import = True
        # use the following line to show the membrane potential change from 1 pre-synaptic spike
        # in_spike = [[10], [], [], []]
        # use the following line to show the membrane potential change from spikes in 1 E-Pockets trajectory
        in_spike = BuildTrainingSpike_with_noise(order, ON_OFF, num_noise)
        sim_time = (2 * len(order) + 1) * animation_time
        
# e_syn_weight = np.random.normal(0.2, 0.003, pre_pop_size*post_pop_size).tolist()
e_syn_weight = [0.2, 0.1, 0.2, 0.0]
i_syn_weight = 15.0


    
print "---------------------------------------------"
print "Simulation Setup & Initialisation"    
#--------------------------------------------------#
# Simulation Setup & Initialisation
#--------------------------------------------------#     
    
setup_cond = {'timestep':  1,
              'min_delay': 1,
              'max_delay': 144}

list = convert_weights_to_list(weights_import, delay)
simulation = TwoLayerTopology(pre_pop_size=pre_pop_size, post_pop_size=post_pop_size,
                              e_syn_weight=e_syn_weight, i_syn_weight=i_syn_weight,
                              cell_params_lif=cell_params_lif, setup_cond=setup_cond,
                              stdp_param=stdp_param, stdp_mode=stdp_mode,
                              inhibitory_spike_mode=inhibitory_spike_mode,
                              allsameweight=allsameweight)

    
simulation.input_spike(in_spike)
simulation.connection_list_converter_uniform(self_import=self_import, conn_list=list, min=0.1, max=0.4, delay=1.0)
simulation.start_sim(sim_time)
simulation.plot_spikes("input", pre_pop_size, "Spike Pattern of Pre-Synaptic Population")
simulation.plot_spikes("output", post_pop_size, "Spike Pattern of Post-Synaptic Population")
'''sim1.display_membrane_potential("Membrane Potential of one Post-Synaptic Neuron",
                                    xmin=0, xmax=400, ymin=-75, ymax=-57)'''
simulation.Plot_WeightDistribution(pre_pop_size,'Histogram of Trained Weight')
