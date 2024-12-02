# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 14:44:41 2024

@author: Kone
"""
import random
import simpy
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

#random.seed(42)

AVG_INTERARRIVAL_TIME = 25
AVG_PREP_TIME = 40
AVG_OPERATION_TIME = 20
AVG_RECOVERY_TIME = 40

N_PREP_UNITS = [3, 3, 4]
N_OPER_UNITS = 1
N_RECOVERY_UNITS = [4, 4, 5]

N_ROOM_KEYS = ['3p4r', '3p5r', '4p5r']
STAT_KEYS = ['preparation_queue', 'operation_queue']

SIMULATION_RUN_TIME = 1000
N_SAMPLES=20
WARMUP=20000

# Create Simulation Monitor for saving statistics
class SimulationMonitor():
    def __init__(self, env, prep_res, oper_res, rec_res):
        self.env = env
        self.prep_res = prep_res
        self.oper_res = oper_res
        self.rec_res = rec_res
        self.waiting_times = {
            'preparation_wait_time': [],
            'recovery_wait_time': []
            }
        
        self.stats = {
            'simulation_time': [],
            'prep_units_used': [],
            'preparation_queue': [],
            'mean_preparation_queue': [],
            'operation_queue': [],
            'recovery_units_used': [],
            'recovery_queue': [],
            }
        self.action = env.process(self.run())
    
    def run(self):
        """
        Monitors key statistics (queue lengths etc.) regularly.
        """
        
        # Wait until simulation has warmed up
        yield self.env.timeout(WARMUP)
        
        while True:
            yield self.env.timeout(1)
            self.stats['simulation_time'].append(self.env.now)
            
            self.stats['prep_units_used'].append(self.prep_res.count)
            self.stats['preparation_queue'].append(len(self.prep_res.queue))
            self.stats['mean_preparation_queue'].append(
                np.mean(self.stats['preparation_queue']))
            
            self.stats['operation_queue'].append(len(self.oper_res.queue))
            self.stats['recovery_units_used'].append(self.rec_res.count)
            
            self.stats['recovery_queue'].append(len(self.rec_res.queue))
    
    def log_waiting_time(self, phase, time):
        """
        Logs waiting times when required.
        """
        self.waiting_times[f"{phase}_wait_time"].append(time)
        
        
def print_stats(monitor):
    #print(monitor.stats)
    
    print("Average usage for preparation rooms: {0:.3f}".format(
        np.mean(monitor.stats['prep_units_used'])/N_PREP_UNITS))
    print("Average usage for recovery rooms: {0:.3f}".format(
        np.mean(monitor.stats['recovery_units_used'])/N_RECOVERY_UNITS))
    
    #print(monitor.waiting_times)
    print("Average queue length for preparation: {0:.3f}".format(
        np.mean(monitor.stats['preparation_queue'])))
    print("Average time spent in queue for preparation: {0:.4f}".format(
        np.mean(monitor.waiting_times['preparation_wait_time'])))
    
    print("Average queue length for recovery: {0:.3f}".format(
        np.mean(monitor.stats['recovery_queue'])))
    print("Average time spent in queue for preparation: {0:.4f}".format(
        np.mean(monitor.waiting_times['recovery_wait_time'])))      
    
        

def patient(env, prep_units, oper_unit, recovery_units, counter, monitor):
    patient_rng = np.random.default_rng()
    name = f"Patient {counter}"
    
    # Get service times for patient
    prep_time = patient_rng.exponential(AVG_PREP_TIME, 1)
    oper_time = patient_rng.exponential(AVG_OPERATION_TIME, 1)
    rec_time = patient_rng.exponential(AVG_RECOVERY_TIME, 1)
    
    # Request preparation room
    wait_time = env.now
    with prep_units.request() as req_prep:
        yield req_prep
        prep_wait_time = env.now - wait_time
        if isinstance(prep_wait_time, np.ndarray):
            prep_wait_time = prep_wait_time[0]
        
        #if prep_wait_time > 0:
        #    print(f'{name} waited preparation for {round(prep_wait_time, 2)}')
        
        # Log time in queue
        monitor.log_waiting_time('preparation', prep_wait_time)
        """
        if isinstance(env.now, np.ndarray):
            print('%s entering preparation at %d' % (name, env.now[0]))
        else:
            print('%s entering preparation at %d' % (name, env.now))
        """
        #print(f'{prep_units.count} of {prep_units.capacity} preparation units are now being used.')
        
        # Spend time in preparation
        yield env.timeout(prep_time)
    
    # Request operation room
    with oper_unit.request() as req_oper:
        yield req_oper
        #print('%s entering operation at %d' % (name, env.now[0]))
        
        # Spend time in operation
        yield env.timeout(oper_time)
    
    # Request recovery room
    wait_time = env.now[0]
    with recovery_units.request() as req_recovery:
        yield req_recovery
        recovery_wait_time = env.now[0] - wait_time
        
        # Log time in queue
        monitor.log_waiting_time('recovery', recovery_wait_time)
        
        #print('%s entering recovery at %d' % (name, env.now[0]))
        #print(f'{name} waited for recovery unit to open for {recovery_wait_time:.2f}')
        #print(f'{recovery_units.count} of {recovery_units.capacity} recovery units are now being used.')
        
        # Spend time in recovery room
        yield env.timeout(rec_time)
    
    #print('%s leaving hospital at %d' % (name, env.now[0]))


def run_simulation(env, monitor, prep_units, oper_units, recovery_units):
    """
    Function for generating patients.
    """
    # Initialize patient counter to differentiate between patients
    patient_counter = 0
    simul_rng = np.random.default_rng()
    
    # We could also run the simulation for a certain number of patients
    while True:
        
        # Get interarrival time
        new_arrival_time = simul_rng.exponential(AVG_INTERARRIVAL_TIME, 1)
        
        # Start patient process and increase patient count
        env.process(
            patient(env, 
                    prep_units, 
                    oper_units, 
                    recovery_units, 
                    patient_counter, 
                    monitor))
        patient_counter += 1
        
        # Wait for the next patient
        yield env.timeout(new_arrival_time)
    

def run_samples(n_samples):
    """
    Function for running independent sample simulations.
    Number of runs is defined with `n_samples`. 
    """
    
    # Initialize data collection
    results = {
        n_room_key: {stat_key: [] for stat_key in STAT_KEYS} for n_room_key in N_ROOM_KEYS}
    
    for i in range(n_samples):
        print(f'Starting simulation round {i}')
        
        for j, n_rooms in enumerate(N_ROOM_KEYS):
            # Create new simulation environment
            env = simpy.Environment()
            
            prep_units = simpy.Resource(env, capacity = N_PREP_UNITS[j])
            oper_units = simpy.Resource(env, capacity = N_OPER_UNITS)
            recovery_units = simpy.Resource(env, capacity = N_RECOVERY_UNITS[j])
            
            # Create monitor
            monitor = SimulationMonitor(env, prep_units, oper_units, recovery_units)
            
            env.process(run_simulation(
                env, monitor, prep_units, oper_units, recovery_units))    
            
            # Warm up the system
            env.run(until=WARMUP)
            
            # Run simulation
            env.run(until=WARMUP+SIMULATION_RUN_TIME)
            
            # Get necessary statistics
            for stat_key in STAT_KEYS:
                result = monitor.stats[stat_key]
                results[n_rooms][stat_key].append(result)
                
    return results
    

def calculate_CI(arr):
    return st.t.interval(
        confidence=0.95, 
        df=len(arr)-1, 
        loc=np.mean(arr), 
        scale=st.sem(arr))

def calculate_statistics(results):
    #for n_rooms in N_ROOM_KEYS:
    n_room_res = results[N_ROOM_KEYS[0]]
    prep_room_queue = n_room_res[STAT_KEYS[0]]
    or_queue = n_room_res[STAT_KEYS[1]]
    
    prep_room_means = np.mean(prep_room_queue, axis=1)

if __name__ == "__main__":    
    results = run_samples(N_SAMPLES)
    
    prep_room_means = np.zeros((N_SAMPLES, len(N_ROOM_KEYS)))
    prep_room_ci = {n_rooms: np.zeros((N_SAMPLES, 2)) for n_rooms in N_ROOM_KEYS}
    #prep_room_ci = np.zeros((N_SAMPLES, len(N_ROOM_KEYS)))
    
    oper_room_means = np.zeros((N_SAMPLES, len(N_ROOM_KEYS)))
    oper_room_ci = {n_rooms: np.zeros((N_SAMPLES, 2)) for n_rooms in N_ROOM_KEYS}
    #oper_room_ci = np.zeros((N_SAMPLES, len(N_ROOM_KEYS)))
    
    for i, n_rooms in enumerate(N_ROOM_KEYS):
        n_room_res = results[n_rooms]
        
        prep_room_queue = n_room_res[STAT_KEYS[0]]
        or_queue = n_room_res[STAT_KEYS[1]]
        
        prep_room_means[:,i] += np.mean(prep_room_queue, axis=1)
        oper_room_means[:,i] += np.mean(or_queue, axis=1)
        
        for j in range(N_SAMPLES):
            prep_ci = calculate_CI(prep_room_queue[j,:])
            oper_ci = calculate_CI(or_queue[j,:])
            
            prep_room_means[n_rooms][j,:] =  
            
            
        
    
    
    
    
    
    """
    print(len(prep_queue_means[0]))
    print(len(prep_queue_means[1]))
    for i in range(len(prep_queue_means)):
        plt.plot(range(len(prep_queue_means[i])), prep_queue_means[i], label=f'sample {i}')
        
    plt.legend()
    plt.show()
    """















