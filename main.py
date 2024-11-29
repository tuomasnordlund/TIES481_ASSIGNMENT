# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 14:44:41 2024

@author: Kone
"""
import simpy
import numpy as np

AVG_INTERARRIVAL_TIME = 25
AVG_PREP_TIME = 40
AVG_OPERATION_TIME = 20
AVG_RECOVERY_TIME = 40
N_PREP_UNITS = 3
N_OPER_UNITS = 1
N_RECOVERY_UNITS = 3
SIMULATION_RUN_TIME = 2000

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
            'operation_queue': [],
            'recovery_units_used': [],
            'recovery_queue': [],
            }
        self.action = env.process(self.run())
    
    # Check number of resources used and their queues regularly
    def run(self):
        while True:
            yield env.timeout(10)
            self.stats['simulation_time'].append(env.now)
            
            self.stats['prep_units_used'].append(self.prep_res.count)
            self.stats['preparation_queue'].append(len(self.prep_res.queue))
            
            self.stats['operation_queue'].append(len(self.oper_res.queue))
            self.stats['recovery_units_used'].append(self.rec_res.count)
            
            self.stats['recovery_queue'].append(len(self.rec_res.queue))
    
    # Log waited times when asked
    def log_waiting_time(self, phase, time):
        self.waiting_times[f"{phase}_wait_time"].append(time)
        
        

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
        
        # Log time in queue
        monitor.log_waiting_time('preparation', prep_wait_time)
        
        if isinstance(env.now, np.ndarray):
            print('%s entering preparation at %d' % (name, env.now[0]))
        else:
            print('%s entering preparation at %d' % (name, env.now))
        #print(f'{prep_units.count} of {prep_units.capacity} preparation units are now being used.')
        
        # Spend time in preparation
        yield env.timeout(prep_time)
    
    # Request operation room
    with oper_unit.request() as req_oper:
        yield req_oper
        print('%s entering operation at %d' % (name, env.now[0]))
        
        # Spend time in operation
        yield env.timeout(oper_time)
    
    # Request recovery room
    wait_time = env.now[0]
    with recovery_units.request() as req_recovery:
        yield req_recovery
        recovery_wait_time = env.now[0] - wait_time
        
        # Log time in queue
        monitor.log_waiting_time('recovery', recovery_wait_time)
        
        print('%s entering recovery at %d' % (name, env.now[0]))
        print(f'{name} waited for recovery unit to open for {recovery_wait_time:.2f}')
        #print(f'{recovery_units.count} of {recovery_units.capacity} recovery units are now being used.')
        
        # Spend time in recovery room
        yield env.timeout(rec_time)
    
    print('%s leaving hospital at %d' % (name, env.now[0]))

def print_stats(res):
    print(f'{res.count} of {res.capacity} slots are allocated.')
    print(f'  Users: {res.users}')
    print(f'  Queued events: {res.queue}')

def run_simulation(env, monitor):
    # Initialize patient counter to differentiate between patients
    patient_counter = 0
    simul_rng = np.random.default_rng()
    
    # We could also run the simulation for a certain number of patients
    while True:
        
        # Get interarrival time
        new_arrival_time = simul_rng.exponential(AVG_INTERARRIVAL_TIME, 1)
        
        # Start patient process and increase patient count
        env.process(patient(env, prep_units, oper_unit, recovery_units, patient_counter, monitor))
        patient_counter += 1
        
        # Wait for the next patient
        yield env.timeout(new_arrival_time)
    
    
print('Starting simulation.')
env = simpy.Environment()

prep_units = simpy.Resource(env, capacity = N_PREP_UNITS)
oper_unit = simpy.Resource(env, capacity = N_OPER_UNITS)
recovery_units = simpy.Resource(env, capacity = N_RECOVERY_UNITS)

monitor = SimulationMonitor(env, prep_units, oper_unit, recovery_units)
env.process(run_simulation(env, monitor))

#env.run()
env.run(until=SIMULATION_RUN_TIME)

#%%
print(monitor.stats)

print("Average usage for preparation rooms: {0:.3f}".format(np.mean(monitor.stats['prep_units_used'])/N_PREP_UNITS))
print("Average usage for recovery rooms: {0:.3f}".format(np.mean(monitor.stats['recovery_units_used'])/N_RECOVERY_UNITS))

#print(monitor.waiting_times)
print("Average queue length for preparation: {0:.3f}".format(np.mean(monitor.stats['preparation_queue'])))
print("Average time spent in queue for preparation: {0:.4f}".format(np.mean(monitor.waiting_times['preparation_wait_time'])))

print("Average queue length for recovery: {0:.3f}".format(np.mean(monitor.stats['recovery_queue'])))
print("Average time spent in queue for preparation: {0:.4f}".format(np.mean(monitor.waiting_times['recovery_wait_time'])))
















