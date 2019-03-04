#https://mlrose.readthedocs.io/en/latest/source/tutorial1.html

#https://mlrose.readthedocs.io/en/latest/source/tutorial3.html

#https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0

#https://github.com/gkhayes/mlrose/blob/master/tests/test_algorithms.py

#https://github.com/scribby182/mlrose/blob/master/tests/test_neural.py





import mlrose
import numpy as np
import time
#from random import randomint
randomhill_state=[]
randomhill_fitness=[]
randomhill_statistics=[]
randomhill_statistics_function_eval=[]
randomhill_statistics_time=[]
randomhill_statistics_fitness=[]

sa_state=[]
sa_fitness=[]
sa_statistics=[]
sa_statistics_function_eval=[]
sa_statistics_time=[]
sa_statistics_fitness=[]

genetic_algo_state=[]
genetic_algo_fitness=[]
genetic_algo_statistics_fitness=[]
genetic_algo_statistics_function_eval=[]
genetic_algo_statistics_time=[]
genetic_algo_statistics=[]

mimic_algo_state=[]
mimic_algo_fitness=[]
mimic_algo_statistics_fitness=[]
mimic_algo_statistics_function_eval=[]
mimic_algo_statistics=[]
mimic_algo_statistics_time=[]

problem_size=[5,10,15,25,50,75,100,125,150]

weight_list=[]
value_list=[]
state_list=[]

for each in problem_size:
    weight=np.random.randint(1,15,size=each) #16
    value=np.random.randint(1,20,size=each)
    state=np.random.randint(0,2,size=each)
    weight_list.append(weight)
    value_list.append(value)
    state_list.append(state)

max_attempt=50 #50
max_iter=1500  #1500
max_weight_pct=0.45 #0.45

ind=0
# Solve problem using random hill climbing
for each in problem_size:
 
     # Define optimization problem object
     init_state = np.random.randint(0, 2, each)
     fitness=mlrose.Knapsack(weight_list[ind],value_list[ind],max_weight_pct)
     problem=mlrose.DiscreteOpt(length = each, fitness_fn = fitness, maximize = True,max_val=2)
     
     
     best_state, best_fitness,statistics = mlrose.random_hill_climb(problem, max_attempts=max_attempt, max_iters=max_iter,restarts=9,return_statistics=True)
     
     randomhill_state.append(best_state)
     randomhill_fitness.append(best_fitness)
     randomhill_statistics.append(statistics)
     randomhill_statistics_function_eval.append(statistics['fitness_evals'])
     randomhill_statistics_time.append(statistics['time'])
     randomhill_statistics_fitness.append(best_fitness)
     ind=ind+1

# Solve problem using simulated annealing

#Define decay schedule

ind=0
for each in problem_size:
    schedule = mlrose.GeomDecay(init_temp=2.3,decay=0.72,min_temp=0.9)
    # Define optimization problem object
    init_state = np.random.randint(0, 2, each)
    fitness=mlrose.Knapsack(weight_list[ind],value_list[ind],max_weight_pct)
    problem=mlrose.DiscreteOpt(length = each, fitness_fn = fitness, maximize = True,max_val=2)
    best_state, best_fitness,statistics = mlrose.simulated_annealing(problem, schedule = schedule,max_attempts = max_attempt, max_iters = max_iter,return_statistics=True)
    sa_state.append(best_state)
    sa_fitness.append(best_fitness)
    sa_statistics.append(statistics)
    sa_statistics_function_eval.append(statistics['fitness_evals'])
    sa_statistics_time.append(statistics['time'])
    sa_statistics_fitness.append(best_fitness) 
    ind=ind+1
    
    
    
# Solve problem using genetic algortihm
ind=0
for each in problem_size:
    #init_state = np.random.randint(0, 2, each)
    fitness=mlrose.Knapsack(weight_list[ind],value_list[ind],max_weight_pct)
    problem=mlrose.DiscreteOpt(length = each, fitness_fn = fitness, maximize = True,max_val=2)
    best_state, best_fitness,statistics = mlrose.genetic_alg(problem,pop_size=200, mutation_prob=0.39,max_attempts=max_attempt, max_iters=max_iter,return_statistics=True)
    
    genetic_algo_state.append(best_state)
    genetic_algo_fitness.append(best_fitness)
    genetic_algo_statistics.append(statistics)
    genetic_algo_statistics_function_eval.append(statistics['fitness_evals'])
    genetic_algo_statistics_time.append(statistics['time'])
    genetic_algo_statistics_fitness.append(best_fitness) 
    ind=ind+1
    

#Solve problem using mimic algortihm
ind=0
for each in problem_size:
    #init_state = np.random.randint(0, 2, each)
    fitness=mlrose.Knapsack(weight_list[ind],value_list[ind],max_weight_pct)
    problem=mlrose.DiscreteOpt(length = each, fitness_fn = fitness, maximize = True)
    best_state, best_fitness,statistics =mlrose.mimic(problem, pop_size=200, keep_pct=0.19, max_attempts=max_attempt, max_iters=max_iter,return_statistics=True)
    mimic_algo_state.append(best_state)
    mimic_algo_fitness.append(best_fitness)
    mimic_algo_statistics.append(statistics)
    mimic_algo_statistics_function_eval.append(statistics['fitness_evals'])
    mimic_algo_statistics_time.append(statistics['time'])
    mimic_algo_statistics_fitness.append(best_fitness) 
    ind=ind+1
    
# summarize function evaluations vs problem size
import matplotlib.pyplot as plt
plt.plot(problem_size[:-1], randomhill_statistics_function_eval[:-1])
plt.plot(problem_size[:-1], sa_statistics_function_eval[:-1])
plt.plot(problem_size[:-1], genetic_algo_statistics_function_eval[:-1])
plt.plot(problem_size[:-1], mimic_algo_statistics_function_eval[:-1])
plt.title('Knapsack - Problem Size Vs No. of function evaluations')
plt.ylabel('# of function evaluations', fontsize = 14)
plt.xlabel('Problem Size', fontsize = 14)
plt.legend(['RHC','SA','GA','MIMIC'], loc='upper left')
plt.ylim(0,28000)
plt.show()   

# summarize fitness value vs problem size
import matplotlib.pyplot as plt
plt.plot(problem_size[:-1], randomhill_statistics_fitness[:-1])
plt.plot(problem_size[:-1], sa_statistics_fitness[:-1])
plt.plot(problem_size[:-1], genetic_algo_statistics_fitness[:-1])
plt.plot(problem_size[:-1], mimic_algo_statistics_fitness[:-1])
plt.title('KnapSack - Problem Size Vs Fitness Value')
plt.ylabel('Fitness Value', fontsize = 14)
plt.xlabel('Problem Size', fontsize = 14)
plt.legend(['RHC','SA','GA','MIMIC'], loc='upper left')
plt.ylim(0,1000)
plt.show()  


# summarize time value vs problem size(No. Of Bits)
import matplotlib.pyplot as plt
plt.plot(problem_size[:-1], randomhill_statistics_time[:-1])
plt.plot(problem_size[:-1], sa_statistics_time[:-1])
plt.plot(problem_size[:-1], genetic_algo_statistics_time[:-1])
plt.plot(problem_size[:-1], mimic_algo_statistics_time[:-1])
plt.title('KnapSack - Problem Size (No. Of Bits) Vs Time')
plt.ylabel('Time Taken', fontsize = 14)
plt.xlabel('Problem Size', fontsize = 14)
plt.legend(['RHC','SA','GA','MIMIC'], loc='upper left')
plt.ylim(0.0001,100)
plt.show()  

# summarize time value vs problem size(No. Of Bits)
import matplotlib.pyplot as plt
plt.plot(problem_size[:-1], randomhill_statistics_time[:-1])

plt.title('KnapSack - Problem Size (No. Of Bits) Vs Time')
plt.ylabel('Time Taken', fontsize = 14)
plt.xlabel('Problem Size', fontsize = 14)
plt.legend(['RHC'], loc='upper left')
plt.ylim(0.01,0.03)
plt.show() 









'''


'''
'''

problem_fit = mlrose.oneMax(length = 8, fitness_fn = fitness_coords, maximize=False)


import unittest
import numpy as np
from mlrose import (OneMax, FlipFlop, FourPeaks, SixPeaks, ContinuousPeaks,
                    Knapsack, TravellingSales, Queens, MaxKColor,
                    CustomFitness)
from mlrose.fitness import head, tail, max_run
'''

'''

class TestFitness(unittest.TestCase):
    """Tests for fitness.py."""

    @staticmethod
    def test_onemax():
        """Test OneMax fitness function"""
        state = np.array([0, 1, 0, 1, 1, 1, 1])
        assert OneMax().evaluate(state) == 5
        
if __name__ == '__main__':
    unittest.main()
    
    
        

#!/bin/bash

#echo "Running tests on test_activation.py"
python test_activation.py

#echo "Running tests on test_decay.py"
python test_decay.py

#echo "Running tests on test_fitness.py"
python test_fitness.py

#echo "Running tests on test_algorithms.py"
python test_algorithms.py

#echo "Running tests on test_opt_probs.py"
python test_opt_probs.py

#echo "Running tests on test_neural.py"
python test_neural.py

echo "Finished all tests"
'''

