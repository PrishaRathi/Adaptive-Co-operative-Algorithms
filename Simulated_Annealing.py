'''
Authored By: Prisha Rathi

Resources: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
           https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
           https://matplotlib.org/stable/gallery/shapes_and_collections/scatter.html#sphx-glr-gallery-shapes-and-collections-scatter-py
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -- For neighbourhood solution, the window
WIDTH = 0.08

# -- Easom Functions, correct solution is -1
def easom(x):
    return -np.cos(x[0])*np.cos(x[1])*np.exp(-((x[0] - np.pi)**2)-((x[1] - np.pi)**2))

# -- Linear Decrease Function
def linear_decrease(t,alpha):
    return t - alpha

# -- Exponential Decrease Function
def exponential_decrease(t,alpha):
    return t * alpha

# -- Slow Decrease Function
def slow_decrease(t,beta):
    return t/(1 + beta*t)

# -- Implemented Simulated Annealing function from psuedocode in ECE 457A slides
def simulated_annealing(s0, t0, temp_decrease_function, alpha, max_iterations=1000):
    current_sol = s0
    current_temp = t0
    
    current_iterations = 0
    
    # -- Stopping Conditions:
    while easom(current_sol) > -1 + 0.001 and current_iterations < max_iterations:
        
        for i in range(5):
            new_sol = current_sol + np.random.uniform(-WIDTH, WIDTH, size=2)
            change_in_cost = easom(new_sol) - easom(current_sol)
            
            if(change_in_cost < 0):
                current_sol = new_sol
            else:
                random_num = np.random.uniform(0,1)
                acceptance_probability = np.exp(-change_in_cost/current_temp)
                
                if(random_num < acceptance_probability):
                    current_sol = new_sol
                    
        current_temp = temp_decrease_function(current_temp,alpha)
        current_iterations = current_iterations + 1
        
    return current_sol


def part_b_i():
    print("\nPart b i:")
    
    all_s0 = []
    all_solutions = []
    all_easom_values = []

    # -- To test for 10 different initial points
    for i in range(10):
        s0 = np.random.uniform(-100, 100, size=2)
        # -- For ideal solution:
            # -- s0 = np.array([3.14,3.14])
        t0 = 10
        alpha = 0.005
        # -- Call implemented SA function
        solution = simulated_annealing(s0,t0,linear_decrease,alpha)
        
        print("\nDifferent initial s0: ", s0.tolist())
        print("Simulated Annealing Solution: ", solution.tolist())
        print("Easom Value: ", easom(solution))
        
        # -- Append all values to list to send into pandas data frame:
        all_s0.append(str(s0.tolist()))
        all_solutions.append(str(solution.tolist()))
        all_easom_values.append(str(easom(solution)))
        
    # -- The commented lines below in this function are for generating the CSV files -- #   
    # df = pd.DataFrame(data=np.asarray([all_s0, all_solutions, all_easom_values]).T, columns=['Initial Solution', 'Final Solution', 'Easom Value'])
    # df.to_csv('A2_Q1_part_b_i.csv', index=False)
    
    
def part_b_ii():
    print("\nPart b ii:")
    
    #-- Best initial point from part i) is (7.344487440898902, 3.579900379961117), easom = -3.0053063153844545e-05
    all_t0 = []
    all_solutions = []
    all_easom_values = []
    
    # -- To test for 10 different initial temperatures 
    for i in range(10):
        # -- Used best initial point from part i)
        s0 = np.array([7.344487440898902, 3.579900379961117])
        t0 = np.random.uniform(4,14)
        alpha = 0.005
        # -- Call implemented SA function
        solution = simulated_annealing(s0,t0,linear_decrease,alpha)
        
        print("\nDifferent initial t0: ", t0)
        print("Simulated Annealing Solution: ", solution.tolist())
        print("Easom Value: ", easom(solution))
        
        # -- Append all values to list to send into pandas data frame
        all_t0.append(str(t0))
        all_solutions.append(str(solution.tolist()))
        all_easom_values.append(str(easom(solution)))
    
    # -- The commented lines below in this function are for generating the CSV files -- #    
    # df = pd.DataFrame(data=np.asarray([all_t0, all_solutions, all_easom_values]).T, columns=['Initial Temperature', 'Final Solution', 'Easom Value'])
    # df.to_csv('A2_Q1_part_b_ii.csv', index=False)
    
    
def part_b_iii():
    print("\nPart b iii:")
    
    #-- Best initial point from part i) is (7.344487440898902, 3.579900379961117), easom = -3.0053063153844545e-05
    # -- Best temperature from part ii) is 8.651934254372923, easom = -0.9999718775472437
    all_temp_decrease_functions = []
    all_alpha_values = []
    all_solutions = []
    all_easom_values = []
    
    # -- Used best initial point and best initial temperature from part i) and ii)
    s0 = np.array([7.344487440898902, 3.579900379961117])
    t0 = 8.651934254372923
    
    # -- To test for 9 different annealing schedules
    # -- Linear Decrease:
    print("\nLinear Decrease: ")
    for alpha in [0.003, 0.005, 0.007]:
        solution = simulated_annealing(s0,t0,linear_decrease,alpha)
        all_temp_decrease_functions.append('Linear Decrease')
        all_alpha_values.append(alpha)
        all_solutions.append(str(solution.tolist()))
        all_easom_values.append(str(easom(solution)))
        print("\nDifferent alpha value: ", alpha)
        print("Simulated Annealing Solution: ", solution.tolist())
        print("Easom Value: ", easom(solution))
        
    # -- Exponential Decrease:
    print("\nExponential Decrease: ")
    for alpha in [0.5, 0.75, 0.9]:
        solution = simulated_annealing(s0,t0,exponential_decrease,alpha)
        all_temp_decrease_functions.append('Exponential Decrease')
        all_alpha_values.append(alpha)
        all_solutions.append(str(solution.tolist()))
        all_easom_values.append(str(easom(solution)))
        print("\nDifferent alpha value: ", alpha)
        print("Simulated Annealing Solution: ", solution.tolist())
        print("Easom Value: ", easom(solution))
        
    # -- Slow Decrease:
    print("\nSlow Decrease: ")
    for beta in [0.0003, 0.0005, 0.0007]:
        solution = simulated_annealing(s0,t0,slow_decrease,beta)
        all_temp_decrease_functions.append('Slow Decrease')
        all_alpha_values.append(beta)
        all_solutions.append(str(solution.tolist()))
        all_easom_values.append(str(easom(solution)))
        print("\nDifferent beta value: ", beta)
        print("Simulated Annealing Solution: ", solution.tolist())
        print("Easom Value: ", easom(solution))
    
    # -- The commented lines below in this function are for generating the CSV files -- #
    # df = pd.DataFrame(data=np.asarray([all_temp_decrease_functions,all_alpha_values, all_solutions, all_easom_values]).T, columns=['Temperature Decrease Function', 'Alpha', 'Solution', 'Easom Value'])
    # df.to_csv('A2_Q1_part_b_iii.csv', index=False)
    
    
def part_d():
    print("\nPart d:")
        
    all_solutions = []
    
    # -- Best Settings #1 -- #
    # -- From part b)
    print("\nSetting #1: ")
    s0 = np.array([7.344487440898902, 3.579900379961117])
    t0 = 8.651934254372923
    alpha = 0.005
    for i in range (10):
        solution = simulated_annealing(s0,t0,linear_decrease,alpha)
        all_solutions.append(solution)
        print("Simulated Annealing Solution: ", solution.tolist())
        print("Easom Value: ", easom(solution))
    
    # -- Best Settings #2 -- #
    # -- From multiple runs -- #
    print("\nSetting #2: ")
    s0 = np.array([4.97950371745615, 4.980513136131341])
    t0 = 8.651934254372923
    alpha = 0.75
    for i in range (10):
        solution = simulated_annealing(s0,t0,exponential_decrease,alpha)
        all_solutions.append(solution)
        print("Simulated Annealing Solution: ", solution.tolist())
        print("Easom Value: ", easom(solution))
        
    # -- Best Settings #3 -- #
    # -- From multiple runs -- #
    print("\nSetting #3: ")
    s0 = np.array([3.1404873037674816, 3.1397871209467434])
    t0 = 8.651934254372923
    alpha = 0.0003
    for i in range (10):
        solution = simulated_annealing(s0,t0,slow_decrease,alpha)
        all_solutions.append(solution)
        print("Simulated Annealing Solution: ", solution.tolist())
        print("Easom Value: ", easom(solution))
    
    # -- The commented lines below in this function are for generating the two figures -- #
        
    # # -- Convert list to numpy array
    # # -- Figure 1
    # all_solutions = np.array(all_solutions)
    # # -- Create first scatter plot
    # plt.scatter(all_solutions[:,0], all_solutions[:,1])
    # plt.xlim(-100,100)
    # plt.ylim(-100,100)
    # plt.xlabel('Values of x1')
    # plt.ylabel('Values of x2')
    # plt.title("Scatter Plot of Best Settings")
    # plt.savefig("A2_Q1_part_d_fig_1")
    
    # # -- Figure 2
    # # -- Create scatter plot zoomed in on (π,π)
    # plt.figure()
    # plt.scatter(all_solutions[:,0], all_solutions[:,1])
    # plt.xlim(np.pi - 3,np.pi + 3)
    # plt.ylim(np.pi - 3,np.pi + 3)
    # plt.title("Zoomed in Plot on (π,π)")
    # plt.xlabel('Values of x1')
    # plt.ylabel('Values of x2')
    # plt.savefig("A2_Q1_part_d_fig_2")


def main():
   part_b_i()
   part_b_ii()
   part_b_iii()
   part_d()     


if __name__ == "__main__":
    main()
