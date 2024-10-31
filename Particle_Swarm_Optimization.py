'''
Authored By: Prisha Rathi

Resources: https://towardsdatascience.com/ways-to-evaluate-regression-models-77a3ff45ba70
           https://www.simplilearn.com/tutorials/statistics-tutorial/mean-squared-error
           https://statisticsbyjim.com/regression/mean-squared-error-mse/
           https://matplotlib.org/stable/gallery/mplot3d/surface3d.html
           https://stackoverflow.com/questions/27449109/adding-legend-to-a-surface-plot
'''

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np


# -- For stopping conditions
EPSILON = 0.01


# -- Givem function f in assignment
def f(x,y,a,b,c):
    return ((a*(x**2) + y**2 + b) * np.sin(c*x + y))


# -- Cost function
def mean_square_error(csv_values,solution):
    each_x = csv_values[:,0]
    each_y = csv_values[:,1]
    # -- Ideal values (what we want)
    each_z = csv_values[:,2]
    
    # -- The resulting values
    surface_z = f(each_x, each_y, solution[0], solution[1],solution[2])
    
    # -- Minimize difference between each_z and surface_z
    difference = ((each_z - surface_z)**2).mean()
    return difference
    

# -- PSO function
def PSO(csv_values, num_particles, max_iterations, particle_inertia, speed_limit, c1, c2):
    # -- Initialize swarm -- #
    # -- Position (solution): n x 3 array
    position = np.zeros((num_particles, 3))
    # -- a values
    position[:,0] = np.random.uniform(-5,5, num_particles)
    # -- b values
    position[:,1] = np.random.uniform(-50,50, num_particles)
    # -- c values
    position[:,2] = np.random.uniform(0.01,10, num_particles)
    
    # -- Velocity (direction, speed)
    velocity = np.zeros((num_particles, 3))
    
    # -- Personal best (n x 3)
    personal_best = position.copy() 
    # -- Stores the objective value of the function (n x 1), keeps track of current personal best of each particle
    personal_best_obj_function_value = np.zeros(num_particles)
    # -- Access index
    for solution in range(len(personal_best_obj_function_value)):
        personal_best_obj_function_value[solution] = mean_square_error(csv_values,personal_best[solution])

    # -- Neighbour best (star topology) -- #
    # -- Get global best
    global_best = personal_best[personal_best_obj_function_value.argmin()]
    # -- For plotting part A
    global_best_cost = []
    
    current_iteration = 0
    
    # -- While termination criteria not met
    while((current_iteration <= max_iterations) and (personal_best_obj_function_value.min() > EPSILON)):
    # -- For each particle (asynchronous algorithm)
        for particle_index in range(num_particles):
            # -- Update the particle's velocity
            velocity[particle_index] = particle_inertia * velocity[particle_index] + (c1/2)*(personal_best[particle_index] - position[particle_index]) + (c2/2)*(global_best - position[particle_index])
            
            # -- Check if speed from velocity exceeds speed limit
            a = velocity[particle_index][0]
            b = velocity[particle_index][1]
            c = velocity[particle_index][2]
            
            # -- Calculate magnitude of speed
            speed = np.sqrt(a**2 + b**2 + c**2)
            
            if(speed > speed_limit):
                # -- velocity has will have a speed of 1
                velocity[particle_index] = velocity[particle_index] / speed
                # -- Multiple speed by a new magnitude (speed limit) for velocity
                velocity[particle_index] = velocity[particle_index] * speed_limit
                
            # -- Update the particle's position
            position[particle_index] = position[particle_index] + velocity[particle_index]
            # -- Update the particle's personal best
            new_possible_personal_best_obj_function_value = mean_square_error(csv_values, position[particle_index])
            if(new_possible_personal_best_obj_function_value < personal_best_obj_function_value[particle_index]):
                personal_best_obj_function_value[particle_index] = new_possible_personal_best_obj_function_value
                personal_best[particle_index] = position[particle_index]
                # -- Update the Nbest
                global_best = personal_best[personal_best_obj_function_value.argmin()]
                
        global_best_cost.append(personal_best_obj_function_value.min())
        current_iteration += 1
        
    return global_best, personal_best_obj_function_value.min(), current_iteration, global_best_cost


# -- Part a Plots
# -- Plot of best cost vs iterations
def part_A_simple_plots(all_global_best_cost):
    plt.plot(all_global_best_cost)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Best Cost')
    plt.title("Plot of Best Cost vs Iterations")
    plt.savefig("A3_Q2_part_a") 
 
 
# -- Plot 3D Surface
def plot_3D_surface(x_bound,y_bound,a,b,c, csv_values, smoothness):
    figure, axis = plt.subplots(subplot_kw={"projection": "3d"})
    
    x = np.arange(x_bound[0], x_bound[1], smoothness)
    y = np.arange(y_bound[0], y_bound[1], smoothness)
    x,y = np.meshgrid(x,y)
    z = f(x,y,a,b,c)
    
    surface = axis.plot_surface(x,y,z,alpha=0.7)
    if csv_values is not None:
        axis.scatter(csv_values[:,0], csv_values[:,1], csv_values[:,2], color='tab:orange')
        
    axis.legend([mpl.lines.Line2D([0],[0], linestyle="none", c='tab:blue', marker='o'), mpl.lines.Line2D([0],[0], linestyle="none", c='tab:orange', marker='o')], ["Surface", "Points from CSV file"], numpoints=1)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    plt.title("3D Plot of Function f With Best θ:")
    plt.savefig("A3_Q2_part_a_best_theta") 


# -- Part d:
# -- Adjusting number of particles
def adjust_num_of_particles(csv_values, num_of_particle_adjustments):
    for num_of_particles in num_of_particle_adjustments:
        _,_,_, all_global_best_cost = PSO(csv_values, num_of_particles, 45, 0.792, 50, 1.4944, 1.4944)
        plt.plot(all_global_best_cost, label=f"Number of particles={num_of_particles}")
        
    plt.legend()
    plt.xlabel('Number of Iterations')
    plt.ylabel('Best Cost')
    plt.title("Plot of Best Cost vs Iterations - With Number of Particle Adjustments")
    plt.show() 
    

# -- Adjusting speed limit    
def adjust_speed_limit(csv_values, speed_limit_adjustments):
    for speed_limit in speed_limit_adjustments:
        _,_,_, all_global_best_cost = PSO(csv_values, 200, 45, 0.792, speed_limit, 1.4944, 1.4944)
        plt.plot(all_global_best_cost, label=f"Speed Limit ={speed_limit}")
        
    plt.legend()
    plt.xlabel('Number of Iterations')
    plt.ylabel('Best Cost')
    plt.title("Plot of Best Cost vs Iterations - With Speed Limit Adjustments")
    plt.show() 
    

# -- Adjusting attraction parameters (c1 and c2)    
def adjust_attraction_param(csv_values, attraction_param_adjustments):
    for attraction_param in attraction_param_adjustments:
        _,_,_, all_global_best_cost = PSO(csv_values, 200, 45, 0.792, 0.5, attraction_param, attraction_param)
        plt.plot(all_global_best_cost, label=f"Attraction Parameters={attraction_param}")
        
    plt.legend()
    plt.xlabel('Number of Iterations')
    plt.ylabel('Best Cost')
    plt.title("Plot of Best Cost vs Iterations - With Attraction Parameter Adjustments")
    plt.show() 


def main():
    csv = pd.read_csv("data_points.csv")
    data = csv.values
    final_solution, final_solution_cost, num_of_iterations, all_global_best_cost = PSO(data, 200, 45, 0.792, 50, 1.4944, 1.4944)
    print(f"\nFinal solution: a = {final_solution[0]}, b = {final_solution[1]}, c = {final_solution[2]}")
    print(f"\nFinal solution cost: {final_solution_cost}")
    print(f"\nNumber of Iterations: {num_of_iterations}\n")
    
    # -- Part a:
    part_A_simple_plots(all_global_best_cost)
    plot_3D_surface((-10,10), (-10,10),final_solution[0], final_solution[1], final_solution[2], data, 0.25)
    plt.clf()
    
    # -- Part c:
    particle_inertia_adjustments = [0.5,0.6,0.792,0.9,1]
    for particle_inertia in particle_inertia_adjustments:
        _,_,_, all_global_best_cost = PSO(data, 200, 45, particle_inertia, 50, 1.4944, 1.4944)
        plt.plot(all_global_best_cost, label=f"Inertia={particle_inertia}")
    plt.legend()
    plt.xlabel('Number of Iterations')
    plt.ylabel('Best Cost')
    plt.title("Plot of Best Cost vs Iterations - With Particle Inertia Adjustments")
    plt.savefig("A3_Q2_part_c") 
    plt.clf()
    
     # -- Part d:
    num_of_particles = [100,150,200,250,300]
    adjust_num_of_particles(data, num_of_particles)
    speed_limit = [10,20,30,40,50]
    adjust_speed_limit(data, speed_limit)
    attraction_param = [1,1.2,1.4944,1.7,2]
    adjust_attraction_param(data, attraction_param)
    
    
if __name__ == "__main__":
    main()

