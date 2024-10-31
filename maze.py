'''
Authored By: Prisha Rathi

Resources: https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
           https://matplotlib.org/stable/tutorials/introductory/images.html
           https://algodaily.com/lessons/what-is-the-manhattan-distance            
'''

import matplotlib.pyplot as plt
import numpy as np
from queue import PriorityQueue


# -- Reads maze.txt file as a single string
with open('maze.txt', 'r') as f:
    text = ''.join(f.readlines())
# -- Converts the string into a 2D list of integers (list of list)
maze_list = eval(text)


GOAL = (17,14)
START = (1,3)
MAX_X = 25
MAX_Y = 25


# -- Account for obstacles and out of maze
def next_possible_states(x,y,maze_list):
    state_list = []
    # -- Within bounds & not a blocked position
    # -- UP
    if ((x-1) >= 0) and (maze_list[x-1][y] == 0):
        state_list.append((x-1,y))
    # -- RIGHT        
    if ((y+1) < MAX_Y) and (maze_list[x][y+1] == 0):
        state_list.append((x,y+1))
    # -- DOWN
    if ((x+1) < MAX_X) and (maze_list[x+1][y] == 0):
        state_list.append((x+1,y))
    # -- LEFT    
    if ((y-1) >= 0) and (maze_list[x][y-1] == 0):
        state_list.append((x,y-1))
    
    return state_list
 

# -- For potting the maze
def plot_maze(states_visited,complete_path):
    # -- Convert maze list to array
    maze_array = np.array(maze_list)
    # -- Need to reverse black and white for maze
    maze_array = 1 - maze_array
    # -- Convert grayscale to RGB (for start and end pixel)
    maze_list_rgb = []
    for i in range(3):
        maze_list_rgb.append(maze_array)
    
    # -- Transposing an image in order to fix dimensions of image
    # -- 255 for RGB
    maze_array_rgb = np.array(maze_list_rgb).transpose([1,2,0]) * 255
    
    # -- For expanded nodes
    for states in states_visited:
        # -- Change maze for expanded nodes (gray)
        maze_array_rgb[states]=(128,128,128)
    
    # -- For path
    for path in complete_path:
        # -- Change maze for complete path (red)
        maze_array_rgb[path]=(255,0,0)
    
    # -- Change maze at start for green
    maze_array_rgb[START[0], START[1],:]=(0,255,0)
    # -- Change maze at goal for red
    maze_array_rgb[GOAL[0], GOAL[1],:]=(255,0,0)
    
    # -- Plot maze
    plt.imshow(maze_array_rgb)


# -- BFS approach
def bfs(maze_list):
    states_visited = set()
    
    def bfs_search(start_x, start_y):
        # -- Extra square bracket is to accumulate past nodes
        queue = [((start_x, start_y),[])]
        
        # -- For each level, if queue is not empty, there is something to explore
        while len(queue) > 0:
            # -- Remove and return first element in queue
            (current_x, current_y), grandparent_nodes = queue.pop(0)
            
            # -- Skips any duplicates so same state is not checked
            if(current_x,current_y) in states_visited:
                continue
            
            # -- If it's different, then add
            states_visited.add((current_x, current_y))
            
            # -- Checks if at goal state
            if (current_x == GOAL[0] and current_y == GOAL[1]):
                return grandparent_nodes + [(current_x, current_y)] 
                
            # -- States left to explore
            states_to_explore = next_possible_states(current_x, current_y, maze_list)
            
            # -- Process states left to explore to account for parent (or grandparent,etc) nodes
            processed_states = []
            for states in states_to_explore:
                processed_states.append((states, grandparent_nodes + [(current_x,current_y)]))
                
            # -- Extending list, not storing a new one
            queue.extend(processed_states)
            
    complete_path = bfs_search(START[0], START[1])
    return states_visited, complete_path, f"For BFS:\nComplete Path is: {complete_path}\nCost of Path is: {len(complete_path)}\nNumber of nodes explored: {len(states_visited)}\n"
    
                     
# -- DFS approach
def dfs(maze_list):
    
    states_visited = set()
    
    def dfs_search(current_x, current_y):
        # -- print(current_x, current_y)
        states_visited.add((current_x,current_y))
        
        # -- Checks if at goal state
        if (current_x == GOAL[0] and current_y == GOAL[1]):
            return [(current_x, current_y)]
        else:
            # -- States left to explore
            states_to_explore = next_possible_states(current_x, current_y, maze_list)
            
            for new_x, new_y in states_to_explore:
                
                # -- Ensure same state isn't checked multiple times
                if ((new_x,new_y)) not in states_visited:
                    result = dfs_search(new_x, new_y)
        
                    # -- In case no states
                    if (result != None):
                        return [(current_x, current_y)] + result
            return None
                    
    complete_path = dfs_search(START[0], START[1])
    return states_visited, complete_path, f"For DFS:\nComplete Path is: {complete_path}\nCost of Path is: {len(complete_path)}\nNumber of nodes explored: {len(states_visited)}\n"             
    

# -- A* approach
def a_star(maze_list):
    
    # -- Heuristic function
    # -- Manhattan Distance Formula, we can't use diagonal moves, limited to up, right, down, left
    def heuristic_function(x,y):
        return (abs(x-GOAL[0]) + abs(y-GOAL[1]))
    
    states_visited = set()
    
    def a_star_search(start_x, start_y):
        
        frontier = PriorityQueue()
        # -- Priority, state, cost, grandparent_nodes 
        frontier.put((0,(start_x, start_y),0,[]))
        
        while frontier.empty() != True:
            priority, (current_x, current_y), cost_so_far, grandparent_nodes = frontier.get()
            
            # -- Skips any duplicates so same state is not checked
            if(current_x,current_y) in states_visited:
                continue
            # -- If it's different, then add
            states_visited.add((current_x, current_y))
            
            # -- Checks if at goal state
            if (current_x == GOAL[0] and current_y == GOAL[1]):
                return grandparent_nodes + [(current_x, current_y)] 
            
            # -- States left to explore
            states_to_explore = next_possible_states(current_x, current_y, maze_list)
            
            # -- Process states left to explore to account for parent (or grandparent, etc) nodes
            for new_x, new_y in states_to_explore:
                priority = (cost_so_far + 1) + heuristic_function(new_x, new_y)
                frontier.put((priority,(new_x,new_y),(cost_so_far + 1),grandparent_nodes + [(current_x,current_y)]))
    
    complete_path = a_star_search(START[0], START[1])
    return states_visited, complete_path, f"For A* Search:\nComplete Path is: {complete_path}\nCost of Path is: {len(complete_path)}\nNumber of nodes explored: {len(states_visited)}\n" 
  
 
def main():
    states_visited_bfs, complete_path_bfs, bfs_print = bfs(maze_list)
    print(bfs_print)
    plot_maze(states_visited_bfs,complete_path_bfs)
    plt.title("BFS")
    plt.axis("off")
    plt.savefig("Maze_bfs")
    
    states_visited_dfs, complete_path_dfs, dfs_print = dfs(maze_list)
    print(dfs_print)
    plot_maze(states_visited_dfs,complete_path_dfs)
    plt.title("DFS")
    plt.axis("off")
    plt.savefig("Maze_dfs")
    
    states_visited_a_star, complete_path_a_star, a_star_print = a_star(maze_list)
    print(a_star_print)
    plot_maze(states_visited_a_star,complete_path_a_star)
    plt.title("A* Search")
    plt.axis("off")
    plt.savefig("Maze_a_star_search")
    
    
if __name__ == "__main__":
    main()