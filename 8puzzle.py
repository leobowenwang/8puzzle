# ------------------------------------------------------------------
# Filename:    8puzzle.py
# By:          Chmel Sebastian & Wang Leo
# ------------------------------------------------------------------
# File description:
# Solves a random generated 8-puzzle board using manhattan distance
# and misplaced tiles heuristics and print the steps.
# ------------------------------------------------------------------

import numpy as np
from copy import deepcopy
from collections import defaultdict


# calculate Manhattan distance cost between each digit of board(start state) and the goal state
def manhattan_distance(board, goal):
    # counts row distance of every numbered tile to its goal position
    row_diff = abs(board // 3 - goal // 3)
    # counts column distance of every numbered tile to its goal position
    col_diff = abs(board % 3 - goal % 3)
    cost = row_diff + col_diff
    # return the sum of sliced cost array (starting at index 1) 
    return sum(cost[1:])

# calculate the number of misplaced tiles in the current state compared to the goal state
def misplaced_tiles(board, goal):
    # count how many input in both arrays are different
    cost = np.sum(board != goal)
    return cost
    
# will identify the coordinate of each goal or initial state values
def coordinate(board):
    position = np.array(range(9))
    for x, y in enumerate(board):
        position[y] = x  
    return position

# solve puzzle!
def solve(board, goal, input):
    steps = np.array(
        [
            ('up', [0, 1, 2], -3),
            ('down', [6, 7, 8], 3),
            ('left', [0, 3, 6], -1),
            ('right', [2, 5, 8], 1)
        ],
        dtype=[
            ('move', str, 1),
            ('position', list),
            ('delta', int)
        ]
    )

    STATE = [
        ('board', list),
        ('parent', int),
        ('gn', int),
        ('hn', int)
    ]
    # gn = cost to reach node n from start state
    # hn = cost to reach from note n to goal node
    # fn = estimated cost of the cheapest solution
    PRIORITY = [
        ('position', int),
        ('fn', int)
    ]

    # With defaultdict we can check the contense in O(1), where with np.all() it is O(n).
    # check if already used
    previous_board = defaultdict(bool)
    
    # initial STATE values
    goalcoordinate = coordinate(goal)
    parent = -1
    gn = 0
    hn = 0

    # implementing the chosen heuristic
    if (input==1):
        hn = manhattan_distance(coordinate(board), goalcoordinate)
    if (input==2):
        hn = misplaced_tiles(coordinate(board), goalcoordinate)

    # set board state
    state = np.array([(board, parent, gn, hn)], STATE)
    # initialize priority queue
    priority = np.array([(0, hn)], PRIORITY)

    while True:
        # sort priority queue using mergesort (first cheapest solution, then position)
        priority = np.sort(priority, kind='mergesort', order=['fn', 'position'])
        # pick out first from sorted to explore
        position = priority[0][0]
        # delete from the priority queue the search node with the minimum priority, and insert onto the priority queue all neighboring search nodes
        priority = np.delete(priority, 0, 0)
        board = state[position][0]
        gn = state[position][2] + 1
        # Identify the blank space in input
        space = int(np.where(board == 0)[0])

        for s in steps:
            if space not in s['position']:
                # generate new state as copy of current
                current = deepcopy(board)
                delta_space = space + s['delta']
                # move the tile
                current[space], current[delta_space] = current[delta_space], current[space]
                current_tuple = tuple(current)
                
                # the current_tuple should be included in previous boards
                if previous_board[current_tuple]:
                    continue
                
                previous_board[current_tuple] = True

                # calls heuristic to calculate cost
                if (input==1):
                    hn = manhattan_distance(coordinate(current_tuple), goalcoordinate)
                if (input==2):
                    hn = misplaced_tiles(coordinate(current_tuple), goalcoordinate)
                
                # append the current move to the list
                state = np.append(state,np.array([(current, position, gn, hn)], STATE))

                # set the priority for the current state (position & cost)
                priority = np.append(priority,np.array([(len(state) - 1, gn + hn)], PRIORITY))

                # Checking if the node in state is matching the goal state.  
                if np.array_equal(current, goal):
                    return state, len(priority)

def getInvCount(arr):
    sum = 0
    # no_zero - is the same array without 0, so that we don't miscount inversions
    no_zero = arr[arr != 0]

    # iterate through array - compare left and right tile
    for i in range(8):
        for j in range(i + 1, 8):
            if (no_zero[i] > no_zero[j]):
                sum += 1
    return sum

def getBestSolution(state):
    optimal = np.array([], int).reshape(-1, 9)
    count = len(state) - 1
    while count != -1:
        optimal = np.insert(optimal, 0, state[count]['board'], 0)
        count = int(state[count]['parent'])
    # reformat the array string without '[]'
    printstate = str(optimal.reshape(-1, 3, 3)).replace('[', ' ').replace(']', '')
    return printstate

def main():
    # define goal state
    goal = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    # generate random intial board state with a size 9 array (0-9 are unique numbers)
    board = np.random.choice(np.arange(0, 9), replace=False, size=(9))
    
    # alternative: define own puzzle
    # board = np.array([1,2,3,4,5,6,7,8,0])

    # showing our generated board
    show_board = board.reshape(-1, 3, 3)
    show_board = str(show_board).replace('[', ' ').replace(']', '')
    print('Random Board: \n'+str(show_board))
    
    print('Inversion Count: ' + str(getInvCount(board)))

    if getInvCount(board) % 2:
        print('Not solvable!\n-------------')
        return
    else:
        print('Solvable!\n-------------')

    n = int(input("Choose the heuristic\n1. Manhattan distance \n2. Misplaced tiles \n"))

    state, explored = solve(board, goal, n)
    optimal = getBestSolution(state)

    print((
        '{}\n'
        '-------------\n'
        'Total nodes generated: {}\n'
        'Total nodes explored:  {}\n'
        'Total steps: {}\n'
    ).format(optimal, len(state), len(state) - explored, len(optimal) - 1))

if __name__ == '__main__':
    main()