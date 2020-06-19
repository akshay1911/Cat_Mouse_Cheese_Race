from __future__ import print_function
import os, sys, time, json, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
import matplotlib.pyplot as plt
from keras.models import model_from_json
%matplotlib inline
import gym
from gym import wrappers
from datetime import datetime

mouse_mark = 0.5    # The current mouse cell
cat_mark=0.0        # The current cat cell except the starting position
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3
LEFT_UP=4
LEFT_DOWN=5
UP_RIGHT=6
RIGHT_DOWN=7
# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down'
}
num_actions = len(actions_dict)
# Exploration factor
epsilon = 0.1

# maze is a 2d Numpy array of floats between 0.0 to 1.0
# 1.0 corresponds to a free cell, and 0.0 an occupied cell
# mouse = (row, col) initial mouse position (defaults to (0,0))

class Maze(object):
    def __init__(self, maze, mouse=(0,0)):
        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        self.target = (nrows-1, ncols-1)   # target cell
        self.cat=self.target
        self.free_cells = [(r,c) for r in range(nrows) for c in range(ncols) if self._maze[r,c] == 1.0]
        self.free_cells.remove(self.target)
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if not mouse in self.free_cells:
            raise Exception("Invalid mouse Location: must sit on a free cell")
        self.reset(mouse)

    def reset(self, mouse):
        self.mouse = mouse
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        self.cat=(nrows-1, ncols-1)
        row, col = mouse
        self.maze[row, col] = mouse_mark
        self.state = (row, col, 'start')
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = mouse_row, mouse_col, mode = self.state
        if self.maze[mouse_row, mouse_col] > 0.0: self.visited.add((mouse_row, mouse_col))  # mark visited cell
        valid_actions = self.valid_actions([0,1,2,3])                
        if action in valid_actions:
            nmode = 'valid'
            if action == LEFT: ncol -= 1
            elif action == UP: nrow -= 1
            elif action == RIGHT: ncol += 1
            elif action == DOWN: nrow += 1
        else: mode = 'invalid'                # invalid action, no change in mouse position
        self.state = (nrow, ncol, nmode)      # new state

    def get_reward(self,caught):
        mouse_row, mouse_col, mode = self.state
        nrows, ncols = self.maze.shape
        if mouse_row == nrows-1 and mouse_col == ncols-1: return 1.0
        if caught == "Y":
            mode="caught"
            self.state=(mouse_row, mouse_col, mode)
            return -1.0
        if (mouse_row, mouse_col) in self.visited: return -0.25                         
        if mode == 'invalid': return -0.75                      
        if mode == 'valid': return -0.05
        
    def chase(self, valid_actions):
        mouse_row, mouse_col, mode = self.state
        nrows, ncols = self.maze.shape
        if self.cat == (mouse_row, mouse_col): return -1
        best_action = -2
        best_dist = (nrows-1)**2 + (ncols-1)**2 + 1
        for action in valid_actions:
            cat_x=self.cat[0]
            cat_y=self.cat[1]
            if action == LEFT: 
                dist = (cat_x - mouse_row) ** 2 + (cat_y-1 - mouse_col) ** 2
                if best_dist>dist:
                    best_dist=dist
                    best_action=LEFT
            elif action == UP:
                dist = (cat_x-1 - mouse_row) ** 2 + (cat_y - mouse_col) ** 2
                if best_dist>dist:
                    best_dist=dist
                    best_action=UP
            elif action == RIGHT: 
                dist = (cat_x - mouse_row) ** 2 + (cat_y+1 - mouse_col) ** 2
                if best_dist>dist:
                    best_dist=dist
                    best_action=RIGHT
            elif action == DOWN: 
                dist = (cat_x+1 - mouse_row) ** 2 + (cat_y - mouse_col) ** 2
                if best_dist>dist:
                    best_dist=dist
                    best_action=DOWN
        #check for special cat movements
        if best_dist == 1: #indicating the cat and mouse are positioned on a diagonal path
            curr_dist=(cat_x - mouse_row) ** 2 + (cat_y - mouse_col) ** 2   #check for diagonality 
            if curr_dist == 2:
                if best_action==LEFT:
                    #for action in valid_actions:
                    if cat_x!=0 and ((cat_x-1 - mouse_row) ** 2 + (cat_y - mouse_col) ** 2)==1: best_action = LEFT_UP
                    elif cat_x!=nrows-1 and ((cat_x+1 - mouse_row) ** 2 + (cat_y - mouse_col) ** 2)==1: best_action = LEFT_DOWN
                elif best_action==UP:
                    #for action in valid_actions:
                    if cat_y!=0 and ((cat_x - mouse_row) ** 2 + (cat_y-1 - mouse_col) ** 2)==1: best_action = LEFT_UP
                    elif cat_y!=ncols-1 and ((cat_x - mouse_row) ** 2 + (cat_y+1 - mouse_col) ** 2)==1: best_action = UP_RIGHT
                elif best_action==RIGHT:
                    #for action in valid_actions:
                    if cat_x!=0 and ((cat_x-1 - mouse_row) ** 2 + (cat_y - mouse_col) ** 2)==1: best_action = UP_RIGHT
                    elif cat_x!=nrows-1 and ((cat_x+1 - mouse_row) ** 2 + (cat_y - mouse_col) ** 2)==1: best_action = RIGHT_DOWN
                elif best_action==DOWN:
                    #for action in valid_actions:
                    if cat_y!=0 and ((cat_x - mouse_row) ** 2 + (cat_y-1 - mouse_col) ** 2)==1: best_action = LEFT_DOWN
                    elif cat_y!=ncols-1 and ((cat_x - mouse_row) ** 2 + (cat_y+1 - mouse_col) ** 2)==1: best_action = RIGHT_DOWN
        return best_action
        
    def cat_movement(self, action):
        valid_actions = self.valid_actions([0,1,2,3,4,5,6,7],self.cat)        
        action=self.chase(valid_actions)
        nrow, ncol = self.cat
        if action in valid_actions:
            if action == LEFT: ncol -= 1
            elif action == UP: nrow -= 1
            elif action == RIGHT: ncol += 1
            elif action == DOWN: nrow += 1
            elif action == -1: return "Y"
            elif action == LEFT_UP:
                ncol -= 1
                nrow -= 1
            elif action == LEFT_DOWN:
                ncol -= 1
                nrow += 1
            elif action == UP_RIGHT:
                ncol += 1
                nrow -= 1
            elif action == RIGHT_DOWN:
                ncol += 1
                nrow += 1
        self.cat = (nrow, ncol)          #new state
        mouse_row, mouse_col, mode = self.state
        #print("mouse state:", mouse_row, mouse_col, "cat state (watching mouse move):",self.cat)
        if self.cat == (mouse_row, mouse_col): return "Y"
        return "N"

    def act(self, action):
        self.update_state(action)
        caught=self.cat_movement(action)
        reward = self.get_reward(caught)
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        canvas = self.draw_maze()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_maze(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > 0.0: canvas[r,c] = 1.0
        # draw the mouse
        row, col, valid = self.state
        canvas[row, col] = mouse_mark
        row, col = self.cat
        if (row,col) != (nrows-1, ncols-1): canvas[row,col] = cat_mark
        return canvas

    def game_status(self):
        mouse_row, mouse_col, mode = self.state
        nrows, ncols = self.maze.shape
        if mode == "caught": return 'lose'
        if mouse_row == nrows-1 and mouse_col == ncols-1: return 'win'
        return 'not_over'

    def valid_actions(self, actions, cell=None):
        if cell is None: row, col, mode = self.state
        else: row, col = cell
        #actions = [0, 1, 2, 3]
        nrows, ncols = self.maze.shape
        if row == 0:actions.remove(1)
        elif row == nrows-1:actions.remove(3)
        if col == 0: actions.remove(0)
        elif col == ncols-1: actions.remove(2)
        if row>0 and self.maze[row-1,col] == 0.0: actions.remove(1)
        if row<nrows-1 and self.maze[row+1,col] == 0.0: actions.remove(3)
        if col>0 and self.maze[row,col-1] == 0.0: actions.remove(0)
        if col<ncols-1 and self.maze[row,col+1] == 0.0: actions.remove(2)
        return actions

class Learn(object):
    def __init__(self, model, max_memory=300, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def record(self, move):
        # move = [envstate, action, reward, envstate_next, game_over]
        # memory[i] = move
        # envstate == flattened 1d maze cells info, including mouse cell (see method: observe)
        self.memory.append(move)
        if len(self.memory) > self.max_memory: del self.memory[0]

    def predict(self, envstate):
        return self.model.predict(envstate)[0]

    def retrieve(self, data_size=10):
        env_size = self.memory[0][0].shape[1]   # envstate 1d size (1st element of move)        
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i in range(data_size):
            envstate, action, reward, envstate_next, game_over = self.memory[i]
            inputs[i] = envstate        
            targets[i] = self.predict(envstate) #There should be no target values for actions not taken.            
            Q_sa = np.max(self.predict(envstate_next)) # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            if game_over: targets[i, action] = reward
            else: targets[i, action] = reward + self.discount * Q_sa # reward + gamma * max_a' Q(s', a')
        return inputs, targets
     
def train(model, maze, **opt):
    global epsilon
    n_epoch = opt.get('n_epoch', 15000)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    weights_file = opt.get('weights_file', "")
    name = opt.get('name', 'model')
    start_time = datetime.now()    
    if weights_file: # to continue training from a previous model, just supply the h5 file name to weights_file option
        print("loading weights from file: %s" % (weights_file,))
        model.load_weights(weights_file)
    # Construct environment/game from numpy array: maze (see above)
    qmaze = Maze(maze)
    # Initialize learn replay object
    learn = Learn(model, max_memory=max_memory)
    win_history = []   # history of win/lose game
    win_rate = 0.00
    policy_update_window=300
    n_moves_across_games = 0
    policy_iteration = 0
    for epoch in range(n_epoch):
        loss = 0.0
        mouse_cell = random.choice(qmaze.free_cells)
        qmaze.reset(mouse_cell)
        game_over = False
        envstate = qmaze.observe()  # get initial envstate (1d flattened canvas)
        moves = 0
        while not game_over:
            valid_actions = qmaze.valid_actions([0,1,2,3])
            if not valid_actions: break
            prev_envstate = envstate
            if np.random.rand() <= epsilon: action = random.choice(valid_actions)   # Get next action
            else: action = np.argmax(learn.predict(prev_envstate))
            envstate, reward, game_status = qmaze.act(action)       # Apply action, get reward and new envstate
            if game_status == 'win':
                win_history.append(1)
                game_over = True
            elif game_status == 'lose':
                win_history.append(0)
                game_over = True     
            move = [prev_envstate, action, reward, envstate, game_over] # Store move
            learn.record(move)
            moves += 1
            n_moves_across_games+=1
            if n_moves_across_games == policy_iteration*policy_update_window + policy_update_window:
                inputs, targets = learn.retrieve(data_size=data_size)       # Train neural network model
                h = model.fit(inputs, targets, epochs=8, batch_size=16, verbose=0)
                loss = model.evaluate(inputs, targets, verbose=0)
                policy_iteration+=1                
        win_rate = sum(win_history) / (epoch+1)   
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Moves: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch-1, loss, moves, sum(win_history), win_rate, format_time((datetime.now() - start_time).total_seconds())))
        print("win_hist (most recent games)",win_history[-50:])
        if epoch >= 29999: epsilon/=10   # we have almost achieved our optimal policy
        elif epoch >= 19999: epsilon/=10    # less exploration, more exploitation of the policy.
        if win_rate > 0.90:
            print("Reached 90%% win rate at epoch: %d" % (epoch,))
            break
    # Save trained model weights and architecture, this will be used by the visualization code
    h5file = name + ".h5"
    json_file = name + ".json"
    model.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as outfile: json.dump(model.to_json(), outfile)
    print('files: %s, %s' % (h5file, json_file))
    print("n_epoch: %d, max_mem: %d, data: %d, policy iterations: %d, time: %s" % (epoch, max_memory, data_size, policy_iteration, format_time((datetime.now() - start_time).total_seconds())))

# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 121: return "%.1f seconds" % (float(seconds),)
    elif seconds < 3661: return "%.2f minutes" % (seconds / 60.0,)
    else:return "%.2f hours" % (seconds / 3600.0,)
    
def show(qmaze):
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row,col in qmaze.visited:
        canvas[row,col] = 0.6
    mouse_row, mouse_col, _ = qmaze.state
    cat_row, cat_col = qmaze.cat    
    canvas[mouse_row, mouse_col] = 0.3   # mouse cell
    canvas[cat_row, cat_col] = 0.8 #cat cell
    canvas[nrows-1, ncols-1] = 0.9 # cheese cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    return img
    
def play_game(model, qmaze, mouse_cell):
    qmaze.reset(mouse_cell)
    envstate = qmaze.observe()
    while True:
        q = model.predict(envstate)         # get next action
        action = np.argmax(q[0])
        envstate, reward, game_status = qmaze.act(action)         # apply action, get rewards and new state
        if game_status == 'win' or game_status == 'lose' : break
    return game_status
    
maze =  np.array([
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  0.,  1.,  1.,  0.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  0.,  1.,  1.,  0.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]

])

qmaze = Maze(maze)
show(qmaze)

def build_model(maze, lr=0.001):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(LeakyReLU())
    model.add(Dense(maze.size))
    model.add(LeakyReLU())
    model.add(Dense(maze.size))
    model.add(LeakyReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model
    
model = build_model(maze)
train(model, maze, n_epoch=5000, max_memory=8*maze.size, data_size=300)

print(play_game(model, qmaze, (5,3)))
show(qmaze)

model = build_model(maze)
train(model, maze, n_epoch=10000, max_memory=8*maze.size, data_size=300)

print(play_game(model, qmaze, (5,0)))
show(qmaze)

model = build_model(maze)
train(model, maze, n_epoch=40000, max_memory=8*maze.size, data_size=300)

print(play_game(model, qmaze, (0,0)))
show(qmaze)

model = build_model(maze)
train(model, maze, n_epoch=120000, max_memory=8*maze.size, data_size=300)

print(play_game(model, qmaze, (5,0)))
show(qmaze)