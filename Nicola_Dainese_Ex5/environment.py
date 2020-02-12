import numpy as np

class Environment:

    state = []
    goal = []
    boundary = []
    action_map = {
        0: [0, 0],
        1: [0, 1],
        2: [0, -1],
        3: [1, 0],
        4: [-1, 0],
    }
    
    def __init__(self, x, y, initial, goal, R0):
        self.boundary = np.asarray([x, y])
        self.state = np.asarray(initial)
        self.goal = goal
        self.R0 = R0
    # the agent makes an action (0 is stay, 1 is up, 2 is down, 3 is right, 4 is left)
    def move(self, action):
        reward = self.R0 
        movement = self.action_map[action]
        if (action == 0 and (self.state == self.goal).all()):
            reward = 1
        next_state = self.state + np.asarray(movement)
        if(self.check_boundaries(next_state)):
            reward = -1
        else:
            self.state = next_state
        return [self.state, reward]

    # map action index to movement
    def check_boundaries(self, state):
        out = len([num for num in state if num < 0])
        out += len([num for num in (self.boundary - np.asarray(state)) if num <= 0])
        return out > 0

    
class SeasideEnv:
    
    def __init__(self, x, y, initial, goal, R0):
        self.boundary = np.asarray([x, y])
        self.state = np.asarray(initial)
        self.goal = goal
        self.R0 = R0
        self.action_map = {
                            0: [0, 0],
                            1: [0, 1],
                            2: [0, -1],
                            3: [1, 0],
                            4: [-1, 0],
                            }
    # the agent makes an action (0 is stay, 1 is up, 2 is down, 3 is right, 4 is left)
    def move(self, action):
        #print("Current state:", self.state)
        if self.state[0] < 5:
            reward = self.R0
        else:
            reward = self.R0 *5
        movement = self.action_map[action]
        if (action == 0 and (self.state == self.goal).all()):
            reward = 1
        next_state = self.state + np.asarray(movement)
        if(self.check_boundaries(next_state)):
            reward = -1
        else:
            self.state = next_state
            
        #print("Current movement:", movement)
        #print("Reward obtained:", reward)
        return [self.state, reward]

    # map action index to movement
    def check_boundaries(self, state):
        out = len([num for num in state if num < 0])
        out += len([num for num in (self.boundary - np.asarray(state)) if num <= 0])
        return out > 0
    
class BridgeEnv:
    
    def __init__(self, x, y, initial, R0):
        self.boundary = np.asarray([x, y])
        self.state = np.asarray(initial)
        self.initial = np.asarray(initial)
        self.goal = [4,9]
        self.R0 = R0
        self.action_map = {
                            0: [0, 0],
                            1: [0, 1],
                            2: [0, -1],
                            3: [1, 0],
                            4: [-1, 0],
                            }
    # the agent makes an action (0 is stay, 1 is up, 2 is down, 3 is right, 4 is left)
    def move(self, action):
        #print("Current state:", self.state)
        reward = self.R0
        movement = self.action_map[action]
        if (action == 0 and (self.state == self.goal).all()):
            reward = 1
        next_state = self.state + np.asarray(movement)
        # boundary case
        if(self.check_boundaries(next_state)):
            reward = -1
        # cliff case -> re-start from beginning + big negative reward
        elif self.check_void(next_state):
            reward = -10
            self.state = self.initial
        else:
            self.state = next_state
            
        #print("Current movement:", movement)
        #print("Reward obtained:", reward)
        return [self.state, reward]

    # map action index to movement
    def check_boundaries(self, state):
        out = len([num for num in state if num < 0])
        out += len([num for num in (self.boundary - np.asarray(state)) if num <= 0])
        return out > 0
    
    def check_void(self, state):
        if state[0] in [i for i in range(1,9)] and state[1] in [i for i in range(2,8)]:
            return True
        else:
            return False