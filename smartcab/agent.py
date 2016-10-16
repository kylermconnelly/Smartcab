import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

actions = [None, 'forward', 'left', 'right']

class LearningAgent(Agent):
	
	def __init__(self, env):
		super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
		self.color = 'red'  # override color
		self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
		
        # TODO: Initialize any additional variables here
		self.total_reward = 0
		self.previous_state = None
		self.previous_action = None
		self.previoud_reward = None
		
		self.epsilon = 1.0	# random/explore
		self.epsilon_step_size = 0.02	# we will reduce epsilon by this each round
		
		self.success_count = 0
		self.total_reward = 0
		
		self.alpha = 0		# learning rate
		self.gamma = 0		# future vs immediate reward
		
		self.Q_table = {}
		default_reward = 0.0
				
		# initialize state -> action table to all 0
		trafficlight = ['red','green']
		oncoming = [None, 'left', 'right', 'forward']
		right = [None, 'left', 'right', 'forward']
		left = [None, 'left', 'right', 'forward']
		waypoints = ['forward', 'left', 'right']
		for t in trafficlight:
			for o in oncoming:
				for r in right:
					for l in left:
						for w in waypoints:
								new_key = (t,o,r,l,w)
								# initialize rewards for each action for each state
								self.Q_table[new_key] = {None:default_reward, 'forward':default_reward, 'right':default_reward, 'left':default_reward}
		
	def reset(self, destination=None):
		self.planner.route_to(destination)
		# TODO: Prepare for a new trip; reset any variables here, if required
		self.previous_state = None
		self.previous_action = None
		self.previoud_reward = None
		# reduce epsilon a little
		self.epsilon = max(self.epsilon - self.epsilon_step_size, 0)
	
	# this method takes the current state and chooses the next state with the highest Q value
	# or, sometimes chooses ramdomly based on epsilon
	def get_policy(self, state):
		# get action with maximum value
		q_max = max(self.Q_table[tuple(state)].items(), key=lambda item:item[1])
		# now get all actions that have the same value as the max
		max_actions = []
		for actions in self.Q_table[tuple(state)].items():
			if actions[1] == q_max[1]:
				max_actions.append(actions[0])
		
		# now choose randomly sometimes so we explore
		if random.random() <= self.epsilon:
			return random.choice(self.Q_table[tuple(state)].items())[0]
		else:
			return random.choice(max_actions)
				
	# Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
	# s = prev state
	# a = prev action
	# s'= curr state
	def update_policy(self, prev_state, new_state, prev_action, prev_reward):		
		# get previous q for this state/action pair
		old_q = self.Q_table[tuple(prev_state)][prev_action]
		
		# get new q value
		new_q = old_q + self.alpha * (prev_reward + self.gamma * max(self.Q_table[tuple(new_state)].values()) - old_q)
		# now adjust q val for previous state with this action
		self.Q_table[tuple(prev_state)][prev_action] = new_q

	def update(self, t):
		# Gather inputs
		self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
		inputs = self.env.sense(self)
		deadline = self.env.get_deadline(self)
		
		# TODO: Update state
		self.state = inputs.values() + [self.next_waypoint]
        
		# TODO: Select action according to your policy
		action = self.get_policy(self.state)

		# Execute action and get reward
		reward = self.env.act(self, action)
		
		if self.epsilon <= 0:
			self.total_reward += reward
		
		# reward for reaching goal is 12 i believe, so this should catch those instances
		if (reward > 10) and (self.epsilon <= 0):
			self.success_count += 1

		# TODO: Learn policy based on state, action, reward
		if self.previous_state != None:
			self.update_policy(self.previous_state, self.state, self.previous_action, self.previoud_reward)
		
		# save this for next time
		self.previous_state = self.state
		self.previous_action = action
		self.previoud_reward = reward

		#print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
	
	def get_success_count(self):
		return self.success_count
	
	def get_reward_total(self):
		return self.total_reward
	
	def set_params(self, alpha, gamma):
		self.gamma = gamma
		self.alpha = alpha

def run(alpha, gamma):
	"""Run the agent for a finite number of trials."""
    # Set up environment and agent
	e = Environment()  # create environment (also adds some dummy traffic)
	a = e.create_agent(LearningAgent)  # create agent
	e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
	
    # Now simulate it
	sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False
	
	a.set_params(alpha, gamma)
	
	sim.run(n_trials=100)  # run for a specified number of trials
	#return a.get_reward_total()
	return a.get_success_count()
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
	
#alpha_array = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#gamma_array = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

alpha_array = [0.7]
gamma_array = [0.3]

if __name__ == '__main__':
	alpha_gamma_matrix = [[0 for i in range(len(alpha_array))] for j in range(len(gamma_array))]
	for alphas in range(len(alpha_array)):
		for gammas in range(len(gamma_array)):
			alpha_gamma_matrix[alphas][gammas] = run(alpha_array[alphas], gamma_array[gammas])
			print "alpha: %f gamma: %f number of successes: %s"%(alpha_array[alphas], gamma_array[gammas], alpha_gamma_matrix[alphas][gammas])
	
	print '     gamma %s'%(' '.join('%07s' % i for i in gamma_array))
	print 'alpha'
	for row, j in zip(alpha_gamma_matrix, range(len(alpha_array))):
		print '     %.2f [%s]' % (alpha_array[j],' '.join('%07s' % i for i in row))
