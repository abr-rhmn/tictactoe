
import copy
import random
import time
import sys
import math
from collections import namedtuple

GameState = namedtuple('GameState', 'to_move, move, utility, board, moves')

def random_player(game, state):
	"""A random player that chooses a legal move at random."""
	return random.choice(game.actions(state)) if game.actions(state) else None

# MonteCarlo Tree Search support
# Total 40 pt for monteCarlo.py
class MCTS:
	class Node:
		def __init__(self, state, par=None):
			self.state = copy.deepcopy(state)

			self.parent = par
			self.children = []
			self.visitCount = 0
			self.winScore = 0

		def getChildWithMaxScore(self):
			
			if not self.children:
				return None
      
			maxScoreChild = max(self.children, key=lambda x: x.visitCount)
			return maxScoreChild

	def __init__(self, game, state):
		self.root = self.Node(state)
		self.state = state
		self.game = game
		self.exploreFactor = math.sqrt(2)

		
	def monteCarloPlayer(self, timelimit = 4):
		"""Entry point for Monte Carlo tree search"""
		start = time.perf_counter()
		end = start + timelimit
		"""
		Use time.perf_counter() to apply iterative deepening strategy.
		 At each iteration we perform 4 stages of MCTS: 
		 SELECT, 
		 EXPEND, 
		 SIMULATE, 
		 and BACKUP. 
		 
		 Once time is up use getChildWithMaxScore() to pick the node to move to
		"""
		
		while time.perf_counter() < end:
			node_to_explore = self.selectNode(self.root)

			if not self.game.terminal_test(node_to_explore.state) and not node_to_explore.children:
				self.expandNode(node_to_explore)
				if node_to_explore.children:
					node_for_simulation = random.choice(node_to_explore.children)
				else:
					node_for_simulation = node_to_explore
			else:
				node_for_simulation = node_to_explore

			winning_player = self.simulateRandomPlay(node_for_simulation)

			self.backPropagation(node_for_simulation, winning_player)

		winnerNode = self.root.getChildWithMaxScore()
		assert(winnerNode is not None)
		return winnerNode.state.move


	"""SELECT stage function. walks down the tree using findBestNodeWithUCT()"""
	def selectNode(self, nd):
		node = nd

		while not self.game.terminal_test(node.state) and (node.children and all(child.visitCount > 0 for child in node.children)):
			node = self.findBestNodeWithUCT(node)
			if node is None:
				break

		return node

	def findBestNodeWithUCT(self, nd):
		unvisited_children = [child for child in nd.children if child.visitCount == 0]
		if unvisited_children:
			return random.choice(unvisited_children)

		
		maxUCT = -float('inf')
		chosenChild = None

		if not nd.children:
			return None

		for child in nd.children:
			uct = self.uctValue(nd.visitCount, child.winScore, child.visitCount)
			if uct > maxUCT:
				maxUCT = uct
				chosenChild = child

		return chosenChild

	def uctValue(self, parentVisit, nodeScore, nodeVisit):
		if nodeVisit == 0:
			return float('inf')
		
		UCT = (nodeScore/nodeVisit) + self.exploreFactor * math.sqrt(math.log(parentVisit) / nodeVisit)
		return UCT
   
	def expandNode(self, nd):
		stat = nd.state
		if self.game.terminal_test(stat):
			return

		possible_moves = self.game.actions(stat)

		for move in possible_moves:
			new_state = self.game.result(stat, move)
			new_node = self.Node(new_state, par=nd)
			nd.children.append(new_node)

	"""SIMULATE stage function"""
	def simulateRandomPlay(self, nd):
		current_state = nd.state
		
		while not self.game.terminal_test(current_state):
			possible_moves = self.game.actions(current_state)

			if not possible_moves:
				break

			random_move = random.choice(possible_moves)
			current_state = self.game.result(current_state, random_move)
		
		utility_for_X = self.game.utility(current_state, 'X')

		if utility_for_X == self.game.k:
			return 'X'
		elif utility_for_X == -self.game.k:
			return 'O'
		else:
			return 0 


	def backPropagation(self, nd, winningPlayer):
		print(f"DEBUG: backPropagation called with nd: {nd}") # Add this line
		node = nd
		node = node.parent

		while node != None:
			node.visitCount += 1

			if winningPlayer == node.state.to_move:
				node.winScore += 1

			elif winningPlayer == 'O':
				node.winScore -= 1

			elif winningPlayer == 0:
				node.winScore += 0.5

			node = node.par

		return


