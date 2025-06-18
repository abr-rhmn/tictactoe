
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
        
		node = self.root
		self.expandNode(node)

		for child in node.children:
			winner = simulateRandomPlay(child)

        winnerNode = self.root.getChildWithMaxScore()
        assert(winnerNode is not None)
        return winnerNode.state.move


    """SELECT stage function. walks down the tree using findBestNodeWithUCT()"""
    def selectNode(self, nd):
        node = nd
        while node.children is not empty:
			node = findBestNodeWithUCT(self, node)

		return

    def findBestNodeWithUCT(self, nd):
        maxUCT = -float('inf')
		chosenChild = nd
		for child in nd.children:
			uct = uctValue(self, child.visitCount, child.winScore)
			if uct > maxUCT:
				maxUCT = uct
				chosenChild = child


        return chosenChild

    def uctValue(self, parentVisit, nodeScore, nodeVisit):
        UCT = (nodeScore/nodeVisit) + self.exploreFactor * sqrt(log(parentVisit) / nodeVisit)
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
        node = nd

		while node != None:
			node.visitCount += 1

			if winningPlayer == node.state.to_move
				node.winScore += 1

			elif winningPlayer == 'O':
				node.winScore -= 1

			node = node.par

		return


