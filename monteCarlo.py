
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
        print("MCTS: your code goes here. 10pt.")

        winnerNode = self.root.getChildWithMaxScore()
        assert(winnerNode is not None)
        return winnerNode.state.move


    """SELECT stage function. walks down the tree using findBestNodeWithUCT()"""
    def selectNode(self, nd):
        node = nd
        print("Your code goes here 5pt.")
        return node

    def findBestNodeWithUCT(self, nd):
        """finds the child node with the highest UCT. 
        Parse nd's children and use uctValue() to collect ucts 
        for the children.....
        Make sure to handle the case when uct value of 2 or more children
        nodes are the same."""
        childUCT = []
        print("Your code goes here 5pt.")
        return None


    def uctValue(self, parentVisit, nodeScore, nodeVisit):
        """compute Upper Confidence Value for a node"""
        print("Your code goes here 3pt.")
        pass

   
   
    def expandNode(self, nd):
        """generate the child nodes for node nd. For convenience, generate
        all the child nodes and attach them to nd."""
        stat = nd.state
        print("Your code goes here 5pt.")
        pass

    """SIMULATE stage function"""
    def simulateRandomPlay(self, nd):
        """
        This function retuns the result of simulating off of node nd a 
        termination node, and returns the winner 'X' or 'O' or 0 if tie.
        Note: pay attention nd may be itself a termination node. Use compute_utility 
        to check for it.
        """
        print("Your code goes here 7pt.")

        pass


    def backPropagation(self, nd, winningPlayer):
        """propagate upword to update score and visit count from
        the current leaf node to the root node."""
        
        print("Your code goes here 5pt.")


