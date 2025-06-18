"""Games or Adversarial Search (Chapter 5)"""

import copy
import random
from collections import namedtuple
import numpy as np
import time

# Total 60 pt for this script
# namedtuple used to generate game state:
GameState = namedtuple('GameState', 'to_move, move, utility, board, moves')

def gen_state(move = '(1, 1)', to_move='X', x_positions=[], o_positions=[], h=3, v=3):
    """
        move = the move that has lead to this state,
        to_move=Whose turn is to move
        x_position=positions on board occupied by X player,
        o_position=positions on board occupied by O player,
        (optionally) number of rows, columns and how many consecutive X's or O's required to win,
    """
    moves = set([(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]) - set(x_positions) - set(o_positions)
    moves = list(moves)
    board = {}
    for pos in x_positions:
        board[pos] = 'X'
    for pos in o_positions:
        board[pos] = 'O'
    return GameState(to_move=to_move, move=move, utility=0, board=board, moves=moves)

# ______________________________________________________________________________
# MinMax Search
def minmax(game, state):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. If there are multiple
    choices for best move, then randomly choose one.
    Students to do: Add support to this function for state caching, so
    to avoid re-evaluating the same state more than once and save on memory.
    Important: Caching should be added to all search algorithms where ever it
    makes sense. without cache support you won't be able to reach the 
    time limit for various tests."""

    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minmax:
    # to be implemented by students
    print("your code goes here 5pt")
    pass


def minmax_cutoff(game, state):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the cutoff depth. At that level use evaluation func.
    """
    print("your code goes here 5pt")
    pass



# ______________________________________________________________________________
def alpha_beta(game, state):
    """Search game to determine best action; use alpha-beta pruning.
     this version searches all the way to the leaves."""
    player = game.to_move(state)

    alpha = -np.inf
    beta = np.inf
    best_action = None
    print("alpha_beta: Your code goes here 10pt.")

    return best_action


def alpha_beta_cutoff(game, state):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""
    player = game.to_move(state)
    print("Your code here")

    # Body of alpha_beta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    alpha = -np.inf
    beta = np.inf
    best_action = None
    print("Your code here 10pt.")

    return best_action



def random_player(game, state):
    """A random player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None


def alpha_beta_player(game, state):
    """uses alphaBeta prunning with minmax, or with cutoff version, for AI player"""
    
    """Use a method to speed up at the start to avoid search down a long tree with not much outcome.
    Hint: for speedup use random_player for start of the game when you see search time is too long"""
    print("Your code goes here 3pt.")
    
    if( game.timer < 0):
        game.d = -1
        return alpha_beta(game, state)

    start = time.perf_counter()
    end = start + game.timer
    """use the above timer to implement iterative deepening using alpha_beta_cutoff() version"""
    move = None
    
    print("Your code goes here 2pt.")

    print("iterative deepening to depth: ", game.d)
    return move


def minmax_player(game, state):
    """uses minmax or minmax with cutoff depth, for AI player"""
    """Use a method to speed up at the start to avoid search down a deep tree with not much outcome."""
    print("Your code goes here.2pt")

    if( game.timer < 0):
        game.d = -1 #no depth cutoff limit of no time limit is set
        return minmax(game, state)

    start = time.perf_counter()
    end = start + game.timer
    """use the above timer to implement iterative deepening loop bellow, using minmax_cutoff(), controlled by the timer"""
    move = None
    print("Your code goes here 3pt.")

    print("minmax_player: iterative deepening to depth: ", game.d)
    return move



# ______________________________________________________________________________
# base class for Games

class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))

class TicTacToe(Game):
    """Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to_move, a cached utility, a list of moves in
    the form of a list of (x, y) positions, and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'.
    depth = -1 means max search tree depth to be used."""

    def __init__(self, size=3, k=3, t=-1):
        self.size = size
        if k <= 0:
            self.k = size
        else:
            self.k = k
        self.d = -1 # d is cutoff depth. Default is -1 meaning no depth limit. It is controlled usually by timer
        self.maxDepth = size * size # max depth possible is width X height of the board
        self.timer = t #timer  in seconds for opponent's search time limit. -1 means unlimited
        moves = [(x, y) for x in range(1, size + 1)
                 for y in range(1, size + 1)]
        self.initial = GameState(to_move='X', move=None, utility=0, board={}, moves=moves)

    def reset(self):
        moves = [(x, y) for x in range(1, self.size + 1)
                 for y in range(1, self.size + 1)]
        self.initial = GameState(to_move='X', move=None, utility=0, board={}, moves=moves)

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    @staticmethod
    def switchPlayer(player):
        assert(player == 'X' or player == 'O')
        return 'O' if player == 'X' else 'X'

    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        try:
            moves = list(state.moves)
            moves.remove(move)
        except (ValueError, IndexError, TypeError) as e:
            print("exception: ", e)

        return GameState(to_move=self.switchPlayer(state.to_move), move=move,
                         utility=self.compute_utility(board, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the state value to player; state.k for win, -state.k for loss, 0 otherwise. k is dimension of the board (3, 4, ...)"""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or lost or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for x in range(0, self.size):
            for y in range(1, self.size + 1):
                print(board.get((self.size - x, y), '.'), end=' ')
            print()

    def compute_utility(self, board, player):
        """If player wins with this move, return k if player is 
        'X' and -k if 'O' else return 0. 
        For students to do: Use k_in_row for checking if there are k cells
        in line in a direction. For example k_in_row(board, 'X', (0, 1), self.k, self.k) returns 
        a pair (true/false, count) indicating if there are k 'X' in column direction."""
        print("To be done by students 5 pt")
        return 0
        
    # evaluation function, version 1
    def eval1(self, state):
        """design and implement evaluation function for state.
        Here's an idea: 
        	 Use the number of k-1 or less matches for X and O For example you can fill up the following
        	 function possibleKComplete() which will use k_in_row(). This function
        	 would return number of say k-1 matches for a specific player, 'X', or 'O'.
        	 Anyhow, this is an idea. We have been able to use such method to get almost
        	 perfect playing experience.
        Again, remember we want this evaluation function to be represent
        how to good the state is for the player to win the game from here,
        and also it needs to be fast to compute.
        """

        print("Your code goes here 15pt.")

        return 0


    #@staticmethod
    def k_in_row(self, board, player, dir, k, size):
        """
        -This function search for k consecutive player ('X' or 'O') in a line and returns the pair (flag, count) in which flag says if it
        found at least one line and count is the number of found lines. For example (false, 0) means no line found.
        -size is the dimension of the board.
        -dir can be one of (1, 0) , (0, 1), (1, 1), and (1, -1) corresponding ot horizontal, vertical, diagonal
        and off-diagonal match of k squares."""
        (delta_y, delta_x) = dir
        n = 0  # n is number of cells in direction dir occupied by player
        opponent = 'X' if player == 'O' else 'O'
        if(delta_y == delta_x): #diagonal match checking:
            for i in range(1, size+1):
                if (board.get((i, i)) == player):
                    n += 1
                if (board.get((i, i)) == opponent):  # means this match is occup````````1ied by an opponent piece.
                    return False, 0
            return n==k, n==k
        n = 0
        if (delta_y == -delta_x):  # off-diagonal match checking:
            for i in range(1, size+1):
                if (board.get((i, size+1-i)) == player):
                    n += 1
                if (board.get((i, size+1-i)) == opponent):  # means this match is occupied by an opponent piece.
                    return False, 0
            return n==k, n==k

        # below deals with horizontal and vertical matches
        count = 0  # number of rows with k Player on them so far.
        if delta_x == 1:  # means looking in row matches
            for i in range(1, size+1):
                n = 0 # n is the number of slots occupied by Player in row i
                for j in range(1, size+1):  # looking in row i
                    if board.get((i, j)) == player:
                        n += 1
                    if board.get((i, j)) == opponent:  # means this slot is occupied by the opponent.
                        break;
                if n >= k: count += 1

        if delta_y == 1:  # means looking in col matches
            for i in range(1, size+1):
                n = 0  # n is number of Players in col i
                for j in range(1, size+1):  # looking in row i
                    if board.get((j, i)) == player:
                        n += 1
                    if board.get((j, i)) == opponent:  # means this match is occuppied by an opponent piece.
                        break;
                if n >= k: count += 1


        return count > 0, count

