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

	player = game.to_move(state)
	memo = {}

	def max_value(cur_state):
		state_key = (frozenset(cur_state.board.items()), cur_state.to_move)

		if state_key in memo:
			return memo[state_key]
		
		if game.terminal_test(cur_state):
			value = game.utility(cur_state, player)
			memo[state_key] = value
			return value
		
		v = -np.inf
		for a in game.actions(cur_state):
			v = max(v, min_value(game.result(cur_state, a)))
		memo[state_key] = v
		print(f"max_v score is {v}")
		return v

	def min_value(cur_state):
		state_key = (frozenset(cur_state.board.items()), cur_state.to_move)

		# Check if the state is already in the cache
		if state_key in memo:
			return memo[state_key]
		
		if game.terminal_test(cur_state):
			value = game.utility(cur_state, player)
			memo[state_key] = value
			return value
		
		v = np.inf
		for a in game.actions(cur_state):
			v = min(v, max_value(game.result(cur_state, a)))
		memo[state_key] = v # Store result in cache
		print(f"min_v score is {v}")
		return v

	best_score = -float('inf')
	best_actions = []
	for action in game.actions(state):
		new_state = game.result(state, action)

		score = max_value(new_state)

		if score > best_score:
			best_score = score
			best_actions = [action]

		elif score == best_score:
			best_actions.append(action)

		if best_actions:
			print(f"Chosen action: {random.choice(best_actions)}")
			return random.choice(best_actions)
		else:
			return None

def minmax_cutoff(game, state):
	player = game.to_move(state)
	memo = {}

	def max_value(cur_state, depth):
		state_key_d = (frozenset(cur_state.board.items()), cur_state.to_move, depth)
		if (state_key_d) in memo:
			return memo[state_key_d]

		if game.terminal_test(cur_state):
			value = game.utility(cur_state, player)
			memo[state_key_d] = value
			return value

		if game.d != -1 and depth >= game.d:
			value = game.eval1(cur_state)
			memo[state_key_d] = value
			return value

		v = -float('inf')
		for action in game.actions(cur_state):
			v = max(v, min_value(game.result(cur_state, action), depth + 1))

		memo[state_key_d] = v
		return v

	def min_value(cur_state, depth):
		state_key_d = (frozenset(cur_state.board.items()), cur_state.to_move, depth)
		if (state_key_d) in memo:
			return memo[state_key_d]

		if game.terminal_test(cur_state):
			value = game.utility(cur_state, player)
			memo[state_key_d] = value
			return value

		if game.d != -1 and depth >= game.d:
			value = -game.eval1(cur_state)
			memo[state_key_d] = value
			return value

		v = float('inf')
		for action in game.actions(cur_state):
			v = min(v, max_value(game.result(cur_state, action), depth + 1))
		memo[state_key_d] = v
		return v

	best_score = -float('inf')
	best_actions = []
	initial_depth = 0

	for action in game.actions(state):
		new_state = game.result(state, action)
		score = max_value(new_state, initial_depth + 1)
		print(f"score is {score}")

		if score > best_score:
			best_score = score
			best_actions = [action]
		elif score == best_score:
			best_actions.append(action)

	if best_actions:
		return random.choice(best_actions)
	else:
		return None

# ______________________________________________________________________________
def alpha_beta(game, state):
	"""Search game to determine best action; use alpha-beta pruning.
	 this version searches all the way to the leaves."""
	player = game.to_move(state)

	alpha = -np.inf
	beta = np.inf
	best_action = None
	memo = {}
		
	def max_value(cur_state, alpha, beta):
		state_key = (frozenset(cur_state.board.items()), cur_state.to_move)
		if state_key in memo:
			return memo[state_key]

		if game.terminal_test(cur_state):
			value = game.utility(cur_state, player)
			memo[state_key] = value
			return value

		v = -float('inf')
		for action in game.actions(cur_state):
			v = max(v, min_value(game.result(cur_state, action), alpha, beta))
			if v >= beta:
				memo[state_key] = v
				return v
			alpha = max(alpha, v)
		memo[state_key] = v
		return v

	def min_value(cur_state, alpha, beta):
		state_key = (frozenset(cur_state.board.items()), cur_state.to_move)
		if state_key in memo:
			return memo[state_key]

		if game.terminal_test(cur_state):
			value = game.utility(cur_state, player)
			memo[state_key] = value
			return value

		v = float('inf')
		for action in game.actions(cur_state):
			v = min(v, max_value(game.result(cur_state, action), alpha, beta))
			if v <= alpha:
				memo[state_key] = v
				return v
			beta = min(beta, v)
		memo[state_key] = v
		return v

	best_score = float('-inf')

	for action in game.actions(state):
		new_state = game.result(state, action)
		score_for_this_action = max_value(new_state, alpha, beta)

		if score_for_this_action > best_score:
			best_score = score_for_this_action
			best_actions = [action]
		elif score_for_this_action == best_score:
			best_actions.append(action)
		
		alpha = max(alpha, best_score)

	if best_actions:
		return random.choice(best_actions)
	else:
		return None

def alpha_beta_cutoff(game, state):
	"""Search game to determine best action; use alpha-beta pruning.
	This version cuts off search and uses an evaluation function."""
	player = game.to_move(state)

	alpha = -np.inf
	beta = np.inf
	best_action = None

	current_best_score = -float('inf')
	candidate_actions = []

	memo = {}
		
	def max_value(cur_state, depth, current_alpha, current_beta):
		state_key_d = (frozenset(cur_state.board.items()), cur_state.to_move, depth)
		if state_key_d in memo:
			return memo[state_key_d]

		if game.terminal_test(cur_state):
			value = game.utility(cur_state, player)
			memo[state_key_d] = value
			return value

		if game.d != -1 and depth >= game.d:
			value = game.eval1(cur_state)
			memo[state_key_d] = value
			return value

		v = -float('inf')
		for action in game.actions(cur_state):
			v = max(v, min_value(game.result(cur_state, action), depth + 1, current_alpha, current_beta))
			if v >= current_beta:
				memo[state_key_d] = v
				return v
			current_alpha = max(current_alpha, v)
		memo[state_key_d] = v
		return v

	def min_value(cur_state, depth, current_alpha, current_beta):
		state_key_d = (frozenset(cur_state.board.items()), cur_state.to_move, depth)
		if state_key_d in memo:
			return memo[state_key_d]

		if game.terminal_test(cur_state):
			value = game.utility(cur_state, player)
			memo[state_key_d] = value
			return value

		if game.d != -1 and depth >= game.d:
			value = -game.eval1(cur_state)
			memo[state_key_d] = value
			return value

		v = float('inf')
		for action in game.actions(cur_state):
			v = min(v, max_value(game.result(cur_state, action), depth + 1, current_alpha, current_beta))
			if v <= current_alpha:
				memo[state_key_d] = v
				return v
			current_beta = min(current_beta, v)
		memo[state_key_d] = v
		return v

	# Body of alpha_beta_cutoff_search starts here:
	# The default test cuts off at depth d or at a terminal state
	initial_depth = 0

	for action in game.actions(state):
		new_state = game.result(state, action)
		score_for_this_action = max_value(new_state, initial_depth + 1, alpha, beta)

		if score_for_this_action > current_best_score:
			current_best_score = score_for_this_action
			candidate_actions = [action]
		elif score_for_this_action == current_best_score:
			candidate_actions.append(action)
		
		alpha = max(alpha, current_best_score)

	if candidate_actions:
		best_action = random.choice(candidate_actions)
	else:
		best_action = None

	return best_action

def random_player(game, state):
	"""A random player that chooses a legal move at random."""
	return random.choice(game.actions(state)) if game.actions(state) else None

def alpha_beta_player(game, state):
	"""uses alphaBeta prunning with minmax, or with cutoff version, for AI player"""
		
	"""Use a method to speed up at the start to avoid search down a long tree with not much outcome.
	Hint: for speedup use random_player for start of the game when you see search time is too long"""
	if len(state.board) < 2:
		return random_player(game, state)
		
	if( game.timer < 0):
		game.d = -1
		return alpha_beta(game, state)

	start = time.perf_counter()
	end = start + game.timer
	"""use the above timer to implement iterative deepening using alpha_beta_cutoff() version"""
	move = None
	cur_depth = 1
		
	while True:
		if time.perf_counter() >= end:
			break

		game.d = cur_depth

		cur_move = alpha_beta_cutoff(game, state)

		if cur_move is not None:
			move = cur_move

		cur_depth += 1

		if cur_depth > game.maxDepth:
			break

		print("iterative deepening to depth: ", game.d)
		
	return move


def minmax_player(game, state):
	"""uses minmax or minmax with cutoff depth, for AI player"""
	"""Use a method to speed up at the start to avoid search down a deep tree with not much outcome."""
	if len(state.board) < 2:
		return random_player(game, state)

	if(game.timer < 0):
		game.d = -1 #no depth cutoff limit of no time limit is set
		return minmax(game, state)

	start = time.perf_counter()
	end = start + game.timer
	"""use the above timer to implement iterative deepening loop bellow, using minmax_cutoff(), controlled by the timer"""
	move = None
	current_depth = 1

	while True:
		if time.perf_counter() >= end:
			break

		game.d = current_depth

		move_at_current_depth = minmax_cutoff(game, state)

		if move_at_current_depth is not None:
			move = move_at_current_depth
		
		current_depth += 1

		if current_depth > game.maxDepth:
			break

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
		
		if player == 'X':
			if (self.k_in_row(board, 'X', (0, 1), self.k, self.size)[0] or
				self.k_in_row(board, 'X', (1, 0), self.k, self.size)[0] or
				self.k_in_row(board, 'X', (1, 1), self.k, self.size)[0] or
				self.k_in_row(board, 'X', (1, -1), self.k, self.size)[0]):
					return self.k

		elif player == 'O':
			if (self.k_in_row(board, 'O', (0, 1), self.k, self.size)[0] or
				self.k_in_row(board, 'O', (1, 0), self.k, self.size)[0] or
				self.k_in_row(board, 'O', (1, 1), self.k, self.size)[0] or
				self.k_in_row(board, 'O', (1, -1), self.k, self.size)[0]):
					return -self.k

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

		board = state.board
		score = 0

		def count_pieces(coords, p_symbol):
			p_count = 0
			o_count = 0
			op_symbol = 'O' if p_symbol == 'X' else 'X'
			
			for r, c in coords:
				if board.get((r, c)) == p_symbol:
					p_count += 1
				elif board.get((r, c)) == op_symbol:
					o_count += 1
			return p_count, o_count

			POINTS_K_MINUS_1 = 100
			POINTS_K_MINUS_2 = 10
			POINTS_K_MINUS_3 = 1

			directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

			for row in range(1, self.size + 1):
				for column in range(1, self.size + 1):
					for dc, dc in directions:
						line_coords = []
						valid_line = True
						for i in range(self.k):
							r, c = row + i * dr, c_start + i * dc
							if not (1 <= r <= self.size and 1 <= c <= self.size):
								valid_line = False
								break
							line_coords.append((r, c))

						if not valid_line:
							continue

						x_count, o_count = count_pieces(line_coords, 'X')
						if o_block_count == 0:
							if x_count == self.k - 1: score += POINTS_K_MINUS_1
							elif x_count == self.k - 2: score += POINTS_K_MINUS_2
							elif x_count == self.k - 3: score += POINTS_K_MINUS_3

						o_count, x_block_count = count_pieces_in_line(line_coords, 'O')
						if x_block_count == 0:
							if o_count == self.k - 1: score -= POINTS_K_MINUS_1
							elif o_count == self.k - 2: score -= POINTS_K_MINUS_2
							elif o_count == self.k - 3: score -= POINTS_K_MINUS_3
		return score


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

# if __name__ == "__main__":
# 	print("Running tests for games.py...")

# 	# Create a TicTacToe game instance for testing
# 	# Assuming default 3x3 for simplicity
# 	test_game = TicTacToe(size=3, k=3)

# 	# --- Test Cases for compute_utility ---
# 	print("\nTesting compute_utility...")

# 	# Test 1: X wins horizontally (top row)
# 	# X . .
# 	# X . .
# 	# X . .
# 	x_win_h_board_1 = {(1, 1): 'X', (1, 2): 'X', (1, 3): 'X'}
# 	assert test_game.compute_utility(x_win_h_board_1, 'X') == test_game.k, "Test 1 Failed: X should win horizontally (row 1)"
# 	print("Test 1 Passed: X horizontal win (row 1)")

# 	# Test 2: O wins vertically (middle column)
# 	# . O .
# 	# . O .
# 	# . O .
# 	o_win_v_board_1 = {(1, 2): 'O', (2, 2): 'O', (3, 2): 'O'}
# 	assert test_game.compute_utility(o_win_v_board_1, 'O') == -test_game.k, "Test 2 Failed: O should win vertically (col 2)"
# 	print("Test 2 Passed: O vertical win (col 2)")

# 	# Test 3: X wins diagonally (top-left to bottom-right)
# 	# X . .
# 	# . X .
# 	# . . X
# 	x_win_diag_board = {(1, 1): 'X', (2, 2): 'X', (3, 3): 'X'}
# 	assert test_game.compute_utility(x_win_diag_board, 'X') == test_game.k, "Test 3 Failed: X should win diagonally"
# 	print("Test 3 Passed: X diagonal win")

# 	# Test 4: O wins anti- diagonally (top-right to bottom-left)
# 	# . . O
# 	# . O .
# 	# O . .
# 	o_win_anti_diag_board = {(1, 3): 'O', (2, 2): 'O', (3, 1): 'O'}
# 	assert test_game.compute_utility(o_win_anti_diag_board, 'O') == -test_game.k, "Test 4 Failed: O should win anti-diagonally"
# 	print("Test 4 Passed: O anti-diagonal win")

# 	# Test 5: A full board draw (no winner)
# 	# X O X
# 	# O O X
# 	# X X O
# 	draw_board = {
# 		(1, 1): 'X', (1, 2): 'O', (1, 3): 'X',
# 		(2, 1): 'O', (2, 2): 'O', (2, 3): 'X',
# 		(3, 1): 'X', (3, 2): 'X', (3, 3): 'O'
# 	}
# 	assert test_game.compute_utility(draw_board, 'X') == 0, "Test 5 Failed: Should be a draw"
# 	assert test_game.compute_utility(draw_board, 'O') == 0, "Test 5 Failed: Should be a draw for O's perspective"
# 	print("Test 5 Passed: Full board draw")

# 	# Test 6: An ongoing game (no winner yet)
# 	# X . O
# 	# . O .
# 	# X . .
# 	ongoing_board = {(1, 1): 'X', (1, 3): 'O', (2, 2): 'O', (3, 1): 'X'}
# 	assert test_game.compute_utility(ongoing_board, 'X') == 0, "Test 6 Failed: Ongoing game should be 0"
# 	print("Test 6 Passed: Ongoing game")


# 	# --- Test Cases for k_in_row (indirectly through compute_utility if you didn't expose it,
# 	#     but if k_in_row is meant to be tested directly, you can) ---
# 	print("\nTesting k_in_row (direct tests, if applicable)...")

# 	# Example: Direct test of k_in_row for a horizontal win
# 	# . . .
# 	# X X X
# 	# . . .
# 	k_in_row_test_board = {(2, 1): 'X', (2, 2): 'X', (2, 3): 'X'}
# 	# Test k_in_row for X horizontally (dir=(0,1))
# 	flag, count = test_game.k_in_row(k_in_row_test_board, 'X', (0, 1), 3, 3)
# 	assert flag is True and count > 0, "Test 7 Failed: k_in_row should detect X horizontal win"
# 	print("Test 7 Passed: k_in_row detects X horizontal win")

# 	# Example: k_in_row where line is blocked by opponent
# 	# X O X
# 	# . . .
# 	# . . .
# 	blocked_line_board = {(1,1):'X', (1,2):'O', (1,3):'X'}
# 	flag, count = test_game.k_in_row(blocked_line_board, 'X', (0,1), 3, 3)
# 	assert flag is False and count == 0, "Test 8 Failed: k_in_row should show blocked line as no win"
# 	print("Test 8 Passed: k_in_row detects blocked line")

# 	print("\nAll tests completed.")