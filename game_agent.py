"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import itertools

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

# Helper Functions for Evaluators
def is_near_walls(move, walls):
    """
    Checks if a move is on near the edges of the board

    Parameters
    ----------
    move : (int, int)
        The input move on the board

    walls : list(list(tuples))
        A nested list of tuples for each edge of the board

    Returns
    -------
    bool
        Returns True if a move lies along the edges else False
    """
    for wall in walls:
        if move in wall:
            return True
    return False

def is_in_corners(move, corners):
    """
        Checks if a move is in the corners of the board

        Parameters
        ----------
        move : (int, int)
            The input move on the board

        corners : list(tuples)
            A list of tuples for each corner of the board

        Returns
        -------
        bool
            Returns True if a move lies in a corner of the board else False
    """
    return move in corners

def percent_occupied(game):
    """
            Checks if a move is in the corners of the board

            Parameters
            ----------
            game : `isolation.Board`
                The game board

            Returns
            -------
            int
                The percentage of occupied space in the board
        """
    blank_spaces = game.get_blank_spaces()
    return int((len(blank_spaces)/(game.width * game.height)) * 100)

# Evaluator Functions
def evaluator_aggresively_chase_opponent(game, player):
    """The "Aggresively Chase Opponent" evaluation function outputs a
        score equal to the difference in the number of moves available to the player
        and twice the numbers of moves available to the opponent

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        player : hashable
            One of the objects registered by the game object as a valid player.
            (i.e., `player` should be either game.__player_1__ or
            game.__player_2__).

        Returns
        ----------
        float
            The heuristic value of the current game state
        """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(abs(own_moves - 2*opp_moves))

def evaluator_check_near_walls(game, player):
    """The "Check Near Walls" evaluation function calculates a cumulative score based on the moves
    and their positions.
    A cumulative score is calculated for both the players.
    A positive score is added to the cumulative score for the player if the board is less than 50% occupied and
    the moves lie on the walls, in case the board is between 50% and 85% occupied a higher score is subtracted from the sum.
    If the occupancy is greater than 85% even more higher score is subtracted
    The process is negated in case of the opponent.
    The difference between both the player cumulative scores is added to the number difference of non-corners move for both players
    and the value returned

            Parameters
            ----------
            game : `isolation.Board`
                An instance of `isolation.Board` encoding the current state of the
                game (e.g., player locations and blocked cells).

            player : hashable
                One of the objects registered by the game object as a valid player.
                (i.e., `player` should be either game.__player_1__ or
                game.__player_2__).

            Returns
            ----------
            float
                The heuristic value of the current game state
            """
    walls = [
        [(0, i) for i in range(game.width)],
        [(i, 0) for i in range(game.height)],
        [(game.width - 1, i) for i in range(game.width)],
        [(i, game.height - 1) for i in range(game.height)]
    ]
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    own_cum_score = 0
    opp_cum_score = 0

    own_moves_left = 0
    opp_moves_left = 0

    own_moves_left = 0
    opp_moves_left = 0
    for move in own_moves:
        if is_near_walls(move, walls) and percent_occupied(game) < 50:
            own_cum_score += 10
        elif is_near_walls(move, walls) and percent_occupied(game) > 50 and percent_occupied(game) < 85:
            own_cum_score -= 20
        elif is_near_walls(move, walls) and percent_occupied(game) > 85:
            own_cum_score -= 30
        else:
            own_moves_left += 5

    for move in opp_moves:
        if is_near_walls(move, walls) and percent_occupied(game) < 50:
            opp_cum_score += 10
        elif is_near_walls(move, walls) and percent_occupied(game) > 50 and percent_occupied(game) < 85:
            opp_cum_score -= 20
        elif is_near_walls(move, walls) and percent_occupied(game) > 85:
            opp_cum_score -= 30
        else:
            opp_moves_left += 5

    return float(own_cum_score - opp_cum_score) + float(own_moves_left - opp_moves_left)

def evaluator_check_in_corners(game, player):
    """The "Check In Corners" evaluation function calculates a cumulative score based on the moves
        and their positions.
        A cumulative score is calculated for both the players.
        A positive score is added to the cumulative score for the player if the board is less than 60% occupied and
        the moves lie in the corners, in case the board is more than 60% occupied a higher score is subtracted from the sum.
        The process is negated in case of the opponent.
        The difference between both the player cumulative scores is added to the number difference of non-corners move for both players
        and the value returned

                Parameters
                ----------
                game : `isolation.Board`
                    An instance of `isolation.Board` encoding the current state of the
                    game (e.g., player locations and blocked cells).

                player : hashable
                    One of the objects registered by the game object as a valid player.
                    (i.e., `player` should be either game.__player_1__ or
                    game.__player_2__).

                Returns
                ----------
                float
                    The heuristic value of the current game state
                """
    corners = [(0,0), (0,game.width-1), (game.height-1,0), (game.height-1,game.width-1)]

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    own_cum_score = 0
    opp_cum_score = 0
    own_moves_left = 0
    opp_moves_left = 0
    for move in own_moves:
        if is_in_corners(move, corners) and percent_occupied(game) < 60:
            own_cum_score += 15
        elif is_in_corners(move, corners) and percent_occupied(game) > 60:
            own_cum_score -= 40
        else:
            own_moves_left += 10

    for move in opp_moves:
        if is_in_corners(move, corners) and percent_occupied(game) < 60:
            opp_cum_score += 15
        elif is_in_corners(move, corners) and percent_occupied(game) > 60:
            opp_cum_score -= 40
        else:
            opp_moves_left += 10

    return float(own_cum_score - opp_cum_score) + float(own_moves_left - opp_moves_left)

def evaluator_check_near_walls_and_corners(game, player):
    """The "Check Near Walls And Corners" evaluation function calculates a cumulative score based on the moves
            and their positions.
            A cumulative score is calculated for both the players.
            The method calculates the score for corners using 'evaluator_check_in_corners',
            and the score for walls using 'evaluator_check_near_walls'. The walls score is multiplied by 0.3 and
            corners score is multiplied y 0.7 and their sum is returned.

                    Parameters
                    ----------
                    game : `isolation.Board`
                        An instance of `isolation.Board` encoding the current state of the
                        game (e.g., player locations and blocked cells).

                    player : hashable
                        One of the objects registered by the game object as a valid player.
                        (i.e., `player` should be either game.__player_1__ or
                        game.__player_2__).

                    Returns
                    ----------
                    float
                        The heuristic value of the current game state
                    """
    walls_score = evaluator_check_near_walls(game, player)
    corners_score = evaluator_check_in_corners(game, player)
    return float(0.3 * walls_score + 0.7 * corners_score)

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    #if game.is_loser(player):
     #   return float("-inf")

#    if game.is_winner(player):
 #       return float("inf")

    #own_moves = len(game.get_legal_moves(player))
    #opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
   # return float(own_moves - opp_moves)
    return evaluator_check_near_walls_and_corners(game, player)

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        if not legal_moves:
            return (-1, -1)

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        search = None
        if self.method == 'minimax':
            search = self.minimax
        elif self.method == 'alphabeta':
            search = self.alphabeta

        score, move = None, None
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            # Perform Iterative Deepening Search
            if self.iterative:
                for i in itertools.count():
                    score, move = search(game, i+1, False)
            else:
                # Perform Non-Iterative Deepening Search
                score, move = search(game, self.search_depth, False)

        except Timeout:
            return move

        return move
        # Return the best move from the last completed search iteration

    def min_max_helper(self, game, max_depth, max_mode, move, current_depth=0):
        """
        Perform Top-Down Recursion for Minimax Algorithm, starting from Depth 0 to the given max_depth

        Parameters
        ----------
        game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

        max_depth : int
        The max_depth specified for the Depth Limited Search in Minimax Algorithm

        max_mode : bool
        Specifies maximizing node or minimizing node

        move : (int, int)
        The move for which the score is being calculated, used as helper parameter for recursion

        current_depth : int
        Starting point for the recursive function. Defaults to zero. Should always be zero.

        Returns
        -------
        (int, (int, int))
        The score for the move, and the move
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        result_move = None
        moves = game.get_legal_moves()

        if not moves:
            return game.utility(self), (-1, -1)

        if max_depth == current_depth:
            return self.score(game, self), (-1, -1)

        if max_mode:
            max_score = float("-inf")
            for m in moves:
                new_game = game.forecast_move(m)
                init_move = m if current_depth is 0 else move
                score, temp_move = self.min_max_helper(new_game, max_depth, False, init_move, current_depth+1)
                if score > max_score:
                    max_score, result_move = score, init_move
            return max_score, result_move
        else:
            min_score = float("inf")
            for m in moves:
                new_game = game.forecast_move(m)
                init_move = m if current_depth is 0 else move
                score, temp_move = self.min_max_helper(new_game, max_depth, True, init_move, current_depth + 1)
                if score < min_score:
                    min_score, result_move = score, init_move
            return min_score, result_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Helper function for Top-Down Recursive Minimax
        score, move = self.min_max_helper(game=game, max_depth=depth, max_mode=maximizing_player, move=None, current_depth=0)
        return score, move

    def alpha_beta_helper(self, game, max_depth, max_mode, move, current_depth=0, alpha=float("-inf"), beta=float("inf")):
        """
        Perform Top-Down Recursion for Minimax Algorithm With Alpha-Beta Pruning,
        starting from Depth 0 to the given max_depth

        Parameters
        ----------

        game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

        max_depth : int
        The max_depth specified for the Depth Limited Search in Minimax Algorithm

        max_mode : bool
        Specifies maximizing node or minimizing node

        move : (int, int)
        The move for which the score is being calculated, used as helper parameter for recursion

        current_depth : int
        Starting point for the recursive function. Defaults to zero. Should always be zero.

        alpha : float
        alpha value for the alpha beta pruning

        beta: float
        beta value for the alpha beta pruning

        Returns
        -------
        (int, (int, int))
        The score for the move, and the move
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        moves = game.get_legal_moves()
        result_move = None

        if not moves:
            return game.utility(self), None

        if current_depth == max_depth:
            return self.score(game, self), None

        if max_mode:
            max_score = float("-inf")
            for m in moves:
                new_game = game.forecast_move(m)
                init_move = m if current_depth is 0 else move
                score, _ = self.alpha_beta_helper(new_game, max_depth, False, init_move, current_depth+1, alpha=alpha, beta=beta)
                if score > max_score:
                    max_score, result_move = score, init_move
                if max_score >= beta:
                    return max_score, result_move
                alpha = max(alpha, max_score)
            return max_score, result_move
        else:
            min_score = float("inf")
            for m in moves:
                new_game = game.forecast_move(m)
                init_move = m if current_depth is 0 else move
                score, _ = self.alpha_beta_helper(new_game, max_depth, True, init_move, current_depth + 1, alpha=alpha, beta=beta)
                if score < min_score:
                    min_score, result_move = score, init_move
                if min_score <= alpha:
                    return min_score, result_move
                beta = min(beta, min_score)
            return min_score, result_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

         alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Helper function for Top-Down Recursive Minimax With Alpha-Beta Pruning

        score, move = self.alpha_beta_helper(game=game, max_depth=depth, max_mode=maximizing_player, move=None, current_depth=0, alpha=alpha, beta=beta)
        return score, move
