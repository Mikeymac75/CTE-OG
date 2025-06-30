import random
import logging
import time
import json # Added for RL Agent state serialization
import os   # Added for saving/loading Q-table
import sqlite3 # Added for SQLite Q-table storage
from flask import Flask, jsonify, render_template, send_from_directory, request

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# --- RL Agent Configuration ---
Q_TABLE_DB_FILE = "q_table.sqlite" # Changed from Q_TABLE_FILE = "q_table.json"
REWARD_WIN_TRICK = 5
REWARD_LOSE_TRICK = -5
REWARD_WIN_ROUND_MAKER_NORMAL = 20
REWARD_WIN_ROUND_MAKER_MARCH = 40
REWARD_WIN_ROUND_MAKER_ALONE_MARCH = 60 # Higher for successful alone
REWARD_EUCHRED_MAKER = -50
REWARD_WIN_ROUND_DEFENSE_EUCHRE = 50
REWARD_LOSE_ROUND_DEFENSE = -20 # If maker makes points
REWARD_WIN_GAME = 100
REWARD_LOSE_GAME = -100
# Intermediate rewards (optional, can be 0)
REWARD_SUCCESSFUL_BID = 2 # e.g. if bid and made points
REWARD_FAILED_BID = -2    # e.g. if bid and got euchred

DEFAULT_LEARNING_RATE = 0.1
DEFAULT_DISCOUNT_FACTOR = 0.9
DEFAULT_EPSILON = 1.0 # Initial exploration rate
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

class RLAgent:
    def __init__(self, player_id, learning_rate=DEFAULT_LEARNING_RATE, discount_factor=DEFAULT_DISCOUNT_FACTOR, epsilon=DEFAULT_EPSILON):
        """
        Initializes the Reinforcement Learning Agent.

        Args:
            player_id: The unique identifier for the player.
            learning_rate: The learning rate (alpha) for Q-learning.
            discount_factor: The discount factor (gamma) for future rewards.
            epsilon: The initial exploration rate for epsilon-greedy strategy.
        """
        self.player_id = player_id
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.training_mode = True # Can be set to False for exploitation only
        self.db_conn = self._init_db()
        # self.q_table cache is removed; direct DB operations will be used.

    def _init_db(self):
        """Initializes SQLite connection and creates table if it doesn't exist."""
        conn = sqlite3.connect(Q_TABLE_DB_FILE, check_same_thread=False) # check_same_thread=False for Flask if accessed across requests by same agent instance
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS q_values (
                state_key TEXT PRIMARY KEY,
                actions_q_values TEXT NOT NULL
            )
        """)
        conn.commit()
        logging.info(f"RLAgent P{self.player_id}: SQLite DB initialized/connected at {Q_TABLE_DB_FILE}")
        return conn

    def set_training_mode(self, mode):
        """
        Sets the agent's training mode.

        Args:
            mode (bool): True to enable training (exploration and learning),
                         False for exploitation-only mode.
        """
        self.training_mode = mode
        if not mode:
            self.epsilon = 0 # No exploration if not training

    # load_q_table and save_q_table are removed as we operate directly on DB

    def _serialize_state(self, state_dict):
        """
        Converts a state dictionary to a canonical JSON string representation.
        This is used as a key in the Q-table. Sorting keys ensures consistency.
        """
        # Sort by keys to ensure canonical representation
        return json.dumps(state_dict, sort_keys=True)

    def get_q_value(self, state_key, action_key):
        """
        Retrieves the Q-value for a given state-action pair from the SQLite database.

        Args:
            state_key (str): The serialized string representation of the state.
            action_key (str): The serialized string representation of the action.

        Returns:
            float: The Q-value, or 0.0 if the state or action is not found.
        """
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT actions_q_values FROM q_values WHERE state_key = ?", (state_key,))
        row = cursor.fetchone()
        if row:
            actions_dict = json.loads(row[0])
            return actions_dict.get(action_key, 0.0)
        return 0.0

    def choose_action(self, state_dict, valid_actions):
        """
        Chooses an action based on epsilon-greedy strategy.
        state_dict: A dictionary representing the current game state relevant to the agent.
        valid_actions: A list of valid action identifiers (e.g., card strings, bid types).
                       Each action should be a simple string or tuple that can be a dict key.
        """
        if not valid_actions:
            logging.warning(f"RLAgent P{self.player_id}: No valid actions provided.")
            return None

        state_key = self._serialize_state(state_dict)

        if self.training_mode and random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random valid action
            action = random.choice(valid_actions)
            logging.debug(f"RLAgent P{self.player_id} (Explore): Chose random action {action} from {valid_actions}")
            return action
        else:
            # Exploit: choose the best action from Q-table
            # Use serialized action for keys in q_values dict if action is a dict (e.g. card)
            # Original action objects are preserved for choosing from best_actions.

            action_q_pairs = []
            for action_obj in valid_actions:
                action_key_for_qtable = self._serialize_action(action_obj)
                q_val = self.get_q_value(state_key, action_key_for_qtable)
                action_q_pairs.append({'action_obj': action_obj, 'q_value': q_val, 'serialized_key': action_key_for_qtable})

            # For logging q_values in a readable way:
            log_q_values = {item['serialized_key']: item['q_value'] for item in action_q_pairs}

            max_q = -float('inf')
            if action_q_pairs: # Ensure there are actions
                 max_q = max(item['q_value'] for item in action_q_pairs)

            best_actions = [item['action_obj'] for item in action_q_pairs if item['q_value'] == max_q]

            chosen_action = random.choice(best_actions) if best_actions else random.choice(valid_actions) # Fallback
            logging.debug(f"RLAgent P{self.player_id} (Exploit): Q-values (serialized): {log_q_values}, Chose action_obj {self._serialize_action(chosen_action)} (Max Q: {max_q}) from best_actions (serialized keys): {[self._serialize_action(a) for a in best_actions]}")
            return chosen_action

    def _serialize_action(self, action):
        """Serializes an action (which could be a simple string or a dict for a card) into a string key."""
        if isinstance(action, dict): # e.g., a card object {'suit': 'H', 'rank': 'A'}
            return f"card_{action['suit']}{action['rank']}"
        return str(action) # For simple actions like "pass_bid", "order_up"

    def update_q_table(self, state_dict, action, reward, next_state_dict, next_valid_actions):
        """
        Updates the Q-table using the Q-learning formula.
        Q(s,a) = Q(s,a) + lr * [reward + gamma * max_a'(Q(s',a')) - Q(s,a)]

        Args:
            state_dict (dict): The state from which the action was taken.
            action: The action taken by the agent.
            reward (float): The reward received for taking the action.
            next_state_dict (dict): The resulting state after the action.
            next_valid_actions (list): A list of valid actions in the next_state.
        """
        if not self.training_mode:
            return

        state_key = self._serialize_state(state_dict)
        action_key = self._serialize_action(action) # Serialized action
        next_state_key = self._serialize_state(next_state_dict)

        # Fetch current Q-values for the state, or an empty dict if state is new
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT actions_q_values FROM q_values WHERE state_key = ?", (state_key,))
        row = cursor.fetchone()
        actions_q_dict = {}
        if row:
            try:
                actions_q_dict = json.loads(row[0])
            except json.JSONDecodeError:
                logging.error(f"RLAgent P{self.player_id}: Failed to decode actions_q_values for state {state_key}. Starting fresh for this state.")
                actions_q_dict = {}

        old_value = actions_q_dict.get(action_key, 0.0)

        # Q-learning: Q(s,a) = Q(s,a) + lr * [reward + gamma * max_a'(Q(s',a')) - Q(s,a)]
        next_max_q = 0.0
        if next_valid_actions: # If there are next actions (not a terminal state for Q-learning)
            # Fetch Q-values for next_state to find max_a'
            cursor.execute("SELECT actions_q_values FROM q_values WHERE state_key = ?", (next_state_key,))
            next_row = cursor.fetchone()
            if next_row:
                try:
                    next_actions_q_dict = json.loads(next_row[0])
                    if next_actions_q_dict: # Check if not empty
                         next_max_q = max(next_actions_q_dict.get(self._serialize_action(next_action), 0.0) for next_action in next_valid_actions)
                except json.JSONDecodeError:
                    logging.error(f"RLAgent P{self.player_id}: Failed to decode actions_q_values for next_state {next_state_key}.")
            # If next_state not in DB or actions empty, next_max_q remains 0.0, which is correct.
            # Also, if next_valid_actions is empty, this loop is skipped and next_max_q is 0.0

        new_value = old_value + self.lr * (reward + self.gamma * next_max_q - old_value)
        actions_q_dict[action_key] = new_value

        # Update the database
        updated_actions_q_json = json.dumps(actions_q_dict)
        try:
            cursor.execute("""
                INSERT INTO q_values (state_key, actions_q_values) VALUES (?, ?)
                ON CONFLICT(state_key) DO UPDATE SET actions_q_values = excluded.actions_q_values
            """, (state_key, updated_actions_q_json))
            self.db_conn.commit()
        except sqlite3.Error as e:
            logging.error(f"RLAgent P{self.player_id}: SQLite error during Q-update: {e}")
            # Optionally, implement a retry mechanism or error handling strategy

        logging.debug(f"RLAgent P{self.player_id} Q-Update (DB): StateKey: {state_key}, ActionKey: {action_key}, Reward: {reward}, OldQ: {old_value:.2f}, NewQ: {new_value:.2f}, NextMaxQ: {next_max_q:.2f}")

        # Epsilon decay after update
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(MIN_EPSILON, self.epsilon)

    def __del__(self):
        """Ensure DB connection is closed when agent instance is garbage collected."""
        if hasattr(self, 'db_conn') and self.db_conn:
            self.db_conn.close()
            logging.info(f"RLAgent P{self.player_id}: SQLite DB connection closed.")

# --- Card and Deck Setup ---
SUITS_MAP = {'H': 'Hearts', 'D': 'Diamonds', 'C': 'Clubs', 'S': 'Spades'}
SUITS = ['H', 'D', 'C', 'S']
RANKS = ['9', '10', 'J', 'Q', 'K', 'A']
VALUES = {'9': 1, '10': 2, 'J': 3, 'Q': 4, 'K': 5, 'A': 6} # Base values, trump adds more

class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return f"{self.rank} of {SUITS_MAP[self.suit]}"

    def to_dict(self):
        return {"suit": self.suit, "rank": self.rank, "name": str(self)}

def create_deck():
    return [Card(s, r) for s in SUITS for r in RANKS]

# --- Game State Management ---

class Game:
    """
    Encapsulates all game state and related logic for a single Euchre game.
    This includes player hands, scores, current phase, deck, RL agents, etc.
    """
    def __init__(self):
        self.game_data = {}
        self.rl_agents = {}
        self._initialize_game_and_agents() # Initialize upon creation

    def _initialize_game_and_agents(self):
        """
        Initializes or resets the game_data and rl_agents for a new game.
        This function sets up the game structure, player identities, scores,
        and RL agent parameters (like epsilon) to their default starting values.
        """
        logging.info("Game class: Initializing game data and RL agents.")

        num_players = 3
        player_identities = {
            0: "Player 0 (You)",
            1: "Player 1 (AI)",
            2: "Player 2 (AI)"
        }

        # Initialize RL agents
        self.rl_agents.clear() # Clear any existing agents if re-initializing
        for pid, name in player_identities.items():
            if pid not in self.rl_agents:
                logging.info(f"Game class: Creating RLAgent for player {pid} ({name})")
                self.rl_agents[pid] = RLAgent(player_id=pid) # RLAgent class needs to be defined before Game

        # Reset epsilon for agents at the start of each new game
        for agent_id, agent in self.rl_agents.items():
            agent.epsilon = DEFAULT_EPSILON # Assuming DEFAULT_EPSILON is accessible
            agent.set_training_mode(True)
            if hasattr(agent, 'last_action_info'):
                del agent.last_action_info

        self.game_data = {
            "deck": [], "hands": {p: [] for p in range(num_players)}, "dummy_hand": [],
            "scores": {p: 0 for p in range(num_players)}, "dealer": random.randint(0, num_players - 1),
            "rl_training_data": {p_id: {} for p_id in self.rl_agents.keys()}, # Store per-agent training info
            "trump_suit": None, "up_card": None, "up_card_visible": False,
            "current_player_turn": -1, "maker": None, "going_alone": False,
            "trick_cards": [], "current_trick_lead_suit": None,
            "trick_leader": -1, "round_tricks_won": {p: 0 for p in range(num_players)},
            "game_phase": "setup", # Initial phase
            "message": "Welcome! Click 'Start New Round'.",
            "player_identities": player_identities,
            "num_players": num_players,
            "passes_on_upcard": [], "passes_on_calling": [],
            "cards_to_discard_count": 0,
            "original_up_card_for_round": None,
            "last_completed_trick": None,
            "played_cards_this_round": [] # Initialize here
        }
        logging.debug(f"Game class: Initialization complete. game_data keys: {list(self.game_data.keys())}")

    # --- Convenience accessors, more can be added as needed ---
    def get_data(self):
        """Returns the raw game_data dictionary."""
        return self.game_data

    def get_agent(self, player_id):
        """Retrieves the RLAgent instance for a given player ID."""
        return self.rl_agents.get(player_id)

# Global instance of the Game class
# This is an intermediate step. Ideally, this would be managed by Flask's app context (g object)
# or passed around explicitly. For now, a single global instance simplifies transition.
_current_game_instance = None

def get_game_instance() -> Game:
    """
    Provides access to the single global Game instance.
    Creates it if it doesn't exist yet.
    """
    global _current_game_instance
    if _current_game_instance is None:
        _current_game_instance = Game()
    return _current_game_instance

# Deprecated global variables - to be removed once all refs are updated
# game_data = {}
# rl_agents = {}

# Helper function to get the RLAgent for the current AI player
# This should now use the Game instance
def get_rl_agent(player_id):
    """Retrieves the RLAgent instance for a given player ID from the current game."""
    game = get_game_instance()
    return game.get_agent(player_id)


def get_player_role(player_id, dealer_id, maker_id, num_players):
    """
    Determines the role of a player in the current round.
    Note: This function is currently independent of the Game class instance,
    but operates on data that would typically come from it.

    Args:
        player_id: The ID of the player.
        dealer_id: The ID of the dealer.
        maker_id: The ID of the player who called trump (maker).
        num_players: The total number of players in the game.

    Returns:
        str: The player's role (e.g., "dealer", "maker", "opponent", "bidder").
    """
    if player_id == dealer_id:
        return "dealer"
    if maker_id is None: # Before trump is called
        return "bidder"
    if player_id == maker_id:
        return "maker"

    # Determine partner (assuming 3 players for now as per current game setup)
    # Player 0: Human, Player 1: AI, Player 2: AI
    # If maker is P1 (AI), P2 (AI) is opponent. Dummy (effectively P0's partner) is also opponent from P1's view.
    # If maker is P2 (AI), P1 (AI) is opponent.
    # If maker is P0 (Human), P1 and P2 are opponents.
    # This simplified logic is for a 3-player game where AI doesn't explicitly team up with another AI against human maker yet.
    # For 3 players, if AI is not maker, it's an opponent to the maker.
    if player_id != maker_id:
        return "opponent"

    # Fallback, though ideally should always be one of the above.
    return "unknown"


def get_rl_state(player_id, current_game_data):
    """
    Constructs the state representation for the RL agent.
    player_id: The ID of the AI player for whom the state is being constructed.
    current_game_data_dict: The game_data dictionary from the Game instance.
    """
    # This function now expects current_game_data_dict to be passed,
    # which is game.game_data from the calling context.
    hand = current_game_data_dict["hands"].get(player_id, [])
    hand_serialized = sorted([card.to_dict() for card in hand], key=lambda c: (c['suit'], c['rank']))

    up_card_dict = current_game_data_dict.get("up_card")
    original_up_card_dict = current_game_data_dict.get("original_up_card_for_round")

    state = {
        "player_id": player_id,
        "game_phase": current_game_data_dict.get("game_phase"),
        "trump_suit": current_game_data_dict.get("trump_suit"),
        "dealer": current_game_data_dict.get("dealer"),
        "maker": current_game_data_dict.get("maker"),
        "current_player_turn": current_game_data_dict.get("current_player_turn"),

        "hand": [c['suit'] + c['rank'] for c in hand_serialized],

        # Bidding specific
        "up_card_suit": up_card_dict.suit if up_card_dict and current_game_data_dict.get("up_card_visible") else None,
        "up_card_rank": up_card_dict.rank if up_card_dict and current_game_data_dict.get("up_card_visible") else None,
        "original_up_card_suit": original_up_card_dict.suit if original_up_card_dict else None,
        "passes_on_upcard": len(current_game_data_dict.get("passes_on_upcard", [])),
        "passes_on_calling": len(current_game_data_dict.get("passes_on_calling", [])),

        # Playing specific
        "current_trick_lead_suit": current_game_data_dict.get("current_trick_lead_suit"),
        "trick_cards_played_count": len(current_game_data_dict.get("trick_cards", [])),
        "trick_leader": current_game_data_dict.get("trick_leader"),

        # Scores and round progress
        "my_score": current_game_data_dict["scores"].get(player_id, 0),
        "my_round_tricks": current_game_data_dict["round_tricks_won"].get(player_id, 0),

        "player_role": get_player_role(player_id, current_game_data_dict.get("dealer"), current_game_data_dict.get("maker"), current_game_data_dict.get("num_players")),
        "going_alone": current_game_data_dict.get("going_alone", False)
    }

    current_trump_for_features = current_game_data_dict.get("trump_suit")
    hand_card_objects = current_game_data_dict["hands"].get(player_id, [])
    calculated_features = get_hand_features(hand_card_objects, current_trump_for_features)
    for feature_name, feature_value in calculated_features.items():
        state[f"feat_{feature_name}"] = feature_value


    # Add hand strength for potential trump suits during bidding
    # This can remain, or we can rely on the agent learning from raw features.
    # For now, let's keep evaluate_potential_trump_strength as it's a useful heuristic.
    # The new "feat_*" will be based on actual trump (or None if no trump).
    # The "strength_as_trump_X" are specific to bidding.

    if state["game_phase"] == "bidding_round_1" and state["up_card_suit"]:
        # Evaluate features and strength if the up-card's suit were trump
        potential_trump_features = get_hand_features(hand_card_objects, state["up_card_suit"])
        for f_name, f_val in potential_trump_features.items():
            state[f"potential_trump_{state['up_card_suit']}_{f_name}"] = f_val
        state[f"eval_strength_as_trump_{state['up_card_suit']}"] = evaluate_potential_trump_strength(hand_card_objects, state["up_card_suit"], current_game_data_dict) # Pass game_data for context if needed by eval

    elif state["game_phase"] == "bidding_round_2":
        # Evaluate features and strength for all other possible trump suits
        for s_option in SUITS:
            if s_option != state["original_up_card_suit"]: # Cannot call the turned down suit
                potential_trump_features_r2 = get_hand_features(hand_card_objects, s_option)
                for f_name, f_val in potential_trump_features_r2.items():
                    state[f"potential_trump_{s_option}_{f_name}"] = f_val
                state[f"eval_strength_as_trump_{s_option}"] = evaluate_potential_trump_strength(hand_card_objects, s_option, current_game_data_dict) # Pass game_data for context

    # Add played card information to the state
    played_cards_in_round = current_game_data_dict.get("played_cards_this_round", [])
    state["played_cards_serialized"] = sorted([f"{c.suit}{c.rank}" for c in played_cards_in_round])

    # Initialize specific flags/counts for played cards, relative to current trump
    state["played_right_bower"] = False
    state["played_left_bower"] = False
    state["played_ace_trump"] = False
    state["played_king_trump"] = False
    state["played_queen_trump"] = False
    state["num_trump_cards_played"] = 0
    for s_key in SUITS:
        state[f"num_suit_played_{s_key}"] = 0

    current_trump_suit = current_game_data_dict.get("trump_suit")
    if current_trump_suit: # Only calculate these if trump is set
        left_bower_actual_suit = get_left_bower_suit(current_trump_suit)
        for card in played_cards_in_round:
            state[f"num_suit_played_{card.suit}"] += 1 # Count natural suit played

            effective_card_suit = get_effective_suit(card, current_trump_suit)
            if effective_card_suit == current_trump_suit:
                state["num_trump_cards_played"] += 1
                if card.rank == 'J':
                    if card.suit == current_trump_suit:
                        state["played_right_bower"] = True
                    elif card.suit == left_bower_actual_suit:
                        state["played_left_bower"] = True
                elif card.rank == 'A':
                    state["played_ace_trump"] = True
                elif card.rank == 'K':
                    state["played_king_trump"] = True
                elif card.rank == 'Q':
                    state["played_queen_trump"] = True
    else: # If trump is not set, still count natural suits played
        for card in played_cards_in_round:
            state[f"num_suit_played_{card.suit}"] += 1

    return state

def initialize_game_data():
    """
    This function is now a wrapper around the Game class re-initialization.
    It ensures that the global game instance is reset to a fresh state.
    """
    global _current_game_instance
    _current_game_instance = Game() # Creates a new Game instance, which auto-initializes
    logging.info("Global game instance has been re-initialized.")
    # The old logic for initializing rl_agents and game_data globals is now in Game._initialize_game_and_agents()


def get_next_valid_actions(player_id, game_phase, game_data_for_next_state_dict):
    """ Helper to determine valid actions for the next state. Essential for Q-learning update. """
    # game_data_for_next_state_dict is expected to be a game_data dictionary
    hand_cards = game_data_for_next_state_dict.get("hands", {}).get(player_id, [])

    if game_phase == "playing_tricks":
        lead_suit = game_data_for_next_state_dict.get("current_trick_lead_suit")
        trump_suit = game_data_for_next_state_dict.get("trump_suit")
        valid_card_objects = get_valid_plays(list(hand_cards), lead_suit, trump_suit)
        return [card.to_dict() for card in valid_card_objects]
    elif game_phase == "bidding_round_1":
        return ["order_up", "pass_bid"]
    elif game_phase == "bidding_round_2":
        original_up_card = game_data_for_next_state_dict.get("original_up_card_for_round")
        turned_down_suit = original_up_card.suit if original_up_card else None
        return [f"call_trump:{s}" for s in SUITS if s != turned_down_suit] + ["pass_call"]
    elif game_phase == "dealer_must_call":
        original_up_card = game_data_for_next_state_dict.get("original_up_card_for_round")
        turned_down_suit = original_up_card.suit if original_up_card else None
        return [f"call_trump:{s}" for s in SUITS if s != turned_down_suit]
    elif game_phase == "prompt_go_alone":
        return ["choose_go_alone", "choose_not_go_alone"]
    elif game_phase == "dealer_discard_one" or game_phase == "dealer_must_discard_after_order_up":
        if game_data_for_next_state_dict.get("cards_to_discard_count") == 1 and game_data_for_next_state_dict.get("current_player_turn") == player_id:
             return [card.to_dict() for card in hand_cards]
        return []
    elif game_phase == "maker_discard":
        if game_data_for_next_state_dict.get("cards_to_discard_count") > 1 and game_data_for_next_state_dict.get("current_player_turn") == player_id:
            return [] # Simplified for now
    return []


def process_rl_update(player_id_acted, event_type, event_data=None):
    """
    Processes RL update for an AI player after an event.
    This function now fetches game_data from the Game instance.
    """
    game = get_game_instance()
    current_game_data = game.game_data # Use game_data from the instance

    agent = game.get_agent(player_id_acted) # Get agent from game instance

    if not agent or not agent.training_mode:
        return

    training_info = current_game_data["rl_training_data"].get(player_id_acted)

    if not training_info or "state" not in training_info or "action" not in training_info:
        return

    prev_state_dict = training_info["state"]
    # action_taken_serialized = agent._serialize_action(training_info["action"]) # Action is already serialized if from agent

    if training_info["action"] is None:
        current_game_data["rl_training_data"][player_id_acted] = {}
        return

    reward = 0
    if event_type == "trick_end":
        trick_winner_idx = event_data.get("trick_winner_idx")
        if trick_winner_idx == player_id_acted: reward += REWARD_WIN_TRICK
        else: reward += REWARD_LOSE_TRICK

    elif event_type == "round_end":
        round_maker = current_game_data.get("maker")
        is_player_maker = (player_id_acted == round_maker)
        if round_maker is None: logging.error(f"RL Update (round_end for P{player_id_acted}): Maker is None.")
        elif is_player_maker:
            maker_tricks_won = current_game_data["round_tricks_won"].get(round_maker, 0)
            was_alone = current_game_data.get("going_alone", False)
            if maker_tricks_won < 3: reward += REWARD_EUCHRED_MAKER
            elif maker_tricks_won == 5: reward += REWARD_WIN_ROUND_MAKER_ALONE_MARCH if was_alone else REWARD_WIN_ROUND_MAKER_MARCH
            else: reward += REWARD_WIN_ROUND_MAKER_NORMAL
        else: # Defender
            maker_tricks_won = current_game_data["round_tricks_won"].get(round_maker, 0)
            if maker_tricks_won < 3: reward += REWARD_WIN_ROUND_DEFENSE_EUCHRE
            else: reward += REWARD_LOSE_ROUND_DEFENSE

    elif event_type == "game_end":
        game_winner_idx = event_data.get("game_winner_idx")
        if game_winner_idx == player_id_acted: reward += REWARD_WIN_GAME
        else: reward += REWARD_LOSE_GAME

    elif event_type == "bid_processed":
        action_type_from_state = training_info.get("action_type")
        is_bid_action = action_type_from_state in ["ai_bidding_round_1", "ai_bidding_round_2", "ai_dealer_stuck_call", "decide_go_alone"]
        if is_bid_action and current_game_data.get("maker") == player_id_acted :
             pass # Rewards for becoming maker are part of round_end rewards

    next_state_dict = get_rl_state(player_id_acted, current_game_data) # Pass current_game_data
    next_player_for_q_learning = current_game_data.get("current_player_turn", -1)
    next_valid_actions_for_q = []
    if current_game_data.get("game_phase") not in ["game_over", "round_over", "setup"]:
        # In simulation, all players are AI. If P0 is AI, this needs to be != -1.
        # Assuming P0 is human for Flask, AI players are > 0.
        # For simulation, this check might need adjustment or all players get agents.
        # Current: if next player is any valid player index (non -1) AND not human (0 in mixed mode)
        if next_player_for_q_learning != -1: # Simplified: get actions if there's a next player.
                                         # Further filtering if P0 is always human can be added.
            # If all players are AI (as in training), then P0 is also an AI.
            # If P0 is human, then next_player_for_q_learning != 0 might be needed.
            # For now, assume all players can have next_valid_actions if they are AI.
            # The RLAgent is only created for AI players, so this is implicitly handled.
            if game.get_agent(next_player_for_q_learning): # Check if next player is an RL agent
                 next_valid_actions_for_q = get_next_valid_actions(next_player_for_q_learning, current_game_data["game_phase"], current_game_data)

    logging.debug(f"RL Update P{player_id_acted}: Event: {event_type}, Action: {training_info['action']}, Reward: {reward}")
    agent.update_q_table(prev_state_dict, training_info["action"], reward, next_state_dict, next_valid_actions_for_q)

    current_game_data["rl_training_data"][player_id_acted] = {}


# --- Core Game Logic ---
def initialize_new_round():
    """
    Sets up the game state for the beginning of a new round.
    Uses the current Game instance.
    """
    game = get_game_instance()
    current_game_data = game.game_data # Work with the game_data from the instance

    logging.info(f"Initializing new round. Current dealer: {current_game_data.get('dealer', 'N/A')}")
    game_data["deck"] = create_deck()
    random.shuffle(game_data["deck"])
    player_hands = {i: [] for i in range(game_data["num_players"])}
    for _ in range(5):
        for i in range(game_data["num_players"]):
            if game_data["deck"]:
                player_hands[i].append(game_data["deck"].pop())
            else:
                game_data["message"] = "Error: Not enough cards for player hands."; return
    game_data["hands"] = player_hands
    game_data["dummy_hand"] = []
    for _ in range(5):
        if game_data["deck"]:
            game_data["dummy_hand"].append(game_data["deck"].pop())
        else:
            game_data["message"] = "Error: Not enough cards for dummy hand."; return
    if not game_data["deck"]: game_data["message"] = "Error: Not enough for up_card."; return
    game_data["up_card"] = game_data["deck"].pop()
    game_data["original_up_card_for_round"] = game_data["up_card"]
    if "kitty" in game_data: del game_data["kitty"]
    game_data["up_card_visible"] = True
    game_data["trump_suit"] = None
    game_data["maker"] = None
    game_data["going_alone"] = False
    game_data["trick_cards"] = []
    game_data["current_trick_lead_suit"] = None
    game_data["round_tricks_won"] = {i: 0 for i in range(game_data["num_players"])}
    game_data["dealer"] = int(game_data["dealer"])
    game_data["current_player_turn"] = (game_data["dealer"] + 1) % game_data["num_players"]
    game_data["trick_leader"] = game_data["current_player_turn"]
    game_data["game_phase"] = "bidding_round_1"
    game_data["passes_on_upcard"] = []
    game_data["passes_on_calling"] = []
    game_data["cards_to_discard_count"] = 0
    game_data["last_completed_trick"] = None
    game_data["played_cards_this_round"] = [] # Initialize played cards for the new round
    game_data["message"] = f"{game_data['player_identities'][game_data['current_player_turn']]}'s turn. Up-card: {str(game_data['up_card'])}."
    logging.info(f"New round initialized. Up-card: {str(game_data['up_card']) if game_data['up_card'] else 'N/A'}. Turn: P{game_data['current_player_turn']}. Phase: {game_data['game_phase']}")

def get_left_bower_suit(trump_suit_char):
    if trump_suit_char not in SUITS_MAP: return None
    return {'H': 'D', 'D': 'H', 'C': 'S', 'S': 'C'}.get(trump_suit_char)

def get_effective_suit(card: Card, trump_suit: str | None) -> str:
    """
    Determines the effective suit of a card, considering the trump suit.
    The Left Bower's suit changes to the trump suit.

    Args:
        card: The Card object.
        trump_suit: The current trump suit character (e.g., 'H') or None.

    Returns:
        The effective suit character of the card.
    """
    if not trump_suit: return card.suit
    left_bower_actual_suit = get_left_bower_suit(trump_suit)
    if card.rank == 'J' and card.suit == left_bower_actual_suit:
        return trump_suit
    return card.suit

def get_card_value(card, trump_suit, lead_suit_for_trick=None):
    rank_base_values = {'9': 1, '10': 2, 'J': 3, 'Q': 4, 'K': 5, 'A': 6}
    if not trump_suit: return rank_base_values.get(card.rank, 0)
    card_effective_suit = get_effective_suit(card, trump_suit)
    if card.rank == 'J':
        if card.suit == trump_suit: return 100
        if card.suit == get_left_bower_suit(trump_suit): return 90
    if card_effective_suit == trump_suit:
        return {'A': 80, 'K': 70, 'Q': 60, '10': 50, '9': 40}.get(card.rank, 0)
    if card.rank == 'A': return 30
    non_trump_rank_values = {'K': 25, 'Q': 20, 'J': 15, '10': 10, '9': 5}
    return non_trump_rank_values.get(card.rank, 0)

# --- Card Sense Feature Detection ---
def get_hand_features(hand: list[Card], trump_suit: str | None) -> dict:
    """
    Analyzes a player's hand and returns a dictionary of "Card Sense" features.
    These features are used to represent the hand's characteristics for the RL agent.

    Args:
        hand: A list of Card objects representing the player's hand.
        trump_suit: The current trump suit (e.g., 'H', 'D', 'C', 'S') or None
                    if trump has not yet been determined (e.g., during bidding).

    Returns:
        A dictionary where keys are feature names (str) and values are
        feature values (int, bool). Features include counts of trump cards,
        presence of specific high trump cards (bowers, Ace), information about
        off-suit Aces, suit voids, and lengths of suits.
    """
    features = {
        "num_trump_cards": 0,
        "has_right_bower": False,
        "has_left_bower": False,
        "has_ace_of_trump": False,
        "has_king_of_trump": False,
        "has_queen_of_trump": False,
        "num_aces_offsuit": 0,
        "num_suits_void_natural": 0, # Void in natural suit (H,D,C,S)
        "is_void_in_suit_H": True,
        "is_void_in_suit_D": True,
        "is_void_in_suit_C": True,
        "is_void_in_suit_S": True,
        "highest_trump_card_rank_value": 0,
        "lowest_trump_card_rank_value": 0, # Effectively non-zero if num_trump_cards > 0
        "num_cards_in_longest_offsuit": 0,
        "shortest_offsuit_length": 5, # Max hand size, will be updated
    }

    if not hand: # Handle empty hand case
        features["shortest_offsuit_length"] = 0
        features["num_suits_void_natural"] = 4 # All suits are void if hand is empty
        # is_void_in_suit_X defaults to True, which is correct for an empty hand.
        return features

    trump_cards_in_hand = []
    offsuit_cards_by_suit = {s: [] for s in SUITS}
    natural_suit_counts = {s: 0 for s in SUITS}

    left_bower_actual_suit = get_left_bower_suit(trump_suit) if trump_suit else None

    for card in hand:
        natural_suit_counts[card.suit] += 1
        features[f"is_void_in_suit_{card.suit}"] = False # Mark suit as not void

        effective_suit = get_effective_suit(card, trump_suit)

        if trump_suit and effective_suit == trump_suit:
            features["num_trump_cards"] += 1
            trump_cards_in_hand.append(card)
            card_value_in_trump = get_card_value(card, trump_suit) # Use consistent trump valuation

            if features["lowest_trump_card_rank_value"] == 0 or card_value_in_trump < features["lowest_trump_card_rank_value"]:
                features["lowest_trump_card_rank_value"] = card_value_in_trump
            if card_value_in_trump > features["highest_trump_card_rank_value"]:
                features["highest_trump_card_rank_value"] = card_value_in_trump

            if card.rank == 'J':
                if card.suit == trump_suit:
                    features["has_right_bower"] = True
                elif card.suit == left_bower_actual_suit: # This is the Left Bower
                    features["has_left_bower"] = True
            elif card.rank == 'A':
                features["has_ace_of_trump"] = True
            elif card.rank == 'K':
                features["has_king_of_trump"] = True
            elif card.rank == 'Q':
                features["has_queen_of_trump"] = True
        else: # Card is off-suit (or trump not yet determined)
            offsuit_cards_by_suit[card.suit].append(card)
            if card.rank == 'A':
                features["num_aces_offsuit"] += 1

    # Calculate void natural suits
    for suit_key in SUITS:
        if natural_suit_counts[suit_key] == 0:
            features["num_suits_void_natural"] += 1
    # is_void_in_suit_X is already correctly set

    # Calculate longest and shortest offsuit (natural suits that are not trump)
    max_len = 0
    min_len = 5 # Start with max hand size
    found_an_offsuit = False

    for s in SUITS:
        # An "offsuit" for this calculation is a natural suit that is NOT the trump suit.
        # The Left Bower's natural suit is also considered "trump" for this purpose.
        is_suit_effectively_trump = (trump_suit and (s == trump_suit or s == left_bower_actual_suit))

        if not is_suit_effectively_trump:
            found_an_offsuit = True
            current_suit_len = natural_suit_counts[s]
            if current_suit_len > max_len:
                max_len = current_suit_len
            if current_suit_len < min_len:
                min_len = current_suit_len

    # max_len, min_len, found_an_offsuit are calculated above based on current trump_suit (if any)

    if trump_suit is None:
        # If no trump suit, all suits are considered for longest/shortest.
        # This overrides the previous loop's context which assumed a trump suit might exist.
        if not hand:
            features["num_cards_in_longest_offsuit"] = 0
            features["shortest_offsuit_length"] = 0
        else:
            all_natural_lengths = [natural_suit_counts[s_key] for s_key in SUITS]
            features["num_cards_in_longest_offsuit"] = max(all_natural_lengths) if all_natural_lengths else 0
            features["shortest_offsuit_length"] = min(all_natural_lengths) if all_natural_lengths else 0
    else:
        # Trump suit is defined. Use max_len and min_len calculated from non-trump suits.
        if found_an_offsuit:
            features["num_cards_in_longest_offsuit"] = max_len
            features["shortest_offsuit_length"] = min_len
        else: # No offsuits found (e.g., all trump hand or hand is empty)
            features["num_cards_in_longest_offsuit"] = 0
            features["shortest_offsuit_length"] = 0
            if not hand: # ensure for empty hand, it's explicitly 0, though covered by initial check
                 features["num_suits_void_natural"] = 4


    return features

def evaluate_potential_trump_strength(hand, potential_trump_suit, game_data=None):
    # This function can now leverage get_hand_features if desired,
    # or be kept separate for its specific scoring logic.
    # For now, keeping it separate as it calculates a "score" rather than discrete features.
    if not potential_trump_suit: return 0
    strength_score = 0; num_trump_cards = 0; has_right_bower = False; has_left_bower = False
    for card in hand:
        strength_score += get_card_value(card, potential_trump_suit)
        effective_suit = get_effective_suit(card, potential_trump_suit)
        if effective_suit == potential_trump_suit:
            num_trump_cards += 1
            if card.rank == 'J':
                if card.suit == potential_trump_suit: has_right_bower = True
                elif card.suit == get_left_bower_suit(potential_trump_suit): has_left_bower = True
    if has_right_bower and has_left_bower: strength_score += 25
    elif has_right_bower: strength_score += 15
    elif has_left_bower: strength_score += 10
    if num_trump_cards >= 3: strength_score += (num_trump_cards * 5)
    elif num_trump_cards == 2: strength_score += 5
    num_aces = sum(1 for card in hand if card.rank == 'A')
    if num_aces >= 2: strength_score += (num_aces * 3)
    suits_in_hand = set(get_effective_suit(c, potential_trump_suit) for c in hand)
    if len(suits_in_hand) <= 2 and num_trump_cards >=1 :
        strength_score += 5
        if len(suits_in_hand) == 1 and potential_trump_suit in suits_in_hand: strength_score += 10
    return strength_score

def get_ai_cards_to_discard(hand: list[Card], num_to_discard: int, trump_suit: str | None) -> list[Card]:
    """
    Heuristic for an AI player to select cards to discard.
    Sorts cards by their value (lowest first, considering trump) and discards the worst ones.

    Args:
        hand: The AI player's current hand (list of Card objects).
        num_to_discard: The number of cards the AI needs to discard.
        trump_suit: The current trump suit, or None if not yet determined.

    Returns:
        A list of Card objects to be discarded.
    """
    hand_copy = list(hand)
    hand_copy.sort(key=lambda c: get_card_value(c, trump_suit, None))
    return hand_copy[:num_to_discard]

def get_ai_stuck_suit_call(hand: list[Card], turned_down_suit: str) -> str:
    """
    Heuristic for an AI dealer who is "stuck" (must call trump).
    Evaluates hand strength for each possible suit (excluding the turned-down suit)
    and calls the suit that yields the highest strength.

    Args:
        hand: The AI dealer's hand.
        turned_down_suit: The suit that was originally turned up and then passed on.

    Returns:
        The character representation of the suit the AI decides to call as trump.
    """
    possible_suits = [s for s in SUITS if s != turned_down_suit]
    if not possible_suits: return random.choice(SUITS) # Should not happen in standard Euchre
    best_suit, max_strength = "", -1
    for s_key in possible_suits:
        strength = sum(get_card_value(c, s_key) for c in hand)
        if strength > max_strength: max_strength, best_suit = strength, s_key
    return best_suit if best_suit else random.choice(possible_suits)

def transition_to_play_phase():
    game_data["game_phase"] = "playing_tricks"
    game_data["current_player_turn"] = (game_data["dealer"] + 1) % game_data["num_players"]
    game_data["trick_leader"] = game_data["current_player_turn"]
    game_data["message"] += f" {game_data['player_identities'][game_data['current_player_turn']]} leads the first trick."
    game_data.pop('passes_on_upcard', None); game_data.pop('passes_on_calling', None)
    if game_data["game_phase"] == "playing_tricks" and game_data["current_player_turn"] != 0:
        process_ai_play_card(game_data["current_player_turn"])

def process_ai_play_card(ai_player_idx: int):
    """
    Manages an AI player's turn to play a card.
    It first checks if an RLAgent exists for the player.
    If yes:
        - Constructs the current game state for the RL agent.
        - Asks the agent to choose a card (action).
        - Applies a heuristic overlay for sloughing (discarding a low-value card
          when unable to follow suit and not wanting to trump or win).
    If no RLAgent (or in fallback mode):
        - Uses a basic heuristic to play a card (e.g., follow suit with highest,
          trump if can win, slough lowest otherwise).

    After determining the card to play, this function updates the game state:
    removes the card from AI's hand, adds it to the trick, checks if the trick
    is complete, determines trick winner, updates scores/tricks won, and transitions
    the turn or game phase accordingly. It also triggers RL updates for agents
    involved in the completed trick.

    Args:
        ai_player_idx: The index of the AI player whose turn it is.
    """
    global game_data #; time.sleep(0.5) # Removed AI delay
    if game_data["game_phase"] != "playing_tricks" or game_data["current_player_turn"] != ai_player_idx: return

    current_agent = get_rl_agent(ai_player_idx)
    if not current_agent:
        logging.error(f"RL Agent for P{ai_player_idx} not found in process_ai_play_card. Falling back to heuristic logic.")
        # --- Start of Fallback Heuristic for missing RL Agent ---
        # This heuristic attempts to make a reasonable play based on common Euchre strategies.
        ai_hand_fallback = game_data["hands"][ai_player_idx]
        lead_suit_fallback = game_data["current_trick_lead_suit"]
        trump_suit_fallback = game_data["trump_suit"]

        valid_cards_fallback = get_valid_plays(list(ai_hand_fallback), lead_suit_fallback, trump_suit_fallback)

        if not valid_cards_fallback:
            game_data["message"] += f" Error: AI {game_data['player_identities'][ai_player_idx]} (Fallback) has no valid cards to play."
            if len(game_data["trick_cards"]) < game_data["num_players"]:
                game_data["current_player_turn"] = (ai_player_idx + 1) % game_data["num_players"]
            return

        card_to_play = None
        # Determine if AI can follow suit
        can_follow_lead_suit = False
        if lead_suit_fallback:
            can_follow_lead_suit = any(get_effective_suit(card, trump_suit_fallback) == lead_suit_fallback for card in ai_hand_fallback)

        if can_follow_lead_suit:
            # Must follow suit. Play the highest card of that suit.
            # valid_cards_fallback will already be filtered to only cards of the lead suit.
            valid_cards_fallback.sort(key=lambda c: get_card_value(c, trump_suit_fallback, lead_suit_fallback), reverse=True)
            card_to_play = valid_cards_fallback[0]
            logging.info(f"AI P{ai_player_idx} (Fallback Heuristic, Following Suit) playing highest: {str(card_to_play)}")
        else:
            # Cannot follow suit. Decide whether to trump or slough.
            trump_options = [card for card in valid_cards_fallback if get_effective_suit(card, trump_suit_fallback) == trump_suit_fallback]
            non_trump_options = [card for card in valid_cards_fallback if get_effective_suit(card, trump_suit_fallback) != trump_suit_fallback]

            # Check if already winning trick or if trick is already trumped higher
            current_trick_winning_info = determine_trick_winner_so_far(game_data["trick_cards"], trump_suit_fallback, lead_suit_fallback)

            can_win_with_trump = False
            best_trump_to_play = None

            if trump_options:
                winning_trumps = []
                for trump_card in trump_options:
                    # Simulate playing this trump card
                    hypothetical_trick_cards = game_data["trick_cards"] + [{'player': ai_player_idx, 'card': trump_card}]
                    # The lead suit for evaluation should be the original lead suit of the trick,
                    # or if no one has played yet (lead_suit_fallback is None), it would be trump.
                    eval_lead_suit = lead_suit_fallback if lead_suit_fallback else trump_suit_fallback
                    if determine_trick_winner(hypothetical_trick_cards, trump_suit_fallback, eval_lead_suit) == ai_player_idx:
                        winning_trumps.append(trump_card)

                if winning_trumps:
                    can_win_with_trump = True
                    # Play the highest value trump if trying to win (original behavior)
                    winning_trumps.sort(key=lambda c: get_card_value(c, trump_suit_fallback, lead_suit_fallback), reverse=True)
                    best_trump_to_play = winning_trumps[0]
                    # Alternative: play lowest necessary trump to win:
                    # winning_trumps.sort(key=lambda c: get_card_value(c, trump_suit_fallback, lead_suit_fallback))
                    # best_trump_to_play = winning_trumps[0]


            if can_win_with_trump and best_trump_to_play:
                 # Consider if it's worth trumping (e.g. partner is winning, points needed, etc.)
                 # For simple heuristic: if can win with trump, do it with the best trump.
                card_to_play = best_trump_to_play
                logging.info(f"AI P{ai_player_idx} (Fallback Heuristic, Trumping) playing {str(card_to_play)}")
            elif non_trump_options:
                # Cannot win with trump (or no trump) or chooses not to, and has non-trump cards to slough.
                # Slough the lowest value non-trump card.
                non_trump_options.sort(key=lambda c: get_card_value(c, trump_suit_fallback, lead_suit_fallback)) # Sort ascending
                card_to_play = non_trump_options[0]
                logging.info(f"AI P{ai_player_idx} (Fallback Heuristic, Sloughing Non-Trump) playing lowest: {str(card_to_play)}")
            elif trump_options: # Has only trump cards left, but none can win (or chose not to win with trump)
                # Slough the lowest value trump card.
                trump_options.sort(key=lambda c: get_card_value(c, trump_suit_fallback, lead_suit_fallback)) # Sort ascending
                card_to_play = trump_options[0]
                logging.info(f"AI P{ai_player_idx} (Fallback Heuristic, Sloughing Trump) playing lowest: {str(card_to_play)}")
            else:
                # Should not happen if valid_cards_fallback was not empty. This means no valid cards.
                # The initial check for empty valid_cards_fallback should catch this.
                # If it somehow reaches here, play the first valid card as a last resort.
                card_to_play = valid_cards_fallback[0]
                logging.error(f"AI P{ai_player_idx} (Fallback Heuristic) in unexpected state. Playing first valid: {str(card_to_play)}")

        if not card_to_play: # Should be set by logic above if valid_cards_fallback is not empty
            logging.error(f"AI P{ai_player_idx} (Fallback) - card_to_play is None unexpectedly. Defaulting to first valid card.")
            card_to_play = valid_cards_fallback[0]

        # logging.info(f"AI P{ai_player_idx} (Fallback Heuristic) playing {str(card_to_play)}") # Covered by more specific logs above
        # --- End of Fallback Heuristic ---
    else:
        # --- RL Agent Logic ---
        state_dict = get_rl_state(ai_player_idx, game_data)
        # Store state and action for future learning update
        game_data["rl_training_data"][ai_player_idx] = {"state": state_dict, "action": None, "action_type": "play_card"}

        ai_hand = game_data["hands"][ai_player_idx] # Get current hand
        lead_suit = game_data["current_trick_lead_suit"]
        trump_suit = game_data["trump_suit"]

        # Get valid actions (cards to play)
        valid_card_objects = get_valid_plays(list(ai_hand), lead_suit, trump_suit)
        if not valid_card_objects:
            game_data["message"] += f" Error: AI {ai_player_idx+1} (RL) has no valid cards to play."
            if len(game_data["trick_cards"]) < game_data["num_players"]: game_data["current_player_turn"] = (ai_player_idx + 1) % game_data["num_players"]
            return

        # Convert Card objects to dictionary representations for the agent
        valid_actions_for_agent = [card.to_dict() for card in valid_card_objects]

        chosen_action_dict = current_agent.choose_action(state_dict, valid_actions_for_agent)
        game_data["rl_training_data"][ai_player_idx]["action"] = chosen_action_dict # Store chosen action (dict form)


        if chosen_action_dict is None: # Should not happen if valid_actions_for_agent is not empty
            logging.error(f"RL Agent P{ai_player_idx} failed to choose an action. Playing first valid card.")
            card_to_play = valid_card_objects[0]
        else:
            # Find the actual Card object in hand that matches the chosen action dict
            card_to_play = next((c for c in ai_hand if c.suit == chosen_action_dict['suit'] and c.rank == chosen_action_dict['rank']), None)
            if not card_to_play:
                logging.error(f"RL Agent P{ai_player_idx} chose card {chosen_action_dict} not found in hand {[str(c) for c in ai_hand]}. Playing first valid.")
                card_to_play = valid_card_objects[0] # Fallback

        # Note: The chosen_action_dict (serialized form) is stored for learning.
        # card_to_play is the actual Card object for execution.

        # --- Heuristic Overlay for RL Agent Sloughing ---
        # Purpose: To guide the RL agent towards more "standard" sloughing behavior
        # (discarding the lowest-value, non-winning, off-suit card) if it chooses to slough.
        # This helps prevent the agent from learning to discard valuable cards unnecessarily
        # when it's not following suit and not trying to win the trick.
        # It does not override decisions to follow suit or to win with trump.
        can_follow_lead_suit_rl = False
        if lead_suit: # lead_suit is current_trick_lead_suit
            can_follow_lead_suit_rl = any(get_effective_suit(c, trump_suit) == lead_suit for c in ai_hand)

        if not can_follow_lead_suit_rl and card_to_play: # Cannot follow suit and has chosen a card
            chosen_card_effective_suit = get_effective_suit(card_to_play, trump_suit)

            # Condition: Chosen card is not trump AND there are other non-trump cards it could have played
            if chosen_card_effective_suit != trump_suit:
                # Is the trick already won by someone else with a higher card or trump that card_to_play cannot beat?
                # Or, more simply, is the AI trying to slough? This is implied by not following suit and not playing trump (or playing a trump that won't win).

                # Let's check if the chosen card can win. If it can, the agent might be trying to win.
                # If it cannot win, and it's an off-suit play, then it's a slough.
                hypothetical_trick_after_play = game_data["trick_cards"] + [{'player': ai_player_idx, 'card': card_to_play}]
                eval_lead_suit_for_rl = lead_suit if lead_suit else chosen_card_effective_suit
                is_chosen_card_winning = determine_trick_winner(hypothetical_trick_after_play, trump_suit, eval_lead_suit_for_rl) == ai_player_idx

                if not is_chosen_card_winning: # If the chosen card does not win the trick (it's a slough or a failed trump attempt)
                    # And the chosen card is not trump (already checked by chosen_card_effective_suit != trump_suit)
                    # Now, find the actual lowest value non-trump card it *could* have played.

                    # valid_card_objects contains all legally playable cards in this situation.
                    # Filter these to non-trump cards.
                    sloughable_cards = [c for c in valid_card_objects if get_effective_suit(c, trump_suit) != trump_suit]

                    if sloughable_cards:
                        sloughable_cards.sort(key=lambda c: get_card_value(c, trump_suit, lead_suit)) # Sort ascending by value
                        lowest_value_slough_card = sloughable_cards[0]

                        # If the agent chose a higher value slough card than the absolute lowest available slough card
                        if get_card_value(card_to_play, trump_suit, lead_suit) > get_card_value(lowest_value_slough_card, trump_suit, lead_suit):
                            original_choice_log = str(card_to_play)
                            card_to_play = lowest_value_slough_card # Override
                                                        # Also update the action in rl_training_data for accurate learning if we were to use this post-decision state.
                            # However, the agent already made its Q-value update based on chosen_action_dict.
                            # This override is purely for play execution. For learning, this means the agent might
                            # not directly learn this heuristic unless rewards reflect it.
                            # For now, we accept this discrepancy: agent learns from its choice, but execution is overridden.
                            # game_data["rl_training_data"][ai_player_idx]["action"] = card_to_play.to_dict() # Update action to reflect override
                            logging.info(f"AI P{ai_player_idx} (RL Agent Heuristic Override): Original choice {original_choice_log} was high slough. Overridden to play lowest slough: {str(card_to_play)}")

        logging.info(f"AI P{ai_player_idx} (RL Agent) chose to play: {str(card_to_play)} (Raw action from agent: {chosen_action_dict})")
        # --- End of RL Agent Logic (with Heuristic Overlay) ---

    # Common logic to execute the play
    ai_hand = game_data["hands"][ai_player_idx] # Re-fetch, as it might be a copy above
    actual_card_to_remove_from_hand = next((c for c in ai_hand if c.suit == card_to_play.suit and c.rank == card_to_play.rank), None)
    if not actual_card_to_remove_from_hand:
        # This can happen if card_to_play is from valid_cards (a copy) and not the original hand object.
        # Find the equivalent card in hand.
        logging.warning(f"AI P{ai_player_idx} couldn't find exact object {str(card_to_play)}. Searching by val in hand: {[str(c) for c in ai_hand]}.")
        # Re-select from valid_cards to ensure it's an object that *could* be in hand if logic was perfect.
        # Then find that card in the actual hand.
        target_card_val = card_to_play # Keep the properties of the chosen card.
        card_to_play = next((h_card for h_card in ai_hand if h_card.suit == target_card_val.suit and h_card.rank == target_card_val.rank), None)
        if card_to_play: # Found it by value in hand.
            actual_card_to_remove_from_hand = card_to_play
            logging.info(f"Found card {str(actual_card_to_remove_from_hand)} by value in hand.")
        else: # Still not found, this is a bigger issue.
            logging.error(f"AI P{ai_player_idx} CRITICALLY tried to play {str(target_card_val)} but not found in hand by value: {[str(c) for c in ai_hand]}. Playing first valid from list as last resort.")
            if valid_cards: # valid_cards was defined in the RL Agent Logic block or fallback
                card_to_play = valid_cards[0]
                actual_card_to_remove_from_hand = next((c for c in ai_hand if c.suit == card_to_play.suit and c.rank == card_to_play.rank), None)
                if not actual_card_to_remove_from_hand: # If even the first valid card isn't in hand, something is deeply wrong
                    logging.error(f"CRITICAL: First valid card {str(card_to_play)} also not in hand. Aborting AI play for P{ai_player_idx}.")
                    return
            else: # No valid cards, already handled, but as a safeguard:
                logging.error(f"CRITICAL: No valid cards and logic failed to find card in hand for P{ai_player_idx}.")
                return

    if actual_card_to_remove_from_hand:
        try:
            ai_hand.remove(actual_card_to_remove_from_hand)
            logging.info(f"AI P{ai_player_idx} successfully removed {str(actual_card_to_remove_from_hand)} from hand.")
            # Record the played card for round tracking
            game_data.get("played_cards_this_round", []).append(actual_card_to_remove_from_hand) # Use the actual card object removed
        except ValueError:
            logging.error(f"CRITICAL: Failed to remove {str(actual_card_to_remove_from_hand)} from AI P{ai_player_idx}'s hand. Hand: {[str(c) for c in ai_hand]}")
            return # Avoid proceeding with inconsistent state
    else: # Should be caught by earlier checks, but as a final safeguard
        logging.error(f"CRITICAL: AI P{ai_player_idx} somehow has no actual_card_to_remove_from_hand. Chosen: {str(card_to_play)}. Hand: {[str(c) for c in ai_hand]}"); return

    game_data["trick_cards"].append({'player': ai_player_idx, 'card': card_to_play}) # Use the card object that was chosen and confirmed in hand
    # Message now directly uses player_identities which includes "(AI)" or "(You)"
    game_data["message"] = f"{game_data['player_identities'][ai_player_idx]} played {str(card_to_play)}."

    # Set current_trick_lead_suit if it's not set (i.e., this is the first card of the trick)
    if not game_data["current_trick_lead_suit"]:
        game_data["current_trick_lead_suit"] = get_effective_suit(card_to_play, game_data["trump_suit"])
        logging.info(f"AI P{ai_player_idx} (as first player in trick) set lead suit to {game_data['current_trick_lead_suit']} with card {str(card_to_play)}")

    num_players_in_trick = len(game_data["trick_cards"]); expected_cards_in_trick = game_data["num_players"]
    if num_players_in_trick == expected_cards_in_trick:
        # Pass the now definitive lead suit for the trick
        winner_idx = determine_trick_winner(game_data["trick_cards"], game_data["trump_suit"], game_data["current_trick_lead_suit"])
        game_data["round_tricks_won"][winner_idx] += 1
        game_data["last_completed_trick"] = {"played_cards": [tc.copy() for tc in game_data["trick_cards"]], "winner_player_idx": winner_idx, "winner_name": game_data['player_identities'][winner_idx]}

        # --- RL Update for all AI players who had pending actions for this trick ---
        # The event_data provides context about the trick's outcome.
        trick_event_data = {"trick_winner_idx": winner_idx, "played_cards_in_trick": game_data["last_completed_trick"]["played_cards"]}
        for p_id in game_data["player_identities"].keys():
            if p_id != 0 and game_data["rl_training_data"].get(p_id) and game_data["rl_training_data"][p_id].get("action_type") == "play_card":
                # This implies P_id made a play_card decision that we have stored state/action for.
                process_rl_update(p_id, "trick_end", event_data=trick_event_data)
        # --- End RL Update ---

        logging.info(f"Trick completed. Winner: P{winner_idx}. Storing last_completed_trick: {game_data['last_completed_trick']}")
        game_data["message"] += f" {game_data['player_identities'][winner_idx]} wins the trick."
        game_data["trick_cards"] = []; game_data["current_trick_lead_suit"] = None; game_data["current_player_turn"] = winner_idx; game_data["trick_leader"] = winner_idx

        if is_round_over():
            score_round() # score_round will handle round_end and game_end RL updates
            return
        else:
            game_data["message"] += f" {game_data['player_identities'][winner_idx]} leads next trick."
            # If the next player is AI, their turn will be processed by the game loop or API call.
            # No immediate call to process_ai_play_card here, as the update for the just-finished action needs to complete.
    else:
        game_data["current_player_turn"] = (ai_player_idx + 1) % game_data["num_players"]
        game_data["message"] += f" Next turn: {game_data['player_identities'][game_data['current_player_turn']]}."

def get_valid_plays(hand, lead_suit, trump_suit):
    if not lead_suit: return list(hand)
    can_follow_suit = any(get_effective_suit(card, trump_suit) == lead_suit for card in hand)
    if can_follow_suit: return [card for card in hand if get_effective_suit(card, trump_suit) == lead_suit]
    else: return list(hand)

# Helper to determine current winner of a potentially incomplete trick
def determine_trick_winner_so_far(trick_cards_played, trump_suit, lead_suit_of_trick):
    if not trick_cards_played: return None # Return None instead of -1 for "no winner yet" / "no cards played"

    winning_player_idx = -1
    highest_value_card_obj = None

    # The lead_suit_of_trick is passed in, as game_data["current_trick_lead_suit"] might not be set if the first player hasn't finished playing.
    # If lead_suit_of_trick is None (e.g. first card of trick being considered), then the first card played sets it.
    actual_lead_suit_for_evaluation = lead_suit_of_trick
    if not actual_lead_suit_for_evaluation and trick_cards_played:
        actual_lead_suit_for_evaluation = get_effective_suit(trick_cards_played[0]['card'], trump_suit)

    for play in trick_cards_played:
        current_player = play['player']
        current_card = play['card']
        current_card_effective_suit = get_effective_suit(current_card, trump_suit)
        current_card_value = get_card_value(current_card, trump_suit, actual_lead_suit_for_evaluation) # Pass lead suit for context

        if highest_value_card_obj is None:
            highest_value_card_obj = current_card
            winning_player_idx = current_player
        else:
            highest_value_card_effective_suit = get_effective_suit(highest_value_card_obj, trump_suit)
            highest_value_card_value_for_comparison = get_card_value(highest_value_card_obj, trump_suit, actual_lead_suit_for_evaluation)

            is_current_card_trump = (current_card_effective_suit == trump_suit)
            is_highest_card_trump = (highest_value_card_effective_suit == trump_suit)

            if is_current_card_trump and not is_highest_card_trump:
                highest_value_card_obj = current_card
                winning_player_idx = current_player
            elif is_current_card_trump and is_highest_card_trump:
                if current_card_value > highest_value_card_value_for_comparison:
                    highest_value_card_obj = current_card
                    winning_player_idx = current_player
            elif not is_current_card_trump and not is_highest_card_trump: # Neither are trump
                if current_card_effective_suit == actual_lead_suit_for_evaluation and highest_value_card_effective_suit != actual_lead_suit_for_evaluation:
                    # Current card followed lead, previous high did not (e.g. sloughed off-suit)
                    highest_value_card_obj = current_card
                    winning_player_idx = current_player
                elif current_card_effective_suit == actual_lead_suit_for_evaluation and highest_value_card_effective_suit == actual_lead_suit_for_evaluation:
                    # Both followed lead suit
                    if current_card_value > highest_value_card_value_for_comparison:
                        highest_value_card_obj = current_card
                        winning_player_idx = current_player
                # If current card is off-suit and highest is on lead suit, current does not win
                # If both are off-suit and not lead, neither wins over a lead-suit card (handled by initial lead_suit_of_trick check)

    if winning_player_idx != -1:
        return {'player': winning_player_idx, 'card': highest_value_card_obj}
    return None

def determine_trick_winner(trick_cards_played, trump_suit, lead_suit_of_trick=None): # Added lead_suit_of_trick parameter
    if not trick_cards_played: return -1 # Keep -1 for final determination if something went wrong

    # If lead_suit_of_trick is not provided, determine it from the first card.
    # This maintains compatibility with old calls but allows override for clarity.
    if lead_suit_of_trick is None:
        if not trick_cards_played: return -1 # Should not happen if called after cards are played.
        lead_card_obj = trick_cards_played[0]['card']
        lead_suit_of_trick = get_effective_suit(lead_card_obj, trump_suit)

    winner_info = determine_trick_winner_so_far(trick_cards_played, trump_suit, lead_suit_of_trick)
    return winner_info['player'] if winner_info else -1

def predict_maker_can_beat_card(maker_idx: int, target_card_to_beat: Card,
                                trump_suit: str, current_lead_suit: str | None,
                                game_data_copy: dict) -> bool:
    """
    Heuristically predicts if the maker is likely to beat a specific target card.
    This function is used by AI opponents to gauge if playing a high card might be
    over-trumped by the maker. It does NOT know the maker's actual hand.
    Prediction is based on:
    - Cards already played in the round.
    - General probabilities of remaining cards.
    - Maker's remaining hand size.

    This is a heuristic based on cards played and general probabilities.
    It does not know the maker's actual hand.

    Args:
        maker_idx: Player ID of the maker.
        target_card_to_beat: The card the opponent is considering playing, which they
                             hope the maker cannot beat.
        trump_suit: The current trump suit.
        current_lead_suit: The lead suit of the current trick (if any).
        game_data_copy: A copy of the current game state.

    Returns:
        True if the heuristic predicts the maker *might* beat the target card,
        False otherwise.
    """
    maker_hand_size = len(game_data_copy["hands"][maker_idx])
    if maker_hand_size == 0:
        return False # Maker has no cards left

    target_card_eff_suit = get_effective_suit(target_card_to_beat, trump_suit)
    target_card_value = get_card_value(target_card_to_beat, trump_suit, current_lead_suit)

    # 1. Can maker follow suit and play a higher card?
    if target_card_eff_suit != trump_suit and current_lead_suit == target_card_eff_suit : # Target card is of the lead suit (non-trump)
        # Check for higher cards of this suit potentially held by maker
        # This is hard without knowing maker's hand. A simple proxy:
        # If target card is not an Ace, assume maker *might* have a higher card of that suit.
        if target_card_to_beat.rank != 'A':
            logging.debug(f"Predict: Maker might beat {str(target_card_to_beat)} (non-trump lead) with higher in suit.")
            return True
        # If target is Ace of lead suit, only trump can beat it.

    # 2. Can maker play trump?
    #    a. If target is not trump, any trump beats it.
    #    b. If target is trump, maker needs higher trump.

    # Get all cards played so far this round to estimate remaining cards
    played_cards_this_round = set()
    # from game_data_copy["last_completed_trick"] and game_data_copy["trick_cards"]
    if game_data_copy.get("last_completed_trick") and game_data_copy["last_completed_trick"].get("played_cards"):
        for tc_info in game_data_copy["last_completed_trick"]["played_cards"]:
            played_cards_this_round.add((tc_info['card']['suit'], tc_info['card']['rank']))
    for tc_info in game_data_copy.get("trick_cards", []): # Current trick's cards
        played_cards_this_round.add((tc_info['card'].suit, tc_info['card'].rank)) # card is Card object here

    # Cards in current AI's hand (who is calling this function) are also known
    # This function is called by an AI player, so game_data_copy["hands"][current_ai_player_idx] are known.
    # However, this function is generic for the maker, so we don't use current AI's hand directly here for maker's probability.

    if target_card_eff_suit != trump_suit:
        # Maker can beat non-trump target by playing any trump.
        # Probability: Does maker have *any* trump?
        # Count remaining trump cards not in played_cards_this_round
        num_unseen_trump = 0
        potential_trumps = [Card(s, r) for s in SUITS for r in RANKS if get_effective_suit(Card(s,r), trump_suit) == trump_suit]
        for pt_card in potential_trumps:
            if (pt_card.suit, pt_card.rank) not in played_cards_this_round:
                num_unseen_trump +=1

        # Simplistic: if there are unseen trumps and maker has cards, they *might* have one.
        if num_unseen_trump > 0 and maker_hand_size > 0 :
             # More refined: if many trumps are out, and maker has few cards, less likely.
            if num_unseen_trump >= maker_hand_size or num_unseen_trump > 2: # Arbitrary: if at least 2-3 trumps are "available"
                logging.debug(f"Predict: Maker might beat {str(target_card_to_beat)} (non-trump) with a trump. Unseen trumps: {num_unseen_trump}")
                return True

    # Target card is trump. Maker needs a higher trump.
    if target_card_eff_suit == trump_suit:
        num_unseen_higher_trumps = 0
        potential_trumps = [Card(s, r) for s in SUITS for r in RANKS if get_effective_suit(Card(s,r), trump_suit) == trump_suit]
        for pt_card in potential_trumps:
            if (pt_card.suit, pt_card.rank) not in played_cards_this_round:
                if get_card_value(pt_card, trump_suit, current_lead_suit) > target_card_value:
                    num_unseen_higher_trumps += 1

        if num_unseen_higher_trumps > 0 and maker_hand_size > 0:
            # If the right/left bower or Ace of trump is still unseen and target isn't it, good chance.
            logging.debug(f"Predict: Maker might beat {str(target_card_to_beat)} (trump) with a higher trump. Unseen higher trumps: {num_unseen_higher_trumps}")
            return True

    # Default: Less certain maker can beat it.
    logging.debug(f"Predict: Maker less likely to beat {str(target_card_to_beat)} based on heuristics.")
    return False


def old_determine_trick_winner(trick_cards_played, trump_suit): # Keep old for reference during refactor if needed
    if not trick_cards_played: return -1
    winning_player = -1; highest_value_card = None
    lead_card_obj = trick_cards_played[0]['card'];ตัดสิน_suit_of_trick = get_effective_suit(lead_card_obj, trump_suit) # Original: lead_suit_of_trick
    for play in trick_cards_played:
        player = play['player']; card = play['card']
        card_effective_suit = get_effective_suit(card, trump_suit); card_value = get_card_value(card, trump_suit,ตัดสิน_suit_of_trick) # Pass lead suit
        if highest_value_card is None: highest_value_card = card; winning_player = player
        else:
            highest_value_card_effective_suit = get_effective_suit(highest_value_card, trump_suit)
            highest_value_card_value = get_card_value(highest_value_card, trump_suit)
            if card_effective_suit == trump_suit and highest_value_card_effective_suit != trump_suit: highest_value_card = card; winning_player = player
            elif card_effective_suit == trump_suit and highest_value_card_effective_suit == trump_suit:
                if card_value > highest_value_card_value: highest_value_card = card; winning_player = player
            elif card_effective_suit != trump_suit and highest_value_card_effective_suit != trump_suit:
                if card_effective_suit == lead_suit_of_trick and highest_value_card_effective_suit != lead_suit_of_trick: highest_value_card = card; winning_player = player
                elif card_effective_suit == lead_suit_of_trick and highest_value_card_effective_suit == lead_suit_of_trick:
                    if card_value > highest_value_card_value: highest_value_card = card; winning_player = player
    return winning_player

def is_round_over():
    """Checks if the current round has concluded (i.e., 5 tricks have been played)."""
    return sum(game_data["round_tricks_won"].values()) >= 5

def score_round():
    """
    Calculates and assigns points at the end of a round.
    Updates player scores based on whether the maker made their bid, got euchred,
    or achieved a march (all 5 tricks) or alone march.
    Also updates the game phase to "round_over" or "game_over" if applicable.
    Triggers RL updates for "round_end" or "game_end" events.
    """
    global game_data
    # Prevent re-scoring if round/game is already marked as over
    if game_data.get("game_phase") in ["round_over", "game_over"]:
        logging.warning(f"score_round() called when game_phase is already {game_data.get('game_phase')}. Points not re-awarded.")
        return

    maker = game_data["maker"]
    if maker is None: # Should not happen if round is properly conducted
        logging.error("CRITICAL: score_round() called but maker is None. Aborting scoring for this round.")
        game_data["message"] = "Error: Round scoring aborted due to missing maker."
        game_data["game_phase"] = "round_over" # Mark round as over to allow progression
        return

    maker_tricks = game_data["round_tricks_won"][maker]
    points_awarded = 0; message_suffix = ""; is_going_alone = game_data.get("going_alone", False)
    if maker_tricks < 3:
        points_awarded = 2
        for p_idx in range(game_data["num_players"]):
            if p_idx != maker: game_data["scores"][p_idx] += points_awarded
        message_suffix = f"Maker euchred! Opponents score {points_awarded} points each."
    elif maker_tricks == 5:
        points_awarded = 4 if is_going_alone else 2
        message_suffix = f"Maker ({game_data['player_identities'][maker]}) {'went alone and ' if is_going_alone else ''}marches, scoring {points_awarded} points!"
        game_data["scores"][maker] += points_awarded
    elif maker_tricks >= 3:
        points_awarded = 1
        message_suffix = f"Maker ({game_data['player_identities'][maker]}) {'went alone and ' if is_going_alone else ''}scores {points_awarded} point."
        game_data["scores"][maker] += points_awarded
    game_data["message"] = f"Round Over. {message_suffix}"

    # First, definitively set phase to round_over.
    logging.info(f"SCORE_ROUND: Current phase before update: {game_data.get('game_phase')}")
    game_data["game_phase"] = "round_over"
    logging.info(f"SCORE_ROUND: Phase updated to: {game_data.get('game_phase')}")

    # Check for game winner
    game_winner_found = False
    for p_idx, score in game_data["scores"].items():
        if score >= 10:
            logging.info(f"SCORE_ROUND: Game over condition met for P{p_idx}. Current phase before update: {game_data.get('game_phase')}")
            game_data["game_phase"] = "game_over"
            logging.info(f"SCORE_ROUND: Phase updated to: {game_data.get('game_phase')}")
            game_data["message"] += f" {game_data['player_identities'][p_idx]} wins the game!"
            # --- RL Update for Game End ---
            game_event_data = {"game_winner_idx": p_idx}
            for ai_p_id in rl_agents.keys(): # Update all AI agents
                process_rl_update(ai_p_id, "game_end", event_data=game_event_data)
            # --- End RL Update ---
            game_winner_found = True
            break # Exit loop once a winner is found and processed

    if game_winner_found:
        return # Game is over, no further round processing needed for RL here

    # If game is not over, it's just round_over. RL update for round end.
    # game_data["game_phase"] is already "round_over" if no winner was found.
    logging.info(f"SCORE_ROUND: Round ended, no game winner yet. Phase: {game_data.get('game_phase')}")
    # --- RL Update for Round End (if game not over) ---
    round_event_data = {"round_tricks_won": game_data["round_tricks_won"].copy(), "maker": maker, "going_alone": is_going_alone}
    for ai_p_id in rl_agents.keys(): # Update all AI agents based on round outcome
        process_rl_update(ai_p_id, "round_end", event_data=round_event_data)
    # --- End RL Update ---


# --- Flask Routes ---
@app.route('/')
def index_route(): return render_template('index.html')
@app.route('/style.css')
def styles_route(): return send_from_directory('.', 'style.css')
@app.route('/script.js')
def scripts_route(): return send_from_directory('.', 'script.js')

@app.route('/api/start_game', methods=['GET'])
def start_game_api():
    """
    API endpoint to start a new game or a new round within an existing game.
    If the game is over or in setup, it initializes global game data.
    It then sets the dealer for the new round and initializes the round data
    (dealing cards, setting up-card, etc.).
    If the first player to bid is an AI, it triggers the AI's bidding process.
    Returns the full game state as JSON.
    """
    global game_data
    # Use .get() for safer access to game_phase. If key is missing, treat as "setup".
    current_phase = game_data.get("game_phase")
    if current_phase is None or current_phase in ["game_over", "setup"]:
        initialize_game_data()
        # initialize_game_data sets dealer randomly, so no need to set it again here
        # game_data["dealer"] = random.randint(0, game_data["num_players"] - 1) # This line is now redundant
    else:
        game_data["dealer"] = (game_data["dealer"] + 1) % game_data["num_players"]
    initialize_new_round()
    if game_data["game_phase"] == "bidding_round_1" and game_data["current_player_turn"] != 0:
        logging.info(f"Starting game/new round: P{game_data['current_player_turn']} (AI) to start bidding_round_1.")
        process_ai_bid_action({'player_index': game_data["current_player_turn"], 'action': 'ai_bidding_round_1'})
    return jsonify(game_data_to_json(game_data))

@app.route('/api/submit_action', methods=['POST'])
def submit_action_api():
    """
    API endpoint for the human player (Player 0) to submit an action.
    Handles various actions based on the current game phase, such as:
    - Bidding: 'order_up', 'pass_bid', 'call_trump', 'pass_call'
    - Discarding: 'dealer_discard_one', 'maker_discard', 'dealer_must_discard_after_order_up'
    - Going Alone: 'choose_go_alone', 'choose_not_go_alone'
    - Playing Cards: 'play_card'

    It validates the action against the current game state and player turn.
    Updates the game state based on the action and then may trigger AI actions
    if it becomes an AI's turn.
    Returns the updated game state as JSON.
    """
    global game_data
    action_data = request.json
    player_idx = action_data.get('player_index')
    action_type = action_data.get('action')
    current_phase = game_data["game_phase"]
    logging.info(f"Received action: {action_type} from P{player_idx}. Current phase: {current_phase}. Current turn: P{game_data['current_player_turn']}.")
    logging.debug(f"Action data: {action_data}")

    # Player validation
    if current_phase == 'maker_discard':
        if player_idx != game_data.get("maker"):
            return jsonify({"error": "Only the maker can discard the dummy hand."}), 400
        if game_data["current_player_turn"] != player_idx: # Also ensure it's their turn
             return jsonify({"error": f"Not your turn to discard dummy hand (P{game_data['current_player_turn']+1}'s turn)."}), 400
    elif current_phase == 'prompt_go_alone':
        if player_idx != game_data.get("maker"):
            return jsonify({"error": "Only the maker can decide to go alone."}), 400
        if game_data["current_player_turn"] != player_idx: # Also ensure it's their turn
             return jsonify({"error": f"Not your turn to decide go alone (P{game_data['current_player_turn']+1}'s turn)."}), 400
    elif current_phase in ['dealer_discard_one', 'dealer_must_discard_after_order_up']:
        if player_idx != game_data["dealer"]:
            return jsonify({"error": f"Only the dealer (P{game_data['dealer']+1}) can discard in this phase."}), 400
        if game_data["current_player_turn"] != player_idx: # Dealer must also be current player
            return jsonify({"error": f"Not your turn to discard as dealer (P{game_data['current_player_turn']+1}'s turn)."}), 400
    elif game_data["current_player_turn"] != player_idx: # General turn check for other phases
        return jsonify({"error": f"Not your turn (P{game_data['current_player_turn']+1} vs P{player_idx+1}). Current phase: {current_phase}"}), 400


    # --- ACTION HANDLERS ---
    if action_type == 'order_up':
        if current_phase != 'bidding_round_1':
            logging.warning(f"Action 'order_up' by P{player_idx} rejected. Phase: {current_phase}, expected 'bidding_round_1'.")
            return jsonify({"error": "Cannot order up now."}), 400

        if not game_data.get("original_up_card_for_round"):
            logging.error("CRITICAL: original_up_card_for_round is None when 'order_up' is called.")
            return jsonify({"error": "Critical error: Up-card data missing."}), 500

        game_data["trump_suit"] = game_data["original_up_card_for_round"].suit
        game_data["maker"] = player_idx # player_idx is the one who ordered up, becomes maker
        logging.info(f"P{player_idx} ({game_data['player_identities'][player_idx]}) ordered up {SUITS_MAP[game_data['trump_suit']]}. Maker set to P{player_idx}. Trump: {game_data['trump_suit']}.")
        current_message = f"{game_data['player_identities'][player_idx]} ordered up {SUITS_MAP[game_data['trump_suit']]}."
        game_data["up_card_visible"] = False
        game_data["up_card"] = None

        # --- Dealer picks up card and discards ---
        current_dealer_idx = game_data["dealer"]
        dealer_hand = game_data["hands"][current_dealer_idx]
        up_card_to_pickup = game_data["original_up_card_for_round"]

        dealer_hand.append(up_card_to_pickup)
        logging.info(f"Dealer P{current_dealer_idx} ({game_data['player_identities'][current_dealer_idx]}) picked up {str(up_card_to_pickup)}. Hand size now: {len(dealer_hand)}.")
        current_message += f" {game_data['player_identities'][current_dealer_idx]} (Dealer) picked up {str(up_card_to_pickup)}."

        # Dealer discards one card
        if current_dealer_idx == 0: # Human Dealer
            game_data["cards_to_discard_count"] = 1
            game_data["current_player_turn"] = current_dealer_idx # Dealer's turn to discard
            if player_idx == current_dealer_idx: # Human dealer ordered themselves up
                game_data["game_phase"] = "dealer_discard_one"
            else: # Another player ordered up, Human dealer (who is not the maker) must discard
                game_data["game_phase"] = "dealer_must_discard_after_order_up"
            current_message += f" {game_data['player_identities'][current_dealer_idx]} (Dealer) must discard 1 card."
            logging.info(f"Human Dealer P{current_dealer_idx} to discard 1. Phase: {game_data['game_phase']}. Maker: P{game_data['maker']}.")
            game_data["message"] = current_message # Update message here for human dealer

        else: # AI Dealer
            logging.info(f"AI Dealer P{current_dealer_idx} ({game_data['player_identities'][current_dealer_idx]}) must discard 1. Maker: P{game_data['maker']}. Trump: {game_data['trump_suit']}.")
            if not dealer_hand:
                logging.error(f"CRITICAL: AI Dealer P{current_dealer_idx} has empty hand after pickup.")
                game_data["current_player_turn"] = game_data["maker"]
                game_data["game_phase"] = "prompt_go_alone"
                game_data["message"] = current_message + f" Critical error with AI dealer hand. {game_data['player_identities'][game_data['maker']]} (Maker) to decide go alone."
                return jsonify(game_data_to_json(game_data))

            cards_to_discard_ai = get_ai_cards_to_discard(list(dealer_hand), 1, game_data["trump_suit"])
            card_to_discard_obj = None
            if not cards_to_discard_ai:
                logging.error(f"AI Dealer P{current_dealer_idx} (hand: {[str(c) for c in dealer_hand]}) failed to select card to discard. Trump: {game_data['trump_suit']}. Discarding first card as fallback.")
                if dealer_hand: card_to_discard_obj = dealer_hand[0]
            else:
                card_to_discard_obj = cards_to_discard_ai[0]

            if card_to_discard_obj:
                try:
                    dealer_hand.remove(card_to_discard_obj)
                    logging.info(f"AI Dealer P{current_dealer_idx} discarded ({str(card_to_discard_obj)}). Hand size now: {len(dealer_hand)}.")
                    current_message += f" {game_data['player_identities'][current_dealer_idx]} (AI Dealer) discarded {str(card_to_discard_obj)}."
                except ValueError:
                    found_card = next((c for c in dealer_hand if c.rank == card_to_discard_obj.rank and c.suit == card_to_discard_obj.suit), None)
                    if found_card:
                        dealer_hand.remove(found_card)
                        logging.info(f"AI Dealer P{current_dealer_idx} discarded ({str(card_to_discard_obj)}) by rank/suit match. Hand size now: {len(dealer_hand)}.")
                        current_message += f" {game_data['player_identities'][current_dealer_idx]} (AI Dealer) discarded {str(card_to_discard_obj)}."
                    else:
                        logging.error(f"AI Dealer P{current_dealer_idx} CRITICALLY failed to find/remove card {str(card_to_discard_obj)} for discard. Hand: {[str(c) for c in dealer_hand]}.")
            else:
                 logging.error(f"AI Dealer P{current_dealer_idx} had no card to discard selected or hand was empty.")

            game_data["current_player_turn"] = game_data["maker"]
            game_data["game_phase"] = "prompt_go_alone"
            current_message += f" {game_data['player_identities'][game_data['maker']]} (Maker) to decide go alone."
            logging.info(f"AI Dealer P{current_dealer_idx} discard processed. Phase 'prompt_go_alone'. Turn for P{game_data['maker']} (Maker).")
            if game_data["maker"] != 0 : # If maker is AI, their logic in process_ai_bid_action would have called ai_decide_go_alone_and_proceed
                 logging.debug(f"AI Maker P{game_data['maker']} will proceed with 'go alone' decision based on game state (likely already triggered if AI ordered up).")

        game_data["message"] = current_message # Assign accumulated message at the end

    elif action_type == 'dealer_must_discard_after_order_up':
        if current_phase != 'dealer_must_discard_after_order_up' or player_idx != game_data["dealer"]:
            return jsonify({"error": "Not time/player for dealer to discard now."}), 400
        if player_idx != 0:
            return jsonify({"error": "dealer_must_discard_after_order_up is only for human player."}), 400

        cards_to_discard_dicts = action_data.get('cards', [])
        if len(cards_to_discard_dicts) != 1:
            return jsonify({"error": "Must discard exactly 1 card."}), 400

        dealer_hand = game_data["hands"][game_data["dealer"]]
        card_to_remove_dict = cards_to_discard_dicts[0]
        actual_card_to_remove = next((c for c in dealer_hand if c.rank == card_to_remove_dict['rank'] and c.suit == card_to_remove_dict['suit']), None)

        if not actual_card_to_remove:
            return jsonify({"error": "Card specified for discard not found in hand."}), 400

        dealer_hand.remove(actual_card_to_remove)
        game_data["message"] = f"{game_data['player_identities'][player_idx]} (Dealer) discarded 1 card. "
        logging.info(f"P{player_idx} (Dealer) discarded 1 in 'dealer_must_discard_after_order_up'. Hand size: {len(dealer_hand)}.")

        game_data["game_phase"] = "prompt_go_alone"
        game_data["cards_to_discard_count"] = 0
        game_data["current_player_turn"] = game_data["maker"]
        game_data["message"] += f"{game_data['player_identities'][game_data['maker']]} (Maker) to decide go alone."
        logging.info(f"Phase changed to prompt_go_alone for P{game_data['maker']} (Maker).")
        if game_data["maker"] != 0:
            logging.info(f"Human dealer P{player_idx} discarded. AI Maker P{game_data['maker']}'s turn to decide go alone. Calling ai_decide_go_alone_and_proceed.")
            ai_decide_go_alone_and_proceed(game_data["maker"])


    elif action_type == 'pass_bid':
        if current_phase != 'bidding_round_1':
            return jsonify({"error": "Cannot pass bid now."}), 400
        is_ai_action = (player_idx != 0)
        game_data["message"] = f"{game_data['player_identities'][player_idx]}{' (AI)' if is_ai_action else ''} passes."
        game_data['passes_on_upcard'].append(player_idx)
        logging.info(f"P{player_idx}{' (AI)' if is_ai_action else ''} passed R1. Passes: {len(game_data['passes_on_upcard'])}/{game_data['num_players']}.")
        if len(game_data['passes_on_upcard']) == game_data["num_players"]:
            game_data["game_phase"] = "bidding_round_2"; game_data["up_card_visible"] = False
            game_data["current_player_turn"] = (game_data["dealer"] + 1) % game_data["num_players"]
            game_data["message"] += f" Up-card turned. {game_data['player_identities'][game_data['current_player_turn']]}'s turn to call."
            game_data['passes_on_calling'] = []
            if game_data["current_player_turn"] != 0:
                process_ai_bid_action({'player_index': game_data["current_player_turn"], 'action': 'ai_bidding_round_2'})
        else:
            bid_order = [(game_data["dealer"] + i + 1) % game_data["num_players"] for i in range(game_data["num_players"])]
            current_bidder_index_in_order = bid_order.index(player_idx)
            game_data["current_player_turn"] = bid_order[(current_bidder_index_in_order + 1) % game_data["num_players"]]
            game_data["message"] += f" {game_data['player_identities'][game_data['current_player_turn']]}'s turn."
            if game_data["current_player_turn"] != 0:
                 process_ai_bid_action({'player_index': game_data["current_player_turn"], 'action': 'ai_bidding_round_1'})

    elif action_type == 'call_trump':
        if current_phase not in ['bidding_round_2', 'dealer_must_call']:
            return jsonify({"error": "Cannot call now."}), 400
        chosen_suit = action_data.get('suit')
        if not chosen_suit or chosen_suit not in SUITS:
            return jsonify({"error": "Invalid suit."}), 400
        turned_down_suit = game_data["original_up_card_for_round"].suit
        if current_phase == 'bidding_round_2' and chosen_suit == turned_down_suit:
            return jsonify({"error": f"Cannot call turned down suit ({SUITS_MAP[turned_down_suit]})."}), 400
        game_data["trump_suit"] = chosen_suit; game_data["maker"] = player_idx
        game_data["message"] = f"{game_data['player_identities'][player_idx]} called {SUITS_MAP[chosen_suit]}."
        game_data["up_card_visible"] = False; game_data["up_card"] = None
        game_data["current_player_turn"] = game_data["maker"]; game_data["game_phase"] = "prompt_go_alone"
        game_data["message"] += f" {game_data['player_identities'][game_data['maker']]} to decide go alone."
        if game_data["maker"] != 0:
            logging.info(f"AI P{game_data['maker']} (Maker after calling) to decide go alone. Calling ai_decide_go_alone_and_proceed.")
            ai_decide_go_alone_and_proceed(game_data["maker"])


    elif action_type == 'dealer_discard_one':
        if current_phase != 'dealer_discard_one' or player_idx != game_data["dealer"] or player_idx != game_data["maker"]:
            return jsonify({"error": "Not time/player for dealer_discard_one."}), 400
        if player_idx != 0:
            return jsonify({"error": "dealer_discard_one is only for human player when they are maker."}), 400
        cards_to_discard_dicts = action_data.get('cards', [])
        if len(cards_to_discard_dicts) != 1:
            return jsonify({"error": "Must discard exactly 1 card."}), 400
        dealer_hand = game_data["hands"][game_data["dealer"]]
        card_to_remove_dict = cards_to_discard_dicts[0]
        actual_card_to_remove = next((c for c in dealer_hand if c.rank == card_to_remove_dict['rank'] and c.suit == card_to_remove_dict['suit']), None)
        if not actual_card_to_remove: return jsonify({"error": "Card specified for discard not found in hand."}), 400
        dealer_hand.remove(actual_card_to_remove)
        game_data["message"] = f"{game_data['player_identities'][player_idx]} discarded 1 card. "
        game_data["game_phase"] = "prompt_go_alone"; game_data["cards_to_discard_count"] = 0
        game_data["current_player_turn"] = game_data["maker"]
        game_data["message"] += "Choose to go alone or play with partner."
        # No explicit AI call needed here as player_idx is human maker.

    elif action_type == 'maker_discard_one':
        logging.warning(f"Deprecated 'maker_discard_one' action called by P{player_idx}.")
        return jsonify({"error": "Deprecated action 'maker_discard_one'."}), 400

    elif action_type == 'pass_call':
        if player_idx == game_data["dealer"] and current_phase == 'dealer_must_call':
            return jsonify({"error": "Dealer must call."}), 400
        if current_phase != 'bidding_round_2':
            return jsonify({"error": "Cannot pass call now."}), 400
        game_data["message"] = f"{game_data['player_identities'][player_idx]} passes."
        game_data['passes_on_calling'].append(player_idx)
        logging.info(f"P{player_idx} passed R2. Passes: {len(game_data['passes_on_calling'])}/{game_data['num_players']-1}.")
        bid_order_r2 = [(game_data["dealer"] + i + 1) % game_data["num_players"] for i in range(game_data["num_players"])]
        if len(game_data['passes_on_calling']) == game_data["num_players"] - 1:
            if all(p == game_data["dealer"] or p in game_data["passes_on_calling"] for p in range(game_data["num_players"])) and game_data["dealer"] not in game_data["passes_on_calling"]:
                game_data["current_player_turn"] = game_data["dealer"]; game_data["game_phase"] = "dealer_must_call"
                game_data["message"] += f" Dealer ({game_data['player_identities'][game_data['dealer']]}) is stuck."
                if game_data["dealer"] != 0:
                    process_ai_bid_action({'player_index': game_data["dealer"], 'action': 'ai_dealer_stuck_call'})
                return jsonify(game_data_to_json(game_data))
        current_passer_index_in_order = bid_order_r2.index(player_idx)
        game_data["current_player_turn"] = bid_order_r2[(current_passer_index_in_order + 1) % game_data["num_players"]]
        game_data["message"] += f" {game_data['player_identities'][game_data['current_player_turn']]}'s turn."
        if game_data["current_player_turn"] != 0:
            process_ai_bid_action({'player_index': game_data["current_player_turn"], 'action': 'ai_bidding_round_2'})

    elif action_type == 'maker_discard':
        if current_phase != 'maker_discard' or player_idx != game_data["maker"]:
            return jsonify({"error": "Not time/player for maker discard."}), 400
        cards_to_discard_dicts = action_data.get('cards', [])
        if len(cards_to_discard_dicts) != game_data["cards_to_discard_count"]:
            return jsonify({"error": f"Must discard {game_data['cards_to_discard_count']} cards."}), 400
        maker_hand = game_data["hands"][game_data["maker"]]
        temp_hand_for_removal = list(maker_hand); cards_removed_success = True
        for card_dict in cards_to_discard_dicts:
            found_card_in_temp = next((c for c in temp_hand_for_removal if c.rank == card_dict['rank'] and c.suit == card_dict['suit']), None)
            if found_card_in_temp:
                temp_hand_for_removal.remove(found_card_in_temp)
                actual_card_to_remove_from_real_hand = next((c for c in maker_hand if c.rank == card_dict['rank'] and c.suit == card_dict['suit']), None)
                if actual_card_to_remove_from_real_hand: maker_hand.remove(actual_card_to_remove_from_real_hand)
                else: cards_removed_success = False; break
            else: cards_removed_success = False; break
        if not cards_removed_success: return jsonify({"error": "Error processing discards."}), 400
        game_data["message"] = f"{game_data['player_identities'][player_idx]} discarded {len(cards_to_discard_dicts)} cards."
        game_data["cards_to_discard_count"] = 0; transition_to_play_phase()

    elif action_type == 'choose_go_alone' or action_type == 'choose_not_go_alone':
        if current_phase != 'prompt_go_alone' or player_idx != game_data["maker"]:
            return jsonify({"error": "Not time/player to choose go alone."}), 400
        chose_to_go_alone = (action_type == 'choose_go_alone')
        game_data["going_alone"] = chose_to_go_alone
        game_data["message"] = f"{game_data['player_identities'][player_idx]} (Maker) chose to {'go alone' if chose_to_go_alone else 'play with dummy hand'}."
        if chose_to_go_alone:
            game_data["dummy_hand"] = []; transition_to_play_phase()
        else: # Chose NOT to go alone
            maker_hand = game_data["hands"][game_data["maker"]]
            if game_data.get("dummy_hand") and len(game_data["dummy_hand"]) == 5 : # Ensure dummy hand is valid
                maker_hand.extend(game_data["dummy_hand"]); game_data["dummy_hand"] = []
                game_data["cards_to_discard_count"] = 5; game_data["game_phase"] = "maker_discard"
                game_data["current_player_turn"] = game_data["maker"]
                game_data["message"] += " Picked up dummy hand. Now discard 5 cards."
                if game_data["maker"] != 0: # AI Maker not going alone
                    logging.info(f"AI P{game_data['maker']} not going alone. Calling ai_discard_five_cards.")
                    ai_discard_five_cards(game_data["maker"])
            else:
                logging.error(f"Maker P{player_idx} chose not to go alone, but dummy_hand invalid or missing. Proceeding as if going alone.")
                game_data["going_alone"] = True # Fallback to going alone
                game_data["message"] += " Error with dummy hand, proceeding as if going alone."
                transition_to_play_phase()


    elif action_type == 'play_card':
        if current_phase != 'playing_tricks': return jsonify({"error": "Cannot play card now."}), 400
        if player_idx != game_data["current_player_turn"]: return jsonify({"error": "Not your turn to play."}), 400
        card_data = action_data.get('card');
        if not card_data: return jsonify({"error": "No card data provided."}), 400
        player_hand = game_data["hands"][player_idx]
        played_card = next((c for c in player_hand if c.rank == card_data['rank'] and c.suit == card_data['suit']), None)
        if not played_card: return jsonify({"error": "Card not in hand."}), 400
        valid_plays = get_valid_plays(player_hand, game_data["current_trick_lead_suit"], game_data["trump_suit"])
        if played_card not in valid_plays:
            lead_suit_name = SUITS_MAP.get(game_data["current_trick_lead_suit"]) if game_data["current_trick_lead_suit"] else ""
            required_suit_msg = f"Must follow suit ({lead_suit_name})." if game_data["current_trick_lead_suit"] and any(c.suit == game_data["current_trick_lead_suit"] for c in player_hand) else "Invalid play."
            return jsonify({"error": f"Invalid play. {required_suit_msg}"}), 400

        player_hand.remove(played_card)
        game_data.get("played_cards_this_round", []).append(played_card) # Record played card

        game_data["trick_cards"].append({'player': player_idx, 'card': played_card})
        game_data["message"] = f"{game_data['player_identities'][player_idx]} played {str(played_card)}."
        if not game_data["current_trick_lead_suit"]: game_data["current_trick_lead_suit"] = get_effective_suit(played_card, game_data["trump_suit"])
        num_players_in_trick = len(game_data["trick_cards"]); expected_cards_in_trick = game_data["num_players"]
        if num_players_in_trick == expected_cards_in_trick:
            winner_idx = determine_trick_winner(game_data["trick_cards"], game_data["trump_suit"], game_data["current_trick_lead_suit"]) # Pass lead suit
            game_data["round_tricks_won"][winner_idx] += 1
            game_data["last_completed_trick"] = {"played_cards": [tc.copy() for tc in game_data["trick_cards"]], "winner_player_idx": winner_idx, "winner_name": game_data['player_identities'][winner_idx]}

            # --- RL Update for all AI players who had pending actions for this trick ---
            trick_event_data = {"trick_winner_idx": winner_idx, "played_cards_in_trick": game_data["last_completed_trick"]["played_cards"]}
            for p_id in game_data["player_identities"].keys():
                if p_id != 0 and game_data["rl_training_data"].get(p_id) and game_data["rl_training_data"][p_id].get("action_type") == "play_card":
                    process_rl_update(p_id, "trick_end", event_data=trick_event_data)
            # --- End RL Update ---

            game_data["message"] += f" {game_data['player_identities'][winner_idx]} wins the trick."
            game_data["trick_cards"] = []; game_data["current_trick_lead_suit"] = None
            game_data["current_player_turn"] = winner_idx; game_data["trick_leader"] = winner_idx
            if is_round_over():
                score_round() # score_round will handle its own RL updates
            else:
                game_data["message"] += f" {game_data['player_identities'][winner_idx]} leads next trick."
        else:
            game_data["current_player_turn"] = (player_idx + 1) % game_data["num_players"]
            game_data["message"] += f" Next turn: {game_data['player_identities'][game_data['current_player_turn']]}."

    # After any action, check if it's now an AI's turn to play a card or make a bidding/go_alone decision
    current_phase_after_action = game_data["game_phase"]
    current_player_after_action = game_data["current_player_turn"]

    if current_player_after_action != 0: # If it's an AI's turn
        if current_phase_after_action == "playing_tricks" and not is_round_over():
            # This check was already here, seems fine.
            # logging.info(f"Action by P{player_idx} resulted in AI P{current_player_after_action}'s turn to play. Processing AI play.")
            # process_ai_play_card(current_player_after_action)
            # Decided to let client poll for AI card plays to simplify server-side pushes.
            pass
        elif current_phase_after_action in ["bidding_round_1", "bidding_round_2", "dealer_must_call"] and \
             not (action_type in ['pass_bid', 'pass_call'] and player_idx == current_player_after_action): # Avoid re-triggering AI that just passed
            # This condition is to trigger AI if human action led to AI bidding turn.
            # AI to AI pass sequence is handled within process_ai_bid_action.
            # The 'pass' check is to prevent an AI that just passed from being immediately re-triggered by this block.
            # logging.info(f"Action by P{player_idx} resulted in AI P{current_player_after_action}'s turn in phase {current_phase_after_action}. Triggering AI bid action.")
            # process_ai_bid_action({'player_index': current_player_after_action, 'action': f'ai_{current_phase_after_action}'})
            # AI bidding actions are now more self-contained or triggered directly.
            pass
        elif current_phase_after_action == "prompt_go_alone" and current_player_after_action == game_data.get("maker"):
            # This was the main fix: if it's AI maker's turn for prompt_go_alone, ensure it's handled.
            # This is now handled directly within submit_action's discard handlers and AI's own order_up/call_trump logic.
            # logging.info(f"Action by P{player_idx} resulted in AI Maker P{current_player_after_action}'s turn for prompt_go_alone. Triggering decision.")
            # ai_decide_go_alone_and_proceed(current_player_after_action)
            pass


    return jsonify(game_data_to_json(game_data))

def ai_discard_five_cards(ai_maker_idx):
    global game_data #; time.sleep(0.5) # Removed AI delay
    ai_hand = game_data["hands"][ai_maker_idx]; trump_suit = game_data["trump_suit"]
    logging.info(f"AI P{ai_maker_idx} (Maker) discarding 5 cards from {len(ai_hand)}. Trump: {trump_suit}.")
    if len(ai_hand) != 10:
        logging.error(f"AI P{ai_maker_idx} in ai_discard_five_cards, hand size {len(ai_hand)} not 10.")
        num_to_discard_corrected = max(0, len(ai_hand) - 5)
        if len(ai_hand) <= 5: logging.error(f"AI hand too small ({len(ai_hand)}) to discard to 5."); transition_to_play_phase(); return
        logging.warning(f"Corrected num_to_discard to {num_to_discard_corrected}.")
        cards_to_discard_ai = get_ai_cards_to_discard(list(ai_hand), num_to_discard_corrected, trump_suit)
    else: cards_to_discard_ai = get_ai_cards_to_discard(list(ai_hand), 5, trump_suit)
    if not cards_to_discard_ai: logging.error(f"AI P{ai_maker_idx} selected no cards to discard."); transition_to_play_phase(); return
    discard_log = [str(c) for c in cards_to_discard_ai]; logging.info(f"AI P{ai_maker_idx} intends to discard: {discard_log}")
    for card_obj_to_discard in cards_to_discard_ai:
        actual_card_in_hand = next((c for c in ai_hand if c.suit == card_obj_to_discard.suit and c.rank == card_obj_to_discard.rank), None)
        if actual_card_in_hand: ai_hand.remove(actual_card_in_hand)
        else: logging.error(f"AI P{ai_maker_idx} failed to find {str(card_obj_to_discard)} in hand for discard_five.")
    logging.info(f"AI P{ai_maker_idx} discarded. Hand size: {len(ai_hand)}. Hand: {[str(c) for c in ai_hand]}.")
    game_data["message"] = f"{game_data['player_identities'][ai_maker_idx]} (AI) discarded {len(cards_to_discard_ai)} cards."
    game_data["cards_to_discard_count"] = 0
    if len(ai_hand) != 5: logging.warning(f"AI P{ai_maker_idx} hand size {len(ai_hand)} after discard, expected 5.")
    transition_to_play_phase()

def ai_decide_go_alone_and_proceed(ai_maker_idx: int):
    """
    Handles an AI maker's decision to go alone or play with the dummy hand.
    - Validates if it's the correct player's turn and game phase.
    - If an RLAgent exists, it queries the agent for a "choose_go_alone" or
      "choose_not_go_alone" action based on the current game state.
    - If no RLAgent, uses a heuristic based on hand strength and trump count.
    - Updates game state (`going_alone`, `message`).
    - Triggers an RL update for the decision.
    - If not going alone and dummy hand is valid, calls `ai_discard_five_cards`.
    - Otherwise, transitions to the play phase.

    Args:
        ai_maker_idx: The player ID of the AI maker.
    """
    global game_data #; time.sleep(0.5) # Removed AI delay
    # Ensure it's actually this AI's turn and correct phase
    if game_data["current_player_turn"] != ai_maker_idx or game_data["game_phase"] != "prompt_go_alone" or game_data["maker"] != ai_maker_idx:
        logging.warning(f"ai_decide_go_alone_and_proceed called for P{ai_maker_idx} out of turn/phase. Current turn: P{game_data['current_player_turn']}, Phase: {game_data['game_phase']}, Maker: P{game_data['maker']}.")
        return

    current_agent = get_rl_agent(ai_maker_idx)
    chose_to_go_alone = False

    if not current_agent:
        logging.error(f"RL Agent for P{ai_maker_idx} not found in ai_decide_go_alone_and_proceed. Using heuristic.")
        # Fallback to heuristic for going alone decision
        ai_hand_fallback = game_data["hands"][ai_maker_idx]; trump_suit_fallback = game_data["trump_suit"]
        GO_ALONE_THRESHOLD = 220 # Heuristic threshold for hand strength to consider going alone
        current_hand_strength = evaluate_potential_trump_strength(ai_hand_fallback, trump_suit_fallback, game_data)
        num_trump_cards_in_hand = sum(1 for card in ai_hand_fallback if get_effective_suit(card, trump_suit_fallback) == trump_suit_fallback)
        has_both_bowers = any(c.rank == 'J' and c.suit == trump_suit_fallback for c in ai_hand_fallback) and \
                          any(c.rank == 'J' and c.suit == get_left_bower_suit(trump_suit_fallback) for c in ai_hand_fallback)
        if current_hand_strength >= GO_ALONE_THRESHOLD:
            if num_trump_cards_in_hand >= 3 or (num_trump_cards_in_hand >=2 and has_both_bowers) : chose_to_go_alone = True
        logging.info(f"AI P{ai_maker_idx} (Maker Heuristic) decided to {'go alone' if chose_to_go_alone else 'play with dummy'}. Strength: {current_hand_strength}, Trumps: {num_trump_cards_in_hand}.")
    else:
        # RL Agent logic for going alone decision
        state_dict = get_rl_state(ai_maker_idx, game_data)
        game_data["rl_training_data"][ai_maker_idx] = {"state": state_dict, "action": None, "action_type": "decide_go_alone"}

        valid_actions = ["choose_go_alone", "choose_not_go_alone"]
        chosen_rl_action = current_agent.choose_action(state_dict, valid_actions)
        game_data["rl_training_data"][ai_maker_idx]["action"] = chosen_rl_action # Store chosen action
        chose_to_go_alone = (chosen_rl_action == "choose_go_alone")
        logging.info(f"AI P{ai_maker_idx} (Maker RL) chose action: {chosen_rl_action}. Go alone: {chose_to_go_alone}")

    # Common logic post-decision
    game_data["going_alone"] = chose_to_go_alone
    game_data["message"] = f"{game_data['player_identities'][ai_maker_idx]} (AI Maker) chose to {'go alone' if chose_to_go_alone else 'play with dummy hand'}."

    # RL Update for the go_alone decision.
    if current_agent: # Ensure agent exists before trying to update
        process_rl_update(ai_maker_idx, "bid_processed", event_data={"action_taken": game_data["rl_training_data"][ai_maker_idx]["action"], "is_go_alone_decision": True})

    if chose_to_go_alone:
        game_data["dummy_hand"] = [] # Ensure dummy is not used
        transition_to_play_phase()
    else: # Not going alone
        ai_hand = game_data["hands"][ai_maker_idx] # Get current hand for modification
        if game_data.get("dummy_hand") and len(game_data["dummy_hand"]) == 5:
            ai_hand.extend(game_data["dummy_hand"]); game_data["dummy_hand"] = []
            logging.info(f"AI P{ai_maker_idx} picked up dummy. Hand size: {len(ai_hand)}.")
            game_data["cards_to_discard_count"] = 5
            ai_discard_five_cards(ai_maker_idx) # This will discard and then call transition_to_play_phase
        else:
            logging.error(f"AI P{ai_maker_idx} chose not to go alone, but dummy_hand invalid or missing. Dummy: {game_data.get('dummy_hand')}. Forcing 'go alone'.")
            game_data["going_alone"] = True # Fallback to going alone
            game_data["message"] += " Error with dummy hand, proceeding as if going alone."
            transition_to_play_phase()


def process_ai_bid_action(ai_action_data: dict):
    """
    Handles an AI player's bidding decision during bidding rounds or when stuck as dealer.
    - Retrieves the RLAgent for the player.
    - Constructs the game state for the agent.
    - Queries the agent for a bidding action ('order_up', 'pass_bid', 'call_trump:S', 'pass_call').
    - Updates the game state based on the agent's decision (e.g., sets trump, maker,
      advances bidding turn, or transitions to dealer discard/go_alone phase).
    - Triggers RL updates for the bidding action.
    - If the AI's action leads to another AI's turn (e.g., AI passes, next AI bids),
      this function is called recursively for the next AI.
    - If dealer (AI or human) picks up the up-card, handles their discard.

    Args:
        ai_action_data: A dictionary containing 'player_index' and 'action' type
                        (e.g., 'ai_bidding_round_1', 'ai_dealer_stuck_call').
    """
    global game_data #; time.sleep(0.5) # Removed AI delay
    player_idx = ai_action_data.get('player_index')
    action_type = ai_action_data.get('action') # e.g., 'ai_bidding_round_1', 'ai_bidding_round_2', 'ai_dealer_stuck_call'

    logging.info(f"RL Processing AI P{player_idx} bid action: {action_type}. Phase: {game_data['game_phase']}.")

    current_agent = get_rl_agent(player_idx)
    if not current_agent:
        logging.error(f"RL Agent for P{player_idx} not found in process_ai_bid_action. AI cannot take bid action.")
        # Potentially advance turn or handle error, for now, it will just not act.
        return

    state_dict = get_rl_state(player_idx, game_data)
    game_data["rl_training_data"][player_idx] = {"state": state_dict, "action": None, "action_type": action_type} # Store state

    # ai_hand = game_data["hands"][player_idx] # Already part of state_dict if needed by agent

    chosen_game_action = None # This will be like 'order_up', 'pass_bid', 'call_trump:H', etc.

    if action_type == 'ai_bidding_round_1':
        # Valid actions: 'order_up' or 'pass_bid'
        valid_actions_for_agent = ["order_up", "pass_bid"]
        chosen_rl_action = current_agent.choose_action(state_dict, valid_actions_for_agent)
        game_data["rl_training_data"][player_idx]["action"] = chosen_rl_action # Store chosen action
        logging.info(f"RL Agent P{player_idx} (R1 Bid) chose: {chosen_rl_action}")

        if chosen_rl_action == "order_up":
            # Perform the 'order_up' logic
            up_card_obj = game_data["original_up_card_for_round"]
            game_data["trump_suit"] = up_card_obj.suit
            game_data["maker"] = player_idx
            current_message = f"{game_data['player_identities'][player_idx]} (AI RL) ordered up {SUITS_MAP[game_data['trump_suit']]}."
            game_data["up_card_visible"] = False; game_data["up_card"] = None
            current_dealer_idx = game_data["dealer"]
            dealer_hand_actual = game_data["hands"][current_dealer_idx]
            up_card_to_pickup_actual = game_data["original_up_card_for_round"]
            dealer_hand_actual.append(up_card_to_pickup_actual)
            current_message += f" {game_data['player_identities'][current_dealer_idx]} (Dealer) picked up {str(up_card_to_pickup_actual)}."
            logging.info(f"AI P{player_idx} (RL) ordered up. Dealer P{current_dealer_idx} picked up {str(up_card_to_pickup_actual)}.")

            if current_dealer_idx == 0: # Human dealer
                game_data["cards_to_discard_count"] = 1
                game_data["current_player_turn"] = current_dealer_idx
                game_data["game_phase"] = "dealer_must_discard_after_order_up"
                current_message += f" {game_data['player_identities'][current_dealer_idx]} (Dealer) must discard 1 card."
                game_data["message"] = current_message
            else: # AI Dealer discards
                ai_dealer_hand = game_data["hands"][current_dealer_idx]
                # AI dealer still uses heuristic for discard FOR NOW
                # TODO: RL agent for discard if dealer is AI
                card_to_discard_ai_dealer = get_ai_cards_to_discard(list(ai_dealer_hand), 1, game_data["trump_suit"])[0]
                try: ai_dealer_hand.remove(card_to_discard_ai_dealer)
                except ValueError: # Handle if card object identity is different
                    found_c = next((c for c in ai_dealer_hand if c.rank == card_to_discard_ai_dealer.rank and c.suit == card_to_discard_ai_dealer.suit), None)
                    if found_c: ai_dealer_hand.remove(found_c)
                current_message += f" {game_data['player_identities'][current_dealer_idx]} (AI Dealer) discarded 1 card."
                game_data["game_phase"] = "prompt_go_alone"
                game_data["current_player_turn"] = player_idx # The AI who ordered up is maker
                current_message += f" {game_data['player_identities'][player_idx]} (Maker) to decide go alone."
                game_data["message"] = current_message
                # RL Update for this bid action
                process_rl_update(player_idx, "bid_processed", event_data={"action_taken": chosen_rl_action, "became_maker": True})
                ai_decide_go_alone_and_proceed(player_idx) # AI maker decides to go alone
            return

        else: # chosen_rl_action == "pass_bid" or fallback
            game_data["message"] = f"{game_data['player_identities'][player_idx]} (AI RL) passes."
            game_data['passes_on_upcard'].append(player_idx)
            logging.info(f"AI P{player_idx} (RL) passed R1. Passes: {len(game_data['passes_on_upcard'])}/{game_data['num_players']}.")

            # RL Update for this pass action
            process_rl_update(player_idx, "bid_processed", event_data={"action_taken": chosen_rl_action, "became_maker": False})

            if len(game_data['passes_on_upcard']) == game_data["num_players"]:
                game_data["game_phase"] = "bidding_round_2"; game_data["up_card_visible"] = False
                game_data["current_player_turn"] = (game_data["dealer"] + 1) % game_data["num_players"]
                game_data["message"] += f" All passed. Up-card turned. {game_data['player_identities'][game_data['current_player_turn']]}'s turn to call."
                game_data['passes_on_calling'] = []
                if game_data["current_player_turn"] != 0: process_ai_bid_action({'player_index': game_data["current_player_turn"], 'action': 'ai_bidding_round_2'})
            else:
                bid_order = [(game_data["dealer"] + i + 1) % game_data["num_players"] for i in range(game_data["num_players"])]
                current_bidder_index_in_order = bid_order.index(player_idx)
                game_data["current_player_turn"] = bid_order[(current_bidder_index_in_order + 1) % game_data["num_players"]]
                game_data["message"] += f" {game_data['player_identities'][game_data['current_player_turn']]}'s turn."
                if game_data["current_player_turn"] != 0: process_ai_bid_action({'player_index': game_data["current_player_turn"], 'action': 'ai_bidding_round_1'})
            return

    elif action_type == 'ai_bidding_round_2':
        turned_down_suit = game_data["original_up_card_for_round"].suit
        valid_actions_for_agent = [f"call_trump:{s}" for s in SUITS if s != turned_down_suit] + ["pass_call"]
        chosen_rl_action = current_agent.choose_action(state_dict, valid_actions_for_agent) # e.g. "call_trump:H" or "pass_call"
        game_data["rl_training_data"][player_idx]["action"] = chosen_rl_action # Store chosen action
        logging.info(f"RL Agent P{player_idx} (R2 Bid) chose: {chosen_rl_action}")

        if chosen_rl_action.startswith("call_trump:"):
            called_suit = chosen_rl_action.split(":")[1]
            game_data["trump_suit"] = called_suit
            game_data["maker"] = player_idx
            game_data["message"] = f"{game_data['player_identities'][player_idx]} (AI RL) called {SUITS_MAP[game_data['trump_suit']]}."
            game_data["up_card_visible"] = False; game_data["up_card"] = None
            game_data["game_phase"] = "prompt_go_alone"
            game_data["current_player_turn"] = player_idx # The AI who called is maker
            process_rl_update(player_idx, "bid_processed", event_data={"action_taken": chosen_rl_action, "became_maker": True})
            ai_decide_go_alone_and_proceed(player_idx) # AI maker decides to go alone
            return

        else: # chosen_rl_action == "pass_call" or fallback
            game_data["message"] = f"{game_data['player_identities'][player_idx]} (AI RL) passes round 2."
            game_data['passes_on_calling'].append(player_idx)
            logging.info(f"AI P{player_idx} (RL) passed R2. Passes: {len(game_data['passes_on_calling'])}/{game_data['num_players']-1}.")
            process_rl_update(player_idx, "bid_processed", event_data={"action_taken": chosen_rl_action, "became_maker": False})

            bid_order_r2 = [(game_data["dealer"] + i + 1) % game_data["num_players"] for i in range(game_data["num_players"])]
            # Check if dealer is stuck (all others passed in R2)
            if len(game_data['passes_on_calling']) == game_data["num_players"] - 1: # All players except dealer passed
                 # This check ensures that it's actually the dealer's turn to be stuck.
                 # It implies all other non-dealer players are in passes_on_calling.
                if game_data["dealer"] not in game_data["passes_on_calling"]: # Dealer hasn't passed (obviously, they can't pass if stuck)
                    game_data["current_player_turn"] = game_data["dealer"]
                    game_data["game_phase"] = "dealer_must_call"
                    game_data["message"] += f" Dealer ({game_data['player_identities'][game_data['dealer']]}) is stuck."
                    if game_data["dealer"] != 0: # If dealer is AI
                        process_ai_bid_action({'player_index': game_data["dealer"], 'action': 'ai_dealer_stuck_call'})
                    # If dealer is human, UI will prompt them.
                    return

            # If not stuck, move to next player for R2 bidding
            current_passer_index_in_order = bid_order_r2.index(player_idx)
            next_player_candidate = bid_order_r2[(current_passer_index_in_order + 1) % game_data["num_players"]]

            if next_player_candidate == game_data["dealer"] and len(game_data['passes_on_calling']) < game_data["num_players"]-1 :
                 # This means we've looped through non-dealers and now it's dealer's turn, but they are not yet stuck
                 # This case might be complex, ensure dealer is not skipped if they are next and not stuck.
                 # The original logic for "dealer_must_call" should handle if it's their turn and they are stuck.
                 # If next is dealer, but not everyone else passed, it's still their turn to bid/pass in R2.
                 pass


            game_data["current_player_turn"] = next_player_candidate
            game_data["message"] += f" {game_data['player_identities'][game_data['current_player_turn']]}'s turn."
            if game_data["current_player_turn"] != 0:
                process_ai_bid_action({'player_index': game_data["current_player_turn"], 'action': 'ai_bidding_round_2'})
            return

    elif action_type == 'ai_dealer_stuck_call':
        turned_down_suit = game_data["original_up_card_for_round"].suit
        valid_actions_for_agent = [f"call_trump:{s}" for s in SUITS if s != turned_down_suit]
        if not valid_actions_for_agent: # Should not happen in Euchre
            logging.error(f"AI Dealer P{player_idx} stuck but no valid suits to call. This is a bug.")
            # Fallback: pick any suit not turned down (should be covered by valid_actions_for_agent)
            chosen_suit_by_ai = random.choice([s for s in SUITS if s != turned_down_suit])
        else:
            chosen_rl_action = current_agent.choose_action(state_dict, valid_actions_for_agent) # e.g. "call_trump:H"
            game_data["rl_training_data"][player_idx]["action"] = chosen_rl_action # Store chosen action
            chosen_suit_by_ai = chosen_rl_action.split(":")[1]

        logging.info(f"RL Agent P{player_idx} (Dealer Stuck) chose to call: {chosen_suit_by_ai}")
        game_data["trump_suit"] = chosen_suit_by_ai
        game_data["maker"] = player_idx
        game_data["message"] = f"{game_data['player_identities'][player_idx]} (AI RL Dealer) called {SUITS_MAP[game_data['trump_suit']]}."
        game_data["game_phase"] = "prompt_go_alone"
        game_data["current_player_turn"] = player_idx # The AI dealer is maker
        process_rl_update(player_idx, "bid_processed", event_data={"action_taken": chosen_rl_action, "became_maker": True})
        ai_decide_go_alone_and_proceed(player_idx) # AI maker decides to go alone
        return

    logging.warning(f"AI P{player_idx} RL bid action {action_type} did not result in a defined outcome.")


@app.route('/api/get_current_state', methods=['GET'])
def get_current_state_api():
    global game_data; logging.info("API: /api/get_current_state called.")
    return jsonify(game_data_to_json(game_data))

@app.route('/api/ai_play_turn', methods=['POST'])
def ai_play_turn_api():
    global game_data
    logging.info("Received request for AI to play turn endpoint.") # Added 'endpoint' for clarity

    # It's crucial to read the game state ONCE at the beginning of the request
    # to ensure all decisions within this function call are based on a consistent snapshot.
    current_game_state_snapshot = game_data.copy() # Shallow copy for consistent read
    current_player = current_game_state_snapshot.get("current_player_turn")
    current_phase = current_game_state_snapshot.get("game_phase")

    logging.info(f"AI Play Turn: Current player P{current_player}, Phase: {current_phase}")

    if current_player == 0:
        logging.warning("AI play rejected: Not AI's turn (P0).")
        return jsonify({"error": "Not AI's turn.", "game_state": game_data_to_json(game_data)}), 400

    if current_phase != "playing_tricks":
        logging.warning(f"AI play rejected: Phase is '{current_phase}', not 'playing_tricks'.")
        # Return current state along with the error, so client can update if needed.
        return jsonify({
            "error": f"AI cannot play. Game phase is '{current_phase}', not 'playing_tricks'.",
            "game_state": game_data_to_json(game_data)
        }), 400 # Crucially, this is a 400 error.

    # If phase is "playing_tricks" but trick count indicates round is over,
    # this implies a state discrepancy that score_round() should resolve.
    # This block handles the case where the round ended numerically but phase hasn't caught up yet
    # for this specific request's view (less likely if state is truly global and consistent).
    if is_round_over(): # is_round_over() uses the global game_data
        logging.warning(f"AI Play Turn: Initial phase was '{current_phase}' but is_round_over() is true. Attempting to correct state.")
        logging.info(f"AI Play Turn: game_phase BEFORE calling score_round() in is_round_over block: {game_data.get('game_phase')}")
        # Ensure score_round is called to update the phase in the global game_data
        # score_round() now has a guard against multiple scoring.
        score_round() # score_round updates global game_data if necessary
        logging.info(f"AI Play Turn: game_phase AFTER calling score_round() in is_round_over block: {game_data.get('game_phase')}")

        # Re-fetch the game phase after score_round() has potentially updated it.
        current_phase = game_data.get("game_phase") # This re-fetch is crucial
        logging.info(f"AI Play Turn: Re-fetched current_phase after score_round() is '{current_phase}'.")

        if current_phase in ["round_over", "game_over"]:
            logging.info(f"AI Play Turn: Round/Game is over (phase: {current_phase}). Not processing AI card play. Returning state.")
            return jsonify(game_data_to_json(game_data))
        else:
            # This case should ideally not be reached if is_round_over() is true and score_round() works.
            # It implies is_round_over() was true, but score_round didn't transition to a terminal phase.
            logging.warning(f"AI Play Turn: is_round_over() was true, but phase is still '{current_phase}'. Proceeding cautiously with AI play.")
            # Fall-through to process_ai_play_card, but this is an unusual state.

    # If we reach here: current_phase should be "playing_tricks" and the round not numerically over,
    # OR is_round_over() was true but phase didn't update to terminal (logged above).
    if current_phase == "playing_tricks": # Explicitly check again
        logging.info(f"Processing AI P{current_player}'s card play. Phase: {current_phase}")
        process_ai_play_card(current_player) # This function will modify global game_data
                                             # It might call score_round() if this play ends the round.
    else:
        logging.warning(f"AI Play Turn: Aborting AI card play because current_phase is '{current_phase}', not 'playing_tricks' after checks.")

    # The response will be based on the state of global game_data AFTER any processing.
    return jsonify(game_data_to_json(game_data))

def game_data_to_json(current_game_data_arg: dict) -> dict:
    """
    Converts the game_data dictionary into a JSON-serializable format.
    Specifically, it converts Card objects within lists or dictionaries
    (like hands, deck, up_card) into their dictionary representations.

    Args:
        current_game_data_arg: The game data dictionary to be converted.
                               Although the global `game_data` is primarily used,
                               this argument allows flexibility if a copy is passed.

    Returns:
        A new dictionary where Card objects are replaced by their dict versions.
    """
    global game_data # Access the true global game_data for the most current state
    # Create a shallow copy to modify for JSON serialization without altering the original Card objects in game_data.
    json_safe_data = game_data.copy() # Ensures we are working with the global state

    # Convert Card objects in various parts of the game state to their dict representations
    if json_safe_data.get('deck'): json_safe_data['deck'] = [card.to_dict() for card in json_safe_data['deck']]
    if json_safe_data.get('hands'): json_safe_data['hands'] = { p_idx: [card.to_dict() for card in hand] for p_idx, hand in json_safe_data['hands'].items()}
    if json_safe_data.get('dummy_hand'): json_safe_data['dummy_hand'] = [card.to_dict() for card in json_safe_data['dummy_hand']]
    if json_safe_data.get('up_card'): json_safe_data['up_card'] = json_safe_data['up_card'].to_dict() if json_safe_data['up_card'] else None
    if json_safe_data.get('original_up_card_for_round'): json_safe_data['original_up_card_for_round'] = json_safe_data['original_up_card_for_round'].to_dict() if json_safe_data['original_up_card_for_round'] else None
    if json_safe_data.get('trick_cards'): json_safe_data['trick_cards'] = [{'player': tc['player'], 'card': tc['card'].to_dict()} for tc in json_safe_data['trick_cards']]
    if json_safe_data.get('last_completed_trick') and json_safe_data['last_completed_trick'].get('played_cards'):
        json_safe_data['last_completed_trick']['played_cards'] = [{'player': pc['player'], 'card': pc['card'].to_dict() if isinstance(pc['card'], Card) else pc['card']} for pc in json_safe_data['last_completed_trick']['played_cards']]
    if json_safe_data.get('played_cards_this_round'):
        json_safe_data['played_cards_this_round'] = [card.to_dict() for card in json_safe_data['played_cards_this_round']]
    return json_safe_data

if __name__ == "__main__":
    # app.run(debug=True, host='0.0.0.0') # Comment out Flask app for training

    # --- Training Simulation ---
    def run_training_simulation(num_games_to_simulate: int, save_interval: int = 10):
        """
        Runs a simulation of multiple Euchre games for training RL agents.
        All players in the simulation are AI agents. The game progresses automatically
        through bidding, playing tricks, and scoring, with AI agents making decisions
        and updating their Q-tables based on outcomes.

        Args:
            num_games_to_simulate: The total number of games to simulate.
            save_interval: (No longer used with SQLite) Interval for saving Q-tables.
        """
        logging.info(f"Starting RL training simulation for {num_games_to_simulate} games.")
        # Ensure agents are created if not already (e.g. if initialize_game_data hasn't run via Flask)
        if not rl_agents:
            initialize_game_data() # This will create agents and set them to training mode.

        for game_num in range(1, num_games_to_simulate + 1):
            logging.info(f"--- Starting Training Game {game_num} ---")
            initialize_game_data() # Resets scores, RL agent data (epsilon), dealer, etc. for a new game.
            game_data["dealer"] = random.randint(0, game_data["num_players"] - 1) # Random dealer for new game

            game_over_flag = False
            round_num = 0
            while not game_over_flag:
                round_num += 1
                logging.info(f"Game {game_num}, Round {round_num} starting...")

                # Determine dealer for the new round
                if game_data["game_phase"] != "setup": # If not the very first round of the game
                    game_data["dealer"] = (game_data["dealer"] + 1) % game_data["num_players"]

                initialize_new_round() # Deals cards, sets up bidding phase, etc.

                round_over_flag = False
                while not round_over_flag:
                    current_player_id = game_data["current_player_turn"]
                    current_phase = game_data["game_phase"]

                    # Check for terminal conditions (game over, round over) before processing turn
                    if current_phase == "game_over":
                        game_over_flag = True; break
                    if current_phase == "round_over":
                        round_over_flag = True; break

                    logging.debug(f"Game {game_num}, R{round_num}, Phase: {current_phase}, Turn: P{current_player_id}")

                    # AI logic for different game phases
                    # Since all players are AI in simulation, no human input is awaited.
                    if current_phase == "bidding_round_1":
                        process_ai_bid_action({'player_index': current_player_id, 'action': 'ai_bidding_round_1'})
                    elif current_phase == "bidding_round_2":
                        process_ai_bid_action({'player_index': current_player_id, 'action': 'ai_bidding_round_2'})
                    elif current_phase == "dealer_must_call":
                        process_ai_bid_action({'player_index': current_player_id, 'action': 'ai_dealer_stuck_call'})
                    elif current_phase == "prompt_go_alone":
                        ai_decide_go_alone_and_proceed(current_player_id) # AI Maker decides
                    elif current_phase == "playing_tricks":
                        process_ai_play_card(current_player_id)
                    elif current_phase == "maker_discard":
                        # This phase is entered if AI maker chooses not to go alone and picks up dummy.
                        # ai_decide_go_alone_and_proceed calls ai_discard_five_cards internally.
                        # This direct call might be redundant if the flow is always through ai_decide_go_alone.
                        if game_data["maker"] == current_player_id and game_data["cards_to_discard_count"] == 5:
                            ai_discard_five_cards(current_player_id)
                    elif current_phase in ["dealer_discard_one", "dealer_must_discard_after_order_up"]:
                        # Handles AI dealer discarding one card after up-card is ordered up.
                        if current_player_id == game_data["dealer"]:
                            logging.info(f"AI Dealer P{current_player_id} in phase {current_phase}. Needs to discard.")
                            dealer_hand = game_data["hands"][current_player_id]
                            trump_suit = game_data["trump_suit"]

                            if not dealer_hand: logging.error(f"CRITICAL: AI Dealer P{current_player_id} has empty hand in {current_phase}.")
                            elif not trump_suit: logging.error(f"CRITICAL: Trump suit not set for AI Dealer P{current_player_id} discard in {current_phase}.")
                            else:
                                cards_to_discard = get_ai_cards_to_discard(list(dealer_hand), 1, trump_suit)
                                if cards_to_discard:
                                    card_to_discard_obj = cards_to_discard[0]
                                    try:
                                        actual_card_to_remove = next(c for c in dealer_hand if c.suit == card_to_discard_obj.suit and c.rank == card_to_discard_obj.rank)
                                        dealer_hand.remove(actual_card_to_remove)
                                        game_data["message"] = f"{game_data['player_identities'][current_player_id]} (AI Dealer) discarded {str(actual_card_to_remove)}."
                                        logging.info(f"AI Dealer P{current_player_id} discarded {str(actual_card_to_remove)}. Hand size: {len(dealer_hand)}.")
                                    except (ValueError, StopIteration):
                                        logging.error(f"AI Dealer P{current_player_id} failed to find/remove card {str(card_to_discard_obj)} from hand for discard. Hand: {[str(c) for c in dealer_hand]}")
                                        if dealer_hand: # Fallback
                                            fallback_discard = dealer_hand.pop(0)
                                            logging.warning(f"AI Dealer P{current_player_id} discarded {str(fallback_discard)} by fallback.")
                                            game_data["message"] = f"{game_data['player_identities'][current_player_id]} (AI Dealer) discarded {str(fallback_discard)} (fallback)."

                                    game_data["game_phase"] = "prompt_go_alone"
                                    game_data["current_player_turn"] = game_data["maker"]
                                    game_data["cards_to_discard_count"] = 0
                                    logging.info(f"AI Dealer P{current_player_id} discard complete. Phase -> 'prompt_go_alone'. Turn for P{game_data['maker']} (Maker).")
                                else:
                                    logging.error(f"AI Dealer P{current_player_id} failed to select card to discard in {current_phase}.")
                                    if dealer_hand: # Fallback to prevent stall
                                        fallback_discard = dealer_hand.pop(0)
                                        logging.warning(f"AI Dealer P{current_player_id} discarded {str(fallback_discard)} by fallback (selection error).")
                                        game_data["game_phase"] = "prompt_go_alone"; game_data["current_player_turn"] = game_data["maker"]; game_data["cards_to_discard_count"] = 0
                        else:
                            logging.warning(f"Phase is {current_phase} but current player P{current_player_id} is not dealer P{game_data['dealer']}. Skipping discard logic.")
                    else:
                        logging.error(f"Training Loop: AI P{current_player_id} in UNHANDLED phase {current_phase} for game {game_num}. Aborting game.")
                        game_over_flag = True; round_over_flag = True # Force exit

                    # Basic stall check: if phase and player haven't changed, log warning.
                    # More robust stall detection might be needed for complex scenarios.
                    if game_data["game_phase"] == current_phase and \
                       game_data["current_player_turn"] == current_player_id and \
                       not game_over_flag and not round_over_flag:
                        logging.warning(f"Potential stall in training: P{current_player_id}, Phase: {current_phase}. Game state may not have progressed.")
                        # Consider adding a counter or forcing a break if stall persists.

                    # Re-check terminal conditions after action, as game/round might end.
                    current_phase_after_action = game_data["game_phase"]
                    if current_phase_after_action == "game_over": game_over_flag = True
                    if current_phase_after_action == "round_over": round_over_flag = True

            logging.info(f"--- Training Game {game_num} ended. Scores: {game_data['scores']} ---")
            # Q-table saving is now continuous with SQLite, so no periodic save needed here.

        logging.info(f"Training simulation finished for {num_games_to_simulate} games. Q-values are stored in {Q_TABLE_DB_FILE}.")

    # Example of how to run it (e.g. for 10 iterations as requested for testing)
    run_training_simulation(10000, save_interval=5) # save_interval is no longer used with SQLite

    # To migrate existing q_table.json to SQLite (run once):
    # migrate_json_to_sqlite()

    # To run the Flask app:
    # app.run(debug=True, host='0.0.0.0')

def migrate_json_to_sqlite(json_file_path="q_table.json", db_file_path=None):
    """
    Migrates Q-table data from an old JSON file format to the SQLite database.
    This is a utility function to be run once if transitioning from a JSON-based
    Q-table storage to SQLite. It reads key-value pairs from the JSON file
    and inserts them into the `q_values` table in the SQLite database.
    """
    if db_file_path is None:
        db_file_path = Q_TABLE_DB_FILE # Use the global constant

    if not os.path.exists(json_file_path):
        logging.info(f"JSON file {json_file_path} not found. No migration needed or possible.")
        return

    try:
        with open(json_file_path, 'r') as f:
            json_q_table = json.load(f)
        logging.info(f"Successfully loaded Q-table from {json_file_path}.")
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Error loading Q-table from {json_file_path}: {e}. Migration aborted.")
        return

    if not json_q_table:
        logging.info("JSON Q-table is empty. Nothing to migrate.")
        return

    conn = None
    try:
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()
        # Ensure the table exists (though RLAgent instantiation usually does this)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS q_values (
                state_key TEXT PRIMARY KEY,
                actions_q_values TEXT NOT NULL
            )
        """)
        conn.commit()

        migrated_count = 0
        skipped_count = 0
        for state_key, actions_dict in json_q_table.items():
            if not isinstance(actions_dict, dict):
                logging.warning(f"Skipping state_key '{state_key}' due to invalid actions_dict format: {type(actions_dict)}")
                skipped_count += 1
                continue

            actions_q_values_json = json.dumps(actions_dict) # Serialize the actions dict to JSON string
            try:
                cursor.execute("""
                    INSERT INTO q_values (state_key, actions_q_values) VALUES (?, ?)
                    ON CONFLICT(state_key) DO UPDATE SET actions_q_values = excluded.actions_q_values
                """, (state_key, actions_q_values_json))
                migrated_count += 1
            except sqlite3.Error as e:
                logging.error(f"Error migrating state_key '{state_key}': {e}")
                skipped_count +=1

        conn.commit()
        logging.info(f"Migration complete. Migrated {migrated_count} states. Skipped {skipped_count} states.")

    except sqlite3.Error as e:
        logging.error(f"SQLite error during migration: {e}")
    finally:
        if conn:
            conn.close()
            logging.info(f"SQLite connection closed for migration utility.")
