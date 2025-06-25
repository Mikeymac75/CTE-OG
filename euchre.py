import random
import logging
import time
import json # Added for RL Agent state serialization
import os   # Added for saving/loading Q-table
import sqlite3 # Added for SQLite Q-table storage
from flask import Flask, jsonify, render_template, send_from_directory, request

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        self.training_mode = mode
        if not mode:
            self.epsilon = 0 # No exploration if not training

    # load_q_table and save_q_table are removed as we operate directly on DB

    def _serialize_state(self, state_dict):
        """Converts a state dictionary to a canonical string representation for Q-table keys."""
        # Sort by keys to ensure canonical representation
        return json.dumps(state_dict, sort_keys=True)

    def get_q_value(self, state_key, action_key):
        """Fetches Q-value from SQLite."""
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

# --- Game State ---
game_data = {}
rl_agents = {} # To store RL agents, keyed by player_id (1 and 2 for AI)

# Helper function to get the RLAgent for the current AI player
def get_rl_agent(player_id):
    return rl_agents.get(player_id)

def get_player_role(player_id, dealer_id, maker_id, num_players):
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
    current_game_data: The global game_data dictionary.
    """
    hand = current_game_data["hands"].get(player_id, [])
    hand_serialized = sorted([card.to_dict() for card in hand], key=lambda c: (c['suit'], c['rank']))

    up_card_dict = current_game_data.get("up_card")
    original_up_card_dict = current_game_data.get("original_up_card_for_round")

    state = {
        "player_id": player_id, # Keep for reference, might not be part of serialized key directly if always from this player's view
        "game_phase": current_game_data.get("game_phase"),
        "trump_suit": current_game_data.get("trump_suit"),
        "dealer": current_game_data.get("dealer"),
        "maker": current_game_data.get("maker"),
        "current_player_turn": current_game_data.get("current_player_turn"), # Useful to know if it's this agent's turn

        "hand": [c['suit'] + c['rank'] for c in hand_serialized], # Simplified representation
        "hand_strength_no_trump": evaluate_potential_trump_strength(hand, None, current_game_data),

        # Bidding specific
        "up_card_suit": up_card_dict.suit if up_card_dict and current_game_data.get("up_card_visible") else None,
        "up_card_rank": up_card_dict.rank if up_card_dict and current_game_data.get("up_card_visible") else None,
        "original_up_card_suit": original_up_card_dict.suit if original_up_card_dict else None, # Suit that was turned down
        "passes_on_upcard": len(current_game_data.get("passes_on_upcard", [])),
        "passes_on_calling": len(current_game_data.get("passes_on_calling", [])),

        # Playing specific
        "current_trick_lead_suit": current_game_data.get("current_trick_lead_suit"),
        "trick_cards_played_count": len(current_game_data.get("trick_cards", [])),
        # "cards_in_current_trick": [(tc['player'], tc['card'].to_dict()) for tc in current_game_data.get("trick_cards", [])], # Too complex for now
        "trick_leader": current_game_data.get("trick_leader"),

        # Scores and round progress
        "my_score": current_game_data["scores"].get(player_id, 0),
        # "opponent_scores": [], # Needs logic to determine opponents
        "my_round_tricks": current_game_data["round_tricks_won"].get(player_id, 0),
        # "maker_round_tricks": current_game_data["round_tricks_won"].get(current_game_data.get("maker"), 0) if current_game_data.get("maker") is not None else 0,

        "player_role": get_player_role(player_id, current_game_data.get("dealer"), current_game_data.get("maker"), current_game_data.get("num_players")),
        "going_alone": current_game_data.get("going_alone", False)
    }

    # Add hand strength for potential trump suits during bidding
    if state["game_phase"] == "bidding_round_1" and state["up_card_suit"]:
        state[f"strength_as_trump_{state['up_card_suit']}"] = evaluate_potential_trump_strength(hand, state["up_card_suit"], current_game_data)
    elif state["game_phase"] == "bidding_round_2":
        for s_option in SUITS:
            if s_option != state["original_up_card_suit"]: # Cannot call the turned down suit
                 state[f"strength_as_trump_{s_option}"] = evaluate_potential_trump_strength(hand, s_option, current_game_data)

    # Clean state: remove None values for more consistent serialization, or handle them in serialization
    # For now, json.dumps handles None as null, which is fine.
    return state

def initialize_game_data():
    global game_data, rl_agents
    logging.info("Initializing game data for the very first time.")

    # Define initial game_data structure first
    num_players = 3 # Default, can be adjusted if game structure changes
    player_identities = {0: "Player 1 (You)", 1: "Player 2 (AI)", 2: "Player 3 (AI)"}

    # Initialize RL agents based on player_identities
    # This ensures agents are created once when the application starts or state is wiped.
    # And their Q-tables are loaded.
    if not rl_agents: # Only create new agent instances if rl_agents is empty
        for pid, name in player_identities.items():
            if "AI" in name and pid != 0: # Assuming player 0 is human
                if pid not in rl_agents:
                    logging.info(f"Creating RLAgent for player {pid}")
                    rl_agents[pid] = RLAgent(player_id=pid)

    # Reset epsilon for agents at the start of each new game (full reset of game_data)
    # This is important for consistent training sessions.
    for agent_id, agent in rl_agents.items():
        agent.epsilon = DEFAULT_EPSILON
        agent.set_training_mode(True) # Default to training mode
        # Clear any lingering last_action_info from a previous game
        if hasattr(agent, 'last_action_info'): # Corrected check
            del agent.last_action_info

    logging.debug(f"DEBUG: Inside initialize_game_data, before main game_data assignment. Current game_data keys: {list(game_data.keys()) if isinstance(game_data, dict) else 'Not a dict'}")
    game_data = {
        "deck": [], "hands": {p: [] for p in range(num_players)}, "dummy_hand": [],
        "scores": {p: 0 for p in range(num_players)}, "dealer": random.randint(0, num_players - 1),
        "rl_training_data": {p_id: {} for p_id in rl_agents.keys()},
        "trump_suit": None, "up_card": None, "up_card_visible": False,
        "current_player_turn": -1, "maker": None, "going_alone": False,
        "trick_cards": [], "current_trick_lead_suit": None,
        "trick_leader": -1, "round_tricks_won": {p: 0 for p in range(num_players)},
        "game_phase": "setup",
        "message": "Welcome! Click 'Start New Round'.",
        "player_identities": player_identities,
        "num_players": num_players,
        "passes_on_upcard": [], "passes_on_calling": [],
        "cards_to_discard_count": 0,
        "original_up_card_for_round": None,
        "last_completed_trick": None
    }
    logging.debug(f"DEBUG: Inside initialize_game_data, AFTER main game_data assignment. New game_data keys: {list(game_data.keys())}")


def get_next_valid_actions(player_id, game_phase, game_data_for_next_state):
    """ Helper to determine valid actions for the next state. Essential for Q-learning update. """
    # Ensure game_data_for_next_state is the complete game_data dictionary
    hand_cards = game_data_for_next_state.get("hands", {}).get(player_id, [])

    if game_phase == "playing_tricks":
        lead_suit = game_data_for_next_state.get("current_trick_lead_suit")
        trump_suit = game_data_for_next_state.get("trump_suit")
        # Ensure hand_cards are actual Card objects if get_valid_plays expects them
        # If they are dicts from a serialized state, they need to be Card objects.
        # However, get_rl_state already uses Card objects from game_data["hands"]
        valid_card_objects = get_valid_plays(list(hand_cards), lead_suit, trump_suit)
        return [card.to_dict() for card in valid_card_objects] # Return dicts as actions
    elif game_phase == "bidding_round_1":
        return ["order_up", "pass_bid"]
    elif game_phase == "bidding_round_2":
        original_up_card = game_data_for_next_state.get("original_up_card_for_round")
        turned_down_suit = original_up_card.suit if original_up_card else None
        return [f"call_trump:{s}" for s in SUITS if s != turned_down_suit] + ["pass_call"]
    elif game_phase == "dealer_must_call":
        original_up_card = game_data_for_next_state.get("original_up_card_for_round")
        turned_down_suit = original_up_card.suit if original_up_card else None
        return [f"call_trump:{s}" for s in SUITS if s != turned_down_suit]
    elif game_phase == "prompt_go_alone":
        return ["choose_go_alone", "choose_not_go_alone"]
    elif game_phase == "dealer_discard_one" or game_phase == "dealer_must_discard_after_order_up":
        if game_data_for_next_state.get("cards_to_discard_count") == 1 and game_data_for_next_state.get("current_player_turn") == player_id:
             return [card.to_dict() for card in hand_cards] # Action is to choose one card to discard
        return []
    elif game_phase == "maker_discard":
        if game_data_for_next_state.get("cards_to_discard_count") > 1 and game_data_for_next_state.get("current_player_turn") == player_id:
            # This is complex: action is a combination of 5 cards.
            # For Q-learning, this might be too large an action space if not simplified.
            # For now, returning empty, as discard logic for 5 cards is heuristic.
            return []
    return []


def process_rl_update(player_id_acted, event_type, event_data=None):
    """
    Processes RL update for an AI player after an event (e.g., trick end, round end).
    player_id_acted: The ID of the player whose action is being learned from.
    event_type: "trick_end", "round_end", "game_end", "bid_processed" (after bid action is fully resolved).
    event_data: Dictionary containing relevant data for reward calculation.
    """
    global game_data
    agent = get_rl_agent(player_id_acted)

    # Ensure agent exists and is in training mode
    if not agent or not agent.training_mode:
        return

    training_info = game_data["rl_training_data"].get(player_id_acted)

    # Validate that essential training information is present
    if not training_info or "state" not in training_info or "action" not in training_info:
        return

    prev_state_dict = training_info["state"]
    action_taken_serialized = agent._serialize_action(training_info["action"]) # Ensure action is serialized for Q-table

    if training_info["action"] is None: # Agent failed to choose a valid action
        game_data["rl_training_data"][player_id_acted] = {} # Clear data
        return

    reward = 0
    # --- Calculate Reward based on event_type ---
    if event_type == "trick_end":
        trick_winner_idx = event_data.get("trick_winner_idx")
        # Simplified: + if player_id_acted won, - if someone else won.
        # TODO: More nuanced team-based reward for trick_end
        if trick_winner_idx == player_id_acted:
            reward += REWARD_WIN_TRICK
        else:
            reward += REWARD_LOSE_TRICK

    elif event_type == "round_end":
        # final_scores_for_round = game_data["round_tricks_won"] # Use current game_data for this
        round_maker = game_data.get("maker") # Maker of the completed round
        is_player_maker = (player_id_acted == round_maker)

        if round_maker is None:
             logging.error(f"RL Update (round_end for P{player_id_acted}): Maker is None. Cannot assign round reward.")
        elif is_player_maker:
            maker_tricks_won = game_data["round_tricks_won"].get(round_maker, 0)
            was_alone = game_data.get("going_alone", False) # From the completed round state
            if maker_tricks_won < 3:
                reward += REWARD_EUCHRED_MAKER
            elif maker_tricks_won == 5:
                reward += REWARD_WIN_ROUND_MAKER_ALONE_MARCH if was_alone else REWARD_WIN_ROUND_MAKER_MARCH
            else: # 3 or 4 tricks
                reward += REWARD_WIN_ROUND_MAKER_NORMAL
        else: # Player was a defender against round_maker
            maker_tricks_won = game_data["round_tricks_won"].get(round_maker, 0)
            if maker_tricks_won < 3:
                reward += REWARD_WIN_ROUND_DEFENSE_EUCHRE
            else:
                reward += REWARD_LOSE_ROUND_DEFENSE

    elif event_type == "game_end":
        game_winner_idx = event_data.get("game_winner_idx")
        if game_winner_idx == player_id_acted: # Simplified: direct win/loss
            reward += REWARD_WIN_GAME
        else:
            reward += REWARD_LOSE_GAME

    elif event_type == "bid_processed": # Intermediate reward for bidding outcome
        action_type_from_state = training_info.get("action_type")
        is_bid_action = action_type_from_state in ["ai_bidding_round_1", "ai_bidding_round_2", "ai_dealer_stuck_call"]

        if is_bid_action and game_data.get("maker") == player_id_acted : # If the current player became the maker
             # Positive reward for becoming maker will be implicitly handled by round_end rewards.
             # Can add a small immediate positive nudge if desired: e.g. reward += REWARD_SUCCESSFUL_BID_ATTEMPT
             pass # Covered by round_end
        # No immediate penalty for passing a bid unless it leads to a bad round outcome.

    # --- Determine next_state and next_valid_actions ---
    # The 'current' game_data is the next_state from the perspective of the action just taken.
    next_state_dict = get_rl_state(player_id_acted, game_data)

    next_player_for_q_learning = game_data.get("current_player_turn", -1)
    next_valid_actions_for_q = []
    if game_data.get("game_phase") not in ["game_over", "round_over", "setup"]:
        if next_player_for_q_learning != 0 and next_player_for_q_learning != -1 : # If next player is an AI
             # Get valid actions for the *next* player in the *new* state.
            next_valid_actions_for_q = get_next_valid_actions(next_player_for_q_learning, game_data["game_phase"], game_data)
        # If next player is human (0) or game is in a non-AI state, next_valid_actions_for_q remains empty, next_max_q = 0.

    logging.debug(f"RL Update P{player_id_acted}: Event: {event_type}, Action: {training_info['action']}, Reward: {reward}")
    agent.update_q_table(prev_state_dict, training_info["action"], reward, next_state_dict, next_valid_actions_for_q)

    game_data["rl_training_data"][player_id_acted] = {} # Clear the processed training data
    # Erroneous game_data re-assignment and initialize_game_data() call removed from here.

# initialize_game_data() # This is the global scope call at the end of the original file structure.
                         # It should be commented out if training is run directly from __main__
                         # to avoid double initialization or state conflicts if not careful.
                         # For now, let's assume it's intended to be active for Flask, and the
                         # training loop's own initialize_game_data calls are sufficient for training.

# --- Core Game Logic ---
def initialize_new_round():
    global game_data
    logging.info(f"Initializing new round. Current dealer: {game_data.get('dealer', 'N/A')}")
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
    game_data["message"] = f"{game_data['player_identities'][game_data['current_player_turn']]}'s turn. Up-card: {str(game_data['up_card'])}."
    logging.info(f"New round initialized. Up-card: {str(game_data['up_card']) if game_data['up_card'] else 'N/A'}. Turn: P{game_data['current_player_turn']}. Phase: {game_data['game_phase']}")

def get_left_bower_suit(trump_suit_char):
    if trump_suit_char not in SUITS_MAP: return None
    return {'H': 'D', 'D': 'H', 'C': 'S', 'S': 'C'}.get(trump_suit_char)

def get_effective_suit(card, trump_suit):
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

def evaluate_potential_trump_strength(hand, potential_trump_suit, game_data=None):
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

def get_ai_cards_to_discard(hand, num_to_discard, trump_suit):
    hand_copy = list(hand)
    hand_copy.sort(key=lambda c: get_card_value(c, trump_suit, None))
    return hand_copy[:num_to_discard]

def get_ai_stuck_suit_call(hand, turned_down_suit):
    possible_suits = [s for s in SUITS if s != turned_down_suit]
    if not possible_suits: return random.choice(SUITS)
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

def process_ai_play_card(ai_player_idx):
    global game_data #; time.sleep(0.5) # Removed AI delay
    if game_data["game_phase"] != "playing_tricks" or game_data["current_player_turn"] != ai_player_idx: return

    current_agent = get_rl_agent(ai_player_idx)
    if not current_agent:
        logging.error(f"RL Agent for P{ai_player_idx} not found in process_ai_play_card. Falling back to old logic.")
        # Fallback to a simplified version of old logic or random play if agent is missing
        # This part of fallback needs to be robust or simply log error and skip turn.
        # For now, let's use the existing heuristic if agent is missing.
        # --- Start of Fallback Heuristic (Simplified) ---
        ai_hand_fallback = game_data["hands"][ai_player_idx]; lead_suit_fallback = game_data["current_trick_lead_suit"]; trump_suit_fallback = game_data["trump_suit"]
        valid_cards_fallback = get_valid_plays(list(ai_hand_fallback), lead_suit_fallback, trump_suit_fallback)
        if not valid_cards_fallback:
            game_data["message"] += f" Error: AI {ai_player_idx+1} (Fallback) has no valid cards to play."
            if len(game_data["trick_cards"]) < game_data["num_players"]: game_data["current_player_turn"] = (ai_player_idx + 1) % game_data["num_players"]
            return
        valid_cards_fallback.sort(key=lambda c: get_card_value(c, trump_suit_fallback, lead_suit_fallback), reverse=True)
        card_to_play = valid_cards_fallback[0]
        logging.info(f"AI P{ai_player_idx} (Fallback Heuristic) playing {str(card_to_play)}")
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

        # Update rl_training_data with the actual card object if chosen_action_dict was valid, for easier processing later.
        # The agent's Q-table uses the serialized dict form, so that's what should have been stored for "action".
        # However, for consistency in how we might process rewards related to the card, let's ensure it's the dict.
        # The chosen_action_dict is already stored. If card_to_play had to fallback, the stored action might not match.
        # This is acceptable for now; the agent learns based on what it *thought* it was choosing from valid_actions_for_agent.

        logging.info(f"AI P{ai_player_idx} (RL Agent) chose to play: {str(card_to_play)} (Raw action: {chosen_action_dict})")
        # --- End of RL Agent Logic ---

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
            if valid_cards:
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
        except ValueError:
            logging.error(f"CRITICAL: Failed to remove {str(actual_card_to_remove_from_hand)} from AI P{ai_player_idx}'s hand. Hand: {[str(c) for c in ai_hand]}")
            return # Avoid proceeding with inconsistent state
    else: # Should be caught by earlier checks, but as a final safeguard
        logging.error(f"CRITICAL: AI P{ai_player_idx} somehow has no actual_card_to_remove_from_hand. Chosen: {str(card_to_play)}. Hand: {[str(c) for c in ai_hand]}"); return

    game_data["trick_cards"].append({'player': ai_player_idx, 'card': card_to_play}) # Use the card object that was chosen and confirmed in hand
    game_data["message"] = f"{game_data['player_identities'][ai_player_idx]} (AI) played {str(card_to_play)}."

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

def predict_maker_can_beat_card(maker_idx, target_card_to_beat, trump_suit, current_lead_suit, game_data_copy):
    """
    Predicts if the maker is likely to beat a specific target_card.
    This is a heuristic based on cards played and general probabilities.
    It does not know the maker's actual hand.
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

def is_round_over(): return sum(game_data["round_tricks_won"].values()) >= 5

def score_round():
    global game_data
    # Prevent re-scoring if round/game is already marked as over
    if game_data.get("game_phase") in ["round_over", "game_over"]:
        logging.warning(f"score_round() called when game_phase is already {game_data.get('game_phase')}. Points not re-awarded.")
        return

    maker = game_data["maker"]; maker_tricks = game_data["round_tricks_won"][maker]
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
    for p_idx, score in game_data["scores"].items():
        if score >= 10:
            game_data["game_phase"] = "game_over"
        game_data["message"] += f" {game_data['player_identities'][p_idx]} wins the game!"
        # --- RL Update for Game End ---
        game_event_data = {"game_winner_idx": p_idx}
        for ai_p_id in rl_agents.keys(): # Update all AI agents
            process_rl_update(ai_p_id, "game_end", event_data=game_event_data)
        # --- End RL Update ---
        return # Game is over, no further round processing needed for RL here

    game_data["game_phase"] = "round_over"
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

def ai_decide_go_alone_and_proceed(ai_maker_idx):
    global game_data #; time.sleep(0.5) # Removed AI delay
    # Ensure it's actually this AI's turn and correct phase
    if game_data["current_player_turn"] != ai_maker_idx or game_data["game_phase"] != "prompt_go_alone" or game_data["maker"] != ai_maker_idx:
        logging.warning(f"ai_decide_go_alone_and_proceed called for P{ai_maker_idx} out of turn/phase. Current turn: P{game_data['current_player_turn']}, Phase: {game_data['game_phase']}, Maker: P{game_data['maker']}.")
        return

    current_agent = get_rl_agent(ai_maker_idx)
    chose_to_go_alone = False

    if not current_agent:
        logging.error(f"RL Agent for P{ai_maker_idx} not found in ai_decide_go_alone_and_proceed. Using heuristic.")
        # Fallback to heuristic
        ai_hand_fallback = game_data["hands"][ai_maker_idx]; trump_suit_fallback = game_data["trump_suit"]
        GO_ALONE_THRESHOLD = 220 # Heuristic threshold
        current_hand_strength = evaluate_potential_trump_strength(ai_hand_fallback, trump_suit_fallback, game_data)
        num_trump_cards_in_hand = sum(1 for card in ai_hand_fallback if get_effective_suit(card, trump_suit_fallback) == trump_suit_fallback)
        has_both_bowers = any(c.rank == 'J' and c.suit == trump_suit_fallback for c in ai_hand_fallback) and \
                          any(c.rank == 'J' and c.suit == get_left_bower_suit(trump_suit_fallback) for c in ai_hand_fallback)
        if current_hand_strength >= GO_ALONE_THRESHOLD:
            if num_trump_cards_in_hand >= 3 or (num_trump_cards_in_hand >=2 and has_both_bowers) : chose_to_go_alone = True
        logging.info(f"AI P{ai_maker_idx} (Maker Heuristic) decided to {'go alone' if chose_to_go_alone else 'play with dummy'}. Strength: {current_hand_strength}, Trumps: {num_trump_cards_in_hand}.")
    else:
        # RL Agent logic
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
    # The reward for this is mostly tied to round outcome, but we link S,A -> S' here.
    # chosen_rl_action was stored in game_data["rl_training_data"][ai_maker_idx]["action"]
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
            # ai_discard_five_cards still uses a heuristic. RL for this is a future step.
            ai_discard_five_cards(ai_maker_idx) # This will discard and then call transition_to_play_phase
        else:
            logging.error(f"AI P{ai_maker_idx} chose not to go alone, but dummy_hand invalid or missing. Dummy: {game_data.get('dummy_hand')}. Forcing 'go alone'.")
            game_data["going_alone"] = True # Fallback to going alone
            game_data["message"] += " Error with dummy hand, proceeding as if going alone."
            transition_to_play_phase()


def process_ai_bid_action(ai_action_data):
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
        # Ensure score_round is called to update the phase in the global game_data
        # score_round() now has a guard against multiple scoring.
        score_round() # score_round updates global game_data if necessary

        # Re-fetch the game phase after score_round() has potentially updated it.
        current_phase = game_data.get("game_phase")
        logging.info(f"AI Play Turn: Phase after score_round() check is '{current_phase}'.")

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

def game_data_to_json(current_game_data_arg): # Renamed arg for clarity
    global game_data # Access the true global game_data
    # Ensure that the data being serialized is from the single global source,
    # especially critical if current_game_data_arg could somehow be a stale copy.
    json_safe_data = game_data.copy()

    if json_safe_data.get('deck'): json_safe_data['deck'] = [card.to_dict() for card in json_safe_data['deck']]
    if json_safe_data.get('hands'): json_safe_data['hands'] = { p_idx: [card.to_dict() for card in hand] for p_idx, hand in json_safe_data['hands'].items()}
    if json_safe_data.get('dummy_hand'): json_safe_data['dummy_hand'] = [card.to_dict() for card in json_safe_data['dummy_hand']]
    if json_safe_data.get('up_card'): json_safe_data['up_card'] = json_safe_data['up_card'].to_dict() if json_safe_data['up_card'] else None
    if json_safe_data.get('original_up_card_for_round'): json_safe_data['original_up_card_for_round'] = json_safe_data['original_up_card_for_round'].to_dict() if json_safe_data['original_up_card_for_round'] else None
    if json_safe_data.get('trick_cards'): json_safe_data['trick_cards'] = [{'player': tc['player'], 'card': tc['card'].to_dict()} for tc in json_safe_data['trick_cards']]
    if json_safe_data.get('last_completed_trick') and json_safe_data['last_completed_trick'].get('played_cards'):
        json_safe_data['last_completed_trick']['played_cards'] = [{'player': pc['player'], 'card': pc['card'].to_dict() if isinstance(pc['card'], Card) else pc['card']} for pc in json_safe_data['last_completed_trick']['played_cards']]
    return json_safe_data

if __name__ == "__main__":
    # app.run(debug=True, host='0.0.0.0') # Comment out Flask app for training

    # --- Training Simulation ---
    def run_training_simulation(num_games_to_simulate, save_interval=10):
        logging.info(f"Starting RL training simulation for {num_games_to_simulate} games.")
        # Ensure agents are created if not already (e.g. if initialize_game_data hasn't run via Flask)
        if not rl_agents:
            initialize_game_data() # This will create agents
            # initialize_game_data sets them to training mode and resets epsilon.

        for game_num in range(1, num_games_to_simulate + 1):
            logging.info(f"--- Starting Training Game {game_num} ---")
            # Initialize game data (resets scores, rl_training_data, agent epsilons etc.)
            initialize_game_data()
            # logging.info(f"DEBUG: game_data keys after initialize_game_data in training loop: {list(game_data.keys())}") # Removed Debug print
            game_data["dealer"] = random.randint(0, game_data["num_players"] - 1)

            game_over_flag = False
            round_num = 0
            while not game_over_flag:
                round_num += 1
                logging.info(f"Game {game_num}, Round {round_num} starting...")
                # Initialize new round (deals cards, sets up bidding)
                if game_data["game_phase"] != "setup": # if not first round of game
                    game_data["dealer"] = (game_data["dealer"] + 1) % game_data["num_players"]
                initialize_new_round()

                round_over_flag = False
                while not round_over_flag:
                    current_player_id = game_data["current_player_turn"]
                    current_phase = game_data["game_phase"] # Get current phase at start of iteration

                    # Check for terminal conditions *before* processing turn
                    if current_phase == "game_over":
                        game_over_flag = True
                        break
                    if current_phase == "round_over": # Indicates round ended, new one should start or game ends
                        round_over_flag = True
                        break

                    logging.debug(f"Game {game_num}, R{round_num}, Phase: {current_phase}, Turn: P{current_player_id}")

                    if current_player_id != 0: # AI Player's turn
                        if current_phase == "bidding_round_1":
                            process_ai_bid_action({'player_index': current_player_id, 'action': 'ai_bidding_round_1'})
                        elif current_phase == "bidding_round_2":
                            process_ai_bid_action({'player_index': current_player_id, 'action': 'ai_bidding_round_2'})
                        elif current_phase == "dealer_must_call":
                             process_ai_bid_action({'player_index': current_player_id, 'action': 'ai_dealer_stuck_call'})
                        elif current_phase == "prompt_go_alone":
                            ai_decide_go_alone_and_proceed(current_player_id)
                        elif current_phase == "playing_tricks":
                            process_ai_play_card(current_player_id)
                        elif current_phase == "maker_discard": # AI Maker discarding 5 cards
                            # This is handled by ai_decide_go_alone_and_proceed if not going alone.
                            # If somehow AI is in this phase directly, call discard logic.
                            if game_data["maker"] == current_player_id and game_data["cards_to_discard_count"] == 5:
                                ai_discard_five_cards(current_player_id)
                        elif current_phase in ["dealer_discard_one", "dealer_must_discard_after_order_up"]: # AI Dealer discarding 1
                             # This is handled by order_up logic if AI dealer.
                             # If AI dealer is here directly, it's likely a state error or needs specific handling.
                             # For now, assume primary flows cover AI discard.
                             pass
                        else:
                            logging.warning(f"Training Loop: AI P{current_player_id} in unhandled phase {current_phase}. Advancing turn by default.")
                            # This might indicate a missed RL update point or game logic needing refinement for simulation.
                            # For now, this simple advance might break game flow if AI was supposed to act.
                            # A robust simulation might need to force a valid (random/heuristic) action.
                            # game_data["current_player_turn"] = (current_player_id + 1) % game_data["num_players"]


                    else: # Player 0 (Human) - automate for simulation
                        logging.debug(f"Training Loop: Simulating P0 (Human) turn in phase {current_phase}")
                        if current_phase == "bidding_round_1":
                            # Simulate P0: pass for simplicity in training AI for now
                            game_data['passes_on_upcard'].append(0)
                            game_data["message"] = f"{game_data['player_identities'][0]} (Simulated) passes."
                            if len(game_data['passes_on_upcard']) == game_data["num_players"]:
                                game_data["game_phase"] = "bidding_round_2"; game_data["up_card_visible"] = False
                                game_data["current_player_turn"] = (game_data["dealer"] + 1) % game_data["num_players"]
                            else:
                                bid_order = [(game_data["dealer"] + i + 1) % game_data["num_players"] for i in range(game_data["num_players"])]
                                game_data["current_player_turn"] = bid_order[(bid_order.index(0) + 1) % game_data["num_players"]]

                        elif current_phase == "bidding_round_2":
                            # Simulate P0: pass
                            game_data['passes_on_calling'].append(0)
                            game_data["message"] = f"{game_data['player_identities'][0]} (Simulated) passes."
                            bid_order_r2 = [(game_data["dealer"] + i + 1) % game_data["num_players"] for i in range(game_data["num_players"])]
                            if len(game_data['passes_on_calling']) == game_data["num_players"] -1 and game_data["dealer"] not in game_data["passes_on_calling"]:
                                game_data["current_player_turn"] = game_data["dealer"]; game_data["game_phase"] = "dealer_must_call"
                            else:
                                game_data["current_player_turn"] = bid_order_r2[(bid_order_r2.index(0) + 1) % game_data["num_players"]]

                        elif current_phase == "dealer_must_call" and game_data["dealer"] == 0:
                            # Simulate P0: call first available suit
                            turned_down = game_data["original_up_card_for_round"].suit
                            suit_to_call = next(s for s in SUITS if s != turned_down)
                            game_data["trump_suit"] = suit_to_call; game_data["maker"] = 0
                            game_data["game_phase"] = "prompt_go_alone"; game_data["current_player_turn"] = 0

                        elif current_phase == "prompt_go_alone" and game_data["maker"] == 0:
                            # Simulate P0: don't go alone
                            game_data["going_alone"] = False
                            if game_data.get("dummy_hand") and len(game_data["dummy_hand"]) == 5 :
                                game_data["hands"][0].extend(game_data["dummy_hand"]); game_data["dummy_hand"] = []
                                game_data["cards_to_discard_count"] = 5; game_data["game_phase"] = "maker_discard"
                            else: transition_to_play_phase()

                        elif current_phase == "maker_discard" and game_data["maker"] == 0:
                            # Simulate P0: discard using heuristic
                            to_discard = get_ai_cards_to_discard(list(game_data["hands"][0]), 5, game_data["trump_suit"])
                            for card_obj in to_discard: # remove from hand
                                actual_card = next((c for c in game_data["hands"][0] if c.rank == card_obj.rank and c.suit == card_obj.suit), None)
                                if actual_card: game_data["hands"][0].remove(actual_card)
                            transition_to_play_phase()

                        elif current_phase in ["dealer_discard_one", "dealer_must_discard_after_order_up"] and game_data["dealer"] == 0:
                             # Simulate P0: discard using heuristic (1 card)
                            to_discard = get_ai_cards_to_discard(list(game_data["hands"][0]), 1, game_data["trump_suit"])
                            if to_discard:
                                actual_card = next((c for c in game_data["hands"][0] if c.rank == to_discard[0].rank and c.suit == to_discard[0].suit), None)
                                if actual_card: game_data["hands"][0].remove(actual_card)

                            game_data["game_phase"] = "prompt_go_alone"
                            game_data["current_player_turn"] = game_data["maker"] # Maker decides go alone next
                            if game_data["maker"] != 0: # If AI is maker
                                ai_decide_go_alone_and_proceed(game_data["maker"])


                        elif current_phase == "playing_tricks":
                            # Simulate P0: play first valid card
                            p0_hand = game_data["hands"][0]
                            logging.debug(f"P0 hand before get_valid_plays: {[str(c) for c in p0_hand]}")
                            if not p0_hand: # If P0 hand is empty but it's their turn to play.
                                logging.error(f"Training Loop: P0 (Human Sim) turn in 'playing_tricks' but hand is empty. Hand: {p0_hand}. Round tricks: {game_data['round_tricks_won']}. Total tricks: {sum(game_data['round_tricks_won'].values())}")
                                # This likely means the round should have already ended.
                                round_over_flag = True
                                break # Break from this player's turn processing in the while loop.

                            valid_plays_p0 = get_valid_plays(p0_hand, game_data["current_trick_lead_suit"], game_data["trump_suit"])
                            if valid_plays_p0:
                                card_played_p0 = valid_plays_p0[0]
                                p0_hand.remove(card_played_p0)
                                game_data["trick_cards"].append({'player': 0, 'card': card_played_p0})
                                if not game_data["current_trick_lead_suit"]: game_data["current_trick_lead_suit"] = get_effective_suit(card_played_p0, game_data["trump_suit"])

                                if len(game_data["trick_cards"]) == game_data["num_players"]:
                                    winner_idx = determine_trick_winner(game_data["trick_cards"], game_data["trump_suit"], game_data["current_trick_lead_suit"])
                                    game_data["round_tricks_won"][winner_idx] += 1
                                    # RL Update for AI players after P0 completes trick
                                    trick_event_data = {"trick_winner_idx": winner_idx}
                                    for p_id_ai in rl_agents.keys():
                                        if game_data["rl_training_data"].get(p_id_ai) and game_data["rl_training_data"][p_id_ai].get("action_type") == "play_card":
                                            process_rl_update(p_id_ai, "trick_end", event_data=trick_event_data)
                                    game_data["trick_cards"] = []; game_data["current_trick_lead_suit"] = None
                                    game_data["current_player_turn"] = winner_idx
                                    if is_round_over(): score_round() # score_round handles its RL updates
                                else:
                                    game_data["current_player_turn"] = (0 + 1) % game_data["num_players"]
                            else: # No valid cards for P0 - should not happen
                                logging.error("Training Loop: P0 has no valid cards to play. Game state error.")
                                round_over_flag = True # End round to prevent infinite loop
                        else:
                            logging.warning(f"Training Loop: P0 (Human Sim) in unhandled phase {current_phase}. Turn advanced.")
                            game_data["current_player_turn"] = (current_player_id + 1) % game_data["num_players"]

                    # Check for state consistency or errors that might stall the game
                    if game_data["game_phase"] == current_phase and game_data["current_player_turn"] == current_player_id and not game_over_flag and not round_over_flag :
                        # If no state change after a player's processing, it might be a loop.
                        # This is a basic check. More sophisticated checks might be needed.
                        # For AI turns, the functions should change current_player_turn or game_phase.
                        # For P0 simulation, it also needs to ensure state progression.
                        # logging.warning(f"Potential stall in training loop: P{current_player_id}, Phase: {current_phase}. Check logic.")
                        # Break if stuck, or implement a timeout for the loop iteration
                        pass # Allow one pass, if it happens again, then it's a problem.

                    # Check for terminal conditions *after* processing turn as well,
                    # because AI functions or P0 simulation might call score_round()
                    # which can change game_phase.
                    current_phase_after_action = game_data["game_phase"]
                    if current_phase_after_action == "game_over":
                        game_over_flag = True
                        # break # Will be caught by outer loop's while condition or next iteration's check
                    if current_phase_after_action == "round_over":
                        round_over_flag = True
                        # break # Will be caught by outer loop's while condition or next iteration's check

            logging.info(f"--- Training Game {game_num} ended. Scores: {game_data['scores']} ---")
            # Removed periodic save_q_table calls as DB is updated continuously.
            # if game_num % save_interval == 0:
            #     for agent_id, agent in rl_agents.items():
            #         pass # agent.save_q_table() removed

        # Final save is also not needed as DB is always up-to-date.
        # for agent_id, agent in rl_agents.items():
        #     pass # agent.save_q_table() removed
        logging.info(f"Training simulation finished for {num_games_to_simulate} games. Q-values are stored in {Q_TABLE_DB_FILE}.")

    # Example of how to run it (e.g. for 10 iterations as requested for testing)
    # run_training_simulation(10, save_interval=5) # save_interval is no longer used with SQLite

    # To migrate existing q_table.json to SQLite (run once):
    # migrate_json_to_sqlite()

    # To run the Flask app:
    app.run(debug=True, host='0.0.0.0')

def migrate_json_to_sqlite(json_file_path="q_table.json", db_file_path=None):
    """
    Migrates data from an old JSON Q-table file to the SQLite database.
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
