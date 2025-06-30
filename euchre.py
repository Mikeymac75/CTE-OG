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
            action_q_pairs = []
            for action_obj in valid_actions:
                action_key_for_qtable = self._serialize_action(action_obj)
                q_val = self.get_q_value(state_key, action_key_for_qtable)
                action_q_pairs.append({'action_obj': action_obj, 'q_value': q_val, 'serialized_key': action_key_for_qtable})

            log_q_values = {item['serialized_key']: item['q_value'] for item in action_q_pairs}

            max_q = -float('inf')
            if action_q_pairs:
                 max_q = max(item['q_value'] for item in action_q_pairs)

            best_actions = [item['action_obj'] for item in action_q_pairs if item['q_value'] == max_q]

            chosen_action = random.choice(best_actions) if best_actions else random.choice(valid_actions)
            logging.debug(f"RLAgent P{self.player_id} (Exploit): Q-values (serialized): {log_q_values}, Chose action_obj {self._serialize_action(chosen_action)} (Max Q: {max_q}) from best_actions (serialized keys): {[self._serialize_action(a) for a in best_actions]}")
            return chosen_action

    def _serialize_action(self, action):
        if isinstance(action, dict):
            return f"card_{action['suit']}{action['rank']}"
        return str(action)

    def update_q_table(self, state_dict, action, reward, next_state_dict, next_valid_actions):
        if not self.training_mode:
            return

        state_key = self._serialize_state(state_dict)
        action_key = self._serialize_action(action)
        next_state_key = self._serialize_state(next_state_dict)

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

        next_max_q = 0.0
        if next_valid_actions:
            cursor.execute("SELECT actions_q_values FROM q_values WHERE state_key = ?", (next_state_key,))
            next_row = cursor.fetchone()
            if next_row:
                try:
                    next_actions_q_dict = json.loads(next_row[0])
                    if next_actions_q_dict:
                         next_max_q = max(next_actions_q_dict.get(self._serialize_action(next_action), 0.0) for next_action in next_valid_actions)
                except json.JSONDecodeError:
                    logging.error(f"RLAgent P{self.player_id}: Failed to decode actions_q_values for next_state {next_state_key}.")

        new_value = old_value + self.lr * (reward + self.gamma * next_max_q - old_value)
        actions_q_dict[action_key] = new_value

        updated_actions_q_json = json.dumps(actions_q_dict)
        try:
            cursor.execute("""
                INSERT INTO q_values (state_key, actions_q_values) VALUES (?, ?)
                ON CONFLICT(state_key) DO UPDATE SET actions_q_values = excluded.actions_q_values
            """, (state_key, updated_actions_q_json))
            self.db_conn.commit()
        except sqlite3.Error as e:
            logging.error(f"RLAgent P{self.player_id}: SQLite error during Q-update: {e}")

        logging.debug(f"RLAgent P{self.player_id} Q-Update (DB): StateKey: {state_key}, ActionKey: {action_key}, Reward: {reward}, OldQ: {old_value:.2f}, NewQ: {new_value:.2f}, NextMaxQ: {next_max_q:.2f}")

        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(MIN_EPSILON, self.epsilon)

    def __del__(self):
        if hasattr(self, 'db_conn') and self.db_conn:
            self.db_conn.close()
            logging.info(f"RLAgent P{self.player_id}: SQLite DB connection closed.")

SUITS_MAP = {'H': 'Hearts', 'D': 'Diamonds', 'C': 'Clubs', 'S': 'Spades'}
SUITS = ['H', 'D', 'C', 'S']
RANKS = ['9', '10', 'J', 'Q', 'K', 'A']
VALUES = {'9': 1, '10': 2, 'J': 3, 'Q': 4, 'K': 5, 'A': 6}

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

class Game:
    def __init__(self):
        self.game_data = {}
        self.rl_agents = {}
        self._initialize_game_and_agents()

    def _initialize_game_and_agents(self):
        logging.info("Game class: Initializing game data and RL agents.")
        num_players = 3
        player_identities = {0: "Player 0 (You)", 1: "Player 1 (AI)", 2: "Player 2 (AI)"}

        self.rl_agents.clear()
        for pid, name in player_identities.items():
            if pid not in self.rl_agents:
                logging.info(f"Game class: Creating RLAgent for player {pid} ({name})")
                self.rl_agents[pid] = RLAgent(player_id=pid)

        for agent_id, agent in self.rl_agents.items():
            agent.epsilon = DEFAULT_EPSILON
            agent.set_training_mode(True)
            if hasattr(agent, 'last_action_info'):
                del agent.last_action_info

        self.game_data = {
            "deck": [], "hands": {p: [] for p in range(num_players)}, "dummy_hand": [],
            "scores": {p: 0 for p in range(num_players)}, "dealer": random.randint(0, num_players - 1),
            "rl_training_data": {p_id: {} for p_id in self.rl_agents.keys()},
            "trump_suit": None, "up_card": None, "up_card_visible": False,
            "current_player_turn": -1, "maker": None, "going_alone": False,
            "trick_cards": [], "current_trick_lead_suit": None,
            "trick_leader": -1, "round_tricks_won": {p: 0 for p in range(num_players)},
            "game_phase": "setup",
            "message": "Welcome! Click 'Start New Round'.",
            "player_identities": player_identities, "num_players": num_players,
            "passes_on_upcard": [], "passes_on_calling": [],
            "cards_to_discard_count": 0, "original_up_card_for_round": None,
            "last_completed_trick": None, "played_cards_this_round": []
        }
        logging.debug(f"Game class: Initialization complete. game_data keys: {list(self.game_data.keys())}")

    def get_data(self):
        return self.game_data

    def get_agent(self, player_id):
        return self.rl_agents.get(player_id)

_current_game_instance = None

def get_game_instance() -> Game:
    global _current_game_instance
    if _current_game_instance is None:
        _current_game_instance = Game()
    return _current_game_instance

def get_rl_agent(player_id):
    game = get_game_instance()
    return game.get_agent(player_id)

def get_player_role(player_id, dealer_id, maker_id, num_players):
    if player_id == dealer_id: return "dealer"
    if maker_id is None: return "bidder"
    if player_id == maker_id: return "maker"
    if player_id != maker_id: return "opponent"
    return "unknown"

def get_rl_state(player_id, current_game_data_dict):
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
        "up_card_suit": up_card_dict.suit if up_card_dict and current_game_data_dict.get("up_card_visible") else None,
        "up_card_rank": up_card_dict.rank if up_card_dict and current_game_data_dict.get("up_card_visible") else None,
        "original_up_card_suit": original_up_card_dict.suit if original_up_card_dict else None,
        "passes_on_upcard": len(current_game_data_dict.get("passes_on_upcard", [])),
        "passes_on_calling": len(current_game_data_dict.get("passes_on_calling", [])),
        "current_trick_lead_suit": current_game_data_dict.get("current_trick_lead_suit"),
        "trick_cards_played_count": len(current_game_data_dict.get("trick_cards", [])),
        "trick_leader": current_game_data_dict.get("trick_leader"),
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

    if state["game_phase"] == "bidding_round_1" and state["up_card_suit"]:
        potential_trump_features = get_hand_features(hand_card_objects, state["up_card_suit"])
        for f_name, f_val in potential_trump_features.items():
            state[f"potential_trump_{state['up_card_suit']}_{f_name}"] = f_val
        state[f"eval_strength_as_trump_{state['up_card_suit']}"] = evaluate_potential_trump_strength(hand_card_objects, state["up_card_suit"], current_game_data_dict)

    elif state["game_phase"] == "bidding_round_2":
        for s_option in SUITS:
            if s_option != state["original_up_card_suit"]:
                potential_trump_features_r2 = get_hand_features(hand_card_objects, s_option)
                for f_name, f_val in potential_trump_features_r2.items():
                    state[f"potential_trump_{s_option}_{f_name}"] = f_val
                state[f"eval_strength_as_trump_{s_option}"] = evaluate_potential_trump_strength(hand_card_objects, s_option, current_game_data_dict)

    played_cards_in_round = current_game_data_dict.get("played_cards_this_round", [])
    state["played_cards_serialized"] = sorted([f"{c.suit}{c.rank}" for c in played_cards_in_round])
    state["played_right_bower"] = False; state["played_left_bower"] = False
    state["played_ace_trump"] = False; state["played_king_trump"] = False; state["played_queen_trump"] = False
    state["num_trump_cards_played"] = 0
    for s_key in SUITS: state[f"num_suit_played_{s_key}"] = 0

    current_trump_suit = current_game_data_dict.get("trump_suit")
    if current_trump_suit:
        left_bower_actual_suit = get_left_bower_suit(current_trump_suit)
        for card in played_cards_in_round:
            state[f"num_suit_played_{card.suit}"] += 1
            effective_card_suit = get_effective_suit(card, current_trump_suit)
            if effective_card_suit == current_trump_suit:
                state["num_trump_cards_played"] += 1
                if card.rank == 'J':
                    if card.suit == current_trump_suit: state["played_right_bower"] = True
                    elif card.suit == left_bower_actual_suit: state["played_left_bower"] = True
                elif card.rank == 'A': state["played_ace_trump"] = True
                elif card.rank == 'K': state["played_king_trump"] = True
                elif card.rank == 'Q': state["played_queen_trump"] = True
    else:
        for card in played_cards_in_round: state[f"num_suit_played_{card.suit}"] += 1
    return state

def initialize_game_data():
    global _current_game_instance
    _current_game_instance = Game()
    logging.info("Global game instance has been re-initialized.")

def get_next_valid_actions(player_id, game_phase, game_data_for_next_state_dict):
    hand_cards = game_data_for_next_state_dict.get("hands", {}).get(player_id, [])
    if game_phase == "playing_tricks":
        valid_card_objects = get_valid_plays(list(hand_cards), game_data_for_next_state_dict.get("current_trick_lead_suit"), game_data_for_next_state_dict.get("trump_suit"))
        return [card.to_dict() for card in valid_card_objects]
    elif game_phase == "bidding_round_1": return ["order_up", "pass_bid"]
    elif game_phase == "bidding_round_2":
        original_up_card = game_data_for_next_state_dict.get("original_up_card_for_round")
        turned_down_suit = original_up_card.suit if original_up_card else None
        return [f"call_trump:{s}" for s in SUITS if s != turned_down_suit] + ["pass_call"]
    elif game_phase == "dealer_must_call":
        original_up_card = game_data_for_next_state_dict.get("original_up_card_for_round")
        turned_down_suit = original_up_card.suit if original_up_card else None
        return [f"call_trump:{s}" for s in SUITS if s != turned_down_suit]
    elif game_phase == "prompt_go_alone": return ["choose_go_alone", "choose_not_go_alone"]
    elif game_phase in ["dealer_discard_one", "dealer_must_discard_after_order_up"]:
        if game_data_for_next_state_dict.get("cards_to_discard_count") == 1 and game_data_for_next_state_dict.get("current_player_turn") == player_id:
             return [card.to_dict() for card in hand_cards]
    return []

def process_rl_update(player_id_acted, event_type, event_data=None):
    game = get_game_instance()
    current_game_data = game.game_data
    agent = game.get_agent(player_id_acted)
    if not agent or not agent.training_mode: return
    training_info = current_game_data["rl_training_data"].get(player_id_acted)
    if not training_info or "state" not in training_info or "action" not in training_info: return
    if training_info["action"] is None:
        current_game_data["rl_training_data"][player_id_acted] = {}; return

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
        else:
            maker_tricks_won = current_game_data["round_tricks_won"].get(round_maker, 0)
            if maker_tricks_won < 3: reward += REWARD_WIN_ROUND_DEFENSE_EUCHRE
            else: reward += REWARD_LOSE_ROUND_DEFENSE
    elif event_type == "game_end":
        game_winner_idx = event_data.get("game_winner_idx")
        if game_winner_idx == player_id_acted: reward += REWARD_WIN_GAME
        else: reward += REWARD_LOSE_GAME
    elif event_type == "bid_processed":
        if training_info.get("action_type") in ["ai_bidding_round_1", "ai_bidding_round_2", "ai_dealer_stuck_call", "decide_go_alone"] and current_game_data.get("maker") == player_id_acted :
             pass

    next_state_dict = get_rl_state(player_id_acted, current_game_data)
    next_player_for_q_learning = current_game_data.get("current_player_turn", -1)
    next_valid_actions_for_q = []
    if current_game_data.get("game_phase") not in ["game_over", "round_over", "setup"]:
        if next_player_for_q_learning != -1:
            if game.get_agent(next_player_for_q_learning):
                 next_valid_actions_for_q = get_next_valid_actions(next_player_for_q_learning, current_game_data["game_phase"], current_game_data)

    logging.debug(f"RL Update P{player_id_acted}: Event: {event_type}, Action: {training_info['action']}, Reward: {reward}")
    agent.update_q_table(training_info["state"], training_info["action"], reward, next_state_dict, next_valid_actions_for_q)
    current_game_data["rl_training_data"][player_id_acted] = {}

def initialize_new_round():
    game = get_game_instance()
    current_game_data = game.game_data
    # IMPORTANT: All direct modifications below should be to current_game_data, NOT the old global 'game_data'
    logging.info(f"Initializing new round. Current dealer: {current_game_data.get('dealer', 'N/A')}")
    current_game_data["deck"] = create_deck()
    random.shuffle(current_game_data["deck"])
    player_hands = {i: [] for i in range(current_game_data["num_players"])}
    for _ in range(5):
        for i in range(current_game_data["num_players"]):
            if current_game_data["deck"]:
                player_hands[i].append(current_game_data["deck"].pop())
            else:
                current_game_data["message"] = "Error: Not enough cards for player hands."; return
    current_game_data["hands"] = player_hands
    current_game_data["dummy_hand"] = []
    for _ in range(5):
        if current_game_data["deck"]:
            current_game_data["dummy_hand"].append(current_game_data["deck"].pop())
        else:
            current_game_data["message"] = "Error: Not enough cards for dummy hand."; return
    if not current_game_data["deck"]: current_game_data["message"] = "Error: Not enough for up_card."; return
    current_game_data["up_card"] = current_game_data["deck"].pop()
    current_game_data["original_up_card_for_round"] = current_game_data["up_card"]
    if "kitty" in current_game_data: del current_game_data["kitty"]
    current_game_data["up_card_visible"] = True
    current_game_data["trump_suit"] = None
    current_game_data["maker"] = None
    current_game_data["going_alone"] = False
    current_game_data["trick_cards"] = []
    current_game_data["current_trick_lead_suit"] = None
    current_game_data["round_tricks_won"] = {i: 0 for i in range(current_game_data["num_players"])}
    current_game_data["dealer"] = int(current_game_data["dealer"])
    current_game_data["current_player_turn"] = (current_game_data["dealer"] + 1) % current_game_data["num_players"]
    current_game_data["trick_leader"] = current_game_data["current_player_turn"]
    current_game_data["game_phase"] = "bidding_round_1"
    current_game_data["passes_on_upcard"] = []
    current_game_data["passes_on_calling"] = []
    current_game_data["cards_to_discard_count"] = 0
    current_game_data["last_completed_trick"] = None
    current_game_data["played_cards_this_round"] = []
    current_game_data["message"] = f"{current_game_data['player_identities'][current_game_data['current_player_turn']]}'s turn. Up-card: {str(current_game_data['up_card'])}."
    logging.info(f"New round initialized. Up-card: {str(current_game_data['up_card']) if current_game_data['up_card'] else 'N/A'}. Turn: P{current_game_data['current_player_turn']}. Phase: {current_game_data['game_phase']}")

def get_left_bower_suit(trump_suit_char):
    if trump_suit_char not in SUITS_MAP: return None
    return {'H': 'D', 'D': 'H', 'C': 'S', 'S': 'C'}.get(trump_suit_char)

def get_effective_suit(card: Card, trump_suit: str | None) -> str:
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

def get_hand_features(hand: list[Card], trump_suit: str | None) -> dict:
    features = {"num_trump_cards": 0, "has_right_bower": False, "has_left_bower": False, "has_ace_of_trump": False, "has_king_of_trump": False, "has_queen_of_trump": False, "num_aces_offsuit": 0, "num_suits_void_natural": 0, "is_void_in_suit_H": True, "is_void_in_suit_D": True, "is_void_in_suit_C": True, "is_void_in_suit_S": True, "highest_trump_card_rank_value": 0, "lowest_trump_card_rank_value": 0, "num_cards_in_longest_offsuit": 0, "shortest_offsuit_length": 5}
    if not hand: features["shortest_offsuit_length"] = 0; features["num_suits_void_natural"] = 4; return features
    natural_suit_counts = {s: 0 for s in SUITS}
    left_bower_actual_suit = get_left_bower_suit(trump_suit) if trump_suit else None
    for card in hand:
        natural_suit_counts[card.suit] += 1
        features[f"is_void_in_suit_{card.suit}"] = False
        effective_suit = get_effective_suit(card, trump_suit)
        if trump_suit and effective_suit == trump_suit:
            features["num_trump_cards"] += 1
            card_value_in_trump = get_card_value(card, trump_suit)
            if features["lowest_trump_card_rank_value"] == 0 or card_value_in_trump < features["lowest_trump_card_rank_value"]: features["lowest_trump_card_rank_value"] = card_value_in_trump
            if card_value_in_trump > features["highest_trump_card_rank_value"]: features["highest_trump_card_rank_value"] = card_value_in_trump
            if card.rank == 'J':
                if card.suit == trump_suit: features["has_right_bower"] = True
                elif card.suit == left_bower_actual_suit: features["has_left_bower"] = True
            elif card.rank == 'A': features["has_ace_of_trump"] = True
            elif card.rank == 'K': features["has_king_of_trump"] = True
            elif card.rank == 'Q': features["has_queen_of_trump"] = True
        elif card.rank == 'A': features["num_aces_offsuit"] += 1
    for suit_key in SUITS:
        if natural_suit_counts[suit_key] == 0: features["num_suits_void_natural"] += 1
    max_len = 0; min_len = 5; found_an_offsuit = False
    for s_key in SUITS:
        is_suit_effectively_trump = (trump_suit and (s_key == trump_suit or s_key == left_bower_actual_suit))
        if not is_suit_effectively_trump:
            found_an_offsuit = True
            current_suit_len = natural_suit_counts[s_key]
            if current_suit_len > max_len: max_len = current_suit_len
            if current_suit_len < min_len: min_len = current_suit_len
    if trump_suit is None:
        if not hand: features["num_cards_in_longest_offsuit"] = 0; features["shortest_offsuit_length"] = 0
        else: all_natural_lengths = [natural_suit_counts[s_k] for s_k in SUITS]; features["num_cards_in_longest_offsuit"] = max(all_natural_lengths) if all_natural_lengths else 0; features["shortest_offsuit_length"] = min(all_natural_lengths) if all_natural_lengths else 0
    else:
        if found_an_offsuit: features["num_cards_in_longest_offsuit"] = max_len; features["shortest_offsuit_length"] = min_len
        else: features["num_cards_in_longest_offsuit"] = 0; features["shortest_offsuit_length"] = 0
        if not hand: features["num_suits_void_natural"] = 4
    return features

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

def get_ai_cards_to_discard(hand: list[Card], num_to_discard: int, trump_suit: str | None) -> list[Card]:
    hand_copy = list(hand)
    hand_copy.sort(key=lambda c: get_card_value(c, trump_suit, None))
    return hand_copy[:num_to_discard]

def get_ai_stuck_suit_call(hand: list[Card], turned_down_suit: str) -> str:
    possible_suits = [s for s in SUITS if s != turned_down_suit]
    if not possible_suits: return random.choice(SUITS)
    best_suit, max_strength = "", -1
    for s_key in possible_suits:
        strength = sum(get_card_value(c, s_key) for c in hand)
        if strength > max_strength: max_strength, best_suit = strength, s_key
    return best_suit if best_suit else random.choice(possible_suits)

def transition_to_play_phase():
    game = get_game_instance()
    current_game_data = game.game_data
    current_game_data["game_phase"] = "playing_tricks"
    current_game_data["current_player_turn"] = (current_game_data["dealer"] + 1) % current_game_data["num_players"]
    current_game_data["trick_leader"] = current_game_data["current_player_turn"]
    current_game_data["message"] += f" {current_game_data['player_identities'][current_game_data['current_player_turn']]} leads the first trick."
    current_game_data.pop('passes_on_upcard', None); current_game_data.pop('passes_on_calling', None)
    if current_game_data["game_phase"] == "playing_tricks" and current_game_data["current_player_turn"] != 0:
        process_ai_play_card(current_game_data["current_player_turn"])

def process_ai_play_card(ai_player_idx: int):
    game = get_game_instance()
    current_game_data = game.game_data
    if current_game_data["game_phase"] != "playing_tricks" or current_game_data["current_player_turn"] != ai_player_idx: return

    current_agent = get_rl_agent(ai_player_idx) # Uses get_game_instance()
    if not current_agent:
        logging.error(f"RL Agent for P{ai_player_idx} not found in process_ai_play_card. Falling back to heuristic logic.")
        ai_hand_fallback = current_game_data["hands"][ai_player_idx]
        lead_suit_fallback = current_game_data["current_trick_lead_suit"]
        trump_suit_fallback = current_game_data["trump_suit"]
        valid_cards_fallback = get_valid_plays(list(ai_hand_fallback), lead_suit_fallback, trump_suit_fallback)
        if not valid_cards_fallback:
            current_game_data["message"] += f" Error: AI {current_game_data['player_identities'][ai_player_idx]} (Fallback) has no valid cards to play."
            if len(current_game_data["trick_cards"]) < current_game_data["num_players"]:
                current_game_data["current_player_turn"] = (ai_player_idx + 1) % current_game_data["num_players"]
            return
        card_to_play = None
        can_follow_lead_suit = False
        if lead_suit_fallback: can_follow_lead_suit = any(get_effective_suit(card, trump_suit_fallback) == lead_suit_fallback for card in ai_hand_fallback)
        if can_follow_lead_suit:
            valid_cards_fallback.sort(key=lambda c: get_card_value(c, trump_suit_fallback, lead_suit_fallback), reverse=True)
            card_to_play = valid_cards_fallback[0]
        else:
            trump_options = [card for card in valid_cards_fallback if get_effective_suit(card, trump_suit_fallback) == trump_suit_fallback]
            non_trump_options = [card for card in valid_cards_fallback if get_effective_suit(card, trump_suit_fallback) != trump_suit_fallback]
            can_win_with_trump = False; best_trump_to_play = None
            if trump_options:
                winning_trumps = [tc for tc in trump_options if determine_trick_winner(current_game_data["trick_cards"] + [{'player': ai_player_idx, 'card': tc}], trump_suit_fallback, lead_suit_fallback if lead_suit_fallback else get_effective_suit(tc, trump_suit_fallback)) == ai_player_idx]
                if winning_trumps: can_win_with_trump = True; winning_trumps.sort(key=lambda c: get_card_value(c, trump_suit_fallback, lead_suit_fallback), reverse=True); best_trump_to_play = winning_trumps[0]
            if can_win_with_trump and best_trump_to_play: card_to_play = best_trump_to_play
            elif non_trump_options: non_trump_options.sort(key=lambda c: get_card_value(c, trump_suit_fallback, lead_suit_fallback)); card_to_play = non_trump_options[0]
            elif trump_options: trump_options.sort(key=lambda c: get_card_value(c, trump_suit_fallback, lead_suit_fallback)); card_to_play = trump_options[0]
            else: card_to_play = valid_cards_fallback[0]
        if not card_to_play: card_to_play = valid_cards_fallback[0]
    else: # RL Agent Logic
        state_dict = get_rl_state(ai_player_idx, current_game_data)
        current_game_data["rl_training_data"][ai_player_idx] = {"state": state_dict, "action": None, "action_type": "play_card"}
        ai_hand = current_game_data["hands"][ai_player_idx]
        lead_suit = current_game_data["current_trick_lead_suit"]
        trump_suit = current_game_data["trump_suit"]
        valid_card_objects = get_valid_plays(list(ai_hand), lead_suit, trump_suit)
        if not valid_card_objects:
            current_game_data["message"] += f" Error: AI {ai_player_idx+1} (RL) has no valid cards to play."
            if len(current_game_data["trick_cards"]) < current_game_data["num_players"]: current_game_data["current_player_turn"] = (ai_player_idx + 1) % current_game_data["num_players"]
            return
        valid_actions_for_agent = [c.to_dict() for c in valid_card_objects]
        chosen_action_dict = current_agent.choose_action(state_dict, valid_actions_for_agent)
        current_game_data["rl_training_data"][ai_player_idx]["action"] = chosen_action_dict
        if chosen_action_dict is None: card_to_play = valid_card_objects[0]
        else:
            card_to_play = next((c for c in ai_hand if c.suit == chosen_action_dict['suit'] and c.rank == chosen_action_dict['rank']), None)
            if not card_to_play: card_to_play = valid_card_objects[0]
        # Heuristic Overlay
        can_follow_lead_suit_rl = any(get_effective_suit(c, trump_suit) == lead_suit for c in ai_hand) if lead_suit else False
        if not can_follow_lead_suit_rl and card_to_play:
            chosen_card_effective_suit = get_effective_suit(card_to_play, trump_suit)
            if chosen_card_effective_suit != trump_suit:
                hypothetical_trick_after_play = current_game_data["trick_cards"] + [{'player': ai_player_idx, 'card': card_to_play}]
                eval_lead_suit_for_rl = lead_suit if lead_suit else chosen_card_effective_suit
                is_chosen_card_winning = determine_trick_winner(hypothetical_trick_after_play, trump_suit, eval_lead_suit_for_rl) == ai_player_idx
                if not is_chosen_card_winning:
                    sloughable_cards = [c_obj for c_obj in valid_card_objects if get_effective_suit(c_obj, trump_suit) != trump_suit]
                    if sloughable_cards:
                        sloughable_cards.sort(key=lambda c_obj: get_card_value(c_obj, trump_suit, lead_suit))
                        lowest_value_slough_card = sloughable_cards[0]
                        if get_card_value(card_to_play, trump_suit, lead_suit) > get_card_value(lowest_value_slough_card, trump_suit, lead_suit):
                            card_to_play = lowest_value_slough_card
                            logging.info(f"AI P{ai_player_idx} (RL Agent Heuristic Override) to play lowest slough: {str(card_to_play)}")
    # Common logic
    ai_hand_to_modify = current_game_data["hands"][ai_player_idx]
    actual_card_to_remove = next((c for c in ai_hand_to_modify if c.suit == card_to_play.suit and c.rank == card_to_play.rank), None)
    if actual_card_to_remove:
        try: ai_hand_to_modify.remove(actual_card_to_remove); current_game_data.get("played_cards_this_round", []).append(actual_card_to_remove)
        except ValueError: logging.error(f"CRITICAL: Failed to remove {str(actual_card_to_remove)} from AI P{ai_player_idx}'s hand."); return
    else: logging.error(f"CRITICAL: AI P{ai_player_idx} card {str(card_to_play)} not found in hand."); return

    current_game_data["trick_cards"].append({'player': ai_player_idx, 'card': card_to_play})
    current_game_data["message"] = f"{current_game_data['player_identities'][ai_player_idx]} played {str(card_to_play)}."
    if not current_game_data["current_trick_lead_suit"]: current_game_data["current_trick_lead_suit"] = get_effective_suit(card_to_play, current_game_data["trump_suit"])

    if len(current_game_data["trick_cards"]) == current_game_data["num_players"]:
        winner_idx = determine_trick_winner(current_game_data["trick_cards"], current_game_data["trump_suit"], current_game_data["current_trick_lead_suit"])
        current_game_data["round_tricks_won"][winner_idx] += 1
        current_game_data["last_completed_trick"] = {"played_cards": [tc.copy() for tc in current_game_data["trick_cards"]], "winner_player_idx": winner_idx, "winner_name": current_game_data['player_identities'][winner_idx]}
        trick_event_data = {"trick_winner_idx": winner_idx} # Simplified
        for p_id in current_game_data["player_identities"].keys():
            if p_id != 0 and current_game_data["rl_training_data"].get(p_id) and current_game_data["rl_training_data"][p_id].get("action_type") == "play_card":
                process_rl_update(p_id, "trick_end", event_data=trick_event_data)
        current_game_data["message"] += f" {current_game_data['player_identities'][winner_idx]} wins the trick."
        current_game_data["trick_cards"] = []; current_game_data["current_trick_lead_suit"] = None; current_game_data["current_player_turn"] = winner_idx; current_game_data["trick_leader"] = winner_idx
        if is_round_over(): score_round() # score_round now uses get_game_instance
        else: current_game_data["message"] += f" {current_game_data['player_identities'][winner_idx]} leads next trick."
    else:
        current_game_data["current_player_turn"] = (ai_player_idx + 1) % current_game_data["num_players"]
        current_game_data["message"] += f" Next turn: {current_game_data['player_identities'][current_game_data['current_player_turn']]}."

def get_valid_plays(hand, lead_suit, trump_suit):
    if not lead_suit: return list(hand)
    can_follow_suit = any(get_effective_suit(card, trump_suit) == lead_suit for card in hand)
    if can_follow_suit: return [card for card in hand if get_effective_suit(card, trump_suit) == lead_suit]
    else: return list(hand)

def determine_trick_winner_so_far(trick_cards_played, trump_suit, lead_suit_of_trick):
    if not trick_cards_played: return None
    winning_player_idx = -1; highest_value_card_obj = None
    actual_lead_suit = lead_suit_of_trick if lead_suit_of_trick else get_effective_suit(trick_cards_played[0]['card'], trump_suit)
    for play in trick_cards_played:
        card_eff_suit = get_effective_suit(play['card'], trump_suit)
        card_val = get_card_value(play['card'], trump_suit, actual_lead_suit)
        if highest_value_card_obj is None:
            highest_value_card_obj = play['card']; winning_player_idx = play['player']
        else:
            high_eff_suit = get_effective_suit(highest_value_card_obj, trump_suit)
            high_val_comp = get_card_value(highest_value_card_obj, trump_suit, actual_lead_suit)
            is_curr_trump = (card_eff_suit == trump_suit)
            is_high_trump = (high_eff_suit == trump_suit)
            if is_curr_trump and not is_high_trump: highest_value_card_obj = play['card']; winning_player_idx = play['player']
            elif is_curr_trump and is_high_trump:
                if card_val > high_val_comp: highest_value_card_obj = play['card']; winning_player_idx = play['player']
            elif not is_curr_trump and not is_high_trump:
                if card_eff_suit == actual_lead_suit and high_eff_suit != actual_lead_suit: highest_value_card_obj = play['card']; winning_player_idx = play['player']
                elif card_eff_suit == actual_lead_suit and high_eff_suit == actual_lead_suit:
                    if card_val > high_val_comp: highest_value_card_obj = play['card']; winning_player_idx = play['player']
    return {'player': winning_player_idx, 'card': highest_value_card_obj} if winning_player_idx != -1 else None

def determine_trick_winner(trick_cards_played, trump_suit, lead_suit_of_trick=None):
    if not trick_cards_played: return -1
    if lead_suit_of_trick is None: lead_suit_of_trick = get_effective_suit(trick_cards_played[0]['card'], trump_suit)
    winner_info = determine_trick_winner_so_far(trick_cards_played, trump_suit, lead_suit_of_trick)
    return winner_info['player'] if winner_info else -1

def is_round_over():
    game = get_game_instance()
    current_game_data = game.game_data
    # This function might be called when current_game_data is not fully populated yet (e.g. during setup)
    if "round_tricks_won" not in current_game_data or not isinstance(current_game_data["round_tricks_won"], dict):
        return False # Or handle as an error, depending on expected call points
    return sum(current_game_data["round_tricks_won"].values()) >= 5

def score_round():
    game = get_game_instance()
    current_game_data = game.game_data
    # Global rl_agents is problematic here, should use game.rl_agents
    # For now, this direct use of current_game_data for scores is okay.

    if current_game_data.get("game_phase") in ["round_over", "game_over"]:
        logging.warning(f"score_round() called when game_phase is already {current_game_data.get('game_phase')}. Points not re-awarded.")
        return

    maker = current_game_data["maker"]
    if maker is None:
        logging.error("CRITICAL: score_round() called but maker is None."); current_game_data["message"] = "Error: Scoring aborted."; current_game_data["game_phase"] = "round_over"; return

    maker_tricks = current_game_data["round_tricks_won"][maker]
    points_awarded = 0; message_suffix = ""; is_going_alone = current_game_data.get("going_alone", False)

    if maker_tricks < 3: # Euchred
        points_awarded = 2
        for p_idx in range(current_game_data["num_players"]):
            if p_idx != maker: current_game_data["scores"][p_idx] += points_awarded
        message_suffix = f"Maker euchred! Opponents score {points_awarded} points each."
    elif maker_tricks == 5: # March
        points_awarded = 4 if is_going_alone else 2
        message_suffix = f"Maker ({current_game_data['player_identities'][maker]}) {'went alone and ' if is_going_alone else ''}marches, scoring {points_awarded} points!"
        current_game_data["scores"][maker] += points_awarded
    elif maker_tricks >= 3: # Made points
        points_awarded = 1
        message_suffix = f"Maker ({current_game_data['player_identities'][maker]}) {'went alone and ' if is_going_alone else ''}scores {points_awarded} point."
        current_game_data["scores"][maker] += points_awarded

    current_game_data["message"] = f"Round Over. {message_suffix}"
    current_game_data["game_phase"] = "round_over"

    game_winner_found = False
    for p_idx, score in current_game_data["scores"].items():
        if score >= 10:
            current_game_data["game_phase"] = "game_over"
            current_game_data["message"] += f" {current_game_data['player_identities'][p_idx]} wins the game!"
            game_event_data = {"game_winner_idx": p_idx}
            # Iterate over agents from the game instance
            for ai_p_id in game.rl_agents.keys(): # Corrected: use game.rl_agents
                process_rl_update(ai_p_id, "game_end", event_data=game_event_data)
            game_winner_found = True; break

    if not game_winner_found:
        round_event_data = {"round_tricks_won": current_game_data["round_tricks_won"].copy(), "maker": maker, "going_alone": is_going_alone}
        for ai_p_id in game.rl_agents.keys(): # Corrected: use game.rl_agents
            process_rl_update(ai_p_id, "round_end", event_data=round_event_data)

@app.route('/')
def index_route(): return render_template('index.html')
@app.route('/style.css')
def styles_route(): return send_from_directory('.', 'style.css')
@app.route('/script.js')
def scripts_route(): return send_from_directory('.', 'script.js')

@app.route('/api/start_game', methods=['GET'])
def start_game_api():
    game = get_game_instance() # Use the game instance
    current_game_data = game.game_data
    current_phase = current_game_data.get("game_phase")
    if current_phase is None or current_phase in ["game_over", "setup"]:
        initialize_game_data() # This re-creates _current_game_instance
        game = get_game_instance() # Get the new instance
        current_game_data = game.game_data
    else:
        current_game_data["dealer"] = (current_game_data["dealer"] + 1) % current_game_data["num_players"]
    initialize_new_round() # This function needs to use game.game_data
    # Re-fetch current_game_data as initialize_new_round modifies it
    current_game_data = get_game_instance().game_data
    if current_game_data["game_phase"] == "bidding_round_1" and current_game_data["current_player_turn"] != 0:
        process_ai_bid_action({'player_index': current_game_data["current_player_turn"], 'action': 'ai_bidding_round_1'})
    return jsonify(game_data_to_json(current_game_data)) # Pass current_game_data

@app.route('/api/submit_action', methods=['POST'])
def submit_action_api():
    game = get_game_instance() # Use the game instance
    current_game_data = game.game_data # Work with the instance's data

    action_data = request.json
    player_idx = action_data.get('player_index')
    action_type = action_data.get('action')
    current_phase = current_game_data["game_phase"] # Use current_game_data
    # Logging and validation using current_game_data...
    # All direct uses of 'game_data' in this function should be 'current_game_data'

    # Example of change needed:
    # if current_phase == 'maker_discard':
    #    if player_idx != current_game_data.get("maker"): ...
    # ... and so on for all references to game_data.

    # --- ACTION HANDLERS (needs full refactor to use current_game_data) ---
    if action_type == 'order_up':
        if current_phase != 'bidding_round_1': return jsonify({"error": "Cannot order up now."}), 400
        if not current_game_data.get("original_up_card_for_round"): return jsonify({"error": "Critical error: Up-card data missing."}), 500
        current_game_data["trump_suit"] = current_game_data["original_up_card_for_round"].suit
        current_game_data["maker"] = player_idx
        # ... (continue refactoring all game_data to current_game_data) ...
        current_dealer_idx = current_game_data["dealer"]
        dealer_hand = current_game_data["hands"][current_dealer_idx]
        up_card_to_pickup = current_game_data["original_up_card_for_round"]
        dealer_hand.append(up_card_to_pickup)
        if current_dealer_idx == 0: # Human Dealer
            current_game_data["cards_to_discard_count"] = 1
            current_game_data["current_player_turn"] = current_dealer_idx
            if player_idx == current_dealer_idx: current_game_data["game_phase"] = "dealer_discard_one"
            else: current_game_data["game_phase"] = "dealer_must_discard_after_order_up"
        else: # AI Dealer
            # ... (refactor game_data to current_game_data) ...
            cards_to_discard_ai = get_ai_cards_to_discard(list(dealer_hand), 1, current_game_data["trump_suit"])
            # ...
            current_game_data["current_player_turn"] = current_game_data["maker"]
            current_game_data["game_phase"] = "prompt_go_alone"
            if current_game_data["maker"] != 0 : ai_decide_go_alone_and_proceed(current_game_data["maker"])
        current_game_data["message"] = f"{current_game_data['player_identities'][player_idx]} ordered up." # Simplified
    # ... (Other action types need similar refactoring) ...
    elif action_type == 'play_card':
        # ... ensure all game_data is current_game_data ...
        player_hand = current_game_data["hands"][player_idx]
        # ...
        if len(current_game_data["trick_cards"]) == current_game_data["num_players"]:
            # ...
            if is_round_over(): score_round() # score_round itself needs to use get_game_instance()
            # ...
    # ...
    # After action, re-fetch current_game_data if it could have been changed by internal calls
    final_game_state_to_return = get_game_instance().game_data
    return jsonify(game_data_to_json(final_game_state_to_return))


@app.route('/api/get_current_state', methods=['GET'])
def get_current_state_api():
    current_game_data = get_game_instance().game_data
    logging.info("API: /api/get_current_state called.")
    return jsonify(game_data_to_json(current_game_data))

@app.route('/api/ai_play_turn', methods=['POST'])
def ai_play_turn_api():
    logging.info("Received request for AI to play turn endpoint.")
    game = get_game_instance()
    current_game_data_snapshot = game.game_data.copy() # Use instance data
    current_player = current_game_data_snapshot.get("current_player_turn")
    current_phase = current_game_data_snapshot.get("game_phase")

    if current_player == 0: return jsonify({"error": "Not AI's turn.", "game_state": game_data_to_json(game.game_data)}), 400
    if current_phase != "playing_tricks": return jsonify({"error": f"AI phase is '{current_phase}'.", "game_state": game_data_to_json(game.game_data)}), 400

    # is_round_over and score_round internally use get_game_instance()
    if is_round_over():
        score_round()
        # Re-check phase from the authoritative source after score_round
        if get_game_instance().game_data.get("game_phase") in ["round_over", "game_over"]:
            return jsonify(game_data_to_json(get_game_instance().game_data))

    if get_game_instance().game_data.get("game_phase") == "playing_tricks": # Check authoritative phase again
        process_ai_play_card(current_player) # This must use get_game_instance() internally for game_data

    return jsonify(game_data_to_json(get_game_instance().game_data))

def game_data_to_json(game_data_arg: dict) -> dict: # game_data_arg is now the authoritative data
    json_safe_data = game_data_arg.copy() # Work with the passed data
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

# --- Training Simulation ---
def run_training_simulation(num_games_to_simulate: int, save_interval: int = 10):
    logging.info(f"Starting RL training simulation for {num_games_to_simulate} games.")
    initialize_game_data()

    for game_num in range(1, num_games_to_simulate + 1):
        logging.info(f"--- Starting Training Game {game_num} ---")
        initialize_game_data()
        game = get_game_instance()
        current_game_data = game.game_data
        current_game_data["dealer"] = random.randint(0, current_game_data["num_players"] - 1)

        game_over_flag = False
        round_num = 0
        while not game_over_flag:
            round_num += 1
            logging.info(f"Game {game_num}, Round {round_num} starting...")
            current_game_data = get_game_instance().game_data # Refresh
            if current_game_data["game_phase"] != "setup":
                current_game_data["dealer"] = (current_game_data["dealer"] + 1) % current_game_data["num_players"]
            initialize_new_round()

            round_over_flag = False
            while not round_over_flag:
                current_game_data = get_game_instance().game_data # Refresh
                current_player_id = current_game_data["current_player_turn"]
                current_phase = current_game_data["game_phase"]

                if current_phase == "game_over": game_over_flag = True; break
                if current_phase == "round_over": round_over_flag = True; break
                logging.debug(f"Game {game_num}, R{round_num}, Phase: {current_phase}, Turn: P{current_player_id}")

                if current_phase == "bidding_round_1": process_ai_bid_action({'player_index': current_player_id, 'action': 'ai_bidding_round_1'})
                elif current_phase == "bidding_round_2": process_ai_bid_action({'player_index': current_player_id, 'action': 'ai_bidding_round_2'})
                elif current_phase == "dealer_must_call": process_ai_bid_action({'player_index': current_player_id, 'action': 'ai_dealer_stuck_call'})
                elif current_phase == "prompt_go_alone": ai_decide_go_alone_and_proceed(current_player_id)
                elif current_phase == "playing_tricks": process_ai_play_card(current_player_id)
                elif current_phase == "maker_discard":
                    if current_game_data["maker"] == current_player_id and current_game_data["cards_to_discard_count"] == 5: ai_discard_five_cards(current_player_id)
                elif current_phase in ["dealer_discard_one", "dealer_must_discard_after_order_up"]:
                    if current_player_id == current_game_data["dealer"]:
                        # ... (AI dealer discard logic, ensure it uses current_game_data)
                        dealer_hand = current_game_data["hands"][current_player_id]
                        trump_suit = current_game_data["trump_suit"]
                        # ... (rest of the logic from previous version, adapted for current_game_data)
                        if not dealer_hand or not trump_suit: logging.error("Dealer hand or trump missing for AI discard"); #... error handling
                        else:
                            cards_to_discard = get_ai_cards_to_discard(list(dealer_hand), 1, trump_suit)
                            if cards_to_discard:
                                card_to_discard_obj = cards_to_discard[0]
                                try:
                                    actual_card_to_remove = next(c for c in dealer_hand if c.suit == card_to_discard_obj.suit and c.rank == card_to_discard_obj.rank)
                                    dealer_hand.remove(actual_card_to_remove)
                                    current_game_data["message"] = f"{current_game_data['player_identities'][current_player_id]} (AI Dealer) discarded."
                                except (ValueError, StopIteration): logging.error("AI Dealer discard failed.")
                                current_game_data["game_phase"] = "prompt_go_alone"
                                current_game_data["current_player_turn"] = current_game_data["maker"]
                                current_game_data["cards_to_discard_count"] = 0
                else: logging.error(f"Training Loop: UNHANDLED phase {current_phase}"); game_over_flag = True; round_over_flag = True

                current_game_data = get_game_instance().game_data # Refresh
                current_phase_after_action = current_game_data["game_phase"]
                if current_game_data.get("current_player_turn") == current_player_id and current_phase_after_action == current_phase and not game_over_flag and not round_over_flag:
                    logging.warning(f"Potential stall in training: P{current_player_id}, Phase: {current_phase}.")
                if current_phase_after_action == "game_over": game_over_flag = True
                if current_phase_after_action == "round_over": round_over_flag = True
            logging.info(f"--- Training Game {game_num} ended. Scores: {get_game_instance().game_data['scores']} ---")
        logging.info(f"Training simulation finished for {num_games_to_simulate} games.")

def migrate_json_to_sqlite(json_file_path="q_table.json", db_file_path=None):
    if db_file_path is None: db_file_path = Q_TABLE_DB_FILE
    if not os.path.exists(json_file_path): logging.info(f"JSON file {json_file_path} not found."); return
    try:
        with open(json_file_path, 'r') as f: json_q_table = json.load(f)
    except Exception as e: logging.error(f"Error loading Q-table from {json_file_path}: {e}"); return
    if not json_q_table: logging.info("JSON Q-table is empty."); return
    conn = None
    try:
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS q_values (state_key TEXT PRIMARY KEY, actions_q_values TEXT NOT NULL)")
        conn.commit()
        migrated_count = 0; skipped_count = 0
        for state_key, actions_dict in json_q_table.items():
            if not isinstance(actions_dict, dict): skipped_count += 1; continue
            actions_q_values_json = json.dumps(actions_dict)
            try:
                cursor.execute("INSERT INTO q_values (state_key, actions_q_values) VALUES (?, ?) ON CONFLICT(state_key) DO UPDATE SET actions_q_values = excluded.actions_q_values", (state_key, actions_q_values_json))
                migrated_count += 1
            except sqlite3.Error as e: logging.error(f"Error migrating state_key '{state_key}': {e}"); skipped_count +=1
        conn.commit()
        logging.info(f"Migration complete. Migrated {migrated_count} states. Skipped {skipped_count} states.")
    except sqlite3.Error as e: logging.error(f"SQLite error during migration: {e}")
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    # To run the Flask app:
    # app.run(debug=True, host='0.0.0.0')

    # To run the training simulation:
    run_training_simulation(10000, save_interval=5)

    # To migrate existing q_table.json to SQLite (run once, if needed):
    # migrate_json_to_sqlite()
