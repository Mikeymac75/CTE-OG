import random
import logging
import time
from flask import Flask, jsonify, render_template, send_from_directory, request

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def initialize_game_data():
    global game_data
    logging.info("Initializing game data for the very first time.")
    game_data = {
        "deck": [], "hands": {p: [] for p in range(3)}, "dummy_hand": [],
        "scores": {p: 0 for p in range(3)}, "dealer": random.randint(0, 2),
        "trump_suit": None, "up_card": None, "up_card_visible": False,
        "current_player_turn": -1, "maker": None, "going_alone": False,
        "trick_cards": [], "current_trick_lead_suit": None,
        "trick_leader": -1, "round_tricks_won": {p: 0 for p in range(3)},
        "game_phase": "setup",
        "message": "Welcome! Click 'Start New Round'.",
        "player_identities": {0: "Player 1 (You)", 1: "Player 2 (AI)", 2: "Player 3 (AI)"},
        "num_players": 3,
        "passes_on_upcard": [], "passes_on_calling": [],
        "cards_to_discard_count": 0,
        "original_up_card_for_round": None,
        "last_completed_trick": None
    }
initialize_game_data()

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
    global game_data; time.sleep(0.5)
    if game_data["game_phase"] != "playing_tricks" or game_data["current_player_turn"] != ai_player_idx: return
    ai_hand = game_data["hands"][ai_player_idx]; lead_suit = game_data["current_trick_lead_suit"]; trump_suit = game_data["trump_suit"]
    valid_cards = get_valid_plays(list(ai_hand), lead_suit, trump_suit)
    if not valid_cards:
        game_data["message"] += f" Error: AI {ai_player_idx+1} has no valid cards to play."
        if len(game_data["trick_cards"]) < game_data["num_players"]: game_data["current_player_turn"] = (ai_player_idx + 1) % game_data["num_players"]
        return
    card_to_play = None
    if not lead_suit:
        valid_cards.sort(key=lambda c: get_card_value(c, trump_suit), reverse=True); card_to_play = valid_cards[0]
        logging.info(f"AI P{ai_player_idx} leading trick, playing highest value card: {str(card_to_play)}")
    else:
        can_follow_lead_suit_cards = [c for c in valid_cards if get_effective_suit(c, trump_suit) == lead_suit]
        cannot_follow_lead_suit_cards = [c for c in valid_cards if get_effective_suit(c, trump_suit) != lead_suit]
        if can_follow_lead_suit_cards:
            can_follow_lead_suit_cards.sort(key=lambda c: get_card_value(c, trump_suit), reverse=True); card_to_play = can_follow_lead_suit_cards[0]
            logging.info(f"AI P{ai_player_idx} following suit {lead_suit}, playing highest valid: {str(card_to_play)}")
        elif cannot_follow_lead_suit_cards:
            trump_cards_in_hand = [c for c in cannot_follow_lead_suit_cards if get_effective_suit(c, trump_suit) == trump_suit]
            non_trump_off_suit_cards = [c for c in cannot_follow_lead_suit_cards if get_effective_suit(c, trump_suit) != trump_suit]
            if trump_cards_in_hand:
                trump_cards_in_hand.sort(key=lambda c: get_card_value(c, trump_suit)); card_to_play = trump_cards_in_hand[0]
                logging.info(f"AI P{ai_player_idx} cannot follow suit {lead_suit}, playing lowest trump: {str(card_to_play)}")
            elif non_trump_off_suit_cards:
                non_trump_off_suit_cards.sort(key=lambda c: get_card_value(c, trump_suit)); card_to_play = non_trump_off_suit_cards[0]
                logging.info(f"AI P{ai_player_idx} cannot follow suit {lead_suit}, no trump, sloughing lowest: {str(card_to_play)}")
            else: logging.error(f"AI P{ai_player_idx} in following logic, but no card selected. Valid: {[str(c) for c in valid_cards]}. Fallback to random valid."); card_to_play = random.choice(valid_cards) if valid_cards else None
    if not card_to_play and valid_cards: logging.warning(f"AI P{ai_player_idx} card selection logic failed. Defaulting to random."); card_to_play = random.choice(valid_cards)
    elif not card_to_play and not valid_cards: logging.error(f"AI P{ai_player_idx} has no valid cards and no card_to_play."); return
    actual_card_to_remove_from_hand = next((c for c in ai_hand if c.suit == card_to_play.suit and c.rank == card_to_play.rank), None)
    if not actual_card_to_remove_from_hand:
        logging.error(f"AI P{ai_player_idx} tried to play {str(card_to_play)} but not in hand: {[str(c) for c in ai_hand]}. Playing first valid.")
        if valid_cards: card_to_play = valid_cards[0]; actual_card_to_remove_from_hand = next((c for c in ai_hand if c.suit == card_to_play.suit and c.rank == card_to_play.rank), None)
        else: return
    if actual_card_to_remove_from_hand: ai_hand.remove(actual_card_to_remove_from_hand)
    else: logging.error(f"CRITICAL: AI P{ai_player_idx} could not find card to play. Hand: {[str(c) for c in ai_hand]}"); return
    game_data["trick_cards"].append({'player': ai_player_idx, 'card': card_to_play})
    game_data["message"] = f"{game_data['player_identities'][ai_player_idx]} (AI) played {str(card_to_play)}."
    if not game_data["current_trick_lead_suit"]: game_data["current_trick_lead_suit"] = get_effective_suit(card_to_play, game_data["trump_suit"])
    num_players_in_trick = len(game_data["trick_cards"]); expected_cards_in_trick = game_data["num_players"]
    if num_players_in_trick == expected_cards_in_trick:
        winner_idx = determine_trick_winner(game_data["trick_cards"], game_data["trump_suit"])
        game_data["round_tricks_won"][winner_idx] += 1
        game_data["last_completed_trick"] = {"played_cards": [tc.copy() for tc in game_data["trick_cards"]], "winner_player_idx": winner_idx, "winner_name": game_data['player_identities'][winner_idx]}
        logging.info(f"Trick completed. Winner: P{winner_idx}. Storing last_completed_trick: {game_data['last_completed_trick']}")
        game_data["message"] += f" {game_data['player_identities'][winner_idx]} wins the trick."
        game_data["trick_cards"] = []; game_data["current_trick_lead_suit"] = None; game_data["current_player_turn"] = winner_idx; game_data["trick_leader"] = winner_idx
        if is_round_over(): score_round(); return
        else: game_data["message"] += f" {game_data['player_identities'][winner_idx]} leads next trick."
    else:
        game_data["current_player_turn"] = (ai_player_idx + 1) % game_data["num_players"]
        game_data["message"] += f" Next turn: {game_data['player_identities'][game_data['current_player_turn']]}."

def get_valid_plays(hand, lead_suit, trump_suit):
    if not lead_suit: return list(hand)
    can_follow_suit = any(get_effective_suit(card, trump_suit) == lead_suit for card in hand)
    if can_follow_suit: return [card for card in hand if get_effective_suit(card, trump_suit) == lead_suit]
    else: return list(hand)

def determine_trick_winner(trick_cards_played, trump_suit):
    if not trick_cards_played: return -1
    winning_player = -1; highest_value_card = None
    lead_card_obj = trick_cards_played[0]['card']; lead_suit_of_trick = get_effective_suit(lead_card_obj, trump_suit)
    for play in trick_cards_played:
        player = play['player']; card = play['card']
        card_effective_suit = get_effective_suit(card, trump_suit); card_value = get_card_value(card, trump_suit)
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
    global game_data; maker = game_data["maker"]; maker_tricks = game_data["round_tricks_won"][maker]
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
            game_data["message"] += f" {game_data['player_identities'][p_idx]} wins the game!"; return
    game_data["game_phase"] = "round_over"

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
    if game_data["game_phase"] in ["game_over", "setup"]:
        initialize_game_data(); game_data["dealer"] = random.randint(0, game_data["num_players"] - 1)
    else: game_data["dealer"] = (game_data["dealer"] + 1) % game_data["num_players"]
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
            winner_idx = determine_trick_winner(game_data["trick_cards"], game_data["trump_suit"])
            game_data["round_tricks_won"][winner_idx] += 1
            game_data["last_completed_trick"] = {"played_cards": [tc.copy() for tc in game_data["trick_cards"]], "winner_player_idx": winner_idx, "winner_name": game_data['player_identities'][winner_idx]}
            game_data["message"] += f" {game_data['player_identities'][winner_idx]} wins the trick."
            game_data["trick_cards"] = []; game_data["current_trick_lead_suit"] = None
            game_data["current_player_turn"] = winner_idx; game_data["trick_leader"] = winner_idx
            if is_round_over(): score_round()
            else: game_data["message"] += f" {game_data['player_identities'][winner_idx]} leads next trick."
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
    global game_data; time.sleep(0.5)
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
    global game_data; time.sleep(0.5)
    # Ensure it's actually this AI's turn and correct phase
    if game_data["current_player_turn"] != ai_maker_idx or game_data["game_phase"] != "prompt_go_alone" or game_data["maker"] != ai_maker_idx:
        logging.warning(f"ai_decide_go_alone_and_proceed called for P{ai_maker_idx} out of turn/phase. Current turn: P{game_data['current_player_turn']}, Phase: {game_data['game_phase']}, Maker: P{game_data['maker']}.")
        return

    ai_hand = game_data["hands"][ai_maker_idx]; trump_suit = game_data["trump_suit"]
    GO_ALONE_THRESHOLD = 220
    current_hand_strength = evaluate_potential_trump_strength(ai_hand, trump_suit, game_data)
    num_trump_cards_in_hand = sum(1 for card in ai_hand if get_effective_suit(card, trump_suit) == trump_suit)
    has_both_bowers = any(c.rank == 'J' and c.suit == trump_suit for c in ai_hand) and \
                      any(c.rank == 'J' and c.suit == get_left_bower_suit(trump_suit) for c in ai_hand)
    chose_to_go_alone = False
    if current_hand_strength >= GO_ALONE_THRESHOLD:
        if num_trump_cards_in_hand >= 3 or (num_trump_cards_in_hand >=2 and has_both_bowers) : chose_to_go_alone = True

    game_data["going_alone"] = chose_to_go_alone
    logging.info(f"AI P{ai_maker_idx} (Maker) decided to {'go alone' if chose_to_go_alone else 'play with dummy'}. Strength: {current_hand_strength}, Trumps: {num_trump_cards_in_hand}.")
    game_data["message"] = f"{game_data['player_identities'][ai_maker_idx]} (AI Maker) chose to {'go alone' if chose_to_go_alone else 'play with dummy hand'}."

    if chose_to_go_alone:
        game_data["dummy_hand"] = [] # Ensure dummy is not used
        transition_to_play_phase()
    else: # Not going alone
        if game_data.get("dummy_hand") and len(game_data["dummy_hand"]) == 5:
            ai_hand.extend(game_data["dummy_hand"]); game_data["dummy_hand"] = []
            logging.info(f"AI P{ai_maker_idx} picked up dummy. Hand size: {len(ai_hand)}.")
            game_data["cards_to_discard_count"] = 5
            # game_data["game_phase"] = "maker_discard" # This will be set by ai_discard_five_cards if needed, or directly to play
            ai_discard_five_cards(ai_maker_idx) # This will discard and then call transition_to_play_phase
        else:
            logging.error(f"AI P{ai_maker_idx} chose not to go alone, but dummy_hand invalid or missing. Dummy: {game_data.get('dummy_hand')}. Forcing 'go alone'.")
            game_data["going_alone"] = True # Fallback to going alone
            game_data["message"] += " Error with dummy hand, proceeding as if going alone."
            transition_to_play_phase()


def process_ai_bid_action(ai_action_data):
    global game_data; time.sleep(0.5)
    player_idx = ai_action_data.get('player_index'); action_type = ai_action_data.get('action')
    logging.info(f"Processing AI P{player_idx} bid action: {action_type}. Phase: {game_data['game_phase']}.")
    ai_hand = game_data["hands"][player_idx]
    ORDER_UP_THRESHOLD = 170; CALL_TRUMP_THRESHOLD = 160

    if action_type == 'ai_bidding_round_1':
        up_card_obj = game_data["original_up_card_for_round"]; up_card_suit = up_card_obj.suit
        strength_if_ordered_up = 0
        if player_idx == game_data["dealer"]:
            temp_eval_hand = list(ai_hand) + [up_card_obj]
            temp_eval_hand.sort(key=lambda c: get_card_value(c, up_card_suit))
            best_five_if_dealer_orders = temp_eval_hand[1:] if len(temp_eval_hand) > 5 else temp_eval_hand
            strength_if_ordered_up = evaluate_potential_trump_strength(best_five_if_dealer_orders, up_card_suit, game_data)
        else:
            strength_if_ordered_up = evaluate_potential_trump_strength(ai_hand, up_card_suit, game_data) + get_card_value(up_card_obj, up_card_suit)

        logging.info(f"AI P{player_idx} R1 considering ordering up {str(up_card_obj)}. Strength: {strength_if_ordered_up} vs Threshold: {ORDER_UP_THRESHOLD}.")

        if strength_if_ordered_up >= ORDER_UP_THRESHOLD:
            logging.info(f"AI P{player_idx} is ordering up {SUITS_MAP[up_card_suit]}.")
            game_data["trump_suit"] = up_card_suit
            game_data["maker"] = player_idx
            current_message = f"{game_data['player_identities'][player_idx]} (AI) ordered up {SUITS_MAP[game_data['trump_suit']]}."
            game_data["up_card_visible"] = False; game_data["up_card"] = None

            current_dealer_idx = game_data["dealer"]
            dealer_hand_actual = game_data["hands"][current_dealer_idx]
            up_card_to_pickup_actual = game_data["original_up_card_for_round"]

            dealer_hand_actual.append(up_card_to_pickup_actual)
            current_message += f" {game_data['player_identities'][current_dealer_idx]} (Dealer) picked up {str(up_card_to_pickup_actual)}."
            logging.info(f"AI P{player_idx} ordered up. Dealer P{current_dealer_idx} picked up {str(up_card_to_pickup_actual)}.")

            if current_dealer_idx == 0:
                game_data["cards_to_discard_count"] = 1
                game_data["current_player_turn"] = current_dealer_idx
                game_data["game_phase"] = "dealer_must_discard_after_order_up"
                current_message += f" {game_data['player_identities'][current_dealer_idx]} (Dealer) must discard 1 card."
                game_data["message"] = current_message
                return

            else:
                ai_dealer_hand = game_data["hands"][current_dealer_idx]
                card_to_discard_ai_dealer = get_ai_cards_to_discard(list(ai_dealer_hand), 1, game_data["trump_suit"])[0]
                try: ai_dealer_hand.remove(card_to_discard_ai_dealer)
                except ValueError:
                    found_c = next((c for c in ai_dealer_hand if c.rank == card_to_discard_ai_dealer.rank and c.suit == card_to_discard_ai_dealer.suit), None)
                    if found_c: ai_dealer_hand.remove(found_c)
                current_message += f" {game_data['player_identities'][current_dealer_idx]} (AI Dealer) discarded 1 card."

                game_data["game_phase"] = "prompt_go_alone"
                game_data["current_player_turn"] = player_idx
                current_message += f" {game_data['player_identities'][player_idx]} (Maker) to decide go alone."
                game_data["message"] = current_message
                ai_decide_go_alone_and_proceed(player_idx)
                return

        else:
            game_data["message"] = f"{game_data['player_identities'][player_idx]} (AI) passes."
            game_data['passes_on_upcard'].append(player_idx)
            logging.info(f"AI P{player_idx} passed R1. Passes: {len(game_data['passes_on_upcard'])}/{game_data['num_players']}.")
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

    elif action_type == 'ai_bidding_round_2':
        possible_suits_to_call = [s for s in SUITS if s != game_data["original_up_card_for_round"].suit]
        best_suit_to_call = None; max_strength_for_call = -1
        for suit_option in possible_suits_to_call:
            current_strength = evaluate_potential_trump_strength(ai_hand, suit_option, game_data)
            if current_strength > max_strength_for_call: max_strength_for_call = current_strength; best_suit_to_call = suit_option
        if best_suit_to_call and max_strength_for_call >= CALL_TRUMP_THRESHOLD:
            game_data["trump_suit"] = best_suit_to_call; game_data["maker"] = player_idx
            game_data["message"] = f"{game_data['player_identities'][player_idx]} (AI) called {SUITS_MAP[game_data['trump_suit']]}."
            game_data["up_card_visible"] = False; game_data["up_card"] = None
            game_data["game_phase"] = "prompt_go_alone"
            game_data["current_player_turn"] = player_idx
            ai_decide_go_alone_and_proceed(player_idx)
            return
        else:
            game_data["message"] = f"{game_data['player_identities'][player_idx]} (AI) passes round 2."
            game_data['passes_on_calling'].append(player_idx)
            logging.info(f"AI P{player_idx} passed R2. Passes: {len(game_data['passes_on_calling'])}/{game_data['num_players']-1}.")
            bid_order_r2 = [(game_data["dealer"] + i + 1) % game_data["num_players"] for i in range(game_data["num_players"])]
            if len(game_data['passes_on_calling']) == game_data["num_players"] - 1:
                if all(p == game_data["dealer"] or p in game_data["passes_on_calling"] for p in range(game_data["num_players"])) and game_data["dealer"] not in game_data["passes_on_calling"]:
                    game_data["current_player_turn"] = game_data["dealer"]; game_data["game_phase"] = "dealer_must_call"
                    game_data["message"] += f" Dealer ({game_data['player_identities'][game_data['dealer']]}) is stuck."
                    if game_data["dealer"] != 0: process_ai_bid_action({'player_index': game_data["dealer"], 'action': 'ai_dealer_stuck_call'})
                    return
            current_passer_index_in_order = bid_order_r2.index(player_idx)
            game_data["current_player_turn"] = bid_order_r2[(current_passer_index_in_order + 1) % game_data["num_players"]]
            game_data["message"] += f" {game_data['player_identities'][game_data['current_player_turn']]}'s turn."
            if game_data["current_player_turn"] != 0: process_ai_bid_action({'player_index': game_data["current_player_turn"], 'action': 'ai_bidding_round_2'})

    elif action_type == 'ai_dealer_stuck_call':
        turned_down_suit = game_data["original_up_card_for_round"].suit
        possible_suits_to_call = [s for s in SUITS if s != turned_down_suit]
        best_suit_to_call_when_stuck = None; max_strength_when_stuck = -1
        for suit_option in possible_suits_to_call:
            current_strength = evaluate_potential_trump_strength(ai_hand, suit_option, game_data)
            if current_strength > max_strength_when_stuck: max_strength_when_stuck = current_strength; best_suit_to_call_when_stuck = suit_option
        chosen_suit_by_ai = best_suit_to_call_when_stuck if best_suit_to_call_when_stuck else random.choice(possible_suits_to_call or SUITS)
        game_data["trump_suit"] = chosen_suit_by_ai; game_data["maker"] = player_idx
        game_data["message"] = f"{game_data['player_identities'][player_idx]} (AI) called {SUITS_MAP[game_data['trump_suit']]}."
        game_data["game_phase"] = "prompt_go_alone"
        game_data["current_player_turn"] = player_idx
        ai_decide_go_alone_and_proceed(player_idx)
        return

    logging.warning(f"AI P{player_idx} bid action {action_type} did not result in defined outcome within its own logic block.")


@app.route('/api/get_current_state', methods=['GET'])
def get_current_state_api():
    global game_data; logging.info("API: /api/get_current_state called.")
    return jsonify(game_data_to_json(game_data))

@app.route('/api/ai_play_turn', methods=['POST'])
def ai_play_turn_api():
    global game_data; logging.info("Received request for AI to play turn.")
    current_player = game_data.get("current_player_turn"); current_phase = game_data.get("game_phase")
    if current_player == 0: return jsonify({"error": "Not AI's turn.", "game_state": game_data_to_json(game_data)}), 400
    if current_phase != "playing_tricks": return jsonify({"error": "AI can only play cards in 'playing_tricks' phase.", "game_state": game_data_to_json(game_data)}), 400
    if is_round_over():
        if game_data["game_phase"] not in ["round_over", "game_over"]: score_round()
        return jsonify(game_data_to_json(game_data))
    logging.info(f"Processing AI P{current_player}'s turn via dedicated endpoint.")
    process_ai_play_card(current_player)
    return jsonify(game_data_to_json(game_data))

def game_data_to_json(current_game_data):
    json_safe_data = current_game_data.copy()
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
    app.run(debug=True, host='0.0.0.0')
