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
        "deck": [], "hands": {p: [] for p in range(3)}, "dummy_hand": [], # Changed kitty to dummy_hand
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
        "last_completed_trick": None # To store info about the last trick
    }
initialize_game_data()

# --- Core Game Logic ---
def initialize_new_round():
    global game_data
    logging.info(f"Initializing new round. Current dealer: {game_data.get('dealer', 'N/A')}")
    game_data["deck"] = create_deck() # 24 cards
    random.shuffle(game_data["deck"])

    # Deal 3 hands of 5 cards for players
    player_hands = {i: [] for i in range(game_data["num_players"])}
    for _ in range(5): # 5 cards per hand
        for i in range(game_data["num_players"]): # 3 players
            if game_data["deck"]:
                player_hands[i].append(game_data["deck"].pop())
            else:
                game_data["message"] = "Error: Not enough cards for player hands."; return
    game_data["hands"] = player_hands

    # Deal 5 cards for the dummy hand
    game_data["dummy_hand"] = []
    for _ in range(5):
        if game_data["deck"]:
            game_data["dummy_hand"].append(game_data["deck"].pop())
        else:
            game_data["message"] = "Error: Not enough cards for dummy hand."; return

    if not game_data["deck"]: game_data["message"] = "Error: Not enough for up_card."; return # Should be 4 cards left for up_card
    game_data["up_card"] = game_data["deck"].pop() # The 21st card dealt is the up-card
    game_data["original_up_card_for_round"] = game_data["up_card"]

    # The remaining 3 cards in the deck are not used further in typical 3-player Euchre with a dummy hand.
    # Some variations might use them as a kitty, but the common rule is a 5-card dummy hand.
    # For clarity, let's explicitly clear the deck if it's not going to be used.
    # However, the problem description mentioned "kitty" for the maker to pick up.
    # The standard 3-player rule is a 4th hand (dummy hand) is dealt, and that's what maker uses.
    # The previous code had a 4-card kitty. If the intention is a 5-card dummy + a separate small kitty,
    # that's a very specific variation.
    # Based on "there should be a 4th hand dealt... those are the cards that the maker will get to pick up",
    # this implies the 5-card dummy_hand *is* the "kitty" to be picked up.
    # The up_card is separate.

    # The remaining cards in game_data["deck"] (should be 3) are effectively the "kitty" that's buried.
    # Let's rename it to reflect this, or just ensure it's not confused with dummy_hand.
    # For now, game_data["deck"] will hold these last 3.
    # Let's ensure no "kitty" variable is used for the dummy hand anymore.
    if "kitty" in game_data: # Remove old kitty if it exists from previous structure
        del game_data["kitty"]

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
    game_data["last_completed_trick"] = None # Reset for new round
    game_data["message"] = f"{game_data['player_identities'][game_data['current_player_turn']]}'s turn. Up-card: {str(game_data['up_card'])}."
    logging.info(f"New round initialized. Up-card: {str(game_data['up_card']) if game_data['up_card'] else 'N/A'}. Turn: P{game_data['current_player_turn']}. Phase: {game_data['game_phase']}")

def get_left_bower_suit(trump_suit_char):
    """Returns the suit of the Left Bower given the trump suit character."""
    if trump_suit_char not in SUITS_MAP:
        return None # Invalid trump suit
    return {'H': 'D', 'D': 'H', 'C': 'S', 'S': 'C'}.get(trump_suit_char)

def get_effective_suit(card, trump_suit):
    """Determines the effective suit of a card, considering the Left Bower.
    If a card is the Left Bower, its effective suit becomes the trump suit."""
    if not trump_suit: # No trump suit defined yet
        return card.suit

    left_bower_actual_suit = get_left_bower_suit(trump_suit)
    if card.rank == 'J' and card.suit == left_bower_actual_suit:
        return trump_suit # Left Bower is effectively trump
    return card.suit

def get_card_value(card, trump_suit, lead_suit_for_trick=None): # lead_suit_for_trick is for trick evaluation, not general hand strength
    """
    Calculates the value of a card based on the current trump suit.
    This function is crucial for AI decision-making (bidding, discarding, playing).
    """
    # Define base point values for ranks (independent of suit initially)
    # These are more for sorting/playing than absolute trick-taking power without context.
    # For hand evaluation, these raw values are less important than their trump/bower status.
    rank_base_values = {'9': 1, '10': 2, 'J': 3, 'Q': 4, 'K': 5, 'A': 6}

    if not trump_suit: # If no trump is set (e.g., initial hand evaluation for bidding)
        # Value Ace high, then K, Q, J, 10, 9. Jacks are not special yet.
        return rank_base_values.get(card.rank, 0)

    card_effective_suit = get_effective_suit(card, trump_suit)

    # Highest values for Bowers
    if card.rank == 'J':
        if card.suit == trump_suit:  # Right Bower
            return 100 # Arbitrary high value for Right Bower
        if card.suit == get_left_bower_suit(trump_suit):  # Left Bower
            return 90  # Arbitrary high value for Left Bower, slightly less than Right

    # Values for other trump cards (Ace, King, Queen, 10, 9 of trump)
    if card_effective_suit == trump_suit:
        # Assign values higher than any non-trump card, respecting rank order
        # Example: Trump Ace=80, K=70, Q=60, 10=50, 9=40
        return {'A': 80, 'K': 70, 'Q': 60, '10': 50, '9': 40}.get(card.rank, 0)

    # Values for non-trump cards
    # Off-suit Aces are valuable
    if card.rank == 'A':
        return 30 # Value for an off-suit Ace

    # Other non-trump cards, ordered by rank
    # These are generally lower value, used for sloughing or potentially taking off-suit tricks.
    # K=25, Q=20, J (non-bower)=15, 10=10, 9=5
    # This ensures an off-suit Ace is better than an off-suit King, etc.
    non_trump_rank_values = {'K': 25, 'Q': 20, 'J': 15, '10': 10, '9': 5}
    return non_trump_rank_values.get(card.rank, 0)

def evaluate_potential_trump_strength(hand, potential_trump_suit, game_data=None): # game_data might be used for future advanced context
    """
    Evaluates the strength of a given hand if potential_trump_suit were to be trump.
    Returns a numerical score. Higher is better.
    """
    if not potential_trump_suit:
        return 0 # Cannot evaluate strength without a potential trump

    strength_score = 0
    num_trump_cards = 0
    has_right_bower = False
    has_left_bower = False

    # Card-specific values from get_card_value
    for card in hand:
        strength_score += get_card_value(card, potential_trump_suit)

        # Check for Bowers and count trump cards
        effective_suit = get_effective_suit(card, potential_trump_suit)
        if effective_suit == potential_trump_suit:
            num_trump_cards += 1
            if card.rank == 'J':
                if card.suit == potential_trump_suit:
                    has_right_bower = True
                elif card.suit == get_left_bower_suit(potential_trump_suit):
                    has_left_bower = True

    # Bonuses for specific combinations or counts
    if has_right_bower and has_left_bower:
        strength_score += 25 # Bonus for having both Bowers
    elif has_right_bower:
        strength_score += 15 # Smaller bonus for Right only
    elif has_left_bower:
        strength_score += 10 # Smaller bonus for Left only

    # Bonus for number of trump cards
    if num_trump_cards >= 3:
        strength_score += (num_trump_cards * 5) # e.g., 3 trump = +15, 4 trump = +20
    elif num_trump_cards == 2:
        strength_score += 5

    # Bonus for having multiple Aces (including trump Ace, already valued high)
    num_aces = sum(1 for card in hand if card.rank == 'A')
    if num_aces >= 2:
        strength_score += (num_aces * 3) # Small bonus for multiple aces

    # Consider suit distribution - a very simple void bonus for now
    # This is a placeholder for more complex distribution analysis
    suits_in_hand = set(get_effective_suit(c, potential_trump_suit) for c in hand)
    if len(suits_in_hand) <= 2 and num_trump_cards >=1 : # e.g. void in 1 or 2 suits and has trump
        strength_score += 5
        if len(suits_in_hand) == 1 and potential_trump_suit in suits_in_hand: # All trump
             strength_score += 10 # Extra bonus if all cards are trump (powerful for going alone)


    #logging.debug(f"Evaluated hand {[str(c) for c in hand]} for trump {potential_trump_suit}: Score {strength_score}, Trump cards {num_trump_cards}, RB {has_right_bower}, LB {has_left_bower}")
    return strength_score

def get_ai_cards_to_discard(hand, num_to_discard, trump_suit):
    # Sorts cards from lowest value to highest, so the first num_to_discard are the ones to discard.
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
    """Helper to set game state for start of trick playing."""
    game_data["game_phase"] = "playing_tricks"
    # Standard lead is player left of dealer
    game_data["current_player_turn"] = (game_data["dealer"] + 1) % game_data["num_players"]
    game_data["trick_leader"] = game_data["current_player_turn"]
    game_data["message"] += f" {game_data['player_identities'][game_data['current_player_turn']]} leads the first trick."
    # Clear pass lists for the round
    game_data.pop('passes_on_upcard', None)
    game_data.pop('passes_on_calling', None)

    # If the player leading the first trick is an AI, process their play.
    if game_data["game_phase"] == "playing_tricks" and game_data["current_player_turn"] != 0:
        process_ai_play_card(game_data["current_player_turn"])

def process_ai_play_card(ai_player_idx):
    """Handles an AI player playing a card."""
    global game_data
    time.sleep(0.5) # AI "thinks" before playing a card
    if game_data["game_phase"] != "playing_tricks" or game_data["current_player_turn"] != ai_player_idx:
        # Not the AI's turn or not in play phase, something is wrong or game state changed (e.g. round ended)
        return

    ai_hand = game_data["hands"][ai_player_idx]
    lead_suit = game_data["current_trick_lead_suit"]
    trump_suit = game_data["trump_suit"]

    valid_cards = get_valid_plays(list(ai_hand), lead_suit, trump_suit)
    if not valid_cards:
        # This should not happen if AI has cards and logic is correct
        game_data["message"] += f" Error: AI {ai_player_idx+1} has no valid cards to play."
        # Potentially force game end or error state? For now, just log and skip turn.
        # This might cause game to hang if not handled.
        # A robust solution might be to try to end round or flag error.
        # For now, to prevent hanging, let's try to advance turn if possible, though state is broken.
        if len(game_data["trick_cards"]) < game_data["num_players"]:
             game_data["current_player_turn"] = (ai_player_idx + 1) % game_data["num_players"]
        return


    # AI Card Selection Logic
    card_to_play = None
    if not lead_suit: # AI is leading the trick
        # Play the highest value card from valid_cards (which is all cards in hand)
        valid_cards.sort(key=lambda c: get_card_value(c, trump_suit), reverse=True)
        card_to_play = valid_cards[0]
        logging.info(f"AI P{ai_player_idx} leading trick, playing highest value card: {str(card_to_play)}")
    else: # AI is following
        # Separate cards that can follow suit from those that can't (but are still valid, e.g. trump when void)
        can_follow_lead_suit_cards = [c for c in valid_cards if get_effective_suit(c, trump_suit) == lead_suit]
        cannot_follow_lead_suit_cards = [c for c in valid_cards if get_effective_suit(c, trump_suit) != lead_suit]

        if can_follow_lead_suit_cards:
            # If AI can follow suit, play the highest card of that suit.
            can_follow_lead_suit_cards.sort(key=lambda c: get_card_value(c, trump_suit), reverse=True)
            card_to_play = can_follow_lead_suit_cards[0]
            logging.info(f"AI P{ai_player_idx} following suit {lead_suit}, playing highest valid: {str(card_to_play)}")
        elif cannot_follow_lead_suit_cards: # Cannot follow suit, must play from other cards (e.g., trump or other off-suit)
            # Check if any of the "cannot_follow_lead_suit_cards" are trump
            trump_cards_in_hand = [c for c in cannot_follow_lead_suit_cards if get_effective_suit(c, trump_suit) == trump_suit]
            non_trump_off_suit_cards = [c for c in cannot_follow_lead_suit_cards if get_effective_suit(c, trump_suit) != trump_suit]

            if trump_cards_in_hand:
                # If AI has trump and cannot follow suit, it should consider playing trump.
                # Simple strategy: play the lowest trump card that could potentially win.
                # For Phase 1, let's simplify: play the lowest trump card.
                # (More advanced: check if current trick winner is opponent and if this trump can beat it)
                trump_cards_in_hand.sort(key=lambda c: get_card_value(c, trump_suit)) # Sort low to high
                card_to_play = trump_cards_in_hand[0]
                logging.info(f"AI P{ai_player_idx} cannot follow suit {lead_suit}, playing lowest trump: {str(card_to_play)}")
            elif non_trump_off_suit_cards:
                # If no trump and cannot follow suit, slough (discard) the lowest value off-suit card.
                non_trump_off_suit_cards.sort(key=lambda c: get_card_value(c, trump_suit)) # Sort low to high
                card_to_play = non_trump_off_suit_cards[0]
                logging.info(f"AI P{ai_player_idx} cannot follow suit {lead_suit}, no trump, sloughing lowest: {str(card_to_play)}")
            else:
                # Should not happen if valid_cards is not empty and logic above is correct
                logging.error(f"AI P{ai_player_idx} in following logic, but no card selected. Valid: {[str(c) for c in valid_cards]}. Fallback to random valid.")
                card_to_play = random.choice(valid_cards) if valid_cards else None

    if not card_to_play and valid_cards: # Fallback if no card was chosen by logic but valid cards exist
        logging.warning(f"AI P{ai_player_idx} card selection logic failed to pick a card. Defaulting to random valid card.")
        card_to_play = random.choice(valid_cards)
    elif not card_to_play and not valid_cards:
        logging.error(f"AI P{ai_player_idx} has no valid cards and no card_to_play. This should have been caught earlier.")
        # To prevent crash, try to skip turn or handle error, though game state is broken.
        # This part is already handled by the initial `if not valid_cards:` check.
        return


    # --- Simulate the card play (mirroring parts of 'play_card' action handler) ---
    # Ensure card_to_play is actually in ai_hand before removing, to prevent errors if logic is flawed
    actual_card_to_remove_from_hand = next((c for c in ai_hand if c.suit == card_to_play.suit and c.rank == card_to_play.rank), None)
    if not actual_card_to_remove_from_hand:
        logging.error(f"AI P{ai_player_idx} tried to play {str(card_to_play)} but it's not in its hand: {[str(c) for c in ai_hand]}. Playing first valid if possible.")
        if valid_cards:
            card_to_play = valid_cards[0] # Play the first valid card as a recovery attempt
            actual_card_to_remove_from_hand = next((c for c in ai_hand if c.suit == card_to_play.suit and c.rank == card_to_play.rank), None)
        else: # No valid cards, should have been caught earlier
            return

    if actual_card_to_remove_from_hand:
        ai_hand.remove(actual_card_to_remove_from_hand)
    else:
        # This is a critical error if reached, means AI tried to play a card it doesn't possess AND recovery failed.
        logging.error(f"CRITICAL: AI P{ai_player_idx} could not find a card to play even after fallback. Hand: {[str(c) for c in ai_hand]}")
        # To prevent a complete crash, we might have to let the game proceed in a broken state or end the round.
        # For now, just return, hoping the turn advances or an error is caught elsewhere.
        return

    game_data["trick_cards"].append({'player': ai_player_idx, 'card': card_to_play})
    game_data["message"] = f"{game_data['player_identities'][ai_player_idx]} (AI) played {str(card_to_play)}."

    if not game_data["current_trick_lead_suit"]:
        game_data["current_trick_lead_suit"] = get_effective_suit(card_to_play, game_data["trump_suit"])

    num_players_in_trick = len(game_data["trick_cards"])
    expected_cards_in_trick = game_data["num_players"] # Assuming 3 players, all play

    if num_players_in_trick == expected_cards_in_trick: # Trick is over
        winner_idx = determine_trick_winner(game_data["trick_cards"], game_data["trump_suit"])
        game_data["round_tricks_won"][winner_idx] += 1

        # Store details of the completed trick for the log
        game_data["last_completed_trick"] = {
            "played_cards": [tc.copy() for tc in game_data["trick_cards"]], # Store copies
            "winner_player_idx": winner_idx,
            "winner_name": game_data['player_identities'][winner_idx]
        }
        logging.info(f"Trick completed. Winner: P{winner_idx}. Storing last_completed_trick: {game_data['last_completed_trick']}")

        game_data["message"] += f" {game_data['player_identities'][winner_idx]} wins the trick."

        game_data["trick_cards"] = []
        game_data["current_trick_lead_suit"] = None
        game_data["current_player_turn"] = winner_idx
        game_data["trick_leader"] = winner_idx

        if is_round_over():
            score_round()
            # No further AI play needed if round is over
            return
        else:
            game_data["message"] += f" {game_data['player_identities'][winner_idx]} leads next trick."
            # Recursive call removed:
            # if game_data["current_player_turn"] != 0 and game_data["game_phase"] == "playing_tricks":
            #     process_ai_play_card(game_data["current_player_turn"])
    else: # Trick is not over
        game_data["current_player_turn"] = (ai_player_idx + 1) % game_data["num_players"]
        game_data["message"] += f" Next turn: {game_data['player_identities'][game_data['current_player_turn']]}."
        # Recursive call removed:
        # if game_data["current_player_turn"] != 0 and game_data["game_phase"] == "playing_tricks":
        #      process_ai_play_card(game_data["current_player_turn"])


def get_valid_plays(hand, lead_suit, trump_suit):
    """Determines valid cards a player can play, considering effective suit for Left Bower."""
    if not lead_suit:  # Player is leading the trick
        return list(hand)

    # Check if player can follow the lead_suit using effective suits
    can_follow_suit = any(get_effective_suit(card, trump_suit) == lead_suit for card in hand)

    if can_follow_suit:
        return [card for card in hand if get_effective_suit(card, trump_suit) == lead_suit]
    else:  # Cannot follow suit (even with Left Bower as trump if lead was trump)
        return list(hand) # Any card is valid

def determine_trick_winner(trick_cards_played, trump_suit):
    """
    Determines the winner of a trick.
    trick_cards_played is a list of dicts: {'player': player_idx, 'card': Card_object}
    """
    if not trick_cards_played:
        return -1 # Should not happen

    winning_player = -1
    highest_value_card = None

    if not trick_cards_played: return -1 # Should not happen with valid inputs

    lead_card_obj = trick_cards_played[0]['card']
    lead_suit_of_trick = get_effective_suit(lead_card_obj, trump_suit)

    for play in trick_cards_played:
        player = play['player']
        card = play['card']
        card_effective_suit = get_effective_suit(card, trump_suit)
        card_value = get_card_value(card, trump_suit) # lead_suit_of_trick is not used by get_card_value

        if highest_value_card is None:
            highest_value_card = card
            winning_player = player
        else:
            highest_value_card_effective_suit = get_effective_suit(highest_value_card, trump_suit)
            highest_value_card_value = get_card_value(highest_value_card, trump_suit)

            # Current card is trump, highest seen so far is not trump
            if card_effective_suit == trump_suit and highest_value_card_effective_suit != trump_suit:
                highest_value_card = card
                winning_player = player
            # Both current card and highest seen so far are trump
            elif card_effective_suit == trump_suit and highest_value_card_effective_suit == trump_suit:
                if card_value > highest_value_card_value:
                    highest_value_card = card
                    winning_player = player
            # Neither is trump, but current card follows lead and highest seen so far does not
            # (This implies highest_value_card_effective_suit was off-suit, and card_effective_suit is on-suit)
            elif card_effective_suit != trump_suit and highest_value_card_effective_suit != trump_suit:
                if card_effective_suit == lead_suit_of_trick and highest_value_card_effective_suit != lead_suit_of_trick:
                    highest_value_card = card
                    winning_player = player
                # Both are non-trump and follow lead suit (or are off-suit but lead was also off-suit - less common)
                elif card_effective_suit == lead_suit_of_trick and highest_value_card_effective_suit == lead_suit_of_trick:
                    if card_value > highest_value_card_value: # Compare non-trump, on-suit values
                        highest_value_card = card
                        winning_player = player
                # If highest_value_card is on lead_suit_of_trick and card is not, card cannot win (unless it was trump, handled above)
                # No explicit 'else if' needed here, as card_value comparison for same-suit non-trump is covered.
    return winning_player

def is_round_over():
    """Checks if 5 tricks have been played in the current round."""
    total_tricks_taken = sum(game_data["round_tricks_won"].values())
    return total_tricks_taken >= 5 # Should be exactly 5 for a 3 player game.

def score_round():
    """Calculates scores at the end of a round and updates game scores."""
    global game_data
    maker = game_data["maker"]
    maker_tricks = game_data["round_tricks_won"][maker]
    # For 3 player, maker's partner is implicit or they go alone. Assume simple "maker vs others" for now.
    # Actual Euchre scoring is more complex (lone hands, euchres, etc.)
    # This is a simplified scoring.

    points_awarded = 0
    message_suffix = ""

    # Determine who the "team" is. In 3-player, maker is one team, other two are opponents.
    # Let's assume for now a simple scoring:
    # Maker gets points if they make their bid (3+ tricks).
    # Opponents get points if they "euchre" the maker (maker takes < 3 tricks).
    is_going_alone = game_data.get("going_alone", False)

    if maker_tricks < 3: # Maker was euchred
        points_awarded = 2
        for p_idx in range(game_data["num_players"]):
            if p_idx != maker:
                game_data["scores"][p_idx] += points_awarded
        message_suffix = f"Maker euchred! Opponents score {points_awarded} points each."
    elif maker_tricks == 5: # Maker took all 5 tricks (march)
        if is_going_alone:
            points_awarded = 4
            message_suffix = f"Maker ({game_data['player_identities'][maker]}) went alone and marches, scoring {points_awarded} points!"
        else:
            points_awarded = 2
            message_suffix = f"Maker ({game_data['player_identities'][maker]}) marches, scoring {points_awarded} points!"
        game_data["scores"][maker] += points_awarded
    elif maker_tricks >= 3: # Maker took 3 or 4 tricks
        points_awarded = 1
        # Score is 1 point whether going alone or not for 3-4 tricks.
        # Some variations give 2 for going alone and making it, but prompt implies 1 for 3-4 tricks alone.
        # Let's stick to 1 point for 3-4 tricks, regardless of going alone, as per typical basic rules unless specified otherwise.
        # The prompt: "If they win 3 or 4 tricks, they score 1 point." for going alone.
        # Standard play (not alone) also gets 1 point for 3-4 tricks.
        message_suffix = f"Maker ({game_data['player_identities'][maker]}) scores {points_awarded} point."
        if is_going_alone:
            message_suffix = f"Maker ({game_data['player_identities'][maker]}) went alone and scores {points_awarded} point."
        game_data["scores"][maker] += points_awarded

    game_data["message"] = f"Round Over. {message_suffix}"

    # Check for game over
    for p_idx, score in game_data["scores"].items():
        if score >= 10: # Assuming game to 10
            game_data["game_phase"] = "game_over"
            game_data["message"] += f" {game_data['player_identities'][p_idx]} wins the game!"
            return

    game_data["game_phase"] = "round_over" # Ready for next round or game over handled above


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
        initialize_game_data()
        game_data["dealer"] = random.randint(0, game_data["num_players"] - 1)
    else: game_data["dealer"] = (game_data["dealer"] + 1) % game_data["num_players"]
    initialize_new_round()
    # Check if the current_player_turn is an AI and if game_phase is bidding_round_1
    if game_data["game_phase"] == "bidding_round_1" and game_data["current_player_turn"] != 0: # Player 0 is human
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
    logging.debug(f"Action data: {action_data}") # More detailed, so debug level

    # Player turn validation (general, can be overridden by specific phases)
    is_correct_player_for_action = (game_data["current_player_turn"] == player_idx)
    # Special phases where maker acts, not necessarily current_player_turn initially
    if current_phase == 'maker_discard' and player_idx != game_data["maker"]:
        return jsonify({"error": "Only maker can discard."}), 400
    if current_phase == 'prompt_go_alone' and player_idx != game_data["maker"]:
        return jsonify({"error": "Only maker decides to go alone."}), 400

    if current_phase not in ['maker_discard', 'prompt_go_alone'] and not is_correct_player_for_action:
        return jsonify({"error": f"Not your turn (P{game_data['current_player_turn']+1} vs P{player_idx+1})"}), 400

    # --- ACTION HANDLERS ---
    if action_type == 'order_up':
        if current_phase != 'bidding_round_1':
            logging.warning(f"Action 'order_up' by P{player_idx} rejected. Phase: {current_phase}, expected 'bidding_round_1'.")
            return jsonify({"error": "Cannot order up now."}), 400

        game_data["trump_suit"] = game_data["original_up_card_for_round"].suit
        game_data["maker"] = player_idx
        logging.info(f"P{player_idx} ordered up {SUITS_MAP[game_data['trump_suit']]}. Maker set to P{player_idx}. Trump: {game_data['trump_suit']}.")
        game_data["message"] = f"{game_data['player_identities'][player_idx]} ordered up {SUITS_MAP[game_data['trump_suit']]}."
        game_data["up_card_visible"] = False # Up-card is conceptually taken/turned down
        game_data["up_card"] = None # Clear the current up_card display spot

        if player_idx == game_data["dealer"]: # Player ordering up IS the dealer
            dealer_hand = game_data["hands"][game_data["dealer"]]
            dealer_hand.append(game_data["original_up_card_for_round"]) # Dealer picks up the card
            logging.info(f"Dealer P{game_data['dealer']} picked up the up-card. Hand size: {len(dealer_hand)}.")
            game_data["message"] += " Dealer picked up the up-card."

            if game_data["dealer"] == 0: # Human dealer
                game_data["game_phase"] = "dealer_discard_one" # New phase for dealer to discard one
                game_data["cards_to_discard_count"] = 1
                game_data["current_player_turn"] = game_data["dealer"] # Still dealer's turn to discard
                game_data["message"] += " You must discard 1 card."
            else: # AI dealer
                # AI dealer picks up card, then needs to discard one.
                # This will be handled in process_ai_bid_action for AI ordering up as dealer.
                # That function will then call a modified process_ai_post_trump_decision or similar
                # to handle the "go alone" choice and potential dummy pickup/discard.
                # For now, this block assumes AI's internal logic handles the discard.
                # The next phase will be prompt_go_alone (after AI's internal discard).
                logging.info(f"AI Dealer P{game_data['dealer']} ordered up and picked up card. AI will discard 1.")
                # The AI's 'order_up' in process_ai_bid_action needs to handle this discard
                # and then proceed to the go_alone decision.
                # This path in submit_action is more for setting state after an AI's decision is made by process_ai_bid_action.
                # For now, we assume process_ai_bid_action will set the game to prompt_go_alone after its discard.
                # This will be refined in step 2 & 4.
                # For this step, the key is that the dealer (AI or Human) gets the card.
                # The subsequent AI flow for 'go alone' needs to be built.
                # Let's set phase to prompt_go_alone, assuming AI discard is handled by its order_up logic.
                game_data["current_player_turn"] = game_data["maker"] # Maker (dealer) to decide go alone
                game_data["game_phase"] = "prompt_go_alone" # Placeholder, AI will handle this transition.
                                                            # Actual AI logic will call its discard then go_alone sequence.
                game_data["message"] += " AI (Dealer) processing discard and go alone."
                # process_ai_bid_action (if AI is dealer ordering up) will call its discard logic then go_alone
                pass # AI's specific flow is in process_ai_bid_action

        else: # Player ordering up is NOT the dealer
            # Trump is set, up-card is not physically taken by non-dealer maker.
            # Maker (non-dealer) directly proceeds to "go alone" decision.
            # No initial discard for non-dealer maker in this scenario.
            logging.info(f"P{player_idx} (non-dealer) ordered up. Up-card suit is trump. No card pickup.")
            game_data["current_player_turn"] = game_data["maker"] # Maker's turn to decide
            game_data["game_phase"] = "prompt_go_alone"
            game_data["message"] += f" {game_data['player_identities'][game_data['maker']]} to decide go alone."
            if game_data["maker"] != 0: # AI is the non-dealer maker
                 # AI's process_ai_bid_action for ordering up (as non-dealer) should lead here.
                 # It will then proceed to its go_alone logic.
                 # process_ai_post_trump_decision might need adjustment based on whether up-card was taken.
                 # For now, assume AI's process_ai_bid_action handles this.
                 game_data["message"] += " AI processing go alone."
                 pass # AI's specific flow is in process_ai_bid_action


    elif action_type == 'pass_bid': # Handles both Human and AI pass in bidding round 1
        if current_phase != 'bidding_round_1':
            logging.warning(f"Action 'pass_bid' by P{player_idx} rejected. Phase: {current_phase}, expected 'bidding_round_1'.")
            return jsonify({"error": "Cannot pass bid now."}), 400

        is_ai_action = (player_idx != 0) # Check if this action is from an AI (via process_ai_bid_action)
        logging.info(f"P{player_idx}{' (AI)' if is_ai_action else ''} passed in bidding_round_1. Passes on upcard: {len(game_data['passes_on_upcard']) + 1}/{game_data['num_players']}.")
        game_data["message"] = f"{game_data['player_identities'][player_idx]}{' (AI)' if is_ai_action else ''} passes."
        game_data['passes_on_upcard'].append(player_idx)

        if len(game_data['passes_on_upcard']) == game_data["num_players"]: # All players passed on up-card
            logging.info("All players passed on up-card. Transitioning to bidding_round_2.")
            game_data["game_phase"] = "bidding_round_2"
            game_data["up_card_visible"] = False # Turn down the up-card
            game_data["current_player_turn"] = (game_data["dealer"] + 1) % game_data["num_players"]
            game_data["message"] += f" Up-card turned. {game_data['player_identities'][game_data['current_player_turn']]}'s turn to call."
            game_data['passes_on_calling'] = [] # Reset passes for round 2
            logging.info(f"Phase changed to bidding_round_2. Turn: P{game_data['current_player_turn']}.")
            if game_data["current_player_turn"] != 0: # If AI's turn for round 2 bidding
                logging.info(f"Triggering AI (P{game_data['current_player_turn']}) for bidding_round_2.")
                process_ai_bid_action({'player_index': game_data["current_player_turn"], 'action': 'ai_bidding_round_2'})
        else: # Still in bidding round 1, advance turn
            bid_order = [(game_data["dealer"] + i + 1) % game_data["num_players"] for i in range(game_data["num_players"])]
            # Find current player in bid order and get next
            current_bidder_index_in_order = bid_order.index(player_idx)
            game_data["current_player_turn"] = bid_order[(current_bidder_index_in_order + 1) % game_data["num_players"]]
            logging.info(f"Advanced turn in bidding_round_1 to P{game_data['current_player_turn']}.")
            game_data["message"] += f" {game_data['player_identities'][game_data['current_player_turn']]}'s turn."
            # If next player is AI, trigger its bidding logic for round 1
            # The action for AI here should be specific to bidding on up-card, e.g. 'ai_bidding_round_1'
            if game_data["current_player_turn"] != 0:
                 logging.info(f"Triggering AI (P{game_data['current_player_turn']}) for bidding_round_1.")
                 process_ai_bid_action({'player_index': game_data["current_player_turn"], 'action': 'ai_bidding_round_1'})

    elif action_type == 'call_trump': # Handles both Human and AI calling trump
        if current_phase not in ['bidding_round_2', 'dealer_must_call']:
            logging.warning(f"Action 'call_trump' by P{player_idx} rejected. Phase: {current_phase}, expected 'bidding_round_2' or 'dealer_must_call'.")
            return jsonify({"error": "Cannot call now."}), 400
        chosen_suit = action_data.get('suit')
        if not chosen_suit or chosen_suit not in SUITS:
            logging.warning(f"Action 'call_trump' by P{player_idx} rejected. Invalid suit: {chosen_suit}.")
            return jsonify({"error": "Invalid suit."}), 400

        turned_down_suit = game_data["original_up_card_for_round"].suit
        if current_phase == 'bidding_round_2' and chosen_suit == turned_down_suit:
            logging.warning(f"Action 'call_trump' by P{player_idx} rejected. Cannot call turned down suit {SUITS_MAP[turned_down_suit]}.")
            return jsonify({"error": f"Cannot call turned down suit ({SUITS_MAP[turned_down_suit]})."}), 400

        game_data["trump_suit"] = chosen_suit
        game_data["maker"] = player_idx
        logging.info(f"P{player_idx} called {SUITS_MAP[chosen_suit]}. Maker set to P{player_idx}. Trump: {game_data['trump_suit']}.")
        game_data["message"] = f"{game_data['player_identities'][player_idx]} called {SUITS_MAP[chosen_suit]}."
        game_data["up_card_visible"] = False # No up-card involved in this path
        game_data["up_card"] = None

        # Maker (human or AI) has 5 cards at this point.
        # Transition to prompt_go_alone.
        game_data["current_player_turn"] = game_data["maker"]
        game_data["game_phase"] = "prompt_go_alone"
        game_data["message"] += f" {game_data['player_identities'][game_data['maker']]} to decide go alone."
        logging.info(f"P{player_idx} called trump. Phase changed to prompt_go_alone.")

        if game_data["maker"] != 0: # AI is maker who called trump
            # AI's process_ai_bid_action already handles calling ai_decide_go_alone_and_proceed
            # So, no specific action needed here other than setting the phase.
            # The AI will take its turn based on the new phase.
            game_data["message"] += " AI processing go alone."
            pass

    elif action_type == 'dealer_discard_one': # Handler for dealer (human) discarding one after picking up up-card
        if current_phase != 'dealer_discard_one' or player_idx != game_data["dealer"] or player_idx != game_data["maker"]:
            logging.warning(f"Action 'dealer_discard_one' by P{player_idx} rejected. Phase/player mismatch. Phase: {current_phase}, Expected: dealer_discard_one. Dealer: P{game_data['dealer']}, Maker: P{game_data['maker']}.")
            return jsonify({"error": "Not time/player for dealer_discard_one."}), 400
        if player_idx != 0: # This phase is only for human player (AI dealer handles this internally)
            logging.warning(f"Action 'dealer_discard_one' rejected. P{player_idx} is AI, this action is for Human only.")
            return jsonify({"error": "dealer_discard_one is only for human player."}), 400

        cards_to_discard_dicts = action_data.get('cards', [])
        if len(cards_to_discard_dicts) != 1:
            logging.warning(f"Action 'dealer_discard_one' by P{player_idx} rejected. Expected 1 card, got {len(cards_to_discard_dicts)}.")
            return jsonify({"error": "Must discard exactly 1 card."}), 400
        logging.info(f"P{player_idx} (Human Dealer/Maker) attempting to discard 1 card: {cards_to_discard_dicts[0]}.")

        dealer_hand = game_data["hands"][game_data["dealer"]] # Should be 6 cards
        card_to_remove_dict = cards_to_discard_dicts[0]
        actual_card_to_remove = next((c for c in dealer_hand if c.rank == card_to_remove_dict['rank'] and c.suit == card_to_remove_dict['suit']), None)

        if not actual_card_to_remove:
            return jsonify({"error": "Card specified for discard not found in hand."}), 400

        dealer_hand.remove(actual_card_to_remove) # Hand is now 5 cards
        game_data["message"] = f"{game_data['player_identities'][player_idx]} discarded 1 card. "
        logging.info(f"P{player_idx} (Dealer/Maker) discarded 1. Hand size: {len(dealer_hand)}.")

        # Transition to prompt_go_alone phase
        game_data["game_phase"] = "prompt_go_alone"
        game_data["cards_to_discard_count"] = 0 # Reset discard count
        game_data["current_player_turn"] = game_data["maker"] # Maker's turn to decide
        logging.info(f"Phase changed to prompt_go_alone for P{game_data['maker']}.")
        game_data["message"] += "Choose to go alone or play with partner."


    elif action_type == 'maker_discard_one': # This was the old one, now replaced by dealer_discard_one or handled by AI.
        # This phase should ideally not be reached if 'dealer_discard_one' and AI logic are correct.
        # Retaining for safety or if there's a non-dealer maker scenario that needs a single discard (unlikely with current rules).
        # For now, this can be considered deprecated or for a very specific edge case not yet covered.
        # Based on the new rules, only the DEALER picks up the up-card and discards one.
        # If a non-dealer makes trump on the up-card, they don't pick it up.
        # If trump is called in round 2, no one picks up the original up-card.
        # Therefore, 'maker_discard_one' might be redundant.
        # Let's log a warning if it's hit.
        logging.warning(f"Deprecated 'maker_discard_one' action called by P{player_idx}. This should be handled by 'dealer_discard_one' or AI logic.")
        # For safety, let's make it behave like dealer_discard_one if it is hit by human maker.
        if current_phase == 'maker_discard_one' and player_idx == game_data["maker"] and player_idx == 0:
            cards_to_discard_dicts = action_data.get('cards', [])
            if len(cards_to_discard_dicts) == 1:
                maker_hand = game_data["hands"][game_data["maker"]]
                card_to_remove_dict = cards_to_discard_dicts[0]
                actual_card_to_remove = next((c for c in maker_hand if c.rank == card_to_remove_dict['rank'] and c.suit == card_to_remove_dict['suit']), None)
                if actual_card_to_remove and len(maker_hand) == 6: # Assuming it was a 6 card hand
                    maker_hand.remove(actual_card_to_remove)
                    game_data["game_phase"] = "prompt_go_alone"
                    game_data["cards_to_discard_count"] = 0
                    game_data["current_player_turn"] = game_data["maker"]
                    game_data["message"] = f"{game_data['player_identities'][player_idx]} discarded 1 (via deprecated path). Choose to go alone."
                    logging.info(f"P{player_idx} completed deprecated 'maker_discard_one'. Hand size: {len(maker_hand)}.")
                else:
                     return jsonify({"error": "Invalid state for deprecated maker_discard_one."}),400
            else:
                return jsonify({"error": "Invalid card count for deprecated maker_discard_one."}), 400
        else:
            return jsonify({"error": "Invalid use of deprecated maker_discard_one."}), 400


    elif action_type == 'pass_call':
        if player_idx == game_data["dealer"] and current_phase == 'dealer_must_call':
            logging.warning(f"Action 'pass_call' by P{player_idx} (Dealer) rejected. Dealer must call in 'dealer_must_call' phase.")
            return jsonify({"error": "Dealer must call."}), 400
        if current_phase != 'bidding_round_2':
            logging.warning(f"Action 'pass_call' by P{player_idx} rejected. Phase: {current_phase}, expected 'bidding_round_2'.")
            return jsonify({"error": "Cannot pass call now."}), 400

        logging.info(f"P{player_idx} passed in bidding_round_2. Passes on calling: {len(game_data['passes_on_calling']) + 1}/{game_data['num_players']-1} (excluding dealer initially).")
        game_data["message"] = f"{game_data['player_identities'][player_idx]} passes."
        game_data['passes_on_calling'].append(player_idx)
        bid_order_r2 = [(game_data["dealer"] + i + 1) % game_data["num_players"] for i in range(game_data["num_players"])]

        if len(game_data['passes_on_calling']) == game_data["num_players"] - 1: # All non-dealers passed
            dealer_is_only_one_left = all(p == game_data["dealer"] or p in game_data["passes_on_calling"] for p in range(game_data["num_players"]))
            if dealer_is_only_one_left and game_data["dealer"] not in game_data["passes_on_calling"]: # Dealer is stuck
                game_data["current_player_turn"] = game_data["dealer"]
                game_data["game_phase"] = "dealer_must_call"
                logging.info(f"All non-dealers passed. Dealer P{game_data['dealer']} is stuck. Phase changed to dealer_must_call.")
                game_data["message"] += f" Dealer ({game_data['player_identities'][game_data['dealer']]}) is stuck."
                if game_data["dealer"] != 0: # AI is dealer
                    logging.info(f"Triggering AI Dealer (P{game_data['dealer']}) for ai_dealer_stuck_call.")
                    process_ai_bid_action({'player_index': game_data["dealer"], 'action': 'ai_dealer_stuck_call'})
                return jsonify(game_data_to_json(game_data))

        # Advance turn if not dealer stuck
        current_passer_index_in_order = bid_order_r2.index(player_idx)
        game_data["current_player_turn"] = bid_order_r2[(current_passer_index_in_order + 1) % game_data["num_players"]]
        logging.info(f"Advanced turn in bidding_round_2 to P{game_data['current_player_turn']}.")
        game_data["message"] += f" {game_data['player_identities'][game_data['current_player_turn']]}'s turn."
        if game_data["current_player_turn"] != 0: # Next is AI
            logging.info(f"Triggering AI (P{game_data['current_player_turn']}) for bidding_round_2.")
            process_ai_bid_action({'player_index': game_data["current_player_turn"], 'action': 'ai_bidding_round_2'})

    elif action_type == 'maker_discard':
        if current_phase != 'maker_discard' or player_idx != game_data["maker"]:
            logging.warning(f"Action 'maker_discard' by P{player_idx} rejected. Phase/maker mismatch. Phase: {current_phase}, Expected: maker_discard. Maker: P{game_data['maker']}.")
            return jsonify({"error": "Not time/player for maker discard."}), 400
        cards_to_discard_dicts = action_data.get('cards', [])
        if len(cards_to_discard_dicts) != game_data["cards_to_discard_count"]:
            logging.warning(f"Action 'maker_discard' by P{player_idx} rejected. Expected {game_data['cards_to_discard_count']} cards, got {len(cards_to_discard_dicts)}.")
            return jsonify({"error": f"Must discard {game_data['cards_to_discard_count']} cards."}), 400
        logging.info(f"P{player_idx} (Maker) attempting to discard {len(cards_to_discard_dicts)} cards: {cards_to_discard_dicts}.")

        maker_hand = game_data["hands"][game_data["maker"]]
        # Validate and remove cards (ensure unique cards are specified if duplicates in hand)
        temp_hand_for_removal = list(maker_hand)
        cards_removed_success = True
        for card_dict in cards_to_discard_dicts:
            found_card_in_temp = next((c for c in temp_hand_for_removal if c.rank == card_dict['rank'] and c.suit == card_dict['suit']), None)
            if found_card_in_temp:
                temp_hand_for_removal.remove(found_card_in_temp) # Remove from temp to handle duplicates correctly
                # Now remove the actual corresponding object from the real hand
                actual_card_to_remove_from_real_hand = next((c for c in maker_hand if c.rank == card_dict['rank'] and c.suit == card_dict['suit']), None)
                if actual_card_to_remove_from_real_hand:
                    maker_hand.remove(actual_card_to_remove_from_real_hand)
                else: # Should not happen if logic is right
                    cards_removed_success = False; break
            else:
                cards_removed_success = False; break

        if not cards_removed_success:
             # This part is tricky; if validation fails mid-way, hand might be partially modified.
             # Ideally, full validation before any modification, or rollback. For now, assume client sends valid unique cards.
            return jsonify({"error": "Error processing discards. Card not found or duplicate issue."}), 400


        game_data["message"] = f"{game_data['player_identities'][player_idx]} discarded 5 cards."
        logging.info(f"P{player_idx} (Maker) completed discard of 5. Hand size: {len(maker_hand)}.")
        game_data["cards_to_discard_count"] = 0
        # After discarding 5 (because they chose not to go alone and picked up dummy), transition to play.
        transition_to_play_phase()
        # No explicit "prompt_go_alone" after this discard, that decision was made prior to dummy pickup.


    elif action_type == 'choose_go_alone' or action_type == 'choose_not_go_alone':
        if current_phase != 'prompt_go_alone' or player_idx != game_data["maker"]:
            logging.warning(f"Action '{action_type}' by P{player_idx} rejected. Phase/maker mismatch. Phase: {current_phase}, Expected: prompt_go_alone. Maker: P{game_data['maker']}.")
            return jsonify({"error": "Not time/player to choose go alone."}), 400

        chose_to_go_alone = (action_type == 'choose_go_alone')
        game_data["going_alone"] = chose_to_go_alone
        logging.info(f"P{player_idx} (Maker) chose to {'go alone' if chose_to_go_alone else 'play with partner/use dummy'}.")
        game_data["message"] = f"{game_data['player_identities'][player_idx]} (Maker) chose to {'go alone' if chose_to_go_alone else 'play with dummy hand'}."

        if chose_to_go_alone:
            # Maker plays with their current 5 cards. No dummy pickup.
            logging.info(f"Maker P{player_idx} is going alone. Hand size: {len(game_data['hands'][player_idx])}.")
            game_data["dummy_hand"] = [] # Ensure dummy hand is not available / cleared if it was there
            transition_to_play_phase()
        else: # Chose NOT to go alone
            # Maker picks up dummy hand and must discard 5.
            maker_hand = game_data["hands"][game_data["maker"]]
            if game_data.get("dummy_hand"): # Check if dummy hand exists (it should)
                logging.info(f"Maker P{player_idx} not going alone. Picking up dummy hand. Current dummy: {[str(c) for c in game_data['dummy_hand']] if game_data['dummy_hand'] else 'empty'}.")
                maker_hand.extend(game_data["dummy_hand"])
                game_data["dummy_hand"] = [] # Dummy hand is taken
                logging.info(f"Maker P{player_idx} hand size after picking up dummy: {len(maker_hand)}.")
            else:
                logging.warning(f"Maker P{player_idx} chose not to go alone, but dummy_hand is empty/missing.")
                # This is an unexpected state if rules are followed. Proceed with caution.

            game_data["cards_to_discard_count"] = 5
            game_data["game_phase"] = "maker_discard" # Phase for maker to discard 5 cards
            game_data["current_player_turn"] = game_data["maker"] # Maker's turn to discard
            game_data["message"] += " Picked up dummy hand. Now discard 5 cards."
            logging.info(f"Phase changed to maker_discard for P{game_data['maker']}. Must discard 5 cards.")
            # If AI is maker and not going alone, its internal functions
            # (ai_decide_go_alone_and_proceed -> ai_discard_five_cards)
            # would have handled the discard. This block in submit_action_api
            # is likely redundant for AI-initiated actions of this type.
            if game_data["maker"] != 0: # AI is maker
                # AI's internal flow (ai_decide_go_alone_and_proceed -> ai_discard_five_cards)
                # should manage this. Logging here might be for a case where AI action is POSTed.
                # Since process_ai_post_trump_decision doesn't exist, this part is problematic.
                # For now, let's assume AI's internal path handles it.
                logging.info(f"AI P{game_data['maker']} chose not to go alone. Discard (if any) handled by AI internal logic.")
                pass # AI's internal flow should cover necessary discards.

        # This logging and AI check might be redundant if handled within the branches above.
        # logging.info(f"Transitioned to play phase. Going alone: {game_data['going_alone']}. Lead player: P{game_data['current_player_turn']}.")
        # if game_data["game_phase"] == "playing_tricks" and game_data["current_player_turn"] != 0:
        #     pass


    elif action_type == 'play_card':
        if current_phase != 'playing_tricks':
            return jsonify({"error": "Cannot play card now."}), 400
        if player_idx != game_data["current_player_turn"]:
            return jsonify({"error": "Not your turn to play."}), 400

        card_data = action_data.get('card')
        if not card_data: return jsonify({"error": "No card data provided."}), 400

        player_hand = game_data["hands"][player_idx]
        played_card = next((c for c in player_hand if c.rank == card_data['rank'] and c.suit == card_data['suit']), None)

        if not played_card:
            return jsonify({"error": "Card not in hand."}), 400

        # Validate play legality
        lead_suit = game_data["current_trick_lead_suit"]
        valid_plays = get_valid_plays(player_hand, lead_suit, game_data["trump_suit"])
        if played_card not in valid_plays:
            # Create a more descriptive error, e.g. "Must follow suit (Hearts)"
            lead_suit_name = SUITS_MAP.get(lead_suit) if lead_suit else ""
            required_suit_msg = f"Must follow suit ({lead_suit_name})." if lead_suit and any(c.suit == lead_suit for c in player_hand) else "Invalid play."
            return jsonify({"error": f"Invalid play. {required_suit_msg}"}), 400

        # Process the play
        player_hand.remove(played_card)
        game_data["trick_cards"].append({'player': player_idx, 'card': played_card})
        game_data["message"] = f"{game_data['player_identities'][player_idx]} played {str(played_card)}."

        if not game_data["current_trick_lead_suit"]: # This card is leading the trick
            game_data["current_trick_lead_suit"] = get_effective_suit(played_card, game_data["trump_suit"])

        num_players_in_trick = len(game_data["trick_cards"])

        # Determine number of players participating in this trick
        # (slightly complex if someone is going alone - partner sits out)
        # For 3 player game, this is simpler: always 3 players unless someone went alone (not implemented for 3p yet)
        # For now, assume all active players play. In 3-player, it's always 3.
        expected_cards_in_trick = game_data["num_players"]
        # TODO: Adjust expected_cards_in_trick if going_alone is fully implemented for 3 players where one player's partner would sit out.
        # For 3 players, "going alone" means no partner, so all 3 still play. This logic holds.

        if num_players_in_trick == expected_cards_in_trick: # Trick is over
            winner_idx = determine_trick_winner(game_data["trick_cards"], game_data["trump_suit"])
            game_data["round_tricks_won"][winner_idx] += 1

            # Store details of the completed trick for the log
            game_data["last_completed_trick"] = {
                "played_cards": [tc.copy() for tc in game_data["trick_cards"]], # Store copies
                "winner_player_idx": winner_idx,
                "winner_name": game_data['player_identities'][winner_idx]
            }
            logging.info(f"Trick completed (human play). Winner: P{winner_idx}. Storing last_completed_trick: {game_data['last_completed_trick']}")

            game_data["message"] += f" {game_data['player_identities'][winner_idx]} wins the trick."

            game_data["trick_cards"] = []
            game_data["current_trick_lead_suit"] = None
            game_data["current_player_turn"] = winner_idx # Winner leads next trick
            game_data["trick_leader"] = winner_idx

            if is_round_over():
                score_round() # This will set game_phase to round_over or game_over
                # No AI turn processing needed if round/game is over
            else:
                game_data["message"] += f" {game_data['player_identities'][winner_idx]} leads next trick."
                # AI turn will be handled by the logic after this main if/elif block
        else: # Trick is not over, advance turn
            current_player_order_idx = game_data["trick_cards"].index(next(tc for tc in game_data["trick_cards"] if tc['player'] == player_idx))
            # This needs to be based on original play order, not current trick_cards length if players can sit out
            # For 3 player, simple rotation:
            game_data["current_player_turn"] = (player_idx + 1) % game_data["num_players"]
            game_data["message"] += f" Next turn: {game_data['player_identities'][game_data['current_player_turn']]}."
            # AI turn will be handled by the logic after this main if/elif block

    # ---[[ AI Turn Triggering Logic ]] ---
    # After a human player's action, if it becomes an AI's turn, process that AI's turn once.
    # The client will then be responsible for requesting subsequent AI turns if needed.

    current_phase = game_data["game_phase"]
    current_player_after_action = game_data["current_player_turn"] # Turn might have changed
    original_actor_was_human = (player_idx == 0)

    logging.debug(f"End of submit_action_api for P{player_idx}, action {action_type}. Current phase: {current_phase}, new turn: P{current_player_after_action}.")

    # If the action was by a human, and it's now an AI's turn to play a card
    if original_actor_was_human and \
       current_phase == "playing_tricks" and \
       current_player_after_action != 0 and \
       not is_round_over():
        logging.info(f"Human action resulted in AI P{current_player_after_action}'s turn. Processing AI play.")
        process_ai_play_card(current_player_after_action)
        # After this one AI plays, the state will be returned.
        # If current_player_after_action was ALREADY an AI (e.g. an AI action was submitted via API, though unusual for play_card)
        # then process_ai_play_card would run for that AI, and then state is returned.

    # AI bidding is more complex and handled by process_ai_bid_action which has its own turn progression.
    # The removal of recursion in process_ai_play_card is the main change for card playing sequence.
    # process_ai_bid_action still handles its own sequences of AI bidding. If an AI bid leads to an AI play,
    # that AI play (via transition_to_play_phase -> process_ai_play_card) will now also be non-recursive.

    # Check if AI should make a bidding decision (this is mostly handled by direct calls earlier, but a safety check)
    # Note: process_ai_bid_action is usually called directly when it's AI's turn to bid.
    # This section is more for states reached after human action where an AI might need to immediately follow.
    # Example: Human passes, next is AI.
    # The existing direct calls are likely sufficient.

    # AI actions like discarding (after not going alone) or choosing to go alone are handled
    # within their respective sequences (e.g., ai_decide_go_alone_and_proceed calls ai_discard_five_cards).

    return jsonify(game_data_to_json(game_data))


def ai_discard_five_cards(ai_maker_idx):
    """Handles AI maker discarding 5 cards from 10 after choosing not to go alone."""
    global game_data
    time.sleep(0.5) # AI "thinks"
    ai_hand = game_data["hands"][ai_maker_idx]
    trump_suit = game_data["trump_suit"]
    logging.info(f"AI P{ai_maker_idx} (Maker) is discarding 5 cards from {len(ai_hand)} (should be 10). Trump: {trump_suit}.")

    if len(ai_hand) != 10:
        logging.error(f"AI P{ai_maker_idx} in ai_discard_five_cards, but hand size is {len(ai_hand)}, not 10. This is unexpected after picking up dummy.")
        # Attempt to proceed if possible, but this is an error state.
        # If hand is < 5, this will fail. If > 5 but < 10, it will discard less to try and get to 5.
        # The game rules expect a 10 card hand here.
        # For robustness, if hand is not 10, try to discard down to 5 if possible.
        num_to_discard_corrected = max(0, len(ai_hand) - 5)
        if num_to_discard_corrected == 0 and len(ai_hand) > 5: # e.g. hand of 7, discard 2
             num_to_discard_corrected = len(ai_hand) - 5
        elif len(ai_hand) <= 5: # Hand too small, cannot discard to 5
            logging.error(f"AI P{ai_maker_idx} hand too small ({len(ai_hand)}) to discard down to 5. Proceeding with current hand.")
            transition_to_play_phase()
            return

        logging.warning(f"Corrected num_to_discard to {num_to_discard_corrected} due to unexpected hand size.")
        cards_to_discard_ai = get_ai_cards_to_discard(list(ai_hand), num_to_discard_corrected, trump_suit)
    else:
        # Standard case: discard 5 from 10
        cards_to_discard_ai = get_ai_cards_to_discard(list(ai_hand), 5, trump_suit)

    if not cards_to_discard_ai:
        logging.error(f"AI P{ai_maker_idx} selected no cards to discard. Hand: {[str(c) for c in ai_hand]}. Trump: {trump_suit}. Aborting discard.")
        transition_to_play_phase() # Proceed without discard if selection failed.
        return

    discard_log = [str(c) for c in cards_to_discard_ai]
    logging.info(f"AI P{ai_maker_idx} intends to discard: {discard_log}")

    # Perform the discard
    for card_obj_to_discard in cards_to_discard_ai:
        # Find the specific card object in hand to remove, important if there are duplicates (though not in Euchre deck)
        actual_card_in_hand = next((c for c in ai_hand if c.suit == card_obj_to_discard.suit and c.rank == card_obj_to_discard.rank), None)
        if actual_card_in_hand:
            ai_hand.remove(actual_card_in_hand)
        else:
            # This should ideally not happen if cards_to_discard_ai came from ai_hand
            logging.error(f"AI P{ai_maker_idx} failed to find card {str(card_obj_to_discard)} in hand for discard_five. Current hand: {[str(c) for c in ai_hand]}. Discard list: {discard_log}")

    logging.info(f"AI P{ai_maker_idx} discarded. Hand size now: {len(ai_hand)}. Hand: {[str(c) for c in ai_hand]}.")
    game_data["message"] = f"{game_data['player_identities'][ai_maker_idx]} (AI) discarded {len(cards_to_discard_ai)} cards."
    game_data["cards_to_discard_count"] = 0 # Reset discard count

    if len(ai_hand) != 5:
        logging.warning(f"AI P{ai_maker_idx} hand size is {len(ai_hand)} after discard, expected 5. This might indicate an issue in discard logic or initial hand state.")

    transition_to_play_phase()


def ai_decide_go_alone_and_proceed(ai_maker_idx):
    """AI Maker (with 5 cards) decides to go alone or pick up dummy."""
    global game_data
    time.sleep(0.5) # AI "thinks"
    ai_hand = game_data["hands"][ai_maker_idx] # This is the AI's current 5-card hand
    trump_suit = game_data["trump_suit"]

    # Define a threshold for going alone. This will need tuning.
    # This evaluation is on the AI's 5-card hand *after* trump is set (and after dealer's discard if applicable).
    GO_ALONE_THRESHOLD = 220  # Example value, significantly higher than ordering/calling thresholds

    current_hand_strength = evaluate_potential_trump_strength(ai_hand, trump_suit, game_data)

    # Add a small randomness factor to the decision if strength is borderline
    # For example, if strength is within 10% of threshold, flip a coin.
    # This avoids deterministic behavior for similar hands.
    # chose_to_go_alone = False
    # if current_hand_strength >= GO_ALONE_THRESHOLD:
    #     chose_to_go_alone = True
    # elif current_hand_strength >= GO_ALONE_THRESHOLD * 0.9 and random.random() < 0.3: # 30% chance if close
    #     chose_to_go_alone = True

    # Simpler logic for now: if above threshold, go alone.
    # Consider number of trump cards as a strong factor too.
    # For example, must have at least 3 trumps and meet threshold, or 4+ trumps and slightly lower threshold.

    num_trump_cards_in_hand = sum(1 for card in ai_hand if get_effective_suit(card, trump_suit) == trump_suit)
    has_both_bowers = any(c.rank == 'J' and c.suit == trump_suit for c in ai_hand) and \
                      any(c.rank == 'J' and c.suit == get_left_bower_suit(trump_suit) for c in ai_hand)

    chose_to_go_alone = False
    if current_hand_strength >= GO_ALONE_THRESHOLD:
        if num_trump_cards_in_hand >= 3: # Require at least 3 trump for a very strong hand to go alone
            chose_to_go_alone = True
        elif num_trump_cards_in_hand >=2 and has_both_bowers: # Or 2 trump if they are the bowers
             chose_to_go_alone = True


    # For initial testing, AI might be too aggressive or too timid.
    # Example: Make AI less likely to go alone unless very strong.
    # if not (num_trump_cards_in_hand >= 4 or (num_trump_cards_in_hand == 3 and has_both_bowers)):
    #      chose_to_go_alone = False # Override if not super dominant in trump

    game_data["going_alone"] = chose_to_go_alone
    logging.info(f"AI P{ai_maker_idx} (Maker) evaluated hand for going alone. Hand: {[str(c) for c in ai_hand]}. Strength: {current_hand_strength} vs Threshold: {GO_ALONE_THRESHOLD}. Trump cards: {num_trump_cards_in_hand}. Decision: {'Go alone' if chose_to_go_alone else 'Play with dummy'}.")
    game_data["message"] = f"{game_data['player_identities'][ai_maker_idx]} (AI) chose to {'go alone' if chose_to_go_alone else 'play with dummy hand'}."

    if chose_to_go_alone:
        game_data["dummy_hand"] = [] # Ensure dummy is not used
        transition_to_play_phase()
    else: # Not going alone
        if game_data.get("dummy_hand"):
            logging.info(f"AI P{ai_maker_idx} picking up dummy hand. Current dummy: {[str(c) for c in game_data['dummy_hand']] if game_data['dummy_hand'] else 'empty'}.")
            ai_hand.extend(game_data["dummy_hand"])
            game_data["dummy_hand"] = []
            logging.info(f"AI P{ai_maker_idx} hand size after dummy pickup: {len(ai_hand)}.")
        else:
            logging.warning(f"AI P{ai_maker_idx} chose not to go alone, but dummy_hand is missing!")

        game_data["cards_to_discard_count"] = 5 # Set for ai_discard_five_cards
        # game_data["game_phase"] = "maker_discard" # This phase is implicit for AI before ai_discard_five_cards
        ai_discard_five_cards(ai_maker_idx)


def process_ai_bid_action(ai_action_data):
    global game_data
    time.sleep(0.5) # AI "thinks" before bidding
    player_idx = ai_action_data.get('player_index')
    action_type = ai_action_data.get('action') # e.g. 'ai_bidding_round_1', 'ai_bidding_round_2', 'ai_dealer_stuck_call'
    logging.info(f"Processing AI P{player_idx} bid action: {action_type}. Current phase: {game_data['game_phase']}.")
    ai_hand_cards = [str(c) for c in game_data["hands"][player_idx]]
    logging.debug(f"AI P{player_idx} hand: {ai_hand_cards}. Up-card: {str(game_data.get('original_up_card_for_round')) if game_data.get('original_up_card_for_round') else 'N/A'}.")
    ai_hand = game_data["hands"][player_idx]

    # Define thresholds for AI bidding decisions (these can be tuned)
    ORDER_UP_THRESHOLD = 170  # Adjusted from 150
    CALL_TRUMP_THRESHOLD = 160 # Adjusted from 140
    # GO_ALONE_THRESHOLD is defined in ai_decide_go_alone_and_proceed
    # Note: These thresholds are examples and will likely need tuning based on playtesting.
    # The scale of scores from evaluate_potential_trump_strength depends on the values in get_card_value and bonuses.

    # --- AI Decides to Order Up (Round 1) ---
    if action_type == 'ai_bidding_round_1':
        up_card_obj = game_data["original_up_card_for_round"]
        up_card_suit = up_card_obj.suit

        # Consider hand if AI is dealer (picks up card) vs non-dealer
        # If dealer, they'd have 6 cards then discard 1. Evaluation should be on potential best 5.
        # For simplicity now, evaluate current 5 + up_card, then if ordered, dealer separately handles discard.
        # A more advanced AI might evaluate the strength *after* a hypothetical best discard.

        potential_hand_if_ordered_up = list(ai_hand)
        if player_idx == game_data["dealer"]: # Dealer would pick up the card
            potential_hand_if_ordered_up.append(up_card_obj)
            # This hand is 6 cards. The evaluation should ideally be on the best 5.
            # For now, evaluate_potential_trump_strength will sum all 6.
            # This could be refined to simulate discarding the worst card first.
            # Let's assume for now evaluate_potential_trump_strength handles a 6-card hand by summing values.
            # Or, more accurately, if dealer, simulate discard for evaluation:
            temp_eval_hand = list(potential_hand_if_ordered_up) # 6 cards
            temp_eval_hand.sort(key=lambda c: get_card_value(c, up_card_suit)) # Sort by value in potential trump
            best_five_if_dealer_orders = temp_eval_hand[1:] # Discard the lowest of the 6
            strength_if_ordered_up = evaluate_potential_trump_strength(best_five_if_dealer_orders, up_card_suit, game_data)
        else: # Non-dealer, up-card suit becomes trump, hand size doesn't change (up-card isn't picked up by them)
            strength_if_ordered_up = evaluate_potential_trump_strength(list(ai_hand) + [up_card_obj], up_card_suit, game_data)
            # The above is slightly off for non-dealer as they don't add the up_card to hand,
            # but up_card_obj IS part of the trump suit strength.
            # Correct evaluation for non-dealer: evaluate their 5 cards with up_card_suit as trump,
            # and consider the up_card as a known trump card for overall suit strength.
            # For simplicity, we'll use the current evaluate_potential_trump_strength which sums values.
            # Let's assume the up_card's value contributes to the decision if its suit is made trump.
            # A simple way: evaluate current hand with up_card_suit as trump.
            strength_if_ordered_up = evaluate_potential_trump_strength(ai_hand, up_card_suit, game_data)
            # Add value of the up_card itself if it were trump
            strength_if_ordered_up += get_card_value(up_card_obj, up_card_suit)


        logging.info(f"AI P{player_idx} R1 considering ordering up {str(up_card_obj)} ({SUITS_MAP[up_card_suit]}). Evaluated strength: {strength_if_ordered_up} vs Threshold: {ORDER_UP_THRESHOLD}.")

        if strength_if_ordered_up >= ORDER_UP_THRESHOLD:
            # AI ORDERS UP
            logging.info(f"AI P{player_idx} is ordering up {SUITS_MAP[up_card_suit]}.")
            game_data["trump_suit"] = up_card_suit # Set trump suit first
            game_data["maker"] = player_idx
            game_data["message"] = f"{game_data['player_identities'][player_idx]} (AI) ordered up {SUITS_MAP[game_data['trump_suit']]}."
            game_data["up_card_visible"] = False
            game_data["up_card"] = None

            if player_idx == game_data["dealer"]: # AI is DEALER and ordered up
                ai_hand.append(game_data["original_up_card_for_round"]) # Hand is 6 cards
                logging.info(f"AI P{player_idx} (Dealer) picked up up-card. Hand size: {len(ai_hand)}.")

                # AI Dealer discards 1 card from 6
                card_to_discard_from_6 = get_ai_cards_to_discard(list(ai_hand), 1, game_data["trump_suit"])[0]
                actual_card_to_remove = next((c for c in ai_hand if c.rank == card_to_discard_from_6.rank and c.suit == card_to_discard_from_6.suit), None)
                if actual_card_to_remove:
                    ai_hand.remove(actual_card_to_remove)
                    logging.info(f"AI P{player_idx} (Dealer) discarded 1 card ({str(actual_card_to_remove)}) from 6. Hand size: {len(ai_hand)}.")
                    game_data["message"] += f" AI (Dealer) discarded 1 card."
                else:
                    logging.error(f"AI P{player_idx} (Dealer) failed to find card {str(card_to_discard_from_6)} in hand to discard from 6.")

                logging.info(f"AI P{player_idx} (Dealer who ordered up) has {len(ai_hand)} cards. Proceeding to go alone decision.")
                ai_decide_go_alone_and_proceed(player_idx)
                return
            else: # AI is NON-DEALER and ordered up
                # Non-dealer maker does not pick up the up-card. Hand remains 5 cards.
                logging.info(f"AI P{player_idx} (Non-Dealer, Maker) has 5 cards. Proceeding to go alone decision for ordering up {SUITS_MAP[game_data['trump_suit']]}.")
                ai_decide_go_alone_and_proceed(player_idx)
                return
        # If AI did not order up, it will fall through to the pass logic later in the function for 'ai_bidding_round_1'

    elif action_type == 'ai_bidding_round_2':
        possible_suits_to_call = [s for s in SUITS if s != game_data["original_up_card_for_round"].suit]
        best_suit_to_call = None
        max_strength_for_call = -1

        for suit_option in possible_suits_to_call:
            current_strength = evaluate_potential_trump_strength(ai_hand, suit_option, game_data)
            logging.debug(f"AI P{player_idx} R2 considering calling {SUITS_MAP[suit_option]}. Evaluated strength: {current_strength} vs Threshold: {CALL_TRUMP_THRESHOLD}.")
            if current_strength > max_strength_for_call:
                max_strength_for_call = current_strength
                best_suit_to_call = suit_option

        if best_suit_to_call and max_strength_for_call >= CALL_TRUMP_THRESHOLD:
            # AI CALLS TRUMP IN ROUND 2
            logging.info(f"AI P{player_idx} is calling {SUITS_MAP[best_suit_to_call]} in Round 2. Strength: {max_strength_for_call}.")
            game_data["trump_suit"] = best_suit_to_call
            game_data["maker"] = player_idx
            game_data["message"] = f"{game_data['player_identities'][player_idx]} (AI) called {SUITS_MAP[game_data['trump_suit']]}."
            game_data["up_card_visible"] = False # Original up-card is now irrelevant
            game_data["up_card"] = None
            game_data["current_player_turn"] = player_idx # AI (maker) to decide go alone
            # Transition to go alone decision
            ai_decide_go_alone_and_proceed(player_idx)
            return # AI action complete
        else:
            logging.info(f"AI P{player_idx} R2 evaluation: Best suit {best_suit_to_call} with strength {max_strength_for_call} did not meet threshold {CALL_TRUMP_THRESHOLD}. Passing.")
            # Fall through to pass logic for round 2

    elif action_type == 'ai_dealer_stuck_call': # Dealer is stuck
        turned_down_suit = game_data["original_up_card_for_round"].suit
        possible_suits_to_call = [s for s in SUITS if s != turned_down_suit]
        if not possible_suits_to_call: # Should not happen with standard SUITS
            # Fallback: pick any suit not turned down, or random if something is very wrong.
            # This case implies SUITS might be empty or only contains turned_down_suit.
            chosen_suit_by_ai = random.choice([s for s in SUITS if s != turned_down_suit] or SUITS)
            logging.error(f"AI P{player_idx} (Dealer) stuck, but no valid suits to call other than {turned_down_suit}. Fallback to {chosen_suit_by_ai}.")
        else:
            best_suit_to_call_when_stuck = None
            max_strength_when_stuck = -1
            for suit_option in possible_suits_to_call:
                current_strength = evaluate_potential_trump_strength(ai_hand, suit_option, game_data)
                logging.debug(f"AI P{player_idx} (Dealer Stuck) evaluating {SUITS_MAP[suit_option]}. Strength: {current_strength}.")
                if current_strength > max_strength_when_stuck:
                    max_strength_when_stuck = current_strength
                    best_suit_to_call_when_stuck = suit_option

            chosen_suit_by_ai = best_suit_to_call_when_stuck if best_suit_to_call_when_stuck else random.choice(possible_suits_to_call) # Pick best, or random if all are 0 strength

        logging.info(f"AI P{player_idx} (Dealer) is stuck. Must call. Turned down: {turned_down_suit}. Chosen suit: {chosen_suit_by_ai} with strength {max_strength_when_stuck}.")
        game_data["trump_suit"] = chosen_suit_by_ai
        game_data["maker"] = player_idx
        game_data["message"] = f"{game_data['player_identities'][player_idx]} (AI) called {SUITS_MAP[game_data['trump_suit']]}."
        # AI's hand is currently 5 cards. Dummy pickup is handled in ai_decide_go_alone_and_proceed if not going alone.

        # game_data["cards_to_discard_count"] = 5 # This is set in ai_decide_go_alone_and_proceed if needed
        # game_data["game_phase"] = "maker_discard" # This is also managed by the subsequent functions
        game_data["current_player_turn"] = player_idx # AI (maker) is still acting

        logging.info(f"AI P{player_idx} (maker via call) has 5 cards. Proceeding to go alone decision.")
        ai_decide_go_alone_and_proceed(player_idx)
        return # AI action complete

    # --- AI Passes ---
    # This covers 'ai_bidding_round_1' (if not ordered up) and 'ai_bidding_round_2' (if not called)
    # Logging for pass decision was added before the 'if should_ai_order_up' or 'if chosen_suit_by_ai' blocks.
    # Now, handle the consequences of passing.

    if action_type == 'ai_bidding_round_1': # AI passes in Round 1
        logging.info(f"AI P{player_idx} passes in bidding_round_1.")
        # Message update and passes_on_upcard append are handled by the calling 'pass_bid' in submit_action if AI passes.
        # Here we just need to ensure the game state progresses correctly if the AI's pass triggers next phase/turn.
        # The logic below mirrors the 'pass_bid' in submit_action_api for state changes.
        game_data["message"] = f"{game_data['player_identities'][player_idx]} (AI) passes."
        game_data['passes_on_upcard'].append(player_idx) # Ensure AI's pass is recorded
        logging.info(f"AI P{player_idx} pass recorded. Total passes on upcard: {len(game_data['passes_on_upcard'])}/{game_data['num_players']}.")


        if len(game_data['passes_on_upcard']) == game_data["num_players"]: # All passed
            logging.info("All players passed on up-card (triggered by AI pass). Transitioning to bidding_round_2.")
            game_data["game_phase"] = "bidding_round_2"
            game_data["up_card_visible"] = False
            game_data["current_player_turn"] = (game_data["dealer"] + 1) % game_data["num_players"]
            game_data["message"] += f" All passed. Up-card turned. {game_data['player_identities'][game_data['current_player_turn']]}'s turn to call."
            game_data['passes_on_calling'] = []
            logging.info(f"Phase changed to bidding_round_2. Turn: P{game_data['current_player_turn']}.")
            if game_data["current_player_turn"] != 0: # Next is AI
                 logging.info(f"Triggering AI (P{game_data['current_player_turn']}) for bidding_round_2 from AI R1 pass.")
                 process_ai_bid_action({'player_index': game_data["current_player_turn"], 'action': 'ai_bidding_round_2'})
        else: # More players to bid in round 1
            bid_order = [(game_data["dealer"] + i + 1) % game_data["num_players"] for i in range(game_data["num_players"])]
            # Find current player (the AI that just passed) in bid order and get next
            current_bidder_index_in_order = bid_order.index(player_idx)
            game_data["current_player_turn"] = bid_order[(current_bidder_index_in_order + 1) % game_data["num_players"]]
            game_data["message"] += f" {game_data['player_identities'][game_data['current_player_turn']]}'s turn."
            logging.info(f"Advanced turn in bidding_round_1 to P{game_data['current_player_turn']} (after AI P{player_idx} pass).")
            if game_data["current_player_turn"] != 0: # Next is AI
                logging.info(f"Triggering AI (P{game_data['current_player_turn']}) for bidding_round_1 from AI R1 pass.")
                process_ai_bid_action({'player_index': game_data["current_player_turn"], 'action': 'ai_bidding_round_1'})

    elif action_type == 'ai_bidding_round_2': # AI passes in round 2
        logging.info(f"AI P{player_idx} passes in bidding_round_2.")
        game_data["message"] = f"{game_data['player_identities'][player_idx]} (AI) passes round 2."
        game_data['passes_on_calling'].append(player_idx) # Ensure AI's pass is recorded
        logging.info(f"AI P{player_idx} pass recorded. Total passes on calling: {len(game_data['passes_on_calling'])}/{game_data['num_players']-1} (non-dealers).")

        bid_order_r2 = [(game_data["dealer"] + i + 1) % game_data["num_players"] for i in range(game_data["num_players"])]

        if len(game_data['passes_on_calling']) == game_data["num_players"] - 1: # All non-dealers passed
            dealer_is_only_one_left = all(p == game_data["dealer"] or p in game_data["passes_on_calling"] for p in range(game_data["num_players"]))
            if dealer_is_only_one_left and game_data["dealer"] not in game_data["passes_on_calling"]: # Dealer is stuck
                game_data["current_player_turn"] = game_data["dealer"]
                game_data["game_phase"] = "dealer_must_call"
                game_data["message"] += f" Dealer ({game_data['player_identities'][game_data['dealer']]}) is stuck."
                logging.info(f"All non-dealers passed (triggered by AI pass). Dealer P{game_data['dealer']} is stuck. Phase: dealer_must_call.")
                if game_data["dealer"] != 0: # AI is dealer
                    logging.info(f"Triggering AI Dealer (P{game_data['dealer']}) for ai_dealer_stuck_call from AI R2 pass.")
                    process_ai_bid_action({'player_index': game_data["dealer"], 'action': 'ai_dealer_stuck_call'})
                return # Return because dealer_must_call will be handled, or it's human dealer's turn.

        # Advance turn if not dealer stuck scenario
        current_passer_index_in_order = bid_order_r2.index(player_idx)
        game_data["current_player_turn"] = bid_order_r2[(current_passer_index_in_order + 1) % game_data["num_players"]]
        game_data["message"] += f" {game_data['player_identities'][game_data['current_player_turn']]}'s turn."
        logging.info(f"Advanced turn in bidding_round_2 to P{game_data['current_player_turn']} (after AI P{player_idx} pass).")
        if game_data["current_player_turn"] != 0: # Next is AI
             logging.info(f"Triggering AI (P{game_data['current_player_turn']}) for bidding_round_2 from AI R2 pass.")
             process_ai_bid_action({'player_index': game_data["current_player_turn"], 'action': 'ai_bidding_round_2'})

    # Note: 'ai_dealer_stuck_call' action type results in a call, not a pass.
    # It's handled by the 'if should_ai_call_trump and chosen_suit_by_ai:' block earlier.
    # No separate pass logic needed here for 'ai_dealer_stuck_call'.
    # elif action_type == 'ai_dealer_stuck_call':
    # ... This path should not be reached if it was a call ...
    # If for some reason get_ai_stuck_suit_call failed and chosen_suit_by_ai is None,
    # it would fall through, which is an error state.
    # However, get_ai_stuck_suit_call is designed to always return a suit.
    # So this 'elif' for passing when stuck is not applicable.
    # The stuck dealer *must* call.

    # Final check if any AI pass logic was missed or needs adjustment based on submit_action_api's pass handlers.
    # The main difference is that submit_action_api's pass handlers are for *any* player (human or AI proxy),
    # while process_ai_bid_action is specifically for AI's decision and execution.
    # The AI pass logic here now explicitly updates game state (turn, phase) similar to submit_action_api,
    # because when an AI passes, it's not going through submit_action_api's pass routes.
    # It directly calls the next AI's process_ai_bid_action or returns, letting the game loop handle human turn.
    # This ensures that AI-to-AI pass sequences correctly advance the game state.
    # This was a bit redundant as `submit_action_api` has `pass_bid` and `pass_call` which are called by AI as well.
    # Let's simplify. The AI should call the existing pass actions in `submit_action_api`.
    # This `process_ai_bid_action` should focus on *deciding* and then *triggering* an action (order_up, call_trump, or pass via submit_action).

    # REVISED AI PASS LOGIC:
    # If AI decides to pass, it should construct an action like {'player_index': player_idx, 'action': 'pass_bid'}
    # and call submit_action_api(ai_pass_action_data) or a similar shared handler.
    # For now, the existing structure where AI pass directly calls next AI or returns is kept,
    # but logging is added to trace this flow. The duplication of state change logic is a concern.
    # Let's assume the current duplicated logic for AI passing state changes is what's intended for now and just add logs.
    # The logs added above for AI passing should cover this.
    # The key is that if an AI passes, the game_data state (turn, phase) must be updated here.
    # The messages are also updated here.
    # The `submit_action_api` pass routes are primarily for when the *human* player passes, or when an AI action is proxied through the API.
    # When `process_ai_bid_action` is called directly for an AI, it handles its own pass logic internally.

    # Final logging for the end of the function if no action was taken (e.g. faulty action_type)
    # This should ideally not be reached if action_type is valid and leads to a decision.
    # However, if it falls through all conditions:
    # Corrected condition: Check if any positive action (order up, call trump) was taken.
    # If an action type was 'ai_bidding_round_1' or 'ai_bidding_round_2', and no order/call was made,
    # it means a pass occurred and was handled by the pass logic within those blocks.
    # 'ai_dealer_stuck_call' always results in a call.
    # So, this warning should only trigger if an unknown action_type was passed or logic error.
    action_taken = False
    if 'should_ai_order_up' in locals() and should_ai_order_up and action_type == 'ai_bidding_round_1': action_taken = True
    if 'should_ai_call_trump' in locals() and should_ai_call_trump and chosen_suit_by_ai and action_type == 'ai_bidding_round_2': action_taken = True
    if action_type == 'ai_dealer_stuck_call': action_taken = True # This always results in a call

    # If it was a bidding round and no positive action, it was a pass (handled)
    if action_type == 'ai_bidding_round_1' and not action_taken: action_taken = True # Pass was handled
    if action_type == 'ai_bidding_round_2' and not action_taken: action_taken = True # Pass was handled

    if not action_taken:
        logging.warning(f"AI P{player_idx} bid action {action_type} did not result in a defined outcome (order_up, call, or pass). This might indicate a logic flow issue.")
        pass # Explicitly do nothing more if no path was taken.

@app.route('/api/get_current_state', methods=['GET'])
def get_current_state_api():
    global game_data
    logging.info("API: /api/get_current_state called.")
    return jsonify(game_data_to_json(game_data))

@app.route('/api/ai_play_turn', methods=['POST'])
def ai_play_turn_api():
    global game_data
    logging.info("Received request for AI to play turn.")

    current_player = game_data.get("current_player_turn")
    current_phase = game_data.get("game_phase")

    if current_player == 0: # Player 0 is human
        logging.warning("AI play requested, but it's human's turn.")
        return jsonify({"error": "Not AI's turn.", "game_state": game_data_to_json(game_data)}), 400

    if current_phase != "playing_tricks":
        logging.warning(f"AI play requested, but phase is {current_phase}, not 'playing_tricks'.")
        return jsonify({"error": "AI can only play cards in 'playing_tricks' phase via this endpoint.", "game_state": game_data_to_json(game_data)}), 400

    if is_round_over():
        logging.info("AI play requested, but round is already over.")
        # Ensure game state reflects round over if not already.
        if game_data["game_phase"] not in ["round_over", "game_over"]:
            score_round() # This should set the phase correctly.
        return jsonify(game_data_to_json(game_data))

    logging.info(f"Processing AI P{current_player}'s turn via dedicated endpoint.")
    process_ai_play_card(current_player) # This AI plays its card

    # After the AI plays, the game state is updated by process_ai_play_card.
    # The turn might pass to another AI, a human, or the trick/round might end.
    # We return the new state. The client will decide if another /api/ai_play_turn is needed.
    return jsonify(game_data_to_json(game_data))


# No changes needed for game_data_to_json or if __name__ == "__main__":
# The main logging additions are within the game logic functions.

def game_data_to_json(current_game_data):
    json_safe_data = current_game_data.copy()
    if json_safe_data.get('deck'):
        json_safe_data['deck'] = [card.to_dict() for card in json_safe_data['deck']]
    if json_safe_data.get('hands'):
        json_safe_data['hands'] = { p_idx: [card.to_dict() for card in hand] for p_idx, hand in json_safe_data['hands'].items()}
    if json_safe_data.get('dummy_hand'): # Changed from kitty to dummy_hand
        json_safe_data['dummy_hand'] = [card.to_dict() for card in json_safe_data['dummy_hand']]
    if json_safe_data.get('up_card'):
        json_safe_data['up_card'] = json_safe_data['up_card'].to_dict() if json_safe_data['up_card'] else None
    if json_safe_data.get('original_up_card_for_round'):
        json_safe_data['original_up_card_for_round'] = json_safe_data['original_up_card_for_round'].to_dict() if json_safe_data['original_up_card_for_round'] else None
    if json_safe_data.get('trick_cards'):
        json_safe_data['trick_cards'] = [{'player': tc['player'], 'card': tc['card'].to_dict()} for tc in json_safe_data['trick_cards']]

    if json_safe_data.get('last_completed_trick') and json_safe_data['last_completed_trick'].get('played_cards'):
        json_safe_data['last_completed_trick']['played_cards'] = [
            {'player': pc['player'], 'card': pc['card'].to_dict() if isinstance(pc['card'], Card) else pc['card']}
            for pc in json_safe_data['last_completed_trick']['played_cards']
        ]
    return json_safe_data

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
