import unittest
from euchre import (
    Card, get_hand_features, SUITS, RANKS, get_left_bower_suit,
    get_rl_state, initialize_new_round, initialize_game_data, get_game_instance, # Removed game_data import, added get_game_instance
    get_effective_suit, get_card_value, # get_player_role is also used by get_rl_state
    get_player_role, evaluate_potential_trump_strength,
    RLAgent # Import RLAgent for TestAISloughingLogic
)


class TestGetHandFeatures(unittest.TestCase):

    def c(self, suit_rank_str):
        """Helper to create a Card object from a string like 'HA' for Ace of Hearts."""
        return Card(suit_rank_str[0], suit_rank_str[1:])

    def test_empty_hand(self):
        hand = []
        features = get_hand_features(hand, 'H')
        self.assertEqual(features["num_trump_cards"], 0)
        self.assertEqual(features["num_aces_offsuit"], 0)
        self.assertEqual(features["num_suits_void_natural"], 4)
        self.assertTrue(features["is_void_in_suit_H"])
        self.assertEqual(features["shortest_offsuit_length"], 0)
        self.assertEqual(features["num_cards_in_longest_offsuit"], 0)


    def test_no_trump_suit_defined(self):
        hand = [self.c('HA'), self.c('CK'), self.c('DQ'), self.c('SJ'), self.c('H9')] # No suit is void
        features = get_hand_features(hand, None)
        self.assertEqual(features["num_trump_cards"], 0) # No trump defined
        self.assertFalse(features["has_right_bower"])
        self.assertFalse(features["has_left_bower"])
        self.assertEqual(features["num_aces_offsuit"], 1) # HA is effectively offsuit
        # Natural suit counts: H:2, C:1, D:1, S:1 (from SJ)
        self.assertEqual(features["num_suits_void_natural"], 0) # No suit is void
        self.assertFalse(features["is_void_in_suit_S"])
        self.assertFalse(features["is_void_in_suit_H"])
        self.assertFalse(features["is_void_in_suit_D"])
        self.assertFalse(features["is_void_in_suit_C"])
        self.assertEqual(features["highest_trump_card_rank_value"], 0)
        self.assertEqual(features["lowest_trump_card_rank_value"], 0)
        self.assertEqual(features["num_cards_in_longest_offsuit"], 2) # Hearts (HA, H9)
        self.assertEqual(features["shortest_offsuit_length"], 1) # C, D, S all have 1 card

    def test_simple_trump_hand_hearts_trump(self):
        hand = [self.c('HA'), self.c('HK'), self.c('HQ'), self.c('DJ'), self.c('S10')] # DJ is Left Bower if H is trump. Clubs void.
        trump_suit = 'H'
        features = get_hand_features(hand, trump_suit)

        self.assertEqual(features["num_trump_cards"], 4) # AH, KH, QH, JD (Left)
        self.assertFalse(features["has_right_bower"]) # No JH
        self.assertTrue(features["has_left_bower"])   # DJ
        self.assertTrue(features["has_ace_of_trump"])
        self.assertTrue(features["has_king_of_trump"])
        self.assertTrue(features["has_queen_of_trump"])
        self.assertEqual(features["num_aces_offsuit"], 0)
        self.assertEqual(features["num_suits_void_natural"], 1) # Clubs is void
        self.assertTrue(features["is_void_in_suit_C"])
        self.assertFalse(features["is_void_in_suit_H"])
        self.assertFalse(features["is_void_in_suit_D"]) # Has DJ
        self.assertFalse(features["is_void_in_suit_S"])
        # Values from get_card_value: AH=80, KH=70, QH=60, LeftBower=90
        self.assertEqual(features["highest_trump_card_rank_value"], 90) # Left Bower
        self.assertEqual(features["lowest_trump_card_rank_value"], 60)  # Queen of Trump
        # Offsuits: S (len 1). C is void (len 0). D is trump (left bower). H is trump.
        self.assertEqual(features["num_cards_in_longest_offsuit"], 1) # S10
        self.assertEqual(features["shortest_offsuit_length"], 0) # Clubs is void

    def test_bowers_and_trump_spades_trump(self):
        hand = [self.c('SJ'), self.c('CJ'), self.c('SA'), self.c('S9'), self.c('HA')] # SJ=Right, CJ=Left. Diamonds void.
        trump_suit = 'S'
        features = get_hand_features(hand, trump_suit)

        self.assertEqual(features["num_trump_cards"], 4) # SJ, CJ, SA, S9
        self.assertTrue(features["has_right_bower"])
        self.assertTrue(features["has_left_bower"])
        self.assertTrue(features["has_ace_of_trump"])
        self.assertFalse(features["has_king_of_trump"])
        self.assertEqual(features["num_aces_offsuit"], 1) # HA
        self.assertEqual(features["num_suits_void_natural"], 1) # Diamonds
        self.assertTrue(features["is_void_in_suit_D"])
        # Values: SJ=100, CJ=90, SA=80, S9=40
        self.assertEqual(features["highest_trump_card_rank_value"], 100)
        self.assertEqual(features["lowest_trump_card_rank_value"], 40)
        # Offsuits: H (len 1). D is void (len 0). C is trump (left bower). S is trump.
        self.assertEqual(features["num_cards_in_longest_offsuit"], 1) # HA
        self.assertEqual(features["shortest_offsuit_length"], 0) # Diamonds is void

    def test_void_in_multiple_natural_suits(self):
        hand = [self.c('HA'), self.c('HK'), self.c('HQ'), self.c('H10'), self.c('H9')]
        trump_suit = 'D' # Diamonds trump, hand is all Hearts
        features = get_hand_features(hand, trump_suit)

        self.assertEqual(features["num_trump_cards"], 0)
        self.assertFalse(features["has_right_bower"])
        self.assertEqual(features["num_aces_offsuit"], 1) # HA
        self.assertEqual(features["num_suits_void_natural"], 3) # D, C, S are void
        self.assertTrue(features["is_void_in_suit_D"])
        self.assertTrue(features["is_void_in_suit_C"])
        self.assertTrue(features["is_void_in_suit_S"])
        self.assertFalse(features["is_void_in_suit_H"])
        # Trump 'D'. Left Bower's suit is 'H'. Hand is all 'H'.
        # So, 'H' is considered an "effectively trump" suit for offsuit length calculation.
        # Thus, no true offsuits. C and S are void.
        self.assertEqual(features["num_cards_in_longest_offsuit"], 0) # Hearts is not counted as offsuit.
        self.assertEqual(features["shortest_offsuit_length"], 0) # C or S are void, and H is not offsuit.

    def test_all_trump_hand(self):
        hand = [self.c('DA'), self.c('DK'), self.c('DQ'), self.c('DJ'), self.c('D10')]
        trump_suit = 'D'
        features = get_hand_features(hand, trump_suit)

        self.assertEqual(features["num_trump_cards"], 5)
        self.assertTrue(features["has_right_bower"]) # DJ
        self.assertFalse(features["has_left_bower"]) # No HJ for Diamonds trump
        self.assertTrue(features["has_ace_of_trump"])
        self.assertEqual(features["num_aces_offsuit"], 0)
        self.assertEqual(features["num_suits_void_natural"], 3) # H, C, S
        # Values: DA=80, DK=70, DQ=60, DJ(Right)=100, D10=50
        self.assertEqual(features["highest_trump_card_rank_value"], 100)
        self.assertEqual(features["lowest_trump_card_rank_value"], 50)
        # No offsuits, as H, C, S are void and D is trump. Left bower (HJ) is not present.
        self.assertEqual(features["num_cards_in_longest_offsuit"], 0)
        self.assertEqual(features["shortest_offsuit_length"], 0)

    def test_mixed_hand_longest_shortest_offsuit(self):
        # Clubs trump. Hand: CA, CK, HQ, H10, D9. Spades void.
        hand = [self.c('CA'), self.c('CK'), self.c('HQ'), self.c('H10'), self.c('D9')]
        trump_suit = 'C'
        features = get_hand_features(hand, trump_suit)

        self.assertEqual(features["num_trump_cards"], 2) # CA, CK
        self.assertTrue(features["has_ace_of_trump"])
        self.assertFalse(features["has_left_bower"]) # No SJ
        self.assertEqual(features["num_aces_offsuit"], 0)
        self.assertEqual(features["num_suits_void_natural"], 1) # Spades
        self.assertTrue(features["is_void_in_suit_S"])

        # Offsuits are Hearts (HQ, H10) and Diamonds (D9). Spades is void.
        # Natural suit counts: C:2, H:2, D:1, S:0
        # Left bower actual suit would be 'S'.
        # Offsuits for length calculation: H (length 2), D (length 1). S is trump (left bower's suit).
        self.assertEqual(features["num_cards_in_longest_offsuit"], 2) # Hearts
        self.assertEqual(features["shortest_offsuit_length"], 1)     # Diamonds. Spades is void but is trump here.

    def test_hand_with_only_left_bower_as_trump(self):
        hand = [self.c('HJ'), self.c('DA'), self.c('DK'), self.c('DQ'), self.c('D10')] # Clubs, Spades void
        trump_suit = 'D' # Diamonds is trump, HJ is Left Bower
        features = get_hand_features(hand, trump_suit)

        self.assertEqual(features["num_trump_cards"], 5) # DA,DK,DQ,D10, HJ (Left)
        self.assertFalse(features["has_right_bower"]) # No DJ
        self.assertTrue(features["has_left_bower"])   # HJ
        self.assertTrue(features["has_ace_of_trump"])
        self.assertEqual(features["num_aces_offsuit"], 0) # DA is trump
        self.assertEqual(features["num_suits_void_natural"], 2) # Clubs, Spades
        self.assertTrue(features["is_void_in_suit_C"])
        self.assertTrue(features["is_void_in_suit_S"])
        # Values: DA=80, DK=70, DQ=60, D10=50, HJ(Left)=90
        self.assertEqual(features["highest_trump_card_rank_value"], 90) # Left Bower
        self.assertEqual(features["lowest_trump_card_rank_value"], 50)  # D10
        # Hearts is effectively trump (due to Left Bower), Diamonds is trump. C and S are void.
        # No offsuits.
        self.assertEqual(features["num_cards_in_longest_offsuit"], 0)
        self.assertEqual(features["shortest_offsuit_length"], 0)

    def test_offsuit_calcs_when_trump_is_none(self):
        hand = [self.c('HA'), self.c('HK'), self.c('DQ'), self.c('DJ'), self.c('C9')] # Spades is void
        features = get_hand_features(hand, None) # No trump suit
        # Natural suit counts: H:2, D:2, C:1, S:0
        self.assertEqual(features["num_cards_in_longest_offsuit"], 2) # H and D both have 2
        self.assertEqual(features["shortest_offsuit_length"], 0)     # Spades is void

        hand_2 = [self.c('HA'), self.c('HK'), self.c('HQ'), self.c('H10'), self.c('H9')] # D, C, S void
        features_2 = get_hand_features(hand_2, None)
        # Natural suit counts: H:5, D:0, C:0, S:0
        self.assertEqual(features_2["num_cards_in_longest_offsuit"], 5) # H
        self.assertEqual(features_2["shortest_offsuit_length"], 0)     # D, C, S are void

if __name__ == '__main__':
    unittest.main()


class TestCardTracking(unittest.TestCase):
    def c(self, suit_rank_str):
        """Helper to create a Card object from a string like 'HA' for Ace of Hearts."""
        return Card(suit_rank_str[0], suit_rank_str[1:])

    def setUp(self):
        """Set up a basic game_data structure for testing get_rl_state with played cards."""
        initialize_game_data() # Ensures a fresh _current_game_instance
        self.game_instance = get_game_instance()
        self.game_data = self.game_instance.game_data
        # Default setup for these tests, individual tests can override
        self.game_data.update({
            "hands": {0: [], 1: [], 2: []},
            "trump_suit": None,
            "played_cards_this_round": [],
            "original_up_card_for_round": None,
            "dealer": 0,
            "maker": None,
            "scores": {0:0, 1:0, 2:0},
            "round_tricks_won": {0:0, 1:0, 2:0},
            "passes_on_upcard": [],
            "passes_on_calling": [],
            "num_players": 3,
            "player_identities": {0: "P0", 1: "P1", 2: "P2"}
        })

    def test_no_cards_played_in_state(self):
        # No trump set yet
        state = get_rl_state(player_id=1, current_game_data=self.game_data)
        self.assertEqual(state["played_cards_serialized"], [])
        self.assertFalse(state["played_right_bower"])
        self.assertFalse(state["played_left_bower"])
        self.assertEqual(state["num_trump_cards_played"], 0)
        self.assertEqual(state["num_suit_played_H"], 0)
        self.assertEqual(state["num_suit_played_D"], 0)
        self.assertEqual(state["num_suit_played_C"], 0)
        self.assertEqual(state["num_suit_played_S"], 0)

    def test_few_non_trump_cards_played(self):
        self.game_data["trump_suit"] = 'S' # Spades trump
        self.game_data["played_cards_this_round"] = [self.c('HA'), self.c('DK')]

        state = get_rl_state(player_id=1, current_game_data=self.game_data)

        self.assertEqual(state["played_cards_serialized"], sorted(['HA', 'DK']))
        self.assertFalse(state["played_right_bower"])
        self.assertFalse(state["played_left_bower"])
        self.assertEqual(state["num_trump_cards_played"], 0)
        self.assertEqual(state["num_suit_played_H"], 1)
        self.assertEqual(state["num_suit_played_D"], 1)
        self.assertEqual(state["num_suit_played_C"], 0)
        self.assertEqual(state["num_suit_played_S"], 0)

    def test_trump_cards_played_including_bowers(self):
        self.game_data["trump_suit"] = 'H' # Hearts trump
        # Played: Right (HJ), Left (DJ), Ace (HA), King (HK), off-suit (S9)
        self.game_data["played_cards_this_round"] = [
            self.c('HJ'), self.c('DJ'), self.c('HA'), self.c('HK'), self.c('S9')
        ]
        state = get_rl_state(player_id=1, current_game_data=self.game_data)

        self.assertEqual(state["played_cards_serialized"], sorted(['HJ', 'DJ', 'HA', 'HK', 'S9']))
        self.assertTrue(state["played_right_bower"])
        self.assertTrue(state["played_left_bower"])
        self.assertTrue(state["played_ace_trump"])
        self.assertTrue(state["played_king_trump"])
        self.assertFalse(state["played_queen_trump"]) # Queen not played
        self.assertEqual(state["num_trump_cards_played"], 4) # JH, JD, AH, KH are all trump

        self.assertEqual(state["num_suit_played_H"], 3) # JH, AH, KH
        self.assertEqual(state["num_suit_played_D"], 1) # JD
        self.assertEqual(state["num_suit_played_C"], 0)
        self.assertEqual(state["num_suit_played_S"], 1) # S9

    def test_mixed_cards_played_different_trump(self):
        self.game_data["trump_suit"] = 'C' # Clubs trump
        # Played: Ace of Trump (CA), Left Bower (SJ), off-suit King (HK), off-suit 9 (D9)
        self.game_data["played_cards_this_round"] = [
            self.c('CA'), self.c('SJ'), self.c('HK'), self.c('D9') # Corrected: CA, SJ
        ]
        state = get_rl_state(player_id=1, current_game_data=self.game_data)

        self.assertEqual(state["played_cards_serialized"], sorted(['CA', 'SJ', 'HK', 'D9']))
        self.assertFalse(state["played_right_bower"]) # JC not played
        self.assertTrue(state["played_left_bower"])   # SC
        self.assertTrue(state["played_ace_trump"])    # AC
        self.assertFalse(state["played_king_trump"])  # KC not played
        self.assertFalse(state["played_queen_trump"]) # QC not played
        self.assertEqual(state["num_trump_cards_played"], 2) # AC, SC are trump

        self.assertEqual(state["num_suit_played_H"], 1) # HK
        self.assertEqual(state["num_suit_played_D"], 1) # D9
        self.assertEqual(state["num_suit_played_C"], 1) # AC
        self.assertEqual(state["num_suit_played_S"], 1) # SC

    def test_played_cards_this_round_cleared_by_initialize_new_round(self):
        # Simulate some cards played in a hypothetical previous round
        self.game_data["played_cards_this_round"] = [self.c('HA'), self.c('SK')]
        self.game_data["dealer"] = 0 # initialize_new_round needs a dealer
        self.game_data["num_players"] = 3
        self.game_data["player_identities"] = {0:"P0",1:"P1",2:"P2"}

        # Call initialize_new_round (which uses get_game_instance internally)
        # The self.game_data should be the one from the current instance
        self.game_data["played_cards_this_round"] = [self.c('HA'), self.c('SK')] # Pre-populate for test

        # Ensure initialize_new_round has enough context if it relies on more than just dealer and num_players
        # from the global game_data structure. Typically, it should operate on the current game instance.
        # For this test, the important part is that initialize_new_round is called on the
        # correct game instance, which should be managed by get_game_instance().
        initialize_new_round() # This should reset played_cards_this_round on the current game instance

        self.assertEqual(self.game_data["played_cards_this_round"], [])
        # No need to manually reset euchre.initialize_game_data() if setUp does it.


class TestAISloughingLogic(unittest.TestCase):
    def c(self, suit_rank_str):
        return Card(suit_rank_str[0], suit_rank_str[1:])

    def setUp(self):
        initialize_game_data()
        self.game_instance = get_game_instance()
        self.game_data = self.game_instance.game_data
        self.rl_agents = self.game_instance.rl_agents # Access agents from the instance

        self.game_data["num_players"] = 3
        self.game_data["player_identities"] = {0: "P0", 1: "P1 (AI)", 2: "P2 (Human)"}
        self.game_data["dealer"] = 0
        self.game_data["scores"] = {0:0, 1:0, 2:0}
        self.game_data["round_tricks_won"] = {0:0, 1:0, 2:0}
        # Clear any default agents and let tests set them up if needed
        self.game_instance.rl_agents.clear()


    def tearDown(self):
        # initialize_game_data() in setUp of the next test will handle fresh state.
        pass


    def test_heuristic_ai_sloughs_lowest_card(self):
        ai_player_idx = 1
        self.game_data["hands"][ai_player_idx] = [self.c('CA'), self.c('C9'), self.c('SK')]
        self.game_data["trump_suit"] = 'H'
        self.game_data["current_trick_lead_suit"] = 'D'
        self.game_data["trick_cards"] = [{'player': 2, 'card': self.c('DQ')}]
        self.game_data["game_phase"] = "playing_tricks"
        self.game_data["current_player_turn"] = ai_player_idx
        # No RL agent for this player, so it uses heuristic
        self.game_instance.rl_agents.pop(ai_player_idx, None)

        # Call as a module function, not a method of game_instance
        # initialize_new_round() was causing the hand to be reset after test setup.
        import euchre
        euchre.process_ai_play_card(ai_player_idx)

        self.assertEqual(len(self.game_data["hands"][ai_player_idx]), 2)
        played_card_in_trick = self.game_data["trick_cards"][-1]['card']
        self.assertEqual(played_card_in_trick.suit, 'C')
        self.assertEqual(played_card_in_trick.rank, '9')

    def test_heuristic_ai_must_follow_suit_plays_high(self):
        ai_player_idx = 1
        self.game_data["hands"][ai_player_idx] = [self.c('DA'), self.c('D9'), self.c('SK')]
        self.game_data["trump_suit"] = 'H'
        self.game_data["current_trick_lead_suit"] = 'D'
        self.game_data["trick_cards"] = []
        self.game_data["game_phase"] = "playing_tricks"
        self.game_data["current_player_turn"] = ai_player_idx
        self.game_instance.rl_agents.pop(ai_player_idx, None)

        import euchre
        euchre.process_ai_play_card(ai_player_idx)

        played_card_in_trick = self.game_data["trick_cards"][-1]['card']
        self.assertEqual(played_card_in_trick.suit, 'D')
        self.assertEqual(played_card_in_trick.rank, 'A')

    def test_heuristic_ai_can_trump_to_win(self):
        ai_player_idx = 1
        self.game_data["hands"][ai_player_idx] = [self.c('H9'), self.c('CA'), self.c('C9')]
        self.game_data["trump_suit"] = 'H'
        self.game_data["current_trick_lead_suit"] = 'D'
        self.game_data["trick_cards"] = [{'player': 2, 'card': self.c('DK')}]
        self.game_data["game_phase"] = "playing_tricks"
        self.game_data["current_player_turn"] = ai_player_idx
        self.game_instance.rl_agents.pop(ai_player_idx, None)

        import euchre
        euchre.process_ai_play_card(ai_player_idx)

        played_card_in_trick = self.game_data["trick_cards"][-1]['card']
        self.assertEqual(played_card_in_trick.suit, 'H')
        self.assertEqual(played_card_in_trick.rank, '9')

    def test_rl_agent_overlay_corrects_bad_slough(self):
        ai_player_idx = 1
        class MockRLAgent(RLAgent): # Inherit from RLAgent
            def choose_action(self, state_dict, valid_actions):
                for action in valid_actions:
                    if action['suit'] == 'C' and action['rank'] == 'A':
                        return action
                return valid_actions[0]

        self.game_instance.rl_agents[ai_player_idx] = MockRLAgent(player_id=ai_player_idx)

        self.game_data["hands"][ai_player_idx] = [self.c('CA'), self.c('C9'), self.c('SK')]
        self.game_data["trump_suit"] = 'H'
        self.game_data["current_trick_lead_suit"] = 'D'
        self.game_data["trick_cards"] = [{'player': 2, 'card': self.c('DQ')}]
        self.game_data["game_phase"] = "playing_tricks"
        self.game_data["current_player_turn"] = ai_player_idx
        self.game_data["rl_training_data"] = {ai_player_idx: {}}

        import euchre
        euchre.process_ai_play_card(ai_player_idx)

        played_card_in_trick = self.game_data["trick_cards"][-1]['card']
        self.assertEqual(played_card_in_trick.suit, 'C')
        self.assertEqual(played_card_in_trick.rank, '9')

    def test_rl_agent_overlay_allows_correct_slough(self):
        ai_player_idx = 1
        class MockRLAgent(RLAgent):
            def choose_action(self, state_dict, valid_actions):
                for action in valid_actions:
                    if action['suit'] == 'C' and action['rank'] == '9':
                        return action
                return valid_actions[0]

        self.game_instance.rl_agents[ai_player_idx] = MockRLAgent(player_id=ai_player_idx)

        self.game_data["hands"][ai_player_idx] = [self.c('CA'), self.c('C9'), self.c('SK')]
        self.game_data["trump_suit"] = 'H'
        self.game_data["current_trick_lead_suit"] = 'D'
        self.game_data["trick_cards"] = [{'player': 2, 'card': self.c('DQ')}]
        self.game_data["game_phase"] = "playing_tricks"
        self.game_data["current_player_turn"] = ai_player_idx
        self.game_data["rl_training_data"] = {ai_player_idx: {}}

        import euchre
        euchre.process_ai_play_card(ai_player_idx)

        played_card_in_trick = self.game_data["trick_cards"][-1]['card']
        self.assertEqual(played_card_in_trick.suit, 'C')
        self.assertEqual(played_card_in_trick.rank, '9')

    def test_rl_agent_overlay_allows_trump_play(self):
        ai_player_idx = 1
        H9 = self.c('H9')
        CA = self.c('CA')
        C9 = self.c('C9')

        class MockRLAgent(RLAgent):
            def choose_action(self, state_dict, valid_actions):
                for action in valid_actions:
                    if action['suit'] == 'H' and action['rank'] == '9':
                        return action
                return valid_actions[0]

        self.game_instance.rl_agents[ai_player_idx] = MockRLAgent(player_id=ai_player_idx)

        self.game_data["hands"][ai_player_idx] = [H9, CA, C9]
        self.game_data["trump_suit"] = 'H'
        self.game_data["current_trick_lead_suit"] = 'D'
        self.game_data["trick_cards"] = [{'player': 2, 'card': self.c('DK')}]
        self.game_data["game_phase"] = "playing_tricks"
        self.game_data["current_player_turn"] = ai_player_idx
        self.game_data["rl_training_data"] = {ai_player_idx: {}}

        import euchre
        euchre.process_ai_play_card(ai_player_idx)

        played_card_in_trick = self.game_data["trick_cards"][-1]['card']
        self.assertEqual(played_card_in_trick.suit, 'H')
        self.assertEqual(played_card_in_trick.rank, '9')
