import logging

# Configure logging to see output from the simulation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Attempt to import the necessary function and other components
    # We need to ensure that global variables like 'game_data' and 'rl_agents'
    # are handled correctly, as they are defined at the module level in euchre.py
    # and modified by initialize_game_data.

    # To simulate the euchre.py environment for globals, we can try importing then initializing.
    # However, direct imports of globals can be tricky if they are expected to be set up by Flask context
    # or specific initializers not run here.

    # The key is that initialize_game_data() inside euchre.py should set up 'rl_agents'
    # before run_training_simulation() tries to use it.

    from euchre import run_training_simulation, initialize_game_data, RLAgent, Game, get_game_instance

    logging.info("Successfully imported components from euchre.py")

    # Explicitly initialize the game instance and agents if needed,
    # though run_training_simulation should now do this.
    # This ensures that if run_training_simulation internally calls initialize_game_data,
    # it operates on a fresh state.
    # get_game_instance() # This will create the Game instance which initializes game_data and agents
    # If RLAgent relies on specific global setup not done by Game(), that might need addressing.

    logging.info("Attempting to run the training simulation for 1 game...")
    # The fix was to ensure run_training_simulation itself calls initialize_game_data()
    # at its beginning.

    # We also need to ensure that the RLAgent class can be instantiated, which means
    # Q_TABLE_DB_FILE needs to be valid. The default is "q_table.sqlite".
    # The _init_db method of RLAgent tries to connect to this.

    # If Flask-specific context is needed for some parts of euchre.py, this test might fail.
    # For now, assuming the core logic of run_training_simulation and RLAgent DB init
    # can run standalone after the NameError fix.

    run_training_simulation(num_games_to_simulate=1, save_interval=1)
    logging.info("Training simulation for 1 game completed successfully.")
    print("\nSUCCESS: test_simulation_start.py ran without the NameError and completed a short simulation.")

except NameError as ne:
    logging.error(f"NameError encountered: {ne}")
    print(f"\nFAILURE: Still encountered a NameError: {ne}")
    print("This indicates the previous fix might not have fully resolved the issue, or there's another NameError.")
except ImportError as ie:
    logging.error(f"ImportError encountered: {ie}")
    print(f"\nFAILURE: Could not import necessary components from euchre.py: {ie}")
    print("Please ensure euchre.py is in the same directory or accessible via PYTHONPATH.")
except Exception as e:
    logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    print(f"\nFAILURE: An unexpected error occurred during the test: {e}")

# Note: This test primarily checks if the NameError for 'rl_agents' is gone
# when starting the simulation. It doesn't validate the full correctness of the simulation
# or the Flask app.
# If RLAgent's __init__ or other parts of the simulation fail due to missing setup
# (e.g., Flask app context for other globals if they were used, or DB issues),
# those would be new errors to investigate.
# The test also assumes that the Q_TABLE_DB_FILE ("q_table.sqlite") can be created/accessed
# in the current directory.
# The `save_interval` is not used by SQLite version but kept for compatibility with function signature.
# The test also relies on `initialize_game_data()` within `run_training_simulation` correctly setting up
# not just `rl_agents` but also the `_current_game_instance` for `get_game_instance()` to work,
# and the `Game` class correctly initializing `rl_agents` within its structure.
# The `RLAgent` class itself also needs to be importable and its `__init__` runnable.
# The `Game` class also creates `RLAgent` instances.
# The `get_game_instance()` function is also used by `get_rl_agent()`.
# The `QLearningAgent`'s `_init_db` must be able to run to create the SQLite table.
# `initialize_new_round` uses `game_data` global, which should be handled by the `Game` class instance.
# `process_ai_bid_action` uses `get_rl_agent` which relies on the `Game` instance.
# `score_round` uses `rl_agents` global, which should be accessed via the `Game` instance.
# `process_rl_update` also uses `rl_agents` and `game_data` globals.
# The refactoring to use a Game class instance (`_current_game_instance`) should mitigate direct global use,
# but the training loop itself might still have some direct global accesses that were missed.
# The critical part is that `initialize_game_data()` in `run_training_simulation` now creates the `Game` instance,
# which in turn creates the `RLAgent` instances and stores them in `self.rl_agents`.
# The global `rl_agents` dictionary reference in `euchre.py` that was causing the NameError
# should now be populated by `initialize_game_data()` before `run_training_simulation`'s main loop.
# The line `from euchre import run_training_simulation, initialize_game_data, RLAgent, Game, get_game_instance`
# ensures these names are available.
# The `run_training_simulation` function itself calls `initialize_game_data()`, which should set up the
# `_current_game_instance` and its internal `rl_agents`.
# The global `rl_agents` dictionary is still referenced in `score_round` and `process_rl_update`.
# The `initialize_game_data` function *does* set `_current_game_instance = Game()`, and `Game.__init__` *does*
# populate `self.rl_agents`. The issue is that `score_round` and `process_rl_update` directly access the global `rl_agents`
# which is *not* the same as `_current_game_instance.rl_agents`. This is a separate bug.

# For THIS test, we are only checking if the NameError in `run_training_simulation` at line 2246 is fixed.
# The line was `if not rl_agents:`. My fix was to call `initialize_game_data()` before this.
# `initialize_game_data()` *should* define the global `rl_agents = {}` then populate it via the Game instance.
# Let's double check `initialize_game_data()`:
# It does: `_current_game_instance = Game()`.
# `Game.__init__` does: `self.rl_agents.clear()` and then `self.rl_agents[pid] = RLAgent(...)`.
# It *does not* assign to the global `rl_agents`. This is the problem.

# The `NameError` fix in `run_training_simulation` was to call `initialize_game_data()` first.
# `initialize_game_data` does: `_current_game_instance = Game()`.
# `Game._initialize_game_and_agents` (called by `Game.__init__`) does: `self.rl_agents.clear()` and then populates `self.rl_agents`.
# The global `rl_agents` is never actually assigned to by `initialize_game_data` or the `Game` class constructor.
# This means the `NameError` on `if not rl_agents:` will persist because the global `rl_agents` is not being defined by `initialize_game_data`.

# The original problem was `NameError: name 'rl_agents' is not defined` in `run_training_simulation`.
# The `run_training_simulation` function looks like this now (after my previous fix attempt):
#
# def run_training_simulation(...):
#     logging.info(...)
#     initialize_game_data() # THIS IS THE KEY CALL
#     # ... (loop)
#
# And `initialize_game_data()` is:
#
# _current_game_instance = None
# def get_game_instance(): ...
# def initialize_game_data():
#     global _current_game_instance
#     _current_game_instance = Game() # Game() constructor calls _initialize_game_and_agents()
#     logging.info("Global game instance has been re-initialized.")
#
# And `Game._initialize_game_and_agents()` does:
#    self.rl_agents.clear()
#    for pid, name in player_identities.items():
#        if pid not in self.rl_agents: # This refers to self.rl_agents
#            self.rl_agents[pid] = RLAgent(player_id=pid)
#
# The global `rl_agents = {}` is defined at the top of euchre.py.
# The `NameError` occurs if `initialize_game_data()` is not called before `if not rl_agents`.
# My previous fix *does* call `initialize_game_data()` before that check.
# So, the global `rl_agents` *should* be an empty dict `{}` at that point, and the `NameError` should be gone.
# The test script should pass this specific `NameError`.
# The *subsequent* logic errors regarding global `rl_agents` vs `_current_game_instance.rl_agents`
# are different issues that this test isn't designed to catch but might surface as other errors.
# This test is narrowly focused on the original `NameError`.
# The globals `game_data` and `rl_agents` are indeed defined at the top level of `euchre.py`.
# `game_data = {}`
# `rl_agents = {}`
# So, when `euchre.py` is first imported, these exist as empty dicts.
# The `NameError` would only occur if `run_training_simulation` was somehow defined and called
# in a scope where these module-level globals were not visible, which is not the case here.
# The original traceback was:
# File "C:\Users\Mclei\cte-og\euchre.py", line 2246, in run_training_simulation
#    if not rl_agents:
# NameError: name 'rl_agents' is not defined
# This implies that at the point of the call to `run_training_simulation` (line 2357),
# when `run_training_simulation` itself was entered, the global `rl_agents` was not defined in its scope.
# This is very strange if `rl_agents = {}` is at the top level of `euchre.py`.

# Let's re-verify the top of euchre.py structure provided earlier.
# The provided euchre.py does NOT show `game_data = {}` or `rl_agents = {}` at the top level.
# It shows:
# _current_game_instance = None
# def get_game_instance(): ...
# def get_rl_agent(player_id): ... (uses get_game_instance())
# def initialize_game_data(): ... (sets _current_game_instance)

# Ah, I see. The user's code snippet in the original message showed `game_data = {}` and `rl_agents = {}`
# being deprecated. My apologies, I was looking at an assumed structure.
# If these are truly removed from the global scope and only exist within the Game instance,
# then `run_training_simulation` indeed cannot access a global `rl_agents`.

# The `run_training_simulation` function will need to be refactored to use `get_game_instance().rl_agents`.
# This is a more involved fix than just calling `initialize_game_data()`.

# The immediate `NameError` on `if not rl_agents:` in `run_training_simulation` (line 2246)
# *after* my previous proposed fix (calling `initialize_game_data()` at the start of `run_training_simulation`)
# means that `initialize_game_data()` does *not* create a global `rl_agents`. This is correct.
# The `rl_agents` now live inside the `_current_game_instance`.

# So, the plan needs to change. The `NameError` is valid.
# The fix is to refactor `run_training_simulation` and any functions it calls
# that try to access global `rl_agents` or `game_data` to use the `get_game_instance()` pattern.

# This test script, as written, will likely still fail with the NameError because
# `run_training_simulation` itself tries to access `rl_agents` globally.
# The test script is fine, but the fix needs to be more substantial in `euchre.py`.

# Let's first confirm the structure of `euchre.py` regarding global `game_data` and `rl_agents`.
# Reading `euchre.py` again:
# - `Q_TABLE_DB_FILE` is global.
# - `RLAgent` class.
# - `Card` class, `create_deck`.
# - `Game` class.
# - `_current_game_instance = None`
# - `get_game_instance()`
# - `get_rl_agent(player_id)`: This correctly uses `game = get_game_instance(); return game.get_agent(player_id);`
# - `initialize_game_data()`: Correctly sets `_current_game_instance = Game()`.
# - `get_rl_state(player_id, current_game_data_dict)`: Takes game_data as arg. Good.
# - `process_rl_update(player_id_acted, event_type, event_data=None)`:
#   - `game = get_game_instance()`
#   - `current_game_data = game.game_data` (GOOD)
#   - `agent = game.get_agent(player_id_acted)` (GOOD)
#   - BUT then, in `score_round` (called by `process_ai_play_card` which is called by simulation loop):
#     - `score_round()`:
#       - `global game_data` (This is problematic if we want to phase out the true global)
#       - `for ai_p_id in rl_agents.keys():` <--- PROBLEM: Accesses global `rl_agents`
# - `initialize_new_round()`:
#   - `game = get_game_instance()`
#   - `current_game_data = game.game_data` (GOOD pattern, but then it assigns to `game_data["deck"]` etc.
#     This implies it's modifying `game.game_data` via the `game_data` alias, which is okay IF `game_data` was `game.game_data`.
#     It looks like `initialize_new_round` does `game_data["deck"] = ...`. This should be `current_game_data["deck"]`.
#     Let's check: `game_data["deck"] = create_deck()`. This refers to the global `game_data` if it exists.
#     The top of `euchre.py` does NOT show `game_data = {}` anymore.
#     So, `initialize_new_round` will also cause a `NameError` for `game_data`.

# The problem is deeper: the script is in a halfway state of refactoring globals.
# Some functions use `get_game_instance().game_data`, others try to use a global `game_data` or `rl_agents` that no longer exist.

# The `run_training_simulation` function itself:
#   - Calls `initialize_game_data()`: This sets `_current_game_instance`.
#   - Then it calls `initialize_new_round()`. This function will try to access global `game_data` which is not defined.

# So the very first `NameError` will be in `initialize_new_round` for `game_data`,
# *before* it even gets to the `if not rl_agents:` line in `run_training_simulation`'s loop.

# The user's traceback points to `if not rl_agents:` in `run_training_simulation`.
# This implies `initialize_new_round()` *didn't* crash due to `game_data` not being defined.
# This means there IS a global `game_data` and `rl_agents` defined somewhere that I missed, or the
# execution path is different.

# Let's re-read the user's exact error:
# File "C:\Users\Mclei\cte-og\euchre.py", line 2246, in run_training_simulation
#    if not rl_agents:
# NameError: name 'rl_agents' is not defined

# This is *after* my proposed (but not yet applied to user's file) fix of adding `initialize_game_data()` at the start of `run_training_simulation`.
# The user is reporting the error *still exists*.
# This means `rl_agents` is not being defined as a global by any means before that line.

# If `rl_agents = {}` is NOT at the top of `euchre.py`, then `NameError` is expected.
# The solution is to refactor `run_training_simulation` to not expect global `rl_agents`.
# It should use `get_game_instance().rl_agents` if it needs to check the agents.

# However, the line `if not rl_agents:` was intended to be a guard *before* `initialize_game_data()`.
# The user's report "i cant run the program at all" and the traceback suggests this is the *first* error they hit.
# My previous fix which I *thought* I applied was to move `initialize_game_data()` *before* this check.
# If the user is re-running the *original* code, then the `NameError` at line 2246 is expected.
# If they are running code *with my previous fix applied*, then the `NameError` at that line means
# that `rl_agents` is not a global that can be checked like that.

# Let's assume the user *has* the code where `initialize_game_data()` is called at the start of `run_training_simulation`.
# `initialize_game_data()` creates `_current_game_instance = Game()`.
# `Game()` creates `self.rl_agents`.
# No global `rl_agents` is created or modified by `initialize_game_data` or `Game`.
# Therefore, the line `if not rl_agents:` will still cause a `NameError` because `rl_agents` (as a global) is not defined.

# The fix for *that specific line* is to remove it, because `initialize_game_data()` is now called unconditionally.
# OR, if the check is desired, it must check `get_game_instance().rl_agents`.

# The test script is less about "making a test program to run the main program" and more about
# "making a script to see if the specific NameError in run_training_simulation is gone".
# The current test script will still fail if `run_training_simulation` has the `if not rl_agents:` line.

# New plan:
# 1. Modify `run_training_simulation` in `euchre.py` again:
#    - Ensure `initialize_game_data()` is called at the beginning.
#    - Remove the line `if not rl_agents:` as it's causing the NameError and `initialize_game_data()` now handles agent creation.
# 2. Then, address the other global uses, starting with `initialize_new_round` not using global `game_data`.
# This will be iterative.

# For now, let's create the test file, but acknowledge it will likely fail until `euchre.py` is further corrected.
# The user wants a test program. This is it. It will highlight the existing error.

# The code block for the test script is fine.
# The key is that the user needs to understand this test script *will likely fail* with the current `euchre.py`
# and that the path to fixing it involves modifying `euchre.py` itself.
# The test script helps confirm *where* it's failing in a controlled way.

# The user is asking for a test program *to run*. This script is that.
# It will demonstrate the error.
# My previous "fix" for `run_training_simulation` was insufficient.
# The line `if not rl_agents:` is the direct cause of the reported `NameError`.
# It needs to be removed or changed to use `get_game_instance()`.
# Simplest is to remove it since `initialize_game_data()` is called right before.
# This is step 1 of the new plan.
# The test script will then be used to verify this specific fix.

# The previous `replace_with_git_merge_diff` for `run_training_simulation` was:
# logging.info(...)
# initialize_game_data() # ADDED THIS
# # The check 'if not rl_agents:' below is now mostly redundant...
# for game_num...
# It *did not remove* `if not rl_agents:`. That's the oversight.
# The `NameError` is occurring on the `if not rl_agents:` line itself.

# Okay, the test script is created.
# Next step is to inform the user how to run it, and then use its (expected) failure
# to guide the next fix in `euchre.py`.
logging.info("`test_simulation_start.py` created.")
plan_step_complete("Created `test_simulation_start.py`.")
