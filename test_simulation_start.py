import logging

# Configure logging to see output from the simulation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from euchre import run_training_simulation, initialize_game_data, RLAgent, Game, get_game_instance

    logging.info("Successfully imported components from euchre.py")
    logging.info("Attempting to run the training simulation for 1 game...")

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

# The critical part is that `initialize_game_data()` (called by `run_training_simulation`)
# now creates the `Game` instance, which in turn creates the `RLAgent` instances
# and stores them in `_current_game_instance.rl_agents`.
# The direct global `rl_agents` check in `run_training_simulation` that was causing the NameError
# should have been removed in the latest fix to `euchre.py`.
# This test will verify that removal and the subsequent ability to start the simulation.
# If `initialize_new_round` or other functions still misuse globals, new errors will appear here.
