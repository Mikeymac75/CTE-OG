import logging

# Configure logging to see output from the simulation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from euchre import run_training_simulation, initialize_game_data, RLAgent, Game, get_game_instance

    logging.info("Successfully imported components from euchre.py")
    logging.info("Attempting to run the training simulation for 1 game...")

    run_training_simulation(num_games_to_simulate=1, save_interval=1)
    logging.info("Training simulation for 1 game completed successfully.")
    print("\nSUCCESS: test_simulation_start.py ran and completed a short simulation.")

except NameError as ne:
    logging.error(f"NameError encountered: {ne}")
    print(f"\nFAILURE: Encountered a NameError: {ne}")
except ImportError as ie:
    logging.error(f"ImportError encountered: {ie}")
    print(f"\nFAILURE: Could not import necessary components from euchre.py: {ie}")
    print("Please ensure euchre.py is in the same directory or accessible via PYTHONPATH.")
except Exception as e:
    logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    print(f"\nFAILURE: An unexpected error occurred during the test: {e}")

# Note: This test primarily checks if the NameError for 'rl_agents' is gone
# when starting the simulation and if imports work. It doesn't validate the full
# correctness of the simulation or the Flask app.
# Further errors within the simulation might indicate ongoing issues with global
# variable refactoring in euchre.py.
