# 3-Player Euchre with RL Agent

## Description

This project implements a web-based 3-player version of the classic card game Euchre. You, as the human player, team up with a "dummy" hand to compete against two AI opponents. The AI players leverage a Q-learning reinforcement learning agent to make strategic decisions throughout the game. The game is played interactively in a web browser and also features a simulation mode for training the AI agent.

The primary goal is to be the first team (Player-Dummy or AI-AI) to reach 10 points.

## Features

-   **3-Player Euchre Gameplay:** Adapts standard Euchre rules (bidding, trump selection, trick-taking) for a three-player scenario.
-   **Interactive Web Interface:** Play the game through a user-friendly interface built with Flask and vanilla JavaScript.
-   **Reinforcement Learning AI:** AI opponents are powered by a Q-learning agent that learns and improves its strategy over time.
-   **Dummy Hand for Human Player:** The human player is partnered with a face-up "dummy" hand, adding a unique strategic element.
-   **Training Mode:** Includes a simulation mode to train the RL agent by playing numerous games, enhancing its Q-table.
-   **Persistent Q-Table:** The AI's learned knowledge (Q-table) is stored persistently in an SQLite database.
-   **Game Statistics:** Tracks overall wins and losses for the human player in local browser storage.
-   **Responsive Design:** The web interface is designed to be usable on various screen sizes.

## Gameplay Details

### Objective
The objective is to be the first player (or their implicit partner, in the case of the AI) to score 10 points. Points are scored by winning tricks during a round.

### The Deck
Euchre is played with a 24-card deck consisting of the 9, 10, Jack, Queen, King, and Ace of all four suits (Hearts, Diamonds, Clubs, Spades).

### Dealing
-   Each of the three players is dealt 5 cards.
-   A "dummy hand" of 5 cards is dealt face-up, which will be used by the human player if they become the maker.
-   The remaining 4 cards form the "kitty". One card from the kitty is turned face up; this is the "up-card".

### Bidding Round 1 (Ordering Up)
1.  Starting with the player to the dealer's left, each player has the option to "order up" the dealer.
    *   If a player orders up, the suit of the up-card becomes trump for the round.
    *   The player who ordered up becomes the "maker".
    *   The dealer must pick up the up-card and discard one card from their hand face down.
2.  If a player does not want the up-card's suit to be trump, they "pass".
3.  If all three players pass, the up-card is turned face down.

### Bidding Round 2 (Calling Trump)
1.  If all players passed in Round 1, another round of bidding occurs, again starting with the player to the dealer's left.
2.  Each player can now "call" (name) any suit as trump, *except* for the suit of the card that was turned down in Round 1.
3.  The first player to call a suit becomes the "maker", and that suit becomes trump.
4.  If all three players pass again in this round, the hand is "thrown in," and the deal passes to the next player. (Note: In this implementation, the dealer is forced to call a suit if everyone else passes in the second round to prevent re-deals).

### Dealer's Options (If Stuck)
If bidding goes around twice and no one has made trump, and it comes back to the dealer, the dealer *must* choose a suit to be trump (they cannot pass again). The chosen suit cannot be the one that was initially turned up and then turned down.

### Going Alone (Maker's Option)
After trump is determined, the maker has the option to "go alone."
-   **Human Maker:** If the human player is the maker, they can choose to play with their own 5 cards plus the 5 cards from the dummy hand (making a 10-card hand, from which they must then discard 5), or they can choose to go alone.
    - If going alone, the dummy hand is not used.
-   **AI Maker:** The AI maker decides whether to go alone based on its Q-learning policy. If it goes alone, its implicit AI partner does not play. (In the 3-player version, "going alone" means the maker plays against the other two opponents without a partner's direct involvement in trick play, though scores are still team-based implicitly).

### Playing Tricks
1.  The player to the dealer's left leads the first trick (unless the maker went alone, in which case the player to the maker's left leads).
2.  Players must follow suit if possible.
3.  If a player cannot follow suit, they can play any card (either a trump card or an off-suit card).
4.  The highest card of the suit led wins the trick, unless a trump card is played, in which case the highest trump card wins.
    *   **Trump Suit Hierarchy:** The Jack of the trump suit (Right Bower) is the highest trump. The other Jack of the same color as the trump suit (Left Bower) is the second-highest trump and is considered part of the trump suit. For example, if Hearts are trump, the Jack of Hearts is Right Bower, and the Jack of Diamonds is Left Bower.
    *   Other trump cards rank A, K, Q, 10, 9.
    *   Non-trump (off-suit) cards rank A, K, Q, J, 10, 9.
5.  The winner of a trick leads the next trick. This continues until all 5 tricks have been played.

### Scoring
Points are awarded at the end of each round:
-   **Maker Makes Points:**
    -   If the maker (and their partner, if not alone) wins 3 or 4 tricks: 1 point.
    -   If the maker (and their partner, if not alone) wins all 5 tricks (a "march"): 2 points.
    -   If the maker goes alone and wins 3 or 4 tricks: 1 point.
    -   If the maker goes alone and wins all 5 tricks: 4 points.
-   **Maker is Euchred:**
    -   If the maker (and their partner, if not alone) fails to win at least 3 tricks, they are "euchred." The opponents score 2 points.

The game continues until one side reaches 10 points.

## Technologies Used

-   **Backend:** Python (v3.x), Flask (for web server and API)
-   **Frontend:** HTML5, CSS3, Vanilla JavaScript (for client-side logic and interactivity)
-   **AI:** Q-learning (Reinforcement Learning algorithm)
-   **Database:** SQLite (for persistent storage of the RL agent's Q-table)
-   **Standard Library Modules:** `random`, `logging`, `json`, `os`, `sqlite3` (Python)

## Setup and Running the Game

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```
    Replace `https://github.com/your-username/your-repository-name.git` with the actual URL of the repository.

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    # venv\\Scripts\\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    The primary external dependency is Flask.
    ```bash
    pip install Flask
    ```
    (A `requirements.txt` file would typically be used for projects with more dependencies: `pip install -r requirements.txt`)

4.  **Run the Flask application:**
    ```bash
    python euchre.py
    ```
    This will start the web server.

5.  **Open your web browser:**
    Navigate to `http://127.0.0.1:5000/`.

## Running the Training Simulation

The `euchre.py` script includes a function `run_training_simulation(num_games_to_simulate)` to train the RL agent by simulating games between AI players. This updates the `q_table.sqlite` file.

1.  **Modify `euchre.py`:**
    -   Open `euchre.py` in a text editor.
    -   Locate the `if __name__ == "__main__":` block towards the end of the file.
    -   Comment out the line `app.run(debug=True, host='0.0.0.0')` (or similar Flask app execution line).
    -   Uncomment or add the line `run_training_simulation(NUMBER_OF_GAMES)`, replacing `NUMBER_OF_GAMES` with the desired number of games for the simulation (e.g., `run_training_simulation(10000)`).

2.  **Run the script from your terminal:**
    ```bash
    python euchre.py
    ```
    The simulation will run in your console, and you'll see log messages indicating its progress. The Q-table in `q_table.sqlite` will be updated during this process.

3.  **Revert changes to run the web app:**
    After training, remember to comment out the `run_training_simulation(...)` line and uncomment the `app.run(...)` line in `euchre.py` to run the interactive web application again.

## Reinforcement Learning Agent

The AI opponents in this game use a Q-learning algorithm, a model-free reinforcement learning technique.

### Q-Learning Overview
Q-learning aims to learn a policy, telling an agent what action to take under what circumstances. It does this by learning a Q-function, Q(s, a), which estimates the expected future reward of taking action 'a' in state 's'. Through exploration (trying random actions) and exploitation (choosing actions with the highest known Q-value), the agent iteratively updates its Q-values based on the rewards it receives for its actions. The core update rule is:

`Q(s, a) = Q(s, a) + α * [r + γ * max_a'(Q(s', a')) - Q(s, a)]`

Where:
-   `α` (alpha) is the learning rate (how much new information overrides old information).
-   `γ` (gamma) is the discount factor (importance of future rewards).
-   `r` is the immediate reward received after taking action 'a' in state 's'.
-   `s'` is the new state.
-   `max_a'(Q(s', a'))` is the maximum Q-value for any action in the new state `s'`.

### State Representation
The "state" is a snapshot of the game from the AI agent's perspective. It's converted into a serialized JSON string to be used as a key in the Q-table. Key pieces of information included in the state are:
-   The agent's current hand (cards).
-   Current game phase (e.g., `bidding_round_1`, `playing_tricks`).
-   Trump suit (if determined).
-   Dealer, maker, and current player turn.
-   Up-card details (suit, rank, visibility).
-   Information about passes made during bidding.
-   Details of the current trick (lead suit, cards played).
-   Scores (self, opponents) and tricks won in the current round.
-   Player's role (dealer, maker, opponent).
-   Whether the maker is going alone.
-   Hand features: number of trump cards, presence of bowers/aces, suit voids, etc. These are calculated for the current hand and also for potential trump suits during bidding.
-   Played cards in the current round to provide some memory of what's out.

### Action Space
The actions an AI agent can take depend on the game phase:
-   **Bidding Round 1:** `order_up`, `pass_bid`.
-   **Bidding Round 2 / Dealer Must Call:** `call_trump:<suit>`, `pass_call` (if not dealer stuck).
-   **Prompt Go Alone:** `choose_go_alone`, `choose_not_go_alone`.
-   **Playing Tricks:** Play any valid card from its hand (represented as a card dictionary like `{'suit': 'H', 'rank': 'A'}`).
-   Discarding actions are currently heuristic for AI but could be integrated into RL.

### Reward Structure
The agent receives numerical rewards (or penalties) for its actions and game outcomes:
-   **Positive Rewards:**
    -   Winning a trick (`REWARD_WIN_TRICK`).
    -   Maker winning the round (normal, march, alone march - varying values).
    -   Defeating the maker (euchring them) (`REWARD_WIN_ROUND_DEFENSE_EUCHRE`).
    -   Winning the game (`REWARD_WIN_GAME`).
-   **Negative Rewards (Penalties):**
    -   Losing a trick (`REWARD_LOSE_TRICK`).
    -   Maker being euchred (`REWARD_EUCHRED_MAKER`).
    -   Defense losing the round (maker makes points) (`REWARD_LOSE_ROUND_DEFENSE`).
    -   Losing the game (`REWARD_LOSE_GAME`).

The specific values for these rewards are defined as constants (e.g., `REWARD_WIN_TRICK`) in `euchre.py`. The agent learns to choose actions that maximize its cumulative reward over time.

## Code Structure Overview

The project is organized into several key files:

-   **`euchre.py`**: This is the core Python file containing:
    -   **Flask Application:** Sets up the web server and defines API routes for frontend-backend communication.
    -   **Game Logic:** Manages the overall game state, rules of Euchre (dealing, bidding, trick play, scoring), card representations, and deck operations.
    -   **`RLAgent` Class:** Implements the Q-learning agent, including Q-table management (loading/saving to SQLite), state serialization, action selection (epsilon-greedy strategy), and Q-value updates.
    -   **Game State Management:** Global `game_data` dictionary holds all current game information. Functions like `initialize_game_data()`, `initialize_new_round()` manage its lifecycle.
    -   **AI Player Logic:** Functions like `process_ai_play_card()`, `process_ai_bid_action()`, `ai_decide_go_alone_and_proceed()` integrate the `RLAgent` with the game flow.
    -   **Training Simulation:** The `run_training_simulation()` function enables AI training by simulating games without the web interface.

-   **`test_euchre.py`**: Contains unit tests for various components of the game logic and AI, such as hand feature extraction and AI sloughing logic. Uses Python's `unittest` module.

-   **`templates/index.html`**: The main HTML file that structures the web page displayed to the user. It defines areas for player hands, scores, game messages, action buttons, etc.

-   **`script.js`**: Handles all client-side interactivity and dynamic content updates for the web interface. Its responsibilities include:
    -   Rendering the game board, player hands, scores, and other game information received from the backend.
    -   Managing user input (clicking on cards, action buttons).
    -   Communicating with the backend API (`/api/...` endpoints) to send player actions and fetch game state updates.
    -   Implementing visual aspects like card selection.
    -   Storing and retrieving game statistics (wins/losses) in the browser's local storage.
    -   Polling the backend for AI moves when it's an AI's turn to play a card.

-   **`style.css`**: Provides the visual styling for the web interface, including layout, colors, fonts, and responsiveness for different screen sizes. It implements a dark theme.

-   **`q_table.sqlite`**: An SQLite database file that stores the Q-table for the reinforcement learning agent. This allows the agent's learned knowledge to persist between game sessions and training runs. (Note: This file is included in the repository for distribution with pre-trained data but might typically be in `.gitignore` during active development if starting fresh.)

-   **`.gitignore`**: A standard Git file that specifies intentionally untracked files that Git should ignore (e.g., Python bytecode cache `__pycache__`).

-   **`README.md`**: This file, providing documentation for the project.

## Potential Future Enhancements

-   **Comprehensive Test Suite:** Expand unit and integration tests to cover more game scenarios, API endpoints, and RL agent behavior.
-   **Improved UI/UX:**
    -   Enhance the visual design and user experience of the web interface (e.g., better card animations, clearer turn indicators).
    -   Add visual cues for why an AI made a certain decision (if feasible without overcomplicating).
-   **Advanced AI Features:**
    -   Implement more sophisticated state representation for the RL agent (e.g., considering card distributions more deeply).
    -   Explore different RL algorithms (e.g., SARSA, Deep Q-Networks if complexity warrants) or fine-tune hyperparameters for Q-learning.
    -   Add opponent modeling capabilities to the AI.
    -   Allow AI to learn discarding strategies when picking up the dummy hand.
-   **4-Player Euchre:** Adapt the game logic to support the traditional 4-player (partners) version of Euchre.
-   **Configuration File:** Manage settings like learning rates, discount factors, reward values, and simulation parameters through an external configuration file (e.g., `config.ini` or `config.json`).
-   **Requirements File:** Add a `requirements.txt` file for easier dependency management (e.g., by running `pip freeze > requirements.txt`).
-   **Improved Game Flow:**
    -   Reduce reliance on client-side polling for AI moves by implementing WebSockets or Server-Sent Events for real-time updates.
-   **Documentation:**
    -   Add more inline code comments for complex sections in `euchre.py` and `script.js`.

## API Endpoints

The backend Flask application exposes several API endpoints that the frontend JavaScript interacts with:

-   **`GET /`**
    -   Serves the main `index.html` page.
-   **`GET /style.css`**
    -   Serves the CSS stylesheet.
-   **`GET /script.js`**
    -   Serves the client-side JavaScript file.
-   **`GET /api/start_game`**
    -   Initializes or resets the game state for a new game/round.
    -   If it's an AI's turn to start bidding, this might also trigger the AI's first action.
    -   Returns the full game state as JSON.
-   **`POST /api/submit_action`**
    -   Allows the human player to submit an action (e.g., bid, call trump, play card, discard, go alone).
    -   Request body: JSON object containing `player_index` (always 0 for human), `action` (string identifier), and any action-specific details (e.g., `card` object for playing, `suit` for calling trump).
    -   Processes the action, updates the game state, and potentially triggers AI actions if it becomes their turn.
    -   Returns the updated full game state as JSON.
-   **`GET /api/get_current_state`**
    -   Retrieves the current complete game state.
    -   Used by the client to refresh or re-sync if needed.
    -   Returns the full game state as JSON.
-   **`POST /api/ai_play_turn`**
    -   Endpoint specifically for the client to poll when it detects it's an AI's turn to play a card during the "playing_tricks" phase.
    -   The server processes the AI's card play.
    -   Returns the updated full game state as JSON.

All game state responses are JSON objects that include details like player hands, scores, current player, game phase, trump suit, etc., which `script.js` uses to render the UI.

## Running Tests

The project includes unit tests in `test_euchre.py`. To run these tests:

1.  Ensure you are in the root directory of the project.
2.  Make sure your virtual environment is activated (if you are using one).
3.  Run the following command in your terminal:
    ```bash
    python -m unittest test_euchre.py
    ```
    Or, for more verbose output:
    ```bash
    python -m unittest -v test_euchre.py
    ```
    The tests will execute, and you'll see output indicating whether they passed or failed.

## Dependencies

-   **Python 3.x** (Developed and tested with Python 3.10+)
-   **Flask:** A micro web framework for Python, used for the backend server and API.
    -   Install via `pip install Flask`.
-   **Frontend:**
    -   HTML5
    -   CSS3
    -   Vanilla JavaScript (ES6+)
-   **Standard Python Libraries:** `random`, `logging`, `json`, `os`, `sqlite3` (used extensively in `euchre.py`).

## How to Contribute

Contributions are welcome! If you'd like to contribute to this project, please follow these general steps:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix:
    ```bash
    git checkout -b feature/your-feature-name
    ```
    or
    ```bash
    git checkout -b fix/issue-description
    ```
3.  **Make your changes.**
4.  **Ensure your changes do not break existing functionality.**
5.  **Run the tests** (`python -m unittest test_euchre.py`) to make sure they all pass.
6.  **Commit your changes** with a clear and descriptive commit message.
7.  **Push your branch** to your forked repository.
8.  **Open a Pull Request** to the main repository, detailing the changes you've made.

Please try to adhere to the existing coding style and provide clear documentation for new features if applicable.

## License

This project is currently unlicensed. All rights are reserved by the original author(s).
You may use this code for personal or educational purposes. For other uses, please contact the project maintainers.
```
