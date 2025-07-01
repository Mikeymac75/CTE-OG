document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const player0YouHandDiv = document.getElementById('player-0-you-hand');
    const player1AiHandDiv = document.getElementById('player-1-ai-hand');
    const player2AiHandDiv = document.getElementById('player-2-ai-hand');
    const dummyHandDisplaySpan = document.getElementById('dummy-hand-display'); // Changed from kittyDisplayDiv
    const upCardDisplaySpan = document.getElementById('up-card-display');
    const trumpSuitSpan = document.getElementById('trump-suit');
    const gameMessageP = document.getElementById('game-message');
    const scorePlayer0Span = document.getElementById('score-player-0');
    const scorePlayer1Span = document.getElementById('score-player-1');
    const scorePlayer2Span = document.getElementById('score-player-2');
    const statsWinsSpan = document.getElementById('stats-wins');
    const statsLossesSpan = document.getElementById('stats-losses');
    const player0YouHandTricksSpan = document.getElementById('player-0-you-hand-tricks');
    const player1AiHandTricksSpan = document.getElementById('player-1-ai-hand-tricks');
    const player2AiHandTricksSpan = document.getElementById('player-2-ai-hand-tricks');
    // const dummyHandDisplaySpan = document.getElementById('dummy-hand-display'); // Changed from kitty - REMOVED DUPLICATE
    const trickAreaDiv = document.getElementById('trick-area'); // Added for current trick

    const orderUpButton = document.getElementById('order-up-button');
    const passBidButton = document.getElementById('pass-bid-button');
    const biddingActionsDiv = document.getElementById('bidding-actions');
    const callTrumpActionsDiv = document.getElementById('call-trump-actions');
    const callSuitButtons = document.querySelectorAll('.call-suit-button');
    const passCallButton = document.getElementById('pass-call-button');

    const playActionsDiv = document.getElementById('play-actions');
    const playCardButton = document.getElementById('play-card-button');
    const goAloneActionsDiv = document.getElementById('go-alone-actions'); // This is the div containing the buttons
    const chooseGoAloneButton = document.getElementById('choose-go-alone-button');
    const previousTrickLogDiv = document.getElementById('previous-trick-log');
    const chooseNotGoAloneButton = document.getElementById('choose-not-go-alone-button');

    const startGameButton = document.createElement('button');
    startGameButton.id = 'start-game-button';
    startGameButton.textContent = 'Start New Game / Round';
    const actionsArea = document.querySelector('.actions-area');
    if (actionsArea) actionsArea.parentNode.insertBefore(startGameButton, actionsArea);
    else document.body.appendChild(startGameButton); // Fallback

    // --- Game State Variables (Client-Side Copy) ---
    let clientGameState = {};
    let gameStatistics = { wins: 0, losses: 0 }; // Overall wins and losses
    let selectedCardsForDiscard = []; // Array to hold card data for multi-discard

    // --- Local Storage Keys ---
    const EUCHRE_GAME_STATE_KEY = 'euchreGameState';
    const EUCHRE_GAME_STATS_KEY = 'euchreGameStatistics';

    // --- Card Definitions & Helpers ---
    const SUIT_NAMES = {'H': 'Hearts', 'D': 'Diamonds', 'C': 'Clubs', 'S': 'Spades'};
    const CARD_IMAGE_BASE_URL = 'https://deckofcardsapi.com/static/img/'; // Base URL for card images from Deck of Cards API

    /**
     * Constructs the filename for a card image based on its suit and rank.
     * Adjusts '10' rank to '0' for compatibility with image naming conventions.
     * @param {object} card - Card object with 'suit' and 'rank' properties.
     * @returns {string} Filename for the card image (e.g., "0H.png" for 10 of Hearts).
     */
    function getCardImageFilename(card) {
        let rankForImage = card.rank;
        if (card.rank === '10') rankForImage = '0'; // API uses '0' for 10
        return `${rankForImage}${card.suit}.png`;
    }

    // --- UI Update Functions ---
    /**
     * Checks if the game is in a terminal state (round over or game over).
     * @param {object} gameState - The current game state object.
     * @returns {boolean} True if the round or game is over, false otherwise.
     */
    function isGameOverOrRoundOver(gameState) {
        return gameState.game_phase === "round_over" || gameState.game_phase === "game_over";
    }

    /**
     * Creates a DOM element for a card.
     * @param {object} cardData - Object containing card details (suit, rank, name).
     * @param {boolean} [isOpponentHandCard=false] - If true, displays the card back.
     * @returns {HTMLElement} The card div element.
     */
    function displayCard(cardData, isOpponentHandCard = false) {
        const cardDiv = document.createElement('div');
        cardDiv.classList.add('card');
        cardDiv.dataset.suit = cardData.suit; // Store suit for potential future use
        cardDiv.dataset.rank = cardData.rank;
        cardDiv.dataset.name = cardData.name; // Store for easy access
        cardDiv.setAttribute('aria-label', cardData.name);
        const imageUrl = CARD_IMAGE_BASE_URL + (isOpponentHandCard ? 'back.png' : getCardImageFilename(cardData));
        cardDiv.style.backgroundImage = `url(${imageUrl})`;
        return cardDiv;
    }

    /**
     * Animates a card moving from its position in the hand to a target container (e.g., trick area).
     * Creates a clone of the card for animation, moves it, then removes the clone.
     * @param {HTMLElement} cardElement - The original card element in the player's hand.
     * @param {HTMLElement} targetContainer - The DOM element where the card should appear to land.
     * @returns {Promise<void>} A promise that resolves when the animation is complete.
     */
    function animateCardPlay(cardElement, targetContainer) {
        const cardRect = cardElement.getBoundingClientRect(); // Get position of card in hand
        const targetRect = targetContainer.getBoundingClientRect(); // Get position of trick area

        const animatedCard = cardElement.cloneNode(true); // Clone card for animation
        animatedCard.classList.add('animated-card');
        document.body.appendChild(animatedCard); // Add to body for unrestricted positioning

        // Set initial position of clone to match original card
        animatedCard.style.position = 'fixed';
        animatedCard.style.left = `${cardRect.left}px`;
        animatedCard.style.top = `${cardRect.top}px`;
        animatedCard.style.width = `${cardRect.width}px`;
        animatedCard.style.height = `${cardRect.height}px`;
        animatedCard.style.zIndex = '1000'; // Ensure it's on top

        cardElement.style.opacity = '0.3'; // Fade out original card slightly

        // Calculate target position (center of the target container)
        const targetX = targetRect.left + window.scrollX + (targetRect.width / 2) - (cardRect.width / 2);
        const targetY = targetRect.top + window.scrollY + (targetRect.height / 2) - (cardRect.height / 2);


        // Perform animation
        requestAnimationFrame(() => {
            animatedCard.style.transition = 'left 0.5s ease-out, top 0.5s ease-in, width 0.5s ease-in-out, height 0.5s ease-in-out';
            animatedCard.style.left = `${targetX}px`;
            animatedCard.style.top = `${targetY}px`;
        });

        // Clean up after animation
        return new Promise(resolve => {
            animatedCard.addEventListener('transitionend', () => {
                animatedCard.remove(); // Remove the animated clone
                cardElement.style.opacity = '1'; // Restore original card opacity (though it will be removed by renderBoard)
                resolve();
            }, { once: true });
        });
    }

    /**
     * Updates the text and disabled state of the main action button (playCardButton)
     * based on the current game phase and number of selected cards for discard.
     * This function is primarily for discard phases.
     */
    function updateSelectedCardsDisplay() {
        const phase = clientGameState.game_phase;
        const isMyTurn = clientGameState.current_player_turn === 0;

        const isHumanDealerMustDiscard = phase === 'dealer_must_discard_after_order_up' && isMyTurn && clientGameState.dealer === 0;
        const isHumanMakerInvolvedDiscard = (phase === 'maker_discard' || phase === 'dealer_discard_one') && isMyTurn && clientGameState.maker === 0;

        if (isHumanDealerMustDiscard || isHumanMakerInvolvedDiscard) {
            const needed = clientGameState.cards_to_discard_count; // Server sets this to 1 or 5
            const selectedCount = selectedCardsForDiscard.length;
            playCardButton.textContent = `Discard ${selectedCount}/${needed} Selected`;
            playCardButton.disabled = selectedCount !== needed;
        } else {
            // If not in a specific discard phase that uses this button, ensure it's in a default state or handled by playing_tricks.
            // This avoids carrying over "Discard X/Y" text to other phases like "playing_tricks" before a card is selected.
            if (phase !== 'playing_tricks') { // playing_tricks has its own text/disable logic in renderBoard
                 playCardButton.textContent = 'Play Selected Card'; // A generic default
                 playCardButton.disabled = true;
            }
        }
    }

    /**
     * Handles click events on cards in the human player's hand.
     * Manages card selection logic based on the current game phase:
     * - Discard phases: Allows selecting/deselecting multiple cards up to the required discard count.
     * - Playing tricks phase: Allows selecting a single card to play.
     * Updates the UI to reflect selected cards and enables/disables action buttons accordingly.
     * @param {HTMLElement} cardElement - The DOM element of the clicked card.
     * @param {object} cardData - Data object for the clicked card (suit, rank, name).
     */
    function handleCardClick(cardElement, cardData) {
        const phase = clientGameState.game_phase;
        const isMyTurn = clientGameState.current_player_turn === 0;

        // Determine if the current phase is one where the human player needs to discard.
        const isHumanDealerMustDiscard = phase === 'dealer_must_discard_after_order_up' && isMyTurn && clientGameState.dealer === 0;
        const isHumanMakerInvolvedDiscard = (phase === 'dealer_discard_one' || phase === 'maker_discard') && isMyTurn && clientGameState.maker === 0;

        if (isHumanDealerMustDiscard || isHumanMakerInvolvedDiscard) {
            // Logic for selecting cards to discard
            const neededToDiscard = clientGameState.cards_to_discard_count; // Server indicates 1 or 5
            const indexInSelection = selectedCardsForDiscard.findIndex(c => c.rank === cardData.rank && c.suit === cardData.suit);

            if (indexInSelection > -1) { // Card is already selected, so de-select it
                selectedCardsForDiscard.splice(indexInSelection, 1);
                cardElement.classList.remove('selected');
            } else { // Card is not currently selected
                if (neededToDiscard === 1) {
                    // If only one card needs to be discarded, clicking a new card deselects the old one.
                    if (selectedCardsForDiscard.length === 1) {
                        const currentlySelectedCardElement = document.querySelector('#player-0-you-hand .card.selected');
                        if (currentlySelectedCardElement) {
                            currentlySelectedCardElement.classList.remove('selected');
                        }
                        selectedCardsForDiscard = []; // Clear current selection
                    }
                    selectedCardsForDiscard.push({ rank: cardData.rank, suit: cardData.suit, name: cardData.name });
                    cardElement.classList.add('selected');
                } else { // For multi-card discard (e.g., maker discards 5)
                    if (selectedCardsForDiscard.length < neededToDiscard) {
                        selectedCardsForDiscard.push({ rank: cardData.rank, suit: cardData.suit, name: cardData.name });
                        cardElement.classList.add('selected');
                    } else {
                        // User tried to select more cards than allowed
                        gameMessageP.textContent = `You can only select ${neededToDiscard} card(s) to discard.`;
                        setTimeout(() => { if(clientGameState && clientGameState.message) gameMessageP.textContent = clientGameState.message; }, 2000); // Revert message after 2s
                    }
                }
            }
            updateSelectedCardsDisplay(); // Update button text/state
        } else if (phase === 'playing_tricks' && isMyTurn) {
            // Logic for selecting a single card to play during a trick
            document.querySelectorAll('#player-0-you-hand .card.selected').forEach(el => el.classList.remove('selected')); // Deselect any other card
            cardElement.classList.add('selected');
            selectedCardsForDiscard = [{ rank: cardData.rank, suit: cardData.suit, name: cardData.name }]; // Store the single selected card
            playCardButton.textContent = 'Play Selected Card';
            playCardButton.disabled = false; // Enable play button
        }
    }

    /**
     * Main function to update the entire game UI based on the provided game state.
     * Clears and redraws player hands, up-card, trick area, scores, messages,
     * and visibility/state of action buttons.
     * Also handles triggering AI turns if applicable.
     * @param {object} gameState - The comprehensive game state object from the server.
     */
    function renderBoard(gameState) {
        if (!gameState || Object.keys(gameState).length === 0) {
            // Initial state or error state, show welcome message and hide game elements.
            gameMessageP.textContent = "Welcome! Click 'Start New Game / Round'.";
            biddingActionsDiv.style.display = 'none';
            callTrumpActionsDiv.style.display = 'none';
            playActionsDiv.style.display = 'none';
            goAloneActionsDiv.style.display = 'none';
            startGameButton.style.display = 'inline-block';
            if (player0YouHandDiv) player0YouHandDiv.innerHTML = '';
            if (player1AiHandDiv) player1AiHandDiv.innerHTML = '';
            if (player2AiHandDiv) player2AiHandDiv.innerHTML = '';
            upCardDisplaySpan.innerHTML = 'None';
            dummyHandDisplaySpan.textContent = '0 cards';
            trickAreaDiv.innerHTML = '<h3>Current Trick</h3>';
            trumpSuitSpan.textContent = 'None';
            if (scorePlayer0Span) scorePlayer0Span.textContent = '0';
            if (scorePlayer1Span) scorePlayer1Span.textContent = '0';
            if (scorePlayer2Span) scorePlayer2Span.textContent = '0';
            if (player0YouHandTricksSpan) player0YouHandTricksSpan.textContent = '0';
            if (player1AiHandTricksSpan) player1AiHandTricksSpan.textContent = '0';
            if (player2AiHandTricksSpan) player2AiHandTricksSpan.textContent = '0';
            return;
        }
        clientGameState = gameState;
        selectedCardsForDiscard = [];

        trickAreaDiv.innerHTML = '<h3>Current Trick</h3>';

        if (gameState.game_phase === 'playing_tricks' &&
            gameState.current_player_turn !== 0 &&
            !isGameOverOrRoundOver(gameState)) {
            setTimeout(() => {
                fetch('/api/ai_play_turn', { method: 'POST' })
                    .then(response => {
                        if (!response.ok) {
                            // Try to get error message from server response body
                            return response.json().then(errData => {
                                throw new Error(errData.error || `Server error: ${response.status}`);
                            }).catch(() => {
                                // Fallback if parsing error response fails
                                throw new Error(`Server error: ${response.status} - Could not parse error details.`);
                            });
                        }
                        return response.json();
                    })
                    .then(data => {
                        // data.error is checked by renderBoard if it receives an error structure from submitPlayerAction
                        // For AI play, a successful response means the AI turn was processed (or determined not to be AI's turn by server)
                        renderBoard(data);
                    })
                    .catch(error => {
                        console.error("Error during AI turn fetch:", error);
                        gameMessageP.textContent = `Error during AI turn: ${error.message}. Consider refreshing if the game is stuck.`;
                        // Optionally, try to re-render with existing clientGameState or fetch fresh state
                        // For now, just display error. A refresh might be the best user action.
                    });
            }, 750); // Delay before AI turn processing
        }

        // Render player hands
        const playerHandDivs = {
            '0': player0YouHandDiv,
            '1': player1AiHandDiv,
            '2': player2AiHandDiv
        };

        Object.keys(playerHandDivs).forEach(playerKey => {
            const handDiv = playerHandDivs[playerKey];
            if (!handDiv) return;
            handDiv.innerHTML = '';
            if (gameState.hands && gameState.hands[playerKey]) {
                gameState.hands[playerKey].forEach(card => {
                    const isOpponent = playerKey !== '0'; // Player '0' is 'You'
                    const cardElement = displayCard(card, isOpponent);
                    if (!isOpponent) { // Only add click listeners for 'You'
                         cardElement.addEventListener('click', () => handleCardClick(cardElement, card));
                         cardElement.addEventListener('dblclick', () => {
                            if (clientGameState.game_phase === 'playing_tricks' && clientGameState.current_player_turn === 0) {
                                const cardDataForAction = { rank: card.rank, suit: card.suit, name: card.name };
                                submitPlayerAction('play_card', { card: cardDataForAction });
                            }
                        });
                    }
                    handDiv.appendChild(cardElement);
                });
            }
        });

        upCardDisplaySpan.innerHTML = '';
        if (gameState.up_card && gameState.up_card_visible) {
            upCardDisplaySpan.appendChild(displayCard(gameState.up_card));
        } else if (gameState.original_up_card_for_round && !gameState.up_card_visible && gameState.game_phase !== 'playing_tricks' && gameState.game_phase !== 'maker_discard' && gameState.game_phase !== 'prompt_go_alone') {
             const turnedDownCard = displayCard(gameState.original_up_card_for_round, true);
             upCardDisplaySpan.appendChild(turnedDownCard);
             upCardDisplaySpan.appendChild(document.createTextNode(" (Turned Down)"));
        } else {
            upCardDisplaySpan.textContent = 'None';
        }

        if (gameState.trick_cards && gameState.trick_cards.length > 0) {
            gameState.trick_cards.forEach(playedCardInfo => {
                if (playedCardInfo && playedCardInfo.card && typeof playedCardInfo.card.suit !== 'undefined' && typeof playedCardInfo.card.rank !== 'undefined') {
                    const cardElement = displayCard(playedCardInfo.card);
                    const playerIdentifier = document.createElement('span');
                    playerIdentifier.classList.add('trick-player-identifier');
                    let playerName = `P${parseInt(playedCardInfo.player) + 1}`;
                    if (gameState.player_identities && gameState.player_identities[playedCardInfo.player]) {
                        playerName = gameState.player_identities[playedCardInfo.player].split(' ')[0] +
                                     ` ${gameState.player_identities[playedCardInfo.player].split(' ')[1]}`;
                    }
                    playerIdentifier.textContent = `${playerName}: `;
                    const trickCardContainer = document.createElement('div');
                    trickCardContainer.classList.add('trick-card-container');
                    trickCardContainer.appendChild(playerIdentifier);
                    trickCardContainer.appendChild(cardElement);
                    trickAreaDiv.appendChild(trickCardContainer);
                }
            });
        }

        if (gameState.dummy_hand && gameState.dummy_hand.length > 0) {
            dummyHandDisplaySpan.textContent = `${gameState.dummy_hand.length} cards in dummy hand`;
        } else if (gameState.game_phase !== 'setup' && gameState.game_phase !== 'round_over' && gameState.game_phase !== 'game_over') {
            dummyHandDisplaySpan.textContent = 'Dummy hand taken by maker';
        } else {
            dummyHandDisplaySpan.textContent = '0 cards';
        }

        trumpSuitSpan.textContent = gameState.trump_suit ? SUIT_NAMES[gameState.trump_suit] : 'None';
        if (scorePlayer0Span) scorePlayer0Span.textContent = gameState.scores['0'];
        if (scorePlayer1Span) scorePlayer1Span.textContent = gameState.scores['1'];
        if (scorePlayer2Span) scorePlayer2Span.textContent = gameState.scores['2'];

        if (gameState.round_tricks_won) {
            if (player0YouHandTricksSpan) player0YouHandTricksSpan.textContent = gameState.round_tricks_won['0'] || '0';
            if (player1AiHandTricksSpan) player1AiHandTricksSpan.textContent = gameState.round_tricks_won['1'] || '0';
            if (player2AiHandTricksSpan) player2AiHandTricksSpan.textContent = gameState.round_tricks_won['2'] || '0';
        } else {
            if (player0YouHandTricksSpan) player0YouHandTricksSpan.textContent = '0';
            if (player1AiHandTricksSpan) player1AiHandTricksSpan.textContent = '0';
            if (player2AiHandTricksSpan) player2AiHandTricksSpan.textContent = '0';
        }

        gameMessageP.textContent = gameState.message || "";

        document.querySelectorAll('.player-area').forEach(area => {
            area.classList.remove('player-area-maker');
            const makerIndicator = area.querySelector('.maker-text-indicator');
            if (makerIndicator) makerIndicator.remove();
        });

        if (gameState.maker !== null && gameState.maker !== undefined) {
            // Adjust ID based on gameState.maker (0, 1, or 2)
            let makerPlayerId;
            if (gameState.maker === 0) makerPlayerId = 'player-0-you';
            else if (gameState.maker === 1) makerPlayerId = 'player-1-ai';
            else if (gameState.maker === 2) makerPlayerId = 'player-2-ai';

            const makerPlayerDiv = document.getElementById(makerPlayerId);
            if (makerPlayerDiv) {
                makerPlayerDiv.classList.add('player-area-maker');
                const makerTitle = makerPlayerDiv.querySelector('h2');
                if (makerTitle) {
                    let indicator = makerTitle.querySelector('.maker-text-indicator');
                    if (!indicator) {
                        indicator = document.createElement('span');
                        indicator.classList.add('maker-text-indicator');
                        indicator.textContent = ' (Maker)';
                        makerTitle.appendChild(indicator);
                    }
                }
            }
        }

        biddingActionsDiv.style.display = 'none';
        callTrumpActionsDiv.style.display = 'none';
        playActionsDiv.style.display = 'none';
        goAloneActionsDiv.style.display = 'none';
        passCallButton.style.display = 'inline-block';

        const isMyTurn = gameState.current_player_turn === 0;
        const amIMaker = gameState.maker === 0;

        if (isMyTurn) {
            switch (gameState.game_phase) {
                case 'bidding_round_1':
                    biddingActionsDiv.style.display = 'block';
                    orderUpButton.style.display = 'inline-block';
                    passBidButton.style.display = 'inline-block';
                    break;
                case 'dealer_must_discard_after_order_up':
                    if (gameState.dealer === 0) {
                        playActionsDiv.style.display = 'block';
                    }
                    break;
                case 'bidding_round_2':
                case 'dealer_must_call':
                    biddingActionsDiv.style.display = 'block';
                    callTrumpActionsDiv.style.display = 'block';
                    const turnedDownSuit = gameState.original_up_card_for_round ? gameState.original_up_card_for_round.suit : null;
                    callSuitButtons.forEach(btn => {
                        const suitKey = Object.keys(SUIT_NAMES).find(k => SUIT_NAMES[k] === btn.dataset.suit);
                        btn.disabled = (turnedDownSuit && suitKey === turnedDownSuit);
                    });
                    if (gameState.game_phase === 'dealer_must_call' && gameState.dealer === 0) {
                        passCallButton.style.display = 'none';
                    }
                    break;
                case 'dealer_discard_one':
                    if (amIMaker && gameState.dealer === 0) {
                        playActionsDiv.style.display = 'block';
                    }
                    break;
                case 'maker_discard':
                     if (amIMaker) {
                        playActionsDiv.style.display = 'block';
                    }
                    break;
                case 'prompt_go_alone':
                    if (amIMaker) {
                        goAloneActionsDiv.style.display = 'block';
                    }
                    break;
                case 'playing_tricks':
                    playActionsDiv.style.display = 'block';
                    playCardButton.textContent = 'Play Selected Card';
                    playCardButton.disabled = true;
                    break;
            }
        }
        startGameButton.style.display = (["setup", "round_over", "game_over"].includes(gameState.game_phase)) ? 'inline-block' : 'none';
        updateSelectedCardsDisplay();
        renderPreviousTrickLog(clientGameState.last_completed_trick);

        if (statsWinsSpan) statsWinsSpan.textContent = gameStatistics.wins;
        if (statsLossesSpan) statsLossesSpan.textContent = gameStatistics.losses;

        if (gameState.game_phase === "game_over" && gameState.scores) {
            const player1Score = gameState.scores['0'];
            let gameJustEnded = !clientGameState.hasOwnProperty('_gameEndedFlag') || !clientGameState._gameEndedFlag;
            if (gameJustEnded) {
                let playerWon = false;
                if (player1Score >= 10) {
                     playerWon = true;
                     for (const player in gameState.scores) {
                        if (player !== '0' && gameState.scores[player] >= 10) {
                            if (gameState.scores[player] > player1Score) playerWon = false;
                        }
                     }
                }
                let otherPlayerWon = false;
                if (!playerWon) {
                    for (const player in gameState.scores) {
                        if (player !== '0' && gameState.scores[player] >= 10) {
                            otherPlayerWon = true;
                            break;
                        }
                    }
                }
                if (playerWon) gameStatistics.wins++;
                else if (otherPlayerWon) gameStatistics.losses++;
                clientGameState._gameEndedFlag = true;
                if (statsWinsSpan) statsWinsSpan.textContent = gameStatistics.wins;
                if (statsLossesSpan) statsLossesSpan.textContent = gameStatistics.losses;
            }
        } else if (gameState.game_phase !== "game_over" && clientGameState.hasOwnProperty('_gameEndedFlag')) {
            delete clientGameState._gameEndedFlag;
        }
        saveStateToLocalStorage();
    }

    /**
     * Renders the log of the previously completed trick.
     * Displays the cards played by each player and the winner of the trick.
     * @param {object|null} lastTrickData - Data for the last completed trick, or null if none.
     */
    function renderPreviousTrickLog(lastTrickData) {
        if (!previousTrickLogDiv) return; // Ensure the log container exists
        previousTrickLogDiv.innerHTML = ''; // Clear previous log content

        if (!lastTrickData || !lastTrickData.played_cards || lastTrickData.played_cards.length === 0) {
            previousTrickLogDiv.innerHTML = '<p>No tricks completed yet in this round.</p>';
            return;
        }

        const entryDiv = document.createElement('div');
        entryDiv.classList.add('trick-entry');

        const playedCardsTitle = document.createElement('p');
        playedCardsTitle.textContent = 'Previously Played Trick:'; // Changed title for clarity
        entryDiv.appendChild(playedCardsTitle);

        lastTrickData.played_cards.forEach(playedInfo => {
            const cardElement = displayCard(playedInfo.card); // Use existing displayCard function
            // Apply some styling to make log cards smaller
            cardElement.style.width = '40px';
            cardElement.style.height = '60px';
            cardElement.style.fontSize = '0.6em'; // Adjust text size on card if any was visible
            cardElement.style.cursor = 'default'; // Not interactive

            const playerIdentifier = document.createElement('span');
            playerIdentifier.classList.add('played-card-info');
            let playerName = `P${parseInt(playedInfo.player) + 1}`; // Default name
            if (clientGameState.player_identities && clientGameState.player_identities[playedInfo.player]) {
                playerName = clientGameState.player_identities[playedInfo.player]; // Use full name if available
            }
            playerIdentifier.textContent = `${playerName}: `;

            const cardContainer = document.createElement('div'); // Container for player name + card
            cardContainer.appendChild(playerIdentifier);
            cardContainer.appendChild(cardElement);
            entryDiv.appendChild(cardContainer);
        });

        const winnerP = document.createElement('p');
        winnerP.classList.add('trick-winner-info');
        winnerP.textContent = `Winner: ${lastTrickData.winner_name}`;
        entryDiv.appendChild(winnerP);

        previousTrickLogDiv.appendChild(entryDiv);
    }

    /**
     * Fetches the initial game state from the server when the "Start New Game / Round" button is clicked.
     * Renders the board with the new game state.
     * Handles potential connection errors.
     */
    async function handleStartGameButtonClick() {
        try {
            let currentPhase = clientGameState ? clientGameState.game_phase : null;
            let data;

            // If game is over, or round is over, or not yet started (null phase),
            // first ensure server is in 'setup' state.
            if (currentPhase === "game_over" || currentPhase === "round_over" || currentPhase === null || currentPhase === "setup") {
                // Call /api/start_game to ensure server is reset to 'setup' if needed
                // This also handles the first click when the game is in 'setup'
                const resetResponse = await fetch('/api/start_game');
                if (!resetResponse.ok) {
                    const errData = await resetResponse.json().catch(() => ({ error: `HTTP error ${resetResponse.status}` }));
                    throw new Error(errData.error || `Failed to reset game to setup: ${resetResponse.status}`);
                }
                data = await resetResponse.json();
                renderBoard(data); // Render the "setup" state

                // If after this call, the phase is "setup", it means we are ready to deal the new round.
                // If it was already "setup", this click means "now deal".
                if (data.game_phase === "setup") {
                    const dealResponse = await fetch('/api/deal_new_round');
                    if (!dealResponse.ok) {
                        const errDataDeal = await dealResponse.json().catch(() => ({ error: `HTTP error ${dealResponse.status}` }));
                        throw new Error(errDataDeal.error || `Failed to deal new round: ${dealResponse.status}`);
                    }
                    data = await dealResponse.json();
                    renderBoard(data); // Render the state after dealing (e.g., bidding_round_1)
                }
                // If /api/start_game somehow didn't return "setup" (e.g. server error, unexpected state),
                // the renderBoard(data) above would have shown that. The next click might try again.
            }
            // Note: If the game was in an active state not covered above (e.g. bidding),
            // and the start button was somehow visible, this logic implies the first click
            // effectively resets to setup, and a subsequent click (if button remains/reappears)
            // would deal. This is generally fine as the button should only be visible in terminal/setup states.

        } catch (error) {
            console.error("Error during start game/deal new round sequence:", error);
            gameMessageP.textContent = `Error: ${error.message}. Please try again or refresh.`;
            // Attempt to render a safe, empty board state on critical error
            renderBoard({});
        }
    }

    /**
     * Submits a player's action to the server.
     * Handles card play animation if applicable.
     * Updates the UI based on the server's response.
     * Manages error display if the action fails.
     * @param {string} action - The type of action being submitted (e.g., 'play_card', 'order_up').
     * @param {object} [details={}] - Additional details for the action (e.g., card played, suit called).
     */
    async function submitPlayerAction(action, details = {}) {
        const payload = { player_index: 0, action: action, ...details }; // Player 0 is always human
        let cardToAnimateElement = null;

        // If playing a card, find its DOM element for animation
        if (action === 'play_card' && details.card) {
            if (player0YouHandDiv) {
                cardToAnimateElement = Array.from(player0YouHandDiv.children).find(cardEl =>
                    cardEl.dataset.rank === details.card.rank && cardEl.dataset.suit === details.card.suit
                );
            }
        }

        try {
            if (cardToAnimateElement) {
                playCardButton.disabled = true; // Disable button during animation
                await animateCardPlay(cardToAnimateElement, trickAreaDiv); // Animate card to trick area
            }

            const response = await fetch('/api/submit_action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                let errorMsg = `Action failed. Status: ${response.status}.`;
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.error || errorData.message || errorMsg; // Use server's specific error message
                } catch (e) {
                    // If JSON parsing of error fails, try to get plain text from response
                    const errorText = await response.text().catch(() => "Server returned an error, but further details could not be retrieved.");
                    errorMsg = errorText || errorMsg; // Use text if available, otherwise the status code message
                }
                throw new Error(errorMsg);
            }
            const data = await response.json();
            renderBoard(data); // Update UI with new state from server
        } catch (error) {
            console.error(`Error submitting action '${action}':`, error);
            gameMessageP.textContent = `Action failed: ${error.message}. Please check the game state or try refreshing.`;

            // Attempt to re-enable the correct button based on the action that failed.
            // This helps if the server error was transient and the user might want to retry.
            if (action === 'play_card') {
                 playCardButton.disabled = selectedCardsForDiscard.length !== 1;
            } else if (action === 'dealer_discard_one' || action === 'dealer_must_discard_after_order_up' || action === 'maker_discard') {
                // For discard actions, re-enable based on discard selection logic defined in updateSelectedCardsDisplay
                updateSelectedCardsDisplay();
            }
            // For other action buttons (bidding, go alone), they are typically hidden/shown by renderBoard.
            // If an error occurs, a full re-render might be necessary or a page refresh by the user.
            // If renderBoard({}) was called on error, it would also hide/show appropriate buttons.
        }
    }

    // --- Event Listeners for Player Actions ---
    startGameButton.addEventListener('click', handleStartGameButtonClick);
    orderUpButton.addEventListener('click', () => submitPlayerAction('order_up'));
    passBidButton.addEventListener('click', () => submitPlayerAction('pass_bid'));

    callSuitButtons.forEach(button => {
        button.addEventListener('click', (e) => {
            const suitKey = Object.keys(SUIT_NAMES).find(key => SUIT_NAMES[key] === e.target.dataset.suit); // Convert suit name to key ('H', 'D', etc.)
            if (suitKey) submitPlayerAction('call_trump', { suit: suitKey });
        });
    });
    passCallButton.addEventListener('click', () => submitPlayerAction('pass_call'));
    chooseGoAloneButton.addEventListener('click', () => submitPlayerAction('choose_go_alone'));
    chooseNotGoAloneButton.addEventListener('click', () => submitPlayerAction('choose_not_go_alone'));

    // Listener for the main action button (Play Card / Discard)
    playCardButton.addEventListener('click', () => {
        const phase = clientGameState.game_phase;
        const count = clientGameState.cards_to_discard_count; // Number of cards server expects to be discarded

        // Logic for 'dealer_discard_one' phase (human dealer, made trump themselves)
        if (phase === 'dealer_discard_one') {
            if (selectedCardsForDiscard.length === 1 && count === 1) {
                submitPlayerAction('dealer_discard_one', { cards: selectedCardsForDiscard });
            } else {
                gameMessageP.textContent = `Please select exactly 1 card to discard.`;
                 setTimeout(() => { if(clientGameState && clientGameState.message) gameMessageP.textContent = clientGameState.message; }, 2000); // Revert message
            }
        }
        // Logic for 'dealer_must_discard_after_order_up' phase (human dealer, someone else ordered up)
        else if (phase === 'dealer_must_discard_after_order_up') {
            if (selectedCardsForDiscard.length === 1 && count === 1) {
                submitPlayerAction('dealer_must_discard_after_order_up', { cards: selectedCardsForDiscard });
            } else {
                gameMessageP.textContent = `Please select exactly 1 card to discard.`;
                setTimeout(() => { if(clientGameState && clientGameState.message) gameMessageP.textContent = clientGameState.message; }, 2000);
            }
        }
        // Logic for 'maker_discard' phase (human maker, picked up dummy hand)
        else if (phase === 'maker_discard') {
            if (selectedCardsForDiscard.length === 5 && count === 5) {
                submitPlayerAction('maker_discard', { cards: selectedCardsForDiscard });
            } else {
                gameMessageP.textContent = `Please select exactly 5 cards to discard.`;
                 setTimeout(() => { if(clientGameState && clientGameState.message) gameMessageP.textContent = clientGameState.message; }, 2000);
            }
        }
        // Logic for 'playing_tricks' phase
        else if (phase === 'playing_tricks') {
            if (selectedCardsForDiscard.length === 1) {
                submitPlayerAction('play_card', { card: selectedCardsForDiscard[0] });
            } else {
                gameMessageP.textContent = "Please select a card to play.";
                 // No timeout here as it's an immediate feedback for an attempt to play without selection.
            }
        }
    });

    /**
     * Saves the current client-side game state and statistics to localStorage.
     * This allows the game to be potentially resumed or stats to persist across sessions.
     */
    function saveStateToLocalStorage() {
        try {
            localStorage.setItem(EUCHRE_GAME_STATE_KEY, JSON.stringify(clientGameState));
            localStorage.setItem(EUCHRE_GAME_STATS_KEY, JSON.stringify(gameStatistics));
        } catch (e) {
            console.error("Error saving state to localStorage:", e);
        }
    }

    /**
     * Loads game state and statistics from localStorage if available.
     * Updates the `clientGameState` and `gameStatistics` variables.
     * @returns {boolean} True if a valid game state was loaded, false otherwise.
     */
    function loadStateFromLocalStorage() {
        try {
            const savedGameState = localStorage.getItem(EUCHRE_GAME_STATE_KEY);
            const savedGameStats = localStorage.getItem(EUCHRE_GAME_STATS_KEY);

            if (savedGameStats) {
                gameStatistics = JSON.parse(savedGameStats);
            } else {
                // Initialize stats if not found
                gameStatistics = { wins: 0, losses: 0 };
            }
            // Update stats display elements
            if (statsWinsSpan) statsWinsSpan.textContent = gameStatistics.wins;
            if (statsLossesSpan) statsLossesSpan.textContent = gameStatistics.losses;

            if (savedGameState) {
                const parsedGameState = JSON.parse(savedGameState);
                // Basic validation of the loaded game state
                if (parsedGameState && typeof parsedGameState === 'object' && parsedGameState.hasOwnProperty('game_phase')) {
                    clientGameState = parsedGameState;
                    return true; // Successfully loaded a potentially valid game state
                } else {
                    // Invalid state found, remove it
                    localStorage.removeItem(EUCHRE_GAME_STATE_KEY);
                    clientGameState = {}; // Reset client state
                }
            } else {
                clientGameState = {}; // No saved state found
            }
        } catch (e) {
            // Error during parsing or loading, clear potentially corrupted storage
            console.error("Error loading state from localStorage:", e);
            localStorage.removeItem(EUCHRE_GAME_STATE_KEY);
            localStorage.removeItem(EUCHRE_GAME_STATS_KEY);
            clientGameState = {};
            gameStatistics = { wins: 0, losses: 0 };
        }
        return false; // No valid game state loaded
    }

    /**
     * Initializes the game when the page loads.
     * Attempts to load state from localStorage. If an active game state is found,
     * it fetches the latest state from the server to ensure consistency.
     * Otherwise, it renders an empty/default board.
     */
    async function initializeGame() {
        loadStateFromLocalStorage(); // Load any persisted stats and potentially a game state

        // If a game was in progress (not in a terminal or setup phase according to localStorage)
        if (clientGameState && clientGameState.game_phase &&
            !["game_over", "round_over", "setup"].includes(clientGameState.game_phase)) {
            try {
                // Fetch the current authoritative state from the server to sync up
                const response = await fetch('/api/get_current_state');
                if (!response.ok) {
                    let errorMsg = `Failed to sync with server. Status: ${response.status}.`;
                    try {
                        const errData = await response.json();
                        errorMsg = errData.error || errData.message || errorMsg;
                    } catch (e) {
                        const errorText = await response.text().catch(() => "Could not retrieve server error details.");
                        errorMsg = errorText || errorMsg;
                    }
                    throw new Error(errorMsg);
                }
                const serverGameState = await response.json();
                if (serverGameState && serverGameState.game_phase) {
                    // If server indicates game is actually in 'setup' (e.g., server restarted),
                    // then we should not resume the locally stored "active" game.
                    if (serverGameState.game_phase === "setup") {
                        console.log("Server is in setup phase. Clearing local active game state.");
                        renderBoard({}); // Render an empty/default board
                    } else {
                        renderBoard(serverGameState); // Sync with server's current state
                    }
                } else {
                    // Server returned an unexpected or incomplete state.
                    console.warn("Server returned invalid game state during initialization. Clearing local active game.");
                    renderBoard({}); // Render an empty/default board
                }
            } catch (error) {
                // Error fetching from server, display message and render an empty board.
                // This might happen if the server is down or there's a network issue.
                console.error("Error fetching current state from server during initialization:", error);
                gameMessageP.textContent = `Could not sync game with server: ${error.message}. Starting fresh or refresh.`;
                renderBoard({}); // Render an empty/default board
            }
        } else {
            // No active game in localStorage, or it was in a terminal/setup state.
            // So, just render an empty/default board.
            renderBoard({});
        }
    }

    // Initialize the game when the DOM is fully loaded.
    initializeGame();
});
