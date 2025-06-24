document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const player1HandDiv = document.getElementById('player-1-hand');
    const player2HandDiv = document.getElementById('player-2-hand');
    const player3HandDiv = document.getElementById('player-3-hand');
    const kittyDisplayDiv = document.getElementById('kitty-display');
    const upCardDisplaySpan = document.getElementById('up-card-display');
    const trumpSuitSpan = document.getElementById('trump-suit');
    const gameMessageP = document.getElementById('game-message');
    const scorePlayer1Span = document.getElementById('score-player-1');
    const scorePlayer2Span = document.getElementById('score-player-2');
    const scorePlayer3Span = document.getElementById('score-player-3');
    const statsWinsSpan = document.getElementById('stats-wins');
    const statsLossesSpan = document.getElementById('stats-losses');
    const player1HandTricksSpan = document.getElementById('player-1-hand-tricks');
    const player2HandTricksSpan = document.getElementById('player-2-hand-tricks');
    const player3HandTricksSpan = document.getElementById('player-3-hand-tricks');
    const dummyHandDisplaySpan = document.getElementById('dummy-hand-display'); // Changed from kitty
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
    const CARD_IMAGE_BASE_URL = 'https://deckofcardsapi.com/static/img/';

    function getCardImageFilename(card) {
        let rankForImage = card.rank;
        if (card.rank === '10') rankForImage = '0';
        return `${rankForImage}${card.suit}.png`;
    }

    // --- UI Update Functions ---
    function isGameOverOrRoundOver(gameState) {
        return gameState.game_phase === "round_over" || gameState.game_phase === "game_over";
    }

    function displayCard(cardData, isOpponentHandCard = false) {
        const cardDiv = document.createElement('div');
        cardDiv.classList.add('card');
        cardDiv.dataset.suit = cardData.suit;
        cardDiv.dataset.rank = cardData.rank;
        cardDiv.dataset.name = cardData.name; // Store for easy access
        cardDiv.setAttribute('aria-label', cardData.name);
        const imageUrl = CARD_IMAGE_BASE_URL + (isOpponentHandCard ? 'back.png' : getCardImageFilename(cardData));
        cardDiv.style.backgroundImage = `url(${imageUrl})`;
        return cardDiv;
    }

    function animateCardPlay(cardElement, targetContainer) {
        const cardRect = cardElement.getBoundingClientRect();
        const targetRect = targetContainer.getBoundingClientRect();

        const animatedCard = cardElement.cloneNode(true);
        animatedCard.classList.add('animated-card');
        document.body.appendChild(animatedCard);

        // Set initial position of the animated card
        animatedCard.style.position = 'fixed'; // Use fixed for viewport-relative positioning during animation
        animatedCard.style.left = `${cardRect.left}px`;
        animatedCard.style.top = `${cardRect.top}px`;
        animatedCard.style.width = `${cardRect.width}px`;
        animatedCard.style.height = `${cardRect.height}px`;
        animatedCard.style.zIndex = '1000'; // Ensure it's on top

        // Hide original card in hand during animation (or mark it visually)
        cardElement.style.opacity = '0.3';


        // Calculate target position relative to the viewport
        // The trick area might scroll, so targetRect is what we need.
        // We want the card to land *inside* the targetContainer.
        // For simplicity, let's aim for the top-left of the trick area,
        // then CSS can arrange it nicely if trickAreaDiv uses flex.
        const targetX = targetRect.left + window.scrollX;
        const targetY = targetRect.top + window.scrollY;


        requestAnimationFrame(() => {
            animatedCard.style.transition = 'left 0.5s ease-out, top 0.5s ease-in, width 0.5s ease-in-out, height 0.5s ease-in-out';
            animatedCard.style.left = `${targetX}px`;
            animatedCard.style.top = `${targetY}px`;
            // Optionally, adjust size if cards in trick area are different
            // animatedCard.style.width = '50px'; // Example: smaller size in trick area
            // animatedCard.style.height = '75px';
        });

        return new Promise(resolve => {
            animatedCard.addEventListener('transitionend', () => {
                animatedCard.remove();
                cardElement.style.opacity = '1'; // Restore original card visibility (it will be removed by renderBoard)
                resolve();
            }, { once: true });
        });
    }


    function updateSelectedCardsDisplay() {
        // Visually indicate how many cards are selected vs needed for discard
        if ((clientGameState.game_phase === 'maker_discard' || clientGameState.game_phase === 'dealer_discard_one') &&
            clientGameState.current_player_turn === 0 && clientGameState.maker === 0) {
            const needed = clientGameState.cards_to_discard_count; // Should be 1 for discard_one, 5 for discard
            const selectedCount = selectedCardsForDiscard.length;
            playCardButton.textContent = `Discard ${selectedCount}/${needed} Selected`;
            playCardButton.disabled = selectedCount !== needed;
        }
    }

    function handleCardClick(cardElement, cardData) {
        const phase = clientGameState.game_phase;
        const isMyTurn = clientGameState.current_player_turn === 0;
        const amIMaker = clientGameState.maker === 0;
        const isDealerDiscardOnePhase = phase === 'dealer_discard_one' && amIMaker && isMyTurn;
        const isMakerDiscardPhase = phase === 'maker_discard' && amIMaker && isMyTurn;


        if (isDealerDiscardOnePhase || isMakerDiscardPhase) {
            const neededToDiscard = clientGameState.cards_to_discard_count;
            const indexInSelection = selectedCardsForDiscard.findIndex(c => c.rank === cardData.rank && c.suit === cardData.suit);

            if (indexInSelection > -1) { // Card is already selected, de-select it
                selectedCardsForDiscard.splice(indexInSelection, 1);
                cardElement.classList.remove('selected');
            } else { // Card not selected, select it if not exceeding discard count
                if (selectedCardsForDiscard.length < neededToDiscard) {
                    selectedCardsForDiscard.push({ rank: cardData.rank, suit: cardData.suit, name: cardData.name });
                    cardElement.classList.add('selected');
                } else {
                    gameMessageP.textContent = `You can only select ${neededToDiscard} card(s) to discard.`;
                    setTimeout(() => { if(clientGameState) gameMessageP.textContent = clientGameState.message; }, 2000);
                }
            }
            updateSelectedCardsDisplay();
        } else if (phase === 'playing_tricks' && isMyTurn) {
            // Single card selection for playing a trick
            document.querySelectorAll('#player-1-hand .card.selected').forEach(el => el.classList.remove('selected'));
            cardElement.classList.add('selected');
            selectedCardsForDiscard = [{ rank: cardData.rank, suit: cardData.suit, name: cardData.name }]; // Store single card
            playCardButton.textContent = 'Play Selected Card';
            playCardButton.disabled = false;
        }
    }


    function renderBoard(gameState) {
        if (!gameState || Object.keys(gameState).length === 0) {
            gameMessageP.textContent = "Welcome! Click 'Start New Game / Round'.";
            biddingActionsDiv.style.display = 'none';
            callTrumpActionsDiv.style.display = 'none';
            playActionsDiv.style.display = 'none';
            goAloneActionsDiv.style.display = 'none';
            startGameButton.style.display = 'inline-block';
            player1HandDiv.innerHTML = player2HandDiv.innerHTML = player3HandDiv.innerHTML = '';
            upCardDisplaySpan.innerHTML = 'None';
            dummyHandDisplaySpan.textContent = '0 cards'; // Updated kitty to dummy
            trickAreaDiv.innerHTML = '<h3>Current Trick</h3>'; // Clear trick area, add back title
            trumpSuitSpan.textContent = 'None';
            scorePlayer1Span.textContent = scorePlayer2Span.textContent = scorePlayer3Span.textContent = '0';
            if (player1HandTricksSpan) player1HandTricksSpan.textContent = '0';
            if (player2HandTricksSpan) player2HandTricksSpan.textContent = '0';
            if (player3HandTricksSpan) player3HandTricksSpan.textContent = '0';
            return;
        }
        clientGameState = gameState; // Store the latest game state
        selectedCardsForDiscard = []; // Reset card selection on every state update

        // Clear trick area before re-rendering
        trickAreaDiv.innerHTML = '<h3>Current Trick</h3>'; // Keep title, clear cards

        // Render Hands
        // Before rendering hands, check if an AI needs to play
        if (gameState.game_phase === 'playing_tricks' &&
            gameState.current_player_turn !== 0 && // It's an AI's turn
            !isGameOverOrRoundOver(gameState)) { // And game/round is not over

            // Small delay before AI plays to allow human to see previous action's result
            setTimeout(() => {
                console.log(`Player ${gameState.current_player_turn + 1} (AI) is playing. Requesting AI turn.`);
                fetch('/api/ai_play_turn', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            console.error("Error during AI play turn:", data.error);
                            gameMessageP.textContent = `Error during AI turn: ${data.error}. Refresh if stuck.`;
                            if(data.game_state) renderBoard(data.game_state); // Render state even on error
                        } else {
                            renderBoard(data); // Re-render with the new state after AI play
                        }
                    })
                    .catch(error => {
                        console.error("Failed to trigger AI play or parse response:", error);
                        gameMessageP.textContent = "Connection error during AI turn. Refresh if stuck.";
                    });
            }, 750); // Delay for AI "thinking" or to see previous card, adjust as needed (was 0.5s server-side + this)

            // While waiting for AI to play, display current state but perhaps indicate AI is thinking
            // For now, just render current state and the AI play will trigger a re-render.
        }


        ['0', '1', '2'].forEach(playerKey => {
            const handDivId = `player-${parseInt(playerKey) + 1}-hand`;
            const handDiv = document.getElementById(handDivId);
            if (!handDiv) {
                console.error(`Could not find handDiv for playerKey ${playerKey} (expected ID ${handDivId})`);
                return; // Skip this player if div not found
            }
            handDiv.innerHTML = '';
            if (gameState.hands && gameState.hands[playerKey]) {
                gameState.hands[playerKey].forEach(card => {
                    const isOpponent = playerKey !== '0';
                    const cardElement = displayCard(card, isOpponent);
                    if (!isOpponent) { // Only human player's cards are clickable
                         cardElement.addEventListener('click', () => handleCardClick(cardElement, card));
                         cardElement.addEventListener('dblclick', () => {
                            // Check if it's the playing tricks phase and player's turn
                            if (clientGameState.game_phase === 'playing_tricks' && clientGameState.current_player_turn === 0) {
                                // Directly submit the card for playing
                                console.log("Double-click detected, playing card:", card);
                                // Ensure the card data is correctly formatted for submitPlayerAction
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
            // Show original up-card as turned down if bidding round 2 or dealer must call
             const turnedDownCard = displayCard(gameState.original_up_card_for_round, true); // Show back
             upCardDisplaySpan.appendChild(turnedDownCard);
             upCardDisplaySpan.appendChild(document.createTextNode(" (Turned Down)"));
        } else {
            upCardDisplaySpan.textContent = 'None';
        }

        // Render Current Trick
        if (gameState.trick_cards && gameState.trick_cards.length > 0) {
            gameState.trick_cards.forEach(playedCardInfo => {
                // Ensure playedCardInfo.card is defined and has properties
                if (playedCardInfo && playedCardInfo.card && typeof playedCardInfo.card.suit !== 'undefined' && typeof playedCardInfo.card.rank !== 'undefined') {
                    const cardElement = displayCard(playedCardInfo.card);
                    // Optionally, add player info to the card or trick area
                    const playerIdentifier = document.createElement('span');
                    playerIdentifier.classList.add('trick-player-identifier');
                    // Use player_identities from gameState if available, otherwise fallback to index
                    let playerName = `P${parseInt(playedCardInfo.player) + 1}`; // Fallback
                    if (gameState.player_identities && gameState.player_identities[playedCardInfo.player]) {
                        playerName = gameState.player_identities[playedCardInfo.player].split(' ')[0] + // "Player"
                                     ` ${gameState.player_identities[playedCardInfo.player].split(' ')[1]}`; // "1" or "2" or "3"
                    }
                    playerIdentifier.textContent = `${playerName}: `;

                    const trickCardContainer = document.createElement('div');
                    trickCardContainer.classList.add('trick-card-container');
                    trickCardContainer.appendChild(playerIdentifier);
                    trickCardContainer.appendChild(cardElement);
                    trickAreaDiv.appendChild(trickCardContainer);
                } else {
                    console.warn("Skipping rendering of an invalid/incomplete card in trick_cards:", playedCardInfo);
                }
            });
        }


        // Update dummy hand display
        if (gameState.dummy_hand && gameState.dummy_hand.length > 0) {
            dummyHandDisplaySpan.textContent = `${gameState.dummy_hand.length} cards in dummy hand`;
        } else if (gameState.game_phase !== 'setup' && gameState.game_phase !== 'round_over' && gameState.game_phase !== 'game_over') {
            // If dummy hand is empty but it's mid-round (implying it was taken)
            dummyHandDisplaySpan.textContent = 'Dummy hand taken by maker';
        } else {
            dummyHandDisplaySpan.textContent = '0 cards';
        }

        trumpSuitSpan.textContent = gameState.trump_suit ? SUIT_NAMES[gameState.trump_suit] : 'None';
        scorePlayer1Span.textContent = gameState.scores['0'];
        scorePlayer2Span.textContent = gameState.scores['1'];
        scorePlayer3Span.textContent = gameState.scores['2'];

        // Update Round Tricks Won
        if (gameState.round_tricks_won) {
            if (player1HandTricksSpan) player1HandTricksSpan.textContent = gameState.round_tricks_won['0'] || '0';
            if (player2HandTricksSpan) player2HandTricksSpan.textContent = gameState.round_tricks_won['1'] || '0';
            if (player3HandTricksSpan) player3HandTricksSpan.textContent = gameState.round_tricks_won['2'] || '0';
        } else {
            if (player1HandTricksSpan) player1HandTricksSpan.textContent = '0';
            if (player2HandTricksSpan) player2HandTricksSpan.textContent = '0';
            if (player3HandTricksSpan) player3HandTricksSpan.textContent = '0';
        }

        gameMessageP.textContent = gameState.message || "";

        // Update Maker Indicator
        document.querySelectorAll('.player-area').forEach(area => {
            area.classList.remove('player-area-maker');
            const makerIndicator = area.querySelector('.maker-text-indicator');
            if (makerIndicator) {
                makerIndicator.remove();
            }
        });

        if (gameState.maker !== null && gameState.maker !== undefined) {
            const makerPlayerDiv = document.getElementById(`player-${parseInt(gameState.maker) + 1}`);
            if (makerPlayerDiv) {
                makerPlayerDiv.classList.add('player-area-maker');
                // Add a text indicator inside the player's h2 title
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

        // Button Visibility
        biddingActionsDiv.style.display = 'none';
        callTrumpActionsDiv.style.display = 'none';
        playActionsDiv.style.display = 'none';
        goAloneActionsDiv.style.display = 'none';
        passCallButton.style.display = 'inline-block'; // Default for bidding round 2

        const isMyTurn = gameState.current_player_turn === 0;
        const amIMaker = gameState.maker === 0;

        if (isMyTurn) { // It must be my turn for most actions
            switch (gameState.game_phase) {
                case 'bidding_round_1':
                    biddingActionsDiv.style.display = 'block';
                    orderUpButton.style.display = 'inline-block';
                    passBidButton.style.display = 'inline-block';
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
                case 'dealer_discard_one': // Human dealer (maker) discards 1 after ordering up
                    if (amIMaker && gameState.dealer === 0) { // Ensure it's the human dealer/maker
                        playActionsDiv.style.display = 'block';
                        playCardButton.textContent = `Discard 0/1 Selected`; // cards_to_discard_count should be 1
                        playCardButton.disabled = true;
                    }
                    break;
                case 'maker_discard': // Human maker discards 5 (after not going alone and picking up dummy)
                     if (amIMaker) {
                        playActionsDiv.style.display = 'block';
                        playCardButton.textContent = `Discard 0/5 Selected`; // cards_to_discard_count should be 5
                        playCardButton.disabled = true;
                    }
                    break;
                case 'prompt_go_alone':
                    if (amIMaker) { // Only maker sees these options
                        goAloneActionsDiv.style.display = 'block';
                        // Message is already set by server: "Choose to go alone or play with partner."
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
        renderPreviousTrickLog(clientGameState.last_completed_trick); // Add this call

    // Update overall stats display
    if (statsWinsSpan) statsWinsSpan.textContent = gameStatistics.wins;
    if (statsLossesSpan) statsLossesSpan.textContent = gameStatistics.losses;

    // Check for game over to update stats
    if (gameState.game_phase === "game_over" && gameState.scores) {
        const player1Score = gameState.scores['0'];
        // Assuming a game is won if player 1's score is 10 or more.
        // And lost if another player's score is 10 or more while player 1's is less.
        // This logic might need adjustment based on specific game rules for winning.
        let gameJustEnded = !clientGameState.hasOwnProperty('_gameEndedFlag') || !clientGameState._gameEndedFlag;

        if (gameJustEnded) { // Process only once per game end
            let playerWon = false;
            if (player1Score >= 10) { // Assuming 10 points to win
                 playerWon = true;
                 for (const player in gameState.scores) {
                    if (player !== '0' && gameState.scores[player] >= 10) {
                        // If another player also reached 10, it's possible player 1 didn't "solely" win
                        // Depending on rules, this might need more complex logic (e.g. who reached 10 first)
                        // For simplicity, if P1 is >= 10, we count it as a win unless another has more.
                        if (gameState.scores[player] > player1Score) playerWon = false;
                    }
                 }
            }

            let otherPlayerWon = false;
            if (!playerWon) { // Only if player 1 hasn't won, check if others did
                for (const player in gameState.scores) {
                    if (player !== '0' && gameState.scores[player] >= 10) {
                        otherPlayerWon = true;
                        break;
                    }
                }
            }

            if (playerWon) {
                gameStatistics.wins++;
                console.log("Player 1 wins! Stats updated.");
            } else if (otherPlayerWon) {
                gameStatistics.losses++;
                console.log("Player 1 loses. Stats updated.");
            }
            // Mark that this game_over state has been processed for stats
            // This temporary flag is added to clientGameState *before* saving
            clientGameState._gameEndedFlag = true;

            // Update display immediately
            if (statsWinsSpan) statsWinsSpan.textContent = gameStatistics.wins;
            if (statsLossesSpan) statsLossesSpan.textContent = gameStatistics.losses;
        }
    } else if (gameState.game_phase !== "game_over" && clientGameState.hasOwnProperty('_gameEndedFlag')) {
        // Reset flag when a new round/game starts
        delete clientGameState._gameEndedFlag;
    }


    saveStateToLocalStorage(); // Save state after every render
    }

    function renderPreviousTrickLog(lastTrickData) {
        if (!previousTrickLogDiv) return;

        previousTrickLogDiv.innerHTML = ''; // Clear previous log

        if (!lastTrickData || !lastTrickData.played_cards || lastTrickData.played_cards.length === 0) {
            previousTrickLogDiv.innerHTML = '<p>No tricks completed yet in this round.</p>';
            return;
        }

        const entryDiv = document.createElement('div');
        entryDiv.classList.add('trick-entry');

        const playedCardsTitle = document.createElement('p');
        playedCardsTitle.textContent = 'Cards Played:';
        entryDiv.appendChild(playedCardsTitle);

        lastTrickData.played_cards.forEach(playedInfo => {
            const cardElement = displayCard(playedInfo.card);
            // Make these cards smaller for the log display
            cardElement.style.width = '40px';
            cardElement.style.height = '60px';
            cardElement.style.fontSize = '0.6em';
            cardElement.style.cursor = 'default'; // Not interactive

            const playerIdentifier = document.createElement('span');
            playerIdentifier.classList.add('played-card-info');
            // Use player_identities from clientGameState if available, otherwise fallback to index
            let playerName = `P${parseInt(playedInfo.player) + 1}`; // Fallback
            if (clientGameState.player_identities && clientGameState.player_identities[playedInfo.player]) {
                playerName = clientGameState.player_identities[playedInfo.player];
            }
            playerIdentifier.textContent = `${playerName}: `;

            const cardContainer = document.createElement('div');
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


    async function fetchInitialGameState() {
        console.log("Fetching initial game state via /api/start_game...");
        try {
            const response = await fetch('/api/start_game');
            if (!response.ok) {
                const errorText = await response.text(); // Try to get more details
                throw new Error(`HTTP error! status: ${response.status}, body: ${errorText}`);
            }
            const data = await response.json();
            console.log("Game state received:", data);
            renderBoard(data);
        } catch (error) {
            console.error("Could not fetch initial game state:", error);
            gameMessageP.textContent = `Error connecting to server: ${error.message}. Click 'Start New Game / Round'.`;
            renderBoard({}); // Render an empty board on error
        }
    }

    async function submitPlayerAction(action, details = {}) {
        const payload = { player_index: 0, action: action, ...details };
        console.log("Submitting action:", payload);

        let cardToAnimateElement = null;
        if (action === 'play_card' && details.card) {
            // Find the card element in player 1's hand that matches details.card
            const player1Hand = document.getElementById('player-1-hand');
            cardToAnimateElement = Array.from(player1Hand.children).find(cardEl =>
                cardEl.dataset.rank === details.card.rank && cardEl.dataset.suit === details.card.suit
            );
        }

        try {
            if (cardToAnimateElement) {
                // Disable UI during animation + fetch
                playCardButton.disabled = true;
                // Animate first, then submit and render
                await animateCardPlay(cardToAnimateElement, trickAreaDiv);
            }

            const response = await fetch('/api/submit_action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({error: "Unknown server error."}));
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();
            renderBoard(data); // This will redraw the board, including the card in the trick area
        } catch (error) {
            console.error("Action failed:", error);
            gameMessageP.textContent = `Action failed: ${error.message}. Check console.`;
            // Re-enable button if animation was not the issue, or if server error occurs
            if (action === 'playing_tricks') playCardButton.disabled = selectedCardsForDiscard.length !== 1;

        }
    }

    startGameButton.addEventListener('click', fetchInitialGameState);
    orderUpButton.addEventListener('click', () => submitPlayerAction('order_up'));
    passBidButton.addEventListener('click', () => submitPlayerAction('pass_bid'));
    callSuitButtons.forEach(button => {
        button.addEventListener('click', (e) => {
            const suitKey = Object.keys(SUIT_NAMES).find(key => SUIT_NAMES[key] === e.target.dataset.suit);
            if (suitKey) submitPlayerAction('call_trump', { suit: suitKey });
        });
    });
    passCallButton.addEventListener('click', () => submitPlayerAction('pass_call'));

    chooseGoAloneButton.addEventListener('click', () => submitPlayerAction('choose_go_alone'));
    chooseNotGoAloneButton.addEventListener('click', () => submitPlayerAction('choose_not_go_alone'));

    playCardButton.addEventListener('click', () => {
        const phase = clientGameState.game_phase;
        const count = clientGameState.cards_to_discard_count; // Server-provided count

        if (phase === 'dealer_discard_one') { // Human dealer discards 1
            if (selectedCardsForDiscard.length === 1 && count === 1) {
                submitPlayerAction('dealer_discard_one', { cards: selectedCardsForDiscard });
            } else {
                gameMessageP.textContent = `Please select exactly 1 card to discard.`;
                 setTimeout(() => { if(clientGameState) gameMessageP.textContent = clientGameState.message; }, 2000);
            }
        } else if (phase === 'maker_discard') { // Human maker discards 5
            // Ensure count is 5 for this phase, as it's specifically for discarding 5 after dummy pickup
            if (selectedCardsForDiscard.length === 5 && count === 5) {
                submitPlayerAction('maker_discard', { cards: selectedCardsForDiscard });
            } else {
                gameMessageP.textContent = `Please select exactly 5 cards to discard.`;
                 setTimeout(() => { if(clientGameState) gameMessageP.textContent = clientGameState.message; }, 2000);
            }
        } else if (phase === 'playing_tricks') {
            if (selectedCardsForDiscard.length === 1) {
                submitPlayerAction('play_card', { card: selectedCardsForDiscard[0] });
            } else {
                gameMessageP.textContent = "Please select a card to play.";
            }
        }
        // selectedCardsForDiscard will be cleared by renderBoard on new state
    });

    // --- Local Storage Functions ---
    function saveStateToLocalStorage() {
        try {
            localStorage.setItem(EUCHRE_GAME_STATE_KEY, JSON.stringify(clientGameState));
            localStorage.setItem(EUCHRE_GAME_STATS_KEY, JSON.stringify(gameStatistics));
            console.log("Game state and stats saved to localStorage.");
        } catch (e) {
            console.error("Error saving state to localStorage:", e);
        }
    }

    function loadStateFromLocalStorage() {
        try {
            const savedGameState = localStorage.getItem(EUCHRE_GAME_STATE_KEY);
            const savedGameStats = localStorage.getItem(EUCHRE_GAME_STATS_KEY);

            if (savedGameStats) {
                gameStatistics = JSON.parse(savedGameStats);
                console.log("Game statistics loaded from localStorage:", gameStatistics);
            } else {
                console.log("No game statistics found in localStorage. Using defaults.");
                gameStatistics = { wins: 0, losses: 0 }; // Initialize if not found
            }
            // Ensure stats display is updated even if no game state is loaded yet
            if (statsWinsSpan) statsWinsSpan.textContent = gameStatistics.wins;
            if (statsLossesSpan) statsLossesSpan.textContent = gameStatistics.losses;


            if (savedGameState) {
                const parsedGameState = JSON.parse(savedGameState);
                // Basic validation: Check if it's an object and has a game_phase
                if (parsedGameState && typeof parsedGameState === 'object' && parsedGameState.hasOwnProperty('game_phase')) {
                    clientGameState = parsedGameState;
                    console.log("Game state loaded from localStorage:", clientGameState);
                    return true; // Indicate successful load of game state
                } else {
                    console.warn("Invalid game state found in localStorage. Ignoring.");
                    localStorage.removeItem(EUCHRE_GAME_STATE_KEY); // Clear invalid data
                    clientGameState = {}; // Reset to empty
                }
            } else {
                console.log("No game state found in localStorage.");
                clientGameState = {}; // Ensure it's empty if nothing is loaded
            }
        } catch (e) {
            console.error("Error loading state from localStorage:", e);
            // Clear potentially corrupted data
            localStorage.removeItem(EUCHRE_GAME_STATE_KEY);
            localStorage.removeItem(EUCHRE_GAME_STATS_KEY);
            clientGameState = {};
            gameStatistics = { wins: 0, losses: 0 };
        }
        return false; // Indicate game state was not loaded
    }

    // --- Initialization ---
    async function initializeGame() {
        loadStateFromLocalStorage(); // Loads both gameStatistics and potentially clientGameState

        if (clientGameState && clientGameState.game_phase &&
            !["game_over", "round_over", "setup"].includes(clientGameState.game_phase)) {
            // LocalStorage has an active game state. Let's verify with server.
            console.log("LocalStorage has active game state. Verifying with server...");
            try {
                const response = await fetch('/api/get_current_state');
                if (!response.ok) {
                    // Consider non-OK responses (like 404, 500) as server not ready or error
                    console.error(`Error fetching current state: ${response.status}. Assuming server is not in a ready state or game is not active.`);
                    throw new Error(`Server status: ${response.status}`);
                }
                const serverGameState = await response.json();

                if (serverGameState && serverGameState.game_phase) {
                    if (serverGameState.game_phase === "setup") {
                        // Server is in setup (e.g. restarted). Client's localStorage is stale.
                        console.log("Server is in setup phase. Discarding stale localStorage game state.");
                        renderBoard({}); // Show startGameButton
                    } else {
                        // Server has an active game. Use server's state.
                        console.log("Server has active game state. Rendering server state:", serverGameState);
                        renderBoard(serverGameState);
                    }
                } else {
                    // Invalid response from server (e.g. empty JSON or missing game_phase)
                    console.warn("Invalid game state received from /api/get_current_state. Rendering empty board.");
                    renderBoard({});
                }
            } catch (error) {
                // Catch fetch errors (network issue) or errors thrown from non-ok responses
                console.error("Failed to get current state from server:", error, "Rendering empty board as fallback.");
                renderBoard({}); // Fallback: show startGameButton
            }
        } else {
            // No active game state in localStorage, or it was terminal/setup.
            // This includes cases where clientGameState is empty or doesn't have a game_phase.
            console.log("No active game in localStorage or state is terminal/setup. Rendering empty board.");
            renderBoard({}); // Shows startGameButton
        }
    }

    initializeGame(); // Initialize on page load
});
