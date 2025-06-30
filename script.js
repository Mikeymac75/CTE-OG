document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const player0YouHandDiv = document.getElementById('player-0-you-hand');
    const player1AiHandDiv = document.getElementById('player-1-ai-hand');
    const player2AiHandDiv = document.getElementById('player-2-ai-hand');
    const kittyDisplayDiv = document.getElementById('kitty-display'); // This ID does not exist in HTML, but keeping var for now
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

        animatedCard.style.position = 'fixed';
        animatedCard.style.left = `${cardRect.left}px`;
        animatedCard.style.top = `${cardRect.top}px`;
        animatedCard.style.width = `${cardRect.width}px`;
        animatedCard.style.height = `${cardRect.height}px`;
        animatedCard.style.zIndex = '1000';

        cardElement.style.opacity = '0.3';

        const targetX = targetRect.left + window.scrollX;
        const targetY = targetRect.top + window.scrollY;

        requestAnimationFrame(() => {
            animatedCard.style.transition = 'left 0.5s ease-out, top 0.5s ease-in, width 0.5s ease-in-out, height 0.5s ease-in-out';
            animatedCard.style.left = `${targetX}px`;
            animatedCard.style.top = `${targetY}px`;
        });

        return new Promise(resolve => {
            animatedCard.addEventListener('transitionend', () => {
                animatedCard.remove();
                cardElement.style.opacity = '1';
                resolve();
            }, { once: true });
        });
    }

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

    function handleCardClick(cardElement, cardData) {
        const phase = clientGameState.game_phase;
        const isMyTurn = clientGameState.current_player_turn === 0;

        const isHumanDealerMustDiscard = phase === 'dealer_must_discard_after_order_up' && isMyTurn && clientGameState.dealer === 0;
        const isHumanMakerInvolvedDiscard = (phase === 'dealer_discard_one' || phase === 'maker_discard') && isMyTurn && clientGameState.maker === 0;

        if (isHumanDealerMustDiscard || isHumanMakerInvolvedDiscard) {
            const neededToDiscard = clientGameState.cards_to_discard_count; // Should be 1 or 5
            const indexInSelection = selectedCardsForDiscard.findIndex(c => c.rank === cardData.rank && c.suit === cardData.suit);

            if (indexInSelection > -1) { // Card is already selected, de-select it
                selectedCardsForDiscard.splice(indexInSelection, 1);
                cardElement.classList.remove('selected');
            } else { // Card not selected
                if (neededToDiscard === 1) { // Special handling for single card selection: auto-swap
                    if (selectedCardsForDiscard.length === 1) {
                        // Deselect the currently selected card
                        const currentlySelectedCardElement = document.querySelector('#player-0-you-hand .card.selected');
                        if (currentlySelectedCardElement) {
                            currentlySelectedCardElement.classList.remove('selected');
                        }
                        selectedCardsForDiscard = []; // Clear selection
                    }
                    selectedCardsForDiscard.push({ rank: cardData.rank, suit: cardData.suit, name: cardData.name });
                    cardElement.classList.add('selected');
                } else { // For multi-card discard (i.e., maker_discard for 5 cards)
                    if (selectedCardsForDiscard.length < neededToDiscard) {
                        selectedCardsForDiscard.push({ rank: cardData.rank, suit: cardData.suit, name: cardData.name });
                        cardElement.classList.add('selected');
                    } else {
                        gameMessageP.textContent = `You can only select ${neededToDiscard} card(s) to discard.`;
                        setTimeout(() => { if(clientGameState && clientGameState.message) gameMessageP.textContent = clientGameState.message; }, 2000);
                    }
                }
            }
            updateSelectedCardsDisplay();
        } else if (phase === 'playing_tricks' && isMyTurn) {
            // Single card selection for playing a trick
            document.querySelectorAll('#player-0-you-hand .card.selected').forEach(el => el.classList.remove('selected'));
            cardElement.classList.add('selected');
            selectedCardsForDiscard = [{ rank: cardData.rank, suit: cardData.suit, name: cardData.name }];
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
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            gameMessageP.textContent = `Error during AI turn: ${data.error}. Refresh if stuck.`;
                            if(data.game_state) renderBoard(data.game_state);
                        } else {
                            renderBoard(data);
                        }
                    })
                    .catch(error => {
                        gameMessageP.textContent = "Connection error during AI turn. Refresh if stuck.";
                    });
            }, 750);
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

    function renderPreviousTrickLog(lastTrickData) {
        if (!previousTrickLogDiv) return;
        previousTrickLogDiv.innerHTML = '';
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
            cardElement.style.width = '40px';
            cardElement.style.height = '60px';
            cardElement.style.fontSize = '0.6em';
            cardElement.style.cursor = 'default';
            const playerIdentifier = document.createElement('span');
            playerIdentifier.classList.add('played-card-info');
            let playerName = `P${parseInt(playedInfo.player) + 1}`;
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
        try {
            const response = await fetch('/api/start_game');
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, body: ${errorText}`);
            }
            const data = await response.json();
            renderBoard(data);
        } catch (error) {
            gameMessageP.textContent = `Error connecting to server: ${error.message}. Click 'Start New Game / Round'.`;
            renderBoard({});
        }
    }

    async function submitPlayerAction(action, details = {}) {
        const payload = { player_index: 0, action: action, ...details };
        let cardToAnimateElement = null;
        if (action === 'play_card' && details.card) {
            const player1Hand = document.getElementById('player-1-hand');
            cardToAnimateElement = Array.from(player1Hand.children).find(cardEl =>
                cardEl.dataset.rank === details.card.rank && cardEl.dataset.suit === details.card.suit
            );
        }
        try {
            if (cardToAnimateElement) {
                playCardButton.disabled = true;
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
            renderBoard(data);
        } catch (error) {
            gameMessageP.textContent = `Action failed: ${error.message}. Check console.`;
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
        const count = clientGameState.cards_to_discard_count;

        if (phase === 'dealer_discard_one') {
            if (selectedCardsForDiscard.length === 1 && count === 1) {
                submitPlayerAction('dealer_discard_one', { cards: selectedCardsForDiscard });
            } else {
                gameMessageP.textContent = `Please select exactly 1 card to discard.`;
                 setTimeout(() => { if(clientGameState && clientGameState.message) gameMessageP.textContent = clientGameState.message; }, 2000);
            }
        } else if (phase === 'dealer_must_discard_after_order_up') {
            if (selectedCardsForDiscard.length === 1 && count === 1) {
                submitPlayerAction('dealer_must_discard_after_order_up', { cards: selectedCardsForDiscard });
            } else {
                gameMessageP.textContent = `Please select exactly 1 card to discard.`;
                setTimeout(() => { if(clientGameState && clientGameState.message) gameMessageP.textContent = clientGameState.message; }, 2000);
            }
        } else if (phase === 'maker_discard') {
            if (selectedCardsForDiscard.length === 5 && count === 5) {
                submitPlayerAction('maker_discard', { cards: selectedCardsForDiscard });
            } else {
                gameMessageP.textContent = `Please select exactly 5 cards to discard.`;
                 setTimeout(() => { if(clientGameState && clientGameState.message) gameMessageP.textContent = clientGameState.message; }, 2000);
            }
        } else if (phase === 'playing_tricks') {
            if (selectedCardsForDiscard.length === 1) {
                submitPlayerAction('play_card', { card: selectedCardsForDiscard[0] });
            } else {
                gameMessageP.textContent = "Please select a card to play.";
            }
        }
    });

    function saveStateToLocalStorage() {
        try {
            localStorage.setItem(EUCHRE_GAME_STATE_KEY, JSON.stringify(clientGameState));
            localStorage.setItem(EUCHRE_GAME_STATS_KEY, JSON.stringify(gameStatistics));
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
            } else {
                gameStatistics = { wins: 0, losses: 0 };
            }
            if (statsWinsSpan) statsWinsSpan.textContent = gameStatistics.wins;
            if (statsLossesSpan) statsLossesSpan.textContent = gameStatistics.losses;

            if (savedGameState) {
                const parsedGameState = JSON.parse(savedGameState);
                if (parsedGameState && typeof parsedGameState === 'object' && parsedGameState.hasOwnProperty('game_phase')) {
                    clientGameState = parsedGameState;
                    return true;
                } else {
                    localStorage.removeItem(EUCHRE_GAME_STATE_KEY);
                    clientGameState = {};
                }
            } else {
                clientGameState = {};
            }
        } catch (e) {
            localStorage.removeItem(EUCHRE_GAME_STATE_KEY);
            localStorage.removeItem(EUCHRE_GAME_STATS_KEY);
            clientGameState = {};
            gameStatistics = { wins: 0, losses: 0 };
        }
        return false;
    }

    async function initializeGame() {
        loadStateFromLocalStorage();

        if (clientGameState && clientGameState.game_phase &&
            !["game_over", "round_over", "setup"].includes(clientGameState.game_phase)) {
            try {
                const response = await fetch('/api/get_current_state');
                if (!response.ok) {
                    throw new Error(`Server status: ${response.status}`);
                }
                const serverGameState = await response.json();
                if (serverGameState && serverGameState.game_phase) {
                    if (serverGameState.game_phase === "setup") {
                        renderBoard({});
                    } else {
                        renderBoard(serverGameState);
                    }
                } else {
                    renderBoard({});
                }
            } catch (error) {
                renderBoard({});
            }
        } else {
            renderBoard({});
        }
    }
    initializeGame();
});
