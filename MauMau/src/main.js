import { MauMauEngine } from './logic/engine.js';
import { CardRenderer } from './components/cardResult.js';
import { GameLogger } from './logic/GameLogger.js';
import { SoundManager } from './logic/SoundManager.js';

// Global Game Instance (exposed for HTML onclick handlers)
window.game = {
    engine: new MauMauEngine(),
    logger: new GameLogger(),
    sound: new SoundManager(),
    mauClicked: false,
    wishSelectionMode: false,
    pendingWishHandIndex: null, // Index of the Jack that triggered the wish
    currentView: 'view-start',

    switchView: function (viewId) {
        // Hide all views
        document.querySelectorAll('.view-screen').forEach(el => el.classList.add('hidden'));
        // Show target view
        const target = document.getElementById(viewId);
        if (target) {
            target.classList.remove('hidden');
            this.currentView = viewId;
        } else {
            console.error(`View ${viewId} not found!`);
        }
    },

    reboot: function () {
        this.switchView('view-start');
        // Optional: Reset internal state if needed, but 'init' handles game state.
    },

    init: function (startingPlayer = null, rules = {}) {
        document.getElementById('game-log').innerHTML = '';
        this.resetMau();

        // Start Game Music
        this.sound.startMusic('game');

        // Ensure Game View is active
        this.switchView('view-game');

        this.engine.initGame(rules);

        const starter = startingPlayer || (Math.random() < 0.5 ? 'player' : 'ai');
        this.engine.currentPlayer = starter;

        // Log Initial State
        this.logger.logTurn(
            this.engine.getGameState(),
            this.engine.hands,
            `Game Start: ${starter}`
        );

        const starterName = starter === 'player' ? 'Human' : 'AI';
        const startMsg = startingPlayer
            ? `${startingPlayer === 'player' ? 'Human' : 'AI'} won last game. ${starterName} starts.`
            : `Game started. Random pick: ${starterName} starts.`;

        this.log(startMsg);

        this.render();

        // Special Start Card Rules
        const top = this.engine.getGameState().topCard;

        if (top.value === '8') {
            this.log(`Start card is 8. ${starterName} is skipped!`, 'log-skip');
            const next = starter === 'player' ? 'ai' : 'player';
            this.passTurnTo(next);
        } else if (top.value === '7') {
            // FIX for "Seven on Start" bug
            this.engine.drawCount = 2;
            this.log(`Start card is 7! ${starterName} must draw 2 or play 7.`, 'log-seven');
            // Turn proceeds to starter normally, but they face the penalty.
            if (starter === 'ai') setTimeout(() => this.aiTurn(), 1000);
        } else if (top.value === 'B') {
            // FIX for "Jack on Start": 
            // The starting player determines the suit (as if they played it).
            // Logic: They can PLAY a card to set suit, or USE MENU to set suit.

            if (starter === 'player') {
                this.log(`Start card is Jack! You decide the suit (Play or Select).`, 'log-seven');
                this.showWishModal(0, true);
            } else {
                // AI Logic: AI just plays its best card.
                // Engine now allows ANY card on Start-Jack, so AI will pick best.
                this.log(`Start card is Jack! AI will determine suit.`, 'log-seven');
                setTimeout(() => this.aiTurn(), 1000);
            }
        } else {
            // Normal Start
            if (starter === 'ai') setTimeout(() => this.aiTurn(), 1000);
        }
    },

    // New: Handle AI Wish at start
    aiStartWish: function () {
        // AI picks best suit from hand (simple logic: most cards of suit)
        const hand = this.engine.hands.ai;
        const counts = { 'Pik': 0, 'Kreuz': 0, 'Herz': 0, 'Karo': 0 };
        hand.forEach(c => counts[c.suit]++);

        // Find suit with max count
        let bestSuit = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);

        this.engine.wishedSuit = bestSuit;
        const suitSpan = `<span class="text-${bestSuit}">${bestSuit}</span>`;
        this.log(`AI chooses ${suitSpan} for Start-Jack.`, 'log-seven');

        // Update UI then AI takes its normal turn (now playing into the wish)
        this.render();
        setTimeout(() => this.aiTurn(), 1000);
    },

    // New: Handle Player Wish from Modal
    // New: Handle Player Wish from Modal
    resolveStartWish: function (suit) {
        // User clicked the "Suit" button instead of playing a card.
        // This sets the Wish Constraint, then they must play.
        this.clearWishPanel();
        this.engine.wishedSuit = suit;
        const suitSpan = `<span class="text-${suit}">${suit}</span>`;
        this.log(`Human chooses ${suitSpan} (Constraint set). Play a card!`, 'log-seven');
        this.render();
    },

    // ... toggleMau / resetMau ...
    toggleMau: function () {
        if (this.engine.currentPlayer !== 'player') return;
        this.mauClicked = !this.mauClicked;
        const btn = document.getElementById('mau-btn');
        if (this.mauClicked) {
            btn.classList.add('active');
            this.log("Human says: MAU!", "log-skip");
            this.sound.play('mau');
        } else {
            btn.classList.remove('active');
        }
    },

    resetMau: function () {
        this.mauClicked = false;
        const btn = document.getElementById('mau-btn');
        if (btn) btn.classList.remove('active');
    },

    /**
     * User Action: Play Card
     */
    playCard: function (handIndex) {
        if (this.engine.currentPlayer !== 'player') {
            this.log("Wait for AI turn!");
            return;
        }

        const hand = this.engine.hands.player;
        const card = hand[handIndex];
        const state = this.engine.getGameState();

        // --- WISH SELECTION MODE INTERCEPT ---
        // --- WISH SELECTION MODE INTERCEPT ---
        if (this.wishSelectionMode) {
            // Special: If this is Start-Jack mode (pendingWishHandIndex === -1)
            // AND the user clicked a card, they are "Choosing by Playing".
            if (this.pendingWishHandIndex === -1) {
                // Implicit Wish by Playing
                this.clearWishPanel();
                // Check Validation happens below naturally (Engine allows any card if wishing is null)
                // If they clicked a button previously, 'engine.wishedSuit' is set, so validation checks that.
                // If they didn't, 'engine.wishedSuit' is null, so validation allows any.
            } else {
                // Normal Combo Mode (Jack Just Played)
                // Must Resolve via Combo
                this.executeCombo(this.pendingWishHandIndex, handIndex);
                return;
            }
        }
        // -------------------------------------

        // Objective 5: Mau Check (Before Validation)
        // Rule: If playing this card leaves 0 cards (i.e. winning move), check Mau.
        // Hand size check: If I have 1 card now, playing it means I have 0.
        if (hand.length === 1) {
            if (!this.mauClicked) {
                // Chance to fail
                // "Low chance that forgetting is not noticed" = High chance of penalty.
                // Let's say 20% chance to get away with it.
                if (Math.random() > 0.2) {
                    this.log("⚠️ FORGOT TO SAY MAU! Penalty!", "log-skip-2");
                    this.drawCardPenalty('player');
                    return; // Turn ends immediately
                } else {
                    this.log("(Lucky! Forgot Mau but nobody noticed...)", "log-skip");
                }
            }
        }

        // Validation with UI Feedback
        const isValid = this.engine.isValidMove(card);
        const top = this.engine.getGameState().topCard;

        if (!isValid) {
            // Enhanced Debugging for "Jack on 8" bug report
            console.warn(`Invalid Move Debug: Card=${card.suit} ${card.value}, Top=${top.suit} ${top.value}`);
            console.warn(`GameState: SkipNext=${this.engine.skipNext}, DrawCount=${this.engine.drawCount}, Wish=${state.wishedSuit}`);

            if (state.wishedSuit && card.suit !== state.wishedSuit && card.value !== 'B') {
                const suitSpan = `<span class="text-${state.wishedSuit}">${state.wishedSuit}</span>`;
                this.log(`Invalid! Must follow wish: ${suitSpan}`, 'log-seven');
            } else if (state.topCard.value === 'B' && card.value === 'B') {
                this.log("Invalid! 'Bube auf Bube' is forbidden!", 'log-seven');
            } else {
                // Generic error - try to explain why
                let reason = "Rules violation.";
                if (this.engine.skipNext) reason = "You are skipped!";
                if (this.engine.drawCount > 0 && card.value !== '7') reason = "Must play 7 or draw!";

                this.log(`Invalid move: ${card.suit} ${card.value} (${reason})`, 'log-seven');
            }
            return;
        }

        if (card.value === 'B') {
            // Winning check for Jack
            if (hand.length === 1) {
                this.executeMove('player', handIndex, null);
                return;
            }
            this.showWishModal(handIndex);

            // Auto Select Suggestion would be nice, but user wants to CLICK card.
            // Modal is shown, but we also enable clicking hand cards.
            return;
        }

        this.executeMove('player', handIndex, null);
    },

    drawCardPenalty: function (player) {
        this.engine.drawCard(player, 1);
        if (player === 'player') this.log(`Human draws 1 penalty card`, 'log-draw');
        else this.log(`AI draws 1 penalty card`, 'log-draw');

        this.sound.play('draw');

        if (player === 'player') this.resetMau();

        const next = player === 'player' ? 'ai' : 'player';
        this.passTurnTo(next);
    },

    /**
     * Internal: Execute the move
     */
    executeMove: function (player, index, wishSuit) {
        const hand = this.engine.hands[player];
        const card = hand[index];
        const prevTop = this.engine.getGameState().topCard;

        this.engine.playCard(player, index, wishSuit);

        // --- AUDIO TRIGGER ---
        const chain = this.engine.drawCount > 0 ? (this.engine.drawCount / 2) - 1 : 0; // Simple chain logic

        if (card.value === '7') {
            this.sound.play('seven', chain);
        } else if (card.value === '8') {
            this.sound.play('eight');
        } else if (card.value === 'B') {
            this.sound.play('jack');
        } else {
            this.sound.play('play');
        }
        // ---------------------

        let logClass = '';
        if (card.value === '7') logClass = 'log-seven';
        if (card.value === '7' && prevTop.value === '7') logClass += ' log-pulse';

        const actorName = player === 'player' ? 'Human' : 'AI';

        // Log Refinement: Do NOT color suit unless it's a wish
        // Just text for normal play
        let msg = `${actorName} plays ${card.suit} ${card.value}`;

        if (wishSuit) {
            const wishSpan = `<span class="text-${wishSuit}">${wishSuit}</span>`;
            msg += ` and wishes for ${wishSpan}`;
        }

        this.log(msg, logClass);

        this.nextTurn(player);
    },

    resolveWish: function (handIndex, suit) {
        // Fallback for button clicks (if user uses menu instead of combo)
        this.clearWishPanel();
        this.executeMove('player', handIndex, suit);
    },

    executeCombo: function (jackIndex, cardIndex) {
        this.clearWishPanel();

        // 1. Get Card Objects
        const hand = this.engine.hands.player;
        const jack = hand[jackIndex];
        const second = hand[cardIndex];

        // 2. Play Jack (Sets Wish)
        // We must handle index shifting. 
        // Always remove valid indices. If cardIndex > jackIndex, it shifts down.

        // Play Jack first
        // Note: engine.playCard DOES splice the hand. 
        this.engine.playCard('player', jackIndex, second.suit);

        // Log Jack
        this.log(`Human plays ${jack.suit} ${jack.value} (Combo)...`, 'log-seven');

        // 3. Play Second Card
        let adjCardIndex = cardIndex;
        if (cardIndex > jackIndex) adjCardIndex--;

        if (adjCardIndex >= 0 && adjCardIndex < this.engine.hands.player.length) {
            this.engine.playCard('player', adjCardIndex, null);
            this.log(`...and follows with ${second.suit} ${second.value}`, 'log-human');
        } else {
            console.error("Combo Error: Index out of bounds", cardIndex, jackIndex);
        }

        // 4. Update UI & Turn
        this.nextTurn('player'); // Checks winning condition too
        this.render();
    },

    showWishModal: function (handIndex, isStart = false) {
        this.wishSelectionMode = true;
        this.pendingWishHandIndex = isStart ? -1 : handIndex;

        const panel = document.getElementById('wish-panel');
        const suits = ['Pik', 'Kreuz', 'Herz', 'Karo'];
        const symbols = { 'Pik': '^', 'Kreuz': '+', 'Herz': '<3', 'Karo': '<>' };

        // If isStart, we call resolveStartWish. Else resolveWish(handIndex, ...)
        let buttonsHtml = suits.map(s => {
            const clickHandler = isStart
                ? `game.resolveStartWish('${s}')`
                : `game.resolveWish(${handIndex}, '${s}')`;

            return `
            <div class="suit-btn" data-suit="${s}" onclick="${clickHandler}">
                ${symbols[s]}
            </div>
            `;
        }).join('');

        const prompt = "SELECT SUIT (CLICK BUTTON OR CARD IN HAND)";

        panel.innerHTML = `
            <div style="width:100%; text-align:center;">
                <div style="color:#7dcfff; margin-bottom:10px;">${prompt}</div>
                <div class="suit-selector">
                    ${buttonsHtml}
                </div>
            </div>
        `;
    },

    clearWishPanel: function () {
        this.wishSelectionMode = false;
        this.pendingWishHandIndex = null;
        document.getElementById('wish-panel').innerHTML =
            '<div class="wish-placeholder">WAITING_FOR_INPUT...</div>';
    },



    drawCard: function () {
        if (this.engine.currentPlayer !== 'player') return;

        const state = this.engine.getGameState();
        const amount = state.drawCount > 0 ? state.drawCount : 1;

        this.engine.drawCard('player', amount);
        this.log(`Human draws <span class="log-draw">+${amount}</span> cards`);
        this.sound.play('draw');

        this.resetMau();
        this.passTurnTo('ai');
    },

    passTurnTo: function (nextPlayer) {
        // Log turn handover (covers Plays and Draws)
        this.logger.logTurn(
            this.engine.getGameState(),
            this.engine.hands,
            `Turn Pass: ${nextPlayer}`
        );

        // this.resetMau(); // REMOVED: Mau status should persist until draw or game end
        this.engine.currentPlayer = nextPlayer;
        this.render();

        if (nextPlayer === 'ai') {
            setTimeout(() => this.aiTurn(), 1000);
        }
    },

    nextTurn: function (justPlayed) {
        const nextPlayer = justPlayed === 'player' ? 'ai' : 'player';

        if (this.engine.hands[justPlayed].length === 0) {
            this.showGameOver(justPlayed);
            return;
        }



        if (this.engine.skipNext) {
            this.engine.skipNext = false;
            const skippedName = nextPlayer === 'player' ? 'Human' : 'AI';
            this.log(`${skippedName} is skipped!`, 'log-skip');
            this.passTurnTo(justPlayed);
            return;
        }

        this.passTurnTo(nextPlayer);
    },

    aiTurn: function () {
        if (this.engine.currentPlayer !== 'ai') return;

        // AI Logic for Mau Checks
        if (this.engine.hands.ai.length === 1) {
            if (Math.random() < 0.05) { // 5% chance to forget
                if (Math.random() > 0.2) {
                    this.log("AI forgot MAU! Penalty!", "log-skip-2");
                    this.drawCardPenalty('ai');
                    return;
                }
            } else {
                this.log("AI says: MAU", "log-skip");
                this.sound.play('mau');
            }
        }

        const moves = this.engine.getValidMoves('ai');

        if (moves.length > 0) {
            let selectedMove = null;
            const state = this.engine.getGameState();

            // 1. STRATEGY: Prioritize Active Wish
            if (state.wishedSuit) {
                // Find moves that match the wish (Suit Match)
                // Note: isValidMove returns true for Jack too, but we want to comply if possible with a suit card.
                // But playing a Jack on a Wish is also valid.
                // Let's filter for exact suit match first to "comply".
                const suitMatches = moves.filter(m => m.card.suit === state.wishedSuit && m.card.value !== 'B');

                if (suitMatches.length > 0) {
                    selectedMove = suitMatches[0]; // Pick first matching suit
                }
            }

            // 2. STRATEGY: Prioritize Jacks or Rank Matches over random?
            // If no wish-match found (or no wish active), pick best valid move.
            if (!selectedMove) {
                // Heuristic: Try to save Jacks? Or play them?
                // Let's simple-sort: Non-Jacks first, then Jacks.
                const nonJacks = moves.filter(m => m.card.value !== 'B');
                if (nonJacks.length > 0) {
                    selectedMove = nonJacks[0];
                } else {
                    selectedMove = moves[0]; // Only Jacks or whatever is left
                }
            }

            // Fallback (redundant but safe)
            if (!selectedMove) selectedMove = moves[0];

            // 3. WISH LOGIC (If playing Jack)
            let wish = null;
            if (selectedMove.card.value === 'B') {
                // Count suits in remaining hand (excluding the played Jack)
                const counts = { 'Pik': 0, 'Kreuz': 0, 'Herz': 0, 'Karo': 0 };
                this.engine.hands.ai.forEach((c, i) => {
                    if (i !== selectedMove.index) { // Index check is safer than object reference
                        counts[c.suit]++;
                    }
                });

                // Find suit with max count
                // If tie or empty, defaults to first found (e.g. Pik) or logic
                wish = Object.keys(counts).reduce((a, b) => counts[a] >= counts[b] ? a : b);
            }

            this.executeMove('ai', selectedMove.index, wish);

        } else {
            const state = this.engine.getGameState();
            const amount = state.drawCount > 0 ? state.drawCount : 1;
            this.engine.drawCard('ai', amount);
            this.log(`AI draws <span class="log-draw">+${amount}</span> cards`);
            this.sound.play('draw');
            this.passTurnTo('player');
        }
    },

    render: function () {
        const state = this.engine.getGameState();

        document.getElementById('discard-pile').innerHTML = CardRenderer.getHTML(state.topCard);

        const playerContainer = document.getElementById('player-hand');
        playerContainer.innerHTML = '';
        this.engine.hands.player.forEach((card, idx) => {
            const div = document.createElement('div');
            const isInteractive = state.currentPlayer === 'player';
            div.innerHTML = CardRenderer.getHTML(card, isInteractive, idx);
            playerContainer.appendChild(div);
        });

        const aiContainer = document.getElementById('ai-hand');
        aiContainer.innerHTML = '';
        for (let i = 0; i < state.handSizeAI; i++) {
            const div = document.createElement('div');
            div.innerHTML = CardRenderer.getBacksideHTML();
            aiContainer.appendChild(div);
        }

        let statusText = `Turn: ${state.currentPlayer === 'player' ? 'HUMAN' : 'AI'}`;
        if (state.wishedSuit) statusText += ` | WISH: ${state.wishedSuit}`;
        if (state.drawCount > 0) statusText += ` | DRAW STACK: ${state.drawCount}`;

        document.getElementById('status-bar').innerText = statusText;
    },

    log: function (msg, classes = '') {
        const el = document.getElementById('game-log');
        const line = document.createElement('div');
        line.className = `log-entry ${classes}`;

        const time = new Date().toLocaleTimeString();
        if (msg.includes('Human')) {
            msg = msg.replace('Human', '<span class="log-human">Human</span>');
        }

        line.innerHTML = `<span style="opacity:0.5">[${time}]</span> ${msg}`;
        el.prepend(line);
        if (el.children.length > 20) el.lastChild.remove();
    },

    showGameOver: function (winner) {
        this.sound.playWin(winner); // Play Win Sound & Stop Music

        if (winner === 'ai') {
            this.switchView('view-end-ai');
        } else {
            this.switchView('view-end-human');
        }

        // Auto-Restart after 8 seconds (to let Audio finish)
        setTimeout(() => {
            console.log("Auto-restarting game...");
            this.logger.printLog(); // Export Log on Game Over
            this.restartGame(winner);
        }, 8000);
    },

    restartGame: function (winner) {
        // PER OBJECTIVE: Start Screen only appears on first load.
        // Subsequent games restart directly to the board.
        this.switchView('view-game');

        const currentRules = this.engine.rules;
        this.init(winner, currentRules);
    },

    startGame: function () {
        // Get Rules from UI
        const jackOnJackForbidden = document.getElementById('rule-jack-on-jack').checked;

        const rules = {
            jackOnJackForbidden: jackOnJackForbidden
        };

        // Switch Logic handled by init() -> switchView('view-game')
        // But strictly, we should switch here? 
        // init() calls switchView('view-game') so we are good.

        // Init Game
        this.init(null, rules);
    }
};

// Event Listener for Start Button
document.getElementById('btn-start-game').addEventListener('click', () => {
    window.game.startGame();
});

// Update init to accept rules
// We need to patch the init function inside the object above. 
// Since I am replacing the END of the file, I can't easily patch the middle `init` function without a massive replace.
// Let's use MultiReplace or just override it here if I could.
// Wait, I should use MultiReplace for the whole file to be clean. 
// BUT, I can't request MultiReplace inside this tool call.
// FOR NOW: I will just finish the object definition and add the init call.
// ACTUALLY: The User Request was to "Start Screen ... only appear when first game begins".
// So `window.game.init()` at the bottom should be REMOVED.
// And `window.game.init` method needs to accept rules.

// Implementation note: I need to use MultiReplace to properly update `init` signature and the bottom of the file.
// The current tool call is `replace_file_content`. I made a mistake using `replace_file_content` for a complex logic change that spans the file.
// I will cancel this tool call effectively by doing a minimal change? No, verification will fail.
// I will rewrite this effectively to ONLY add the listener/bottom logic, and use a separate call for the `init` signature.
// OR I can try to do the right thing, but I can't see the top of the file here.
// I will perform the bottom change (remove auto init) and add the listener.
// Then I will do a second ID to update `init`.

// --- INITIALIZATION & EVENT BINDING ---

try {
    console.log("System Ready. Waiting for user start.");

    // 1. Audio Toggle Logic
    const soundToggle = document.getElementById('sound-toggle');
    if (soundToggle) {
        // Sync Game State with UI State (UI defaults to Unchecked/OFF)
        window.game.sound.setEnabled(soundToggle.checked);

        soundToggle.addEventListener('change', (e) => {
            const isEnabled = e.target.checked;
            window.game.sound.setEnabled(isEnabled);

            // If turned ON, play appropriate music
            if (isEnabled) {
                if (window.game.currentView === 'view-start') {
                    window.game.sound.startMusic('menu');
                } else if (window.game.currentView === 'view-game') {
                    window.game.sound.startMusic('game');
                }
            }
        });
    }

    // 2. Start Music Init (if enabled)
    // Attempt playback if enabled (e.g. if user checked it and reloaded, though browsers usually reset inputs on hard reload, soft reload might keep it)
    if (window.game.sound.enabled && window.game.currentView === 'view-start') {
        window.game.sound.startMusic('menu');
    }

    // 3. Fallback Click Listener for Autoplay
    document.addEventListener('click', () => {
        // Only try to start music if it's supposed to be on but generated an error or hasn't started
        if (window.game.sound.enabled && window.game.currentView === 'view-start') {
            // startMusic checks internally if already playing, so safe to call
            window.game.sound.startMusic('menu');
        }
    });

} catch (e) {
    document.getElementById('game-log').innerHTML = `<div class="log-entry" style="color:red">ERROR: ${e.message}</div>`;
    console.error(e);
}
