import { MauMauEngine } from './logic/engine.js';
import { CardRenderer } from './components/cardResult.js';

// Global Game Instance (exposed for HTML onclick handlers)
window.game = {
    engine: new MauMauEngine(),
    mauClicked: false,

    init: function (startingPlayer = null) {
        document.getElementById('game-log').innerHTML = '';
        this.resetMau();

        this.engine.initGame();

        const starter = startingPlayer || (Math.random() < 0.5 ? 'player' : 'ai');
        this.engine.currentPlayer = starter;

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
        } else if (top.value === 'B') {
            this.log(`Start card is Jack! ${starterName} chooses suit.`, 'log-seven');
            // Force Wish Setup
            // We pretend the Jack was just played "by the deck", 
            // but current player determines the wish.
            if (starter === 'player') {
                // Open Modal but DO NOT play card yet
                // We pass 'null' as index because we aren't playing a card from hand.
                // We need a special mode for "Start Wish".
                this.showWishModal(null, true);
            } else {
                // AI Start Wish
                setTimeout(() => this.aiStartWish(), 1000);
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
    resolveStartWish: function (suit) {
        this.clearWishPanel();
        this.engine.wishedSuit = suit;
        const suitSpan = `<span class="text-${suit}">${suit}</span>`;
        this.log(`Human chooses ${suitSpan} for Start-Jack.`, 'log-seven');
        this.render();
        // Now it's player's turn to play (or draw) 
    },

    // ... toggleMau / resetMau ...
    toggleMau: function () {
        if (this.engine.currentPlayer !== 'player') return;
        this.mauClicked = !this.mauClicked;
        const btn = document.getElementById('mau-btn');
        if (this.mauClicked) {
            btn.classList.add('active');
            this.log("Human says: MAU!", "log-skip");
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
        if (!this.engine.isValidMove(card)) {
            if (state.wishedSuit && card.suit !== state.wishedSuit && card.value !== 'B') {
                const suitSpan = `<span class="text-${state.wishedSuit}">${state.wishedSuit}</span>`;
                this.log(`Invalid! Must follow wish: ${suitSpan}`, 'log-seven');
            } else if (state.topCard.value === 'B' && card.value === 'B') {
                this.log("Invalid! 'Bube auf Bube' is forbidden!", 'log-seven');
            } else {
                this.log(`Invalid move: ${card.suit} ${card.value}`, 'log-seven');
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
            return;
        }

        this.executeMove('player', handIndex, null);
    },

    drawCardPenalty: function (player) {
        this.engine.drawCard(player, 1);
        if (player === 'player') this.log(`Human draws 1 penalty card`, 'log-draw');
        else this.log(`AI draws 1 penalty card`, 'log-draw');

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
        this.clearWishPanel();
        this.executeMove('player', handIndex, suit);
    },

    showWishModal: function (handIndex, isStart = false) {
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

        panel.innerHTML = `
            <div style="width:100%; text-align:center;">
                <div style="color:#7dcfff; margin-bottom:10px;">SELECT WISH_MATRIX:</div>
                <div class="suit-selector">
                    ${buttonsHtml}
                </div>
            </div>
        `;
    },

    clearWishPanel: function () {
        document.getElementById('wish-panel').innerHTML =
            '<div class="wish-placeholder">WAITING_FOR_INPUT...</div>';
    },



    drawCard: function () {
        if (this.engine.currentPlayer !== 'player') return;

        const state = this.engine.getGameState();
        const amount = state.drawCount > 0 ? state.drawCount : 1;

        this.engine.drawCard('player', amount);
        this.log(`Human draws <span class="log-draw">+${amount}</span> cards`);

        this.passTurnTo('ai');
    },

    passTurnTo: function (nextPlayer) {
        this.resetMau(); // Reset Mau state for new turn (or previous turn end)
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
            // Rare chance to forget (e.g. 5%)
            if (Math.random() < 0.05) {
                // AI Forgot!
                // Chance to be caught (same 80% rule?) - or assume Human always notices?
                // Let's use same RNG for fairness
                if (Math.random() > 0.2) {
                    this.log("AI forgot MAU! Penalty!", "log-skip-2");
                    this.drawCardPenalty('ai');
                    return;
                }
            } else {
                this.log("AI says: MAU", "log-skip");
            }
        }

        const moves = this.engine.getValidMoves('ai');

        if (moves.length > 0) {
            const move = moves[0];
            let wish = null;
            if (move.card.value === 'B' && this.engine.hands.ai.length > 1) {
                wish = ['Pik', 'Kreuz', 'Herz', 'Karo'][Math.floor(Math.random() * 4)];
            }
            this.executeMove('ai', move.index, wish);

        } else {
            const state = this.engine.getGameState();
            const amount = state.drawCount > 0 ? state.drawCount : 1;
            this.engine.drawCard('ai', amount);
            this.log(`AI draws <span class="log-draw">+${amount}</span> cards`);
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
        // Clear log immediately? User said "Clear the log when restarting". 
        // End screen should probably persist until restart.
        const layer = document.getElementById('modal-layer');
        if (winner === 'ai') {
            layer.innerHTML = `
                <div class="terminator-overlay">
                    <div class="glitch-text">
                        HUMANITY TERMINATED.<br>
                        SYSTEM OVERRIDE.
                    </div>
                </div>
            `;
            setTimeout(() => this.restartGame(winner), 5000);
        } else {
            layer.innerHTML = `
            <div class="modal-activator" style="border-color: #9ece6a;">
                <div class="modal-title" style="color: #9ece6a;">YOU WIN!</div>
                <div style="color: #a9b1d6;">HACK SUCCESSFUL. SYSTEMS COMPROMISED.</div>
            </div>
            `;
            setTimeout(() => this.restartGame(winner), 5000);
        }
    },

    restartGame: function (winner) {
        document.getElementById('modal-layer').innerHTML = '';
        this.init(winner);
    }
};

try {
    window.game.init();
} catch (e) {
    document.getElementById('game-log').innerHTML = `<div class="log-entry" style="color:red">ERROR: ${e.message}</div>`;
    console.error(e);
}
