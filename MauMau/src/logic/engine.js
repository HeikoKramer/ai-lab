/**
 * Mau Mau Core Game Engine
 * Handles deck management, rules, validation, and game state.
 */

export const SUITS = ['Pik', 'Kreuz', 'Herz', 'Karo'];
export const VALUES = ['7', '8', '9', '10', 'B', 'D', 'K', 'A'];

export class MauMauEngine {
    constructor() {
        this.deck = [];
        this.discardPile = [];
        this.hands = {
            player: [],
            ai: []
        };
        this.currentPlayer = 'player'; // 'player' or 'ai'

        // Game State Flags
        this.drawCount = 0; // Accumulated draw count from 7s
        this.skipNext = false; // If 8 was played
        this.wishedSuit = null; // If Jack was played
    }

    /**
     * Initialize a new game.
     */
    initGame() {
        // Reset State
        this.discardPile = [];
        this.drawCount = 0;
        this.skipNext = false;
        this.wishedSuit = null;
        this.hands = { player: [], ai: [] };

        this.deck = this.createDeck();
        this.shuffleDeck();
        this.dealCards();

        // Start discard pile with one card
        let firstCard = this.deck.pop();
        this.discardPile.push(firstCard);

        // Handle special case (if first card is special)
        // For simplicity: No effects on start card yet.
    }

    createDeck() {
        let deck = [];
        for (let suit of SUITS) {
            for (let value of VALUES) {
                deck.push({ suit, value });
            }
        }
        console.log(`[Engine] Deck created. Size: ${deck.length} cards.`);
        return deck;
    }

    shuffleDeck() {
        // Fisher-Yates Shuffle
        for (let i = this.deck.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [this.deck[i], this.deck[j]] = [this.deck[j], this.deck[i]];
        }
    }

    dealCards() {
        this.hands.player = [];
        this.hands.ai = [];

        // Deal 5 cards each (classic rule vary, 5 or 6 is standard)
        for (let i = 0; i < 5; i++) {
            this.hands.player.push(this.deck.pop());
            this.hands.ai.push(this.deck.pop());
        }
    }

    /**
     * Check if a move is valid.
     * @param {Object} card - The card attempting to be played
     * @param {string} [wishedSuit] - If playing a Jack, the suit wishing for (optional validation context)
     */
    isValidMove(card, isJackWish = null) {
        const topCard = this.discardPile[this.discardPile.length - 1];

        // 1. Check strict 7 punishment state
        // If drawCount > 0, player MUST play a 7 or draw.
        if (this.drawCount > 0) {
            if (card.value === '7') return true;
            return false;
        }

        // 2. Check 8 skip state
        // (Usually handled by skipping the turn before allowing a move, but if we are here, it's a move attempt)
        if (this.skipNext) {
            // Player should have been skipped. 
            // This suggests engine flow issue if we reach here, usually skip is auto-resolved.
            return false;
        }

        // 3. Jack Rules

        // Prevent Jack on Jack (if top card is Jack, you cannot play a Jack)
        // Note: topCard is Jack allows non-Jack moves normally, but here we strictly forbid playing Jack on Jack.
        // Wait, standard rules say: "Bube auf Bube stinkt" -> forbidden.
        if (topCard.value === 'B' && card.value === 'B') {
            return false;
        }

        // If a suit is wished (from previous Jack), allow only that suit OR another Jack? 
        // Spec says "Außer Bube auf Bube". So if wish is active, and I have a Jack, can I play it?
        // No, because that would be Jack on Jack.
        // So if wishedSuit is active, effectively top card is a virtual "Jack of wishedSuit".
        // My previous logic allowed Jack on Wish. 
        // Correct logic based on spec "außer Bube auf Bube":
        // If Jack is played, wish is made. Next player must follow suit. Next player CANNOT play Jack.

        if (this.wishedSuit) {
            if (card.value === 'B') return false; // explicit forbid Jack on Wish
            if (card.suit === this.wishedSuit) return true;
            return false;
        }

        // Normal play (no wish active): Jack allowed on anything (except on Jack, handled above)
        if (card.value === 'B') {
            return true;
        }

        // 4. Standard Matching
        if (card.suit === topCard.suit || card.value === topCard.value) {
            return true;
        }

        return false;
    }

    /**
     * Execute a move. Assumes validation passed.
     */
    playCard(player, cardIndex, wishSuit = null) {
        let hand = this.hands[player];
        let card = hand[cardIndex];

        // Remove from hand
        hand.splice(cardIndex, 1);

        // Add to discard
        this.discardPile.push(card);

        // Apply Effects
        this.wishedSuit = null; // Reset previous wish

        if (card.value === '7') {
            this.drawCount += 2;
        } else if (card.value === '8') {
            // In 2 player game, skip means current player goes again? 
            // Or opponent skipped? Standard: Next player skipped.
            this.skipNext = true;
        } else if (card.value === 'B') {
            this.wishedSuit = wishSuit; // AI/Player must specify this
        }

        // Next turn logic handled by game loop / controller
    }

    drawCard(player, amount = 1) {
        for (let i = 0; i < amount; i++) {
            if (this.deck.length === 0) {
                this.reshuffleDiscard();
            }
            if (this.deck.length > 0) {
                this.hands[player].push(this.deck.pop());
            }
        }
        // Reset accumulation if drawing
        if (this.drawCount > 0) {
            this.drawCount = 0;
        }
    }

    reshuffleDiscard() {
        if (this.discardPile.length <= 1) return;

        // Keep top card
        const topCard = this.discardPile.pop();

        // Move rest to deck
        this.deck = [...this.discardPile];
        this.discardPile = [topCard];

        this.shuffleDeck();
    }

    /**
     * Get all valid moves for a specific hand.
     * Useful for AI and UI highlighting.
     */
    getValidMoves(player) {
        const hand = this.hands[player];
        return hand.map((card, index) => {
            return {
                index,
                card,
                valid: this.isValidMove(card)
            };
        }).filter(m => m.valid);
    }

    getGameState() {
        return {
            topCard: this.discardPile[this.discardPile.length - 1],
            handSizePlayer: this.hands.player.length,
            handSizeAI: this.hands.ai.length, // Don't allow peeking AI cards in generic state
            drawCount: this.drawCount,
            wishedSuit: this.wishedSuit,
            currentPlayer: this.currentPlayer,
            skipNext: this.skipNext
        };
    }
}
