/**
 * GameLogger - "Black Box" recorder for Mau Mau
 * Captures game state snapshots for debugging AI behavior.
 */
export class GameLogger { // Turbo-all
    constructor() {
        this.history = [];
    }

    /**
     * Log the state before a turn
     * @param {Object} gameState - From engine.getGameState()
     * @param {Object} hands - Full hands object { player: [], ai: [] }
     * @param {String} actionTaken - Optional description of action about to be taken (or taken)
     */
    logTurn(gameState, hands, actionTaken = "") {
        const snapshot = {
            turn: this.history.length + 1,
            timestamp: new Date().toISOString(),
            activePlayer: gameState.currentPlayer,
            topCard: gameState.topCard,
            activeSuitWished: gameState.wishedSuit,
            gameDrawCount: gameState.drawCount,
            skipNext: gameState.skipNext,

            // Full state visibility
            aiHand: JSON.parse(JSON.stringify(hands.ai)), // Deep copy
            playerHandCount: hands.player.length,

            // Optional: Log player hand too if needed for full replay
            playerHand: JSON.parse(JSON.stringify(hands.player)),

            actionTaken: actionTaken
        };

        this.history.push(snapshot);
    }

    exportLog() {
        return this.history;
    }

    printLog() {
        console.group("/// GAME LOG EXPORT ///");
        console.table(this.history.map(entry => ({
            turn: entry.turn,
            activeInfo: `${entry.activePlayer} (Wish: ${entry.activeSuitWished || '-'})`,
            topCard: `${entry.topCard.suit} ${entry.topCard.value}`,
            aiHand: entry.aiHand.map(c => c.suit.charAt(0) + c.value).join(','),
            action: entry.actionTaken
        })));
        console.log("Full JSON (Copy Object below):");
        console.log(JSON.stringify(this.history, null, 2));
        console.groupEnd();
    }
}
