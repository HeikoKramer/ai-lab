# Player Behavior & AI Logic

## AI Decision Tree
The AI follows a prioritized decision making process to determine the best card to play:

1.  **Wish Adherence (Highest Priority)**:
    - If a Suit Wish is active (from a previous Jack), the AI searches its hand for a card matching that suit (excluding Jacks if possible to save them).
    - If found, it plays the matching suit card immediately.

2.  **Standard Play**:
    - If no wish is active (or no suit match found), the AI evaluates all valid moves.
    - **Priority:**
        1.  **Non-Jacks**: AI prefers to play Rank matches or Suit matches using normal cards (7, 8, 9, 10, A, K, Q).
        2.  **Jacks**: AI reserves Jacks as "Wildcards" for when no other move is possible, unless it only has Jacks.

3.  **Drawing**:
    - If no valid move is available, the AI draws cards.

## Special Handling

### Jacks (Bube)
- **Wish Logic**: When the AI plays a Jack, it calculates which suit it holds the **most of** in its remaining hand.
- It then wishes for that suit to maximize its chances for the next turn.
- If the hand is empty after playing the Jack, the wish defaults to the strongest count (effectively random/first found).

### 7s (Seven)
- If a 7 is played against the AI, the engine enforces a "Must Play 7 or Draw" rule.
- The AI will automatically play a 7 if it has one to stack the penalty.
- If it has no 7, it accepts the draw penalty.

### 8s (Eight)
- If an 8 is played, the AI is skipped (enforced by Engine).

## Interaction Logic
- **Game Logger**: The AI's introspection data is logged to the browser console at the end of every game for debugging purposes.
