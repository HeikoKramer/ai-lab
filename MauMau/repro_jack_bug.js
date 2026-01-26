
import { MauMauEngine } from './src/logic/engine.js';

const engine = new MauMauEngine();
engine.initGame();

// Force Top Card to be an 8 of Spades (Pik 8)
engine.discardPile = [{ suit: 'Pik', value: '8' }];
engine.currentPlayer = 'player';
engine.skipNext = false; // Assuming main.js handled the skip logic without setting engine flag

console.log("State: Top Card = Pik 8");

// Test 1: Play Jack of Hearts (Herz Bube)
const jackHearts = { suit: 'Herz', value: 'B' };
const valid1 = engine.isValidMove(jackHearts);
console.log(`Test 1: Play Herz Bube on Pik 8 -> Valid? ${valid1}`);

// Test 2: Play Jack of Spades (Pik Bube)
const jackSpades = { suit: 'Pik', value: 'B' };
const valid2 = engine.isValidMove(jackSpades);
console.log(`Test 2: Play Pik Bube on Pik 8 -> Valid? ${valid2}`);

// Test 3: Play Regular Spades (Pik 10)
const tenSpades = { suit: 'Pik', value: '10' };
const valid3 = engine.isValidMove(tenSpades);
console.log(`Test 3: Play Pik 10 on Pik 8 -> Valid? ${valid3}`);

if (!valid1 || !valid2) {
    console.error("FAIL: Jack should be valid on 8.");
} else {
    console.log("SUCCESS: Jack Logic seems correct in Engine isolation.");
}
