export class CardRenderer {
    static getHTML(card, isInteractive = false, index = -1) {
        if (!card) return ''; // Backside or empty?

        const { suit, value } = card;
        const suitData = this.getSuitData(suit);
        const isTwoDigits = value.length > 1;

        // Interactive attributes
        const clickAttr = isInteractive ? `onclick="game.playCard(${index})"` : '';
        const cursorStyle = isInteractive ? 'cursor: pointer;' : '';

        return `
        <div class="card-container" style="${cursorStyle}" ${clickAttr}>
            <svg width="150" height="220" viewBox="0 0 150 220" xmlns="http://www.w3.org/2000/svg">
                <rect x="0" y="0" width="150" height="220" rx="10" fill="#1a1b26" stroke="#787c99" stroke-width="2" />
                
                <defs>
                    <filter id="glow-${suitData.name}-${index}">
                        <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
                        <feMerge>
                            <feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/>
                        </feMerge>
                    </filter>
                </defs>

                <g fill="${suitData.color}" font-family="Ubuntu Mono" font-weight="700" style="filter: url(#glow-${suitData.name}-${index});">
                    <text x="10" y="25" font-size="22">${value}</text>
                    <text x="10" y="45" font-size="12">${suitData.symbol}</text>
                    
                    <g transform="rotate(180, 75, 110)">
                        <text x="10" y="25" font-size="22">${value}</text>
                        <text x="10" y="45" font-size="12">${suitData.symbol}</text>
                    </g>
                </g>

                <g fill="${suitData.color}" font-family="Ubuntu Mono" font-weight="700" text-anchor="middle">
                    <text x="75" y="${isTwoDigits ? 95 : 95}" font-size="50" style="filter: url(#glow-${suitData.name}-${index});">${value}</text>
                    
                    <text x="75" y="125" font-size="7" font-weight="400" letter-spacing="1">
                        ${suitData.ascii.split('\n').map((line, i) => `<tspan x="75" dy="${i === 0 ? 0 : 8}">${line}</tspan>`).join('')}
                    </text>
                </g>
            </svg>
        </div>`;
    }

    static getBacksideHTML() {
        return `
        <div class="card-container">
            <svg width="150" height="220" viewBox="0 0 150 220" xmlns="http://www.w3.org/2000/svg">
                <rect x="0" y="0" width="150" height="220" rx="10" fill="#16161e" stroke="#565f89" stroke-width="2" />
                <text x="75" y="110" fill="#2ac3de" font-family="Ubuntu Mono" font-size="20" text-anchor="middle" dominant-baseline="middle">TOKYO</text>
                <text x="75" y="130" fill="#2ac3de" font-family="Ubuntu Mono" font-size="20" text-anchor="middle" dominant-baseline="middle">NIGHT</text>
            </svg>
        </div>`;
    }

    static getSuitData(suitName) {
        const suits = [
            { name: 'Pik', symbol: '^', color: '#7dcfff', ascii: `\n   .x.   \n .:::::. \n(:^:^:^:)\n 'x:|:x' \n  /:::\\  \n ------- ` },
            { name: 'Kreuz', symbol: '+', color: '#7dcfff', ascii: `\n   .+.   \n --:+:-- \n   |:|   \n  /:::\\  \n ------- ` },
            { name: 'Herz', symbol: '<3', color: '#f7768e', ascii: `\n .xxx. .xxx.\nxxxxxxxxx\nxxxxxxxxx\n 'xxxxx' \n   'x'   ` },
            { name: 'Karo', symbol: '<>', color: '#f7768e', ascii: `\n   x   \n .:::::. \n.:::::::\n':::::::'\n ':::::' \n   x   ` }
        ];
        // Note: Ascii simplified for concise code integration, original had multiline string literals
        // Re-using logic to match existing style roughly or just mapping names
        const specific = suits.find(s => s.name === suitName);
        if (specific) return specific;

        // Fallback or exact match from original code
        return { name: suitName, symbol: '?', color: '#ccc', ascii: '' };
    }
}
