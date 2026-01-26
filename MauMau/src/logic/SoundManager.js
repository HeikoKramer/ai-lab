/**
 * SoundManager
 * Handles all audio playback, effects, and music loops for the game.
 */
export class SoundManager {
    constructor() {
        this.basePath = './src/assets/audio/';
        this.enabled = true;

        // Sound Banks
        this.sounds = {
            play: [
                'Device Beacon 005.wav',
                'Device Beacon 006.wav',
                'Device Beacon 007.wav',
                'Device Beacon 008.wav'
            ],
            seven: 'Device Short Circuit 002.wav',
            seven_parry_1: 'Device Short Circuit 003.wav',
            seven_parry_2: 'Device Short Circuit 004.wav',
            seven_parry_3: 'Device Short Circuit 005.wav',

            eight: [
                // 'Hacking Downloaded Successfully 001.wav', // MISSING FILE
                // 'Hacking Downloaded Successfully 002.wav', // MISSING FILE
                // 'Hacking Downloaded Successfully 003.wav', // MISSING FILE
                // 'Hacking Downloaded Successfully 004.wav'  // MISSING FILE
                // FALLBACK: Use short circuit or similar until files provided?
                'Device Short Circuit 002.wav'
            ],

            jack: [
                'Armor Reactive Camo 001.wav',
                'Armor Reactive Camo 002.wav',
                'Armor Reactive Camo 003.wav',
                'Armor Reactive Camo 004.wav'
            ],

            // NEW SOUNDS
            mau: 'Aeolian Combo G.wav',

            draw: [
                'Hacking Breach 006.wav',
                'Hacking Breach 001.wav',
                'Hacking Breach 003.wav',
                'Hacking Breach 004.wav'
            ],

            win_ai: 'Hacking Zeroing 004.wav',
            win_human: 'Hacking Mind Scraper 007.wav',

            bg_menu: 'Cyberspace 013.wav',
            bg_game: 'Cyberspace 005.wav'
        };

        this.currentBgTrack = null;
        this.bgAudioObj = null;
    }

    setEnabled(status) {
        this.enabled = status;
        if (!status) {
            this.stopMusic();
        }
    }

    /**
     * Play a one-shot sound effect
     * @param {String} type - 'play', 'seven', 'eight', 'jack', 'win_ai', 'win_human'
     * @param {Number} chainCount - For Seven chains (0 = first, 1 = parry 1, etc.)
     */
    play(type, chainCount = 0) {
        if (!this.enabled) return;

        let filename = null;

        switch (type) {
            case 'play':
                filename = this.getRandom(this.sounds.play);
                break;
            case 'seven':
                if (chainCount === 0) filename = this.sounds.seven;
                else if (chainCount === 1) filename = this.sounds.seven_parry_1;
                else if (chainCount === 2) filename = this.sounds.seven_parry_2;
                else filename = this.sounds.seven_parry_3; // Cap at max
                break;
            case 'eight':
                filename = this.getRandom(this.sounds.eight);
                break;
            case 'jack':
                filename = this.getRandom(this.sounds.jack);
                break;
            default:
                if (this.sounds[type]) {
                    filename = Array.isArray(this.sounds[type])
                        ? this.getRandom(this.sounds[type])
                        : this.sounds[type];
                }
        }

        if (filename) {
            const audio = new Audio(this.basePath + filename);
            audio.volume = 0.6; // Norm volume
            audio.play().catch(e => console.warn("Audio play blocked:", e));
        }
    }

    playWin(winner) {
        this.stopMusic();
        if (!this.enabled) return;
        const type = winner === 'ai' ? 'win_ai' : 'win_human';
        const filename = this.sounds[type];

        if (filename) {
            const audio = new Audio(this.basePath + filename);
            audio.volume = 0.8;
            audio.play().catch(e => console.warn("Win Audio blocked:", e));
            // Return duration for animation sync? 
            // We'll estimate or just let it play.
        }
    }

    startMusic(trackType) {
        if (!this.enabled) return;
        if (this.currentBgTrack === trackType) return; // Already playing

        this.stopMusic();

        const filename = trackType === 'menu' ? this.sounds.bg_menu : this.sounds.bg_game;
        this.bgAudioObj = new Audio(this.basePath + filename);
        this.bgAudioObj.loop = true;
        this.bgAudioObj.volume = 0.4; // Background level

        // Optimistically set track type, but rollback if fails
        this.currentBgTrack = trackType;

        this.bgAudioObj.play()
            .then(() => {
                console.log(`[Audio] Playing ${trackType}`);
            })
            .catch(e => {
                console.warn("BG Music blocked (User interaction needed):", e);
                // CRITICAL FIX: Reset state so subsequent clicks can try again
                if (this.currentBgTrack === trackType) {
                    this.currentBgTrack = null;
                }
            });
    }

    stopMusic() {
        if (this.bgAudioObj) {
            this.bgAudioObj.pause();
            this.bgAudioObj = null;
            this.currentBgTrack = null;
        }
    }

    getRandom(list) {
        return list[Math.floor(Math.random() * list.length)];
    }
}
