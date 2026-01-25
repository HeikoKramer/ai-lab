import logging
from deep_translator import GoogleTranslator
from deep_translator.exceptions import TranslationNotFound, RequestError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def translate_text(text: str, target_lang: str = "en") -> str:
    """
    Translates the given text to the target language.
    Defaults to English ('en').
    
    Args:
        text (str): The text to translate.
        target_lang (str): The target language code (e.g., 'en', 'de', 'fr').
        
    Returns:
        str: The translated text, or the original text if translation fails.
    """
    if not text or not isinstance(text, str):
        return text

    try:
        # Check if text is empty or just whitespace
        if not text.strip():
            return text

        translator = GoogleTranslator(source='auto', target=target_lang)
        translated = translator.translate(text)
        
        logger.info(f"Translated '{text}' -> '{translated}' ({target_lang})")
        return translated

    except (TranslationNotFound, RequestError, Exception) as e:
        logger.error(f"Translation failed for '{text}': {e}")
        return text
