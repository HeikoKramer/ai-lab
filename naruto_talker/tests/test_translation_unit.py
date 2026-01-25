import sys
import os
import unittest

# Add project root to sys.path to ensure we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.translation import translate_text

class TestTranslation(unittest.TestCase):
    def test_german_to_english(self):
        input_text = "Eine raue Stimme"
        expected_keywords = ["rough", "voice", "harsh"] # 'harsh' or 'rough' depending on translation
        
        translated = translate_text(input_text, "en")
        print(f"Input: {input_text}")
        print(f"Output: {translated}")
        
        # Check against lower case to be safe
        translated_lower = translated.lower()
        self.assertTrue(any(kw in translated_lower for kw in expected_keywords), 
                        f"Translation '{translated}' did not contain expected keywords.")

    def test_english_to_german(self):
        input_text = "Hello World"
        translated = translate_text(input_text, "de")
        print(f"Input: {input_text}")
        print(f"Output: {translated}")
        self.assertIn("Hallo", translated)

    def test_empty_input(self):
        self.assertEqual(translate_text("", "en"), "")
        self.assertEqual(translate_text(None, "en"), None)

if __name__ == '__main__':
    unittest.main()
