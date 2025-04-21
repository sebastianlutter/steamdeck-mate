import re
from fuzzywuzzy import process
import nltk
from nltk.corpus import swadesh
from nltk.tokenize import word_tokenize
import string

# Ensure the necessary NLTK data is downloaded
nltk.download('punkt', quiet=False)
nltk.download('swadesh', quiet=False)
nltk.download('punkt_tab', quiet=False)

# Load German words from the Swadesh corpus
GERMAN_WORDS = set(word.lower() for word in swadesh.words('de'))

def is_sane_input_german(input_str: str, threshold: float = 0.15) -> bool:
    """
    Determines if the input string is sane (contains a sufficient proportion of valid German words).

    Args:
        input_str (str): The input string from the STT engine.
        threshold (float): The minimum proportion of valid words required to consider the input sane.

    Returns:
        bool: True if the input is sane, False otherwise.
    """
    if not input_str or not input_str.strip():
        return False

    # Tokenize the input string into words
    tokens = word_tokenize(input_str, language='german')

    if not tokens:
        return False

    # Common German words that might not be in the dictionary but are valid
    common_german_words = {
        "wie", "was", "wer", "wo", "wann", "warum", "welche", "welcher", "welches",
        "mir", "dir", "uns", "euch", "ihnen", "ihm", "ihr", "du", "ich", "er", "sie", "es", "wir", "ihr", "sie",
        "ein", "eine", "einen", "einem", "einer", "eines", "der", "die", "das", "den", "dem", "des",
        "ist", "sind", "war", "waren", "wird", "werden", "würde", "würden", "kann", "können", "könnte", "könnten",
        "hat", "haben", "hatte", "hatten", "geht", "gehen", "ging", "gingen",
        "über", "unter", "vor", "nach", "bei", "mit", "ohne", "für", "gegen", "um", "zu", "aus", "von", "auf",
        "erzähle", "erzähl", "sage", "sag", "zeige", "zeig", "mache", "mach", "gib", "gebe",
        "bitte", "danke", "ja", "nein", "vielleicht", "heute", "morgen", "gestern",
        "uhr", "zeit", "tag", "woche", "monat", "jahr",
        "schön", "gut", "schlecht", "groß", "klein", "alt", "neu", "kurz", "lang",
        "witz", "gedicht", "geschichte", "lied", "musik", "film", "buch"
    }

    valid_word_count = 0
    total_word_count = 0

    for token in tokens:
        # Remove punctuation from the token
        word = token.lower().strip(string.punctuation)
        if word.isalpha() and len(word) > 1:  # Consider only alphabetic words with length > 1
            total_word_count += 1
            if word in GERMAN_WORDS or word in common_german_words:
                valid_word_count += 1
            # Check for common German word endings
            elif (word.endswith("en") or word.endswith("st") or word.endswith("et") or
                  word.endswith("te") or word.endswith("ten") or word.endswith("er") or
                  word.endswith("ung") or word.endswith("keit") or word.endswith("heit") or
                  word.endswith("lich") or word.endswith("bar")):
                # These are common German word endings, so likely a valid German word
                valid_word_count += 0.5  # Count as partially valid

    if total_word_count == 0:
        return False

    # For very short inputs (commands), be more lenient
    if total_word_count <= 5:
        threshold = 0.1

    proportion = valid_word_count / total_word_count
    return proportion >= threshold

def is_conversation_ending(sentence):
    # Define phrases that indicate the end of a conversation in both English and German
    end_phrases = [
        "stop chat", "exit", "bye", "finish",
        "halt stoppen", "chat beenden", "auf wiedersehen", "tschüss", "ende", "schluss",
    ]
    # Use fuzzy matching to find the closest match to the input sentence and get the match score
    highest_match = process.extractOne(sentence.lower(), end_phrases)
    # Define a threshold for deciding if the sentence means to end the conversation
    threshold = 80  # You can adjust the threshold based on testing
    # Check if the highest match score is above the threshold
    if highest_match[1] >= threshold:
        return True
    else:
        return False

def clean_str_from_markdown(text: str):
    # but first clean the string from newline chars. Add a . to each
    buffer = text.replace('\n', '. ')
    # remove the point we added if there is a mark char before
    buffer = re.sub(r'([?:!.,])\.', r'\1', buffer)
    # insert a space between sentences with no whitespace but a .
    buffer = re.sub(r"(?<!\d)\.(?![\d\s])", ". ", buffer)
    # remove all enumeration fragements (.1. and so on)
    buffer = re.sub(r'\.\d+\.', '.', buffer)
    return buffer

def is_sane_input_german(input_str: str, threshold: float = 0.15) -> bool:
    """
    Determines if the input string is sane (contains a sufficient proportion of valid German words).

    Args:
        input_str (str): The input string from the STT engine.
        threshold (float): The minimum proportion of valid words required to consider the input sane.

    Returns:
        bool: True if the input is sane, False otherwise.
    """
    if not input_str or not input_str.strip():
        return False

    # Tokenize the input string into words
    tokens = word_tokenize(input_str, language='german')

    if not tokens:
        return False

    valid_word_count = 0
    total_word_count = 0

    for token in tokens:
        # Remove punctuation from the token
        word = token.lower().strip(string.punctuation)
        if word.isalpha():  # Consider only alphabetic words
            total_word_count += 1
            if word in GERMAN_WORDS:
                valid_word_count += 1

    if total_word_count == 0:
        return False

    proportion = valid_word_count / total_word_count
    return proportion >= threshold