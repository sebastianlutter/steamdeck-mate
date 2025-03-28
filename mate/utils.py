import re
from fuzzywuzzy import process
import nltk
from nltk.corpus import swadesh
from nltk.tokenize import word_tokenize
import string

# Ensure the necessary NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('swadesh', quiet=True)

# Load German words from the Swadesh corpus
GERMAN_WORDS = set(word.lower() for word in swadesh.words('de'))

def is_sane_input_german(input_str: str, threshold: float = 0.3) -> bool:
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