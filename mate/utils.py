import re
from fuzzywuzzy import process
import nltk
from nltk.corpus import swadesh
from nltk.tokenize import word_tokenize
import string
import logging

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
    logger = logging.getLogger(__name__)
    logger.debug(f"Checking if input is sane German: '{input_str}'")

    if not input_str or not input_str.strip():
        logger.debug("Input is empty or whitespace only")
        return False

    # Tokenize the input string into words
    tokens = word_tokenize(input_str, language='german')
    logger.debug(f"Tokenized input: {tokens}")

    if not tokens:
        logger.debug("No tokens found after tokenization")
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
        "witz", "gedicht", "geschichte", "lied", "musik", "film", "buch",
        "mal", "einmal", "zweimal", "noch", "schon", "jetzt", "später", "früher",
        "hallo", "tschüss", "wiedersehen", "morgen", "abend", "mittag",
        "einen", "eine", "einem", "eines", "einer",
        "mein", "dein", "sein", "ihr", "unser", "euer",
        "meine", "deine", "seine", "ihre", "unsere", "eure",
        "meinen", "deinen", "seinen", "ihren", "unseren", "euren",
        "meinem", "deinem", "seinem", "ihrem", "unserem", "eurem",
        "meiner", "deiner", "seiner", "ihrer", "unserer", "eurer",
        "meines", "deines", "seines", "ihres", "unseres", "eures"
    }

    # Common German prefixes and suffixes
    german_prefixes = {"ge", "be", "ver", "er", "ent", "zer", "ab", "an", "auf", "aus", "ein", "vor", "zu", "über", "unter", "um"}
    german_suffixes = {"en", "st", "t", "e", "et", "est", "te", "ten", "er", "ung", "keit", "heit", "lich", "bar", "ig", "isch", "sam"}

    valid_word_count = 0
    total_word_count = 0
    word_analysis = []  # For detailed logging

    for token in tokens:
        # Remove punctuation from the token
        word = token.lower().strip(string.punctuation)

        # Skip non-alphabetic words and single letters (except common ones like "a" and "i")
        if not word.isalpha():
            logger.debug(f"Skipping non-alphabetic token: '{word}'")
            continue

        if len(word) <= 1 and word not in {"a", "i", "o", "u"}:
            logger.debug(f"Skipping single letter token: '{word}'")
            continue

        total_word_count += 1
        word_valid = False
        reason = ""

        # Check if word is in our dictionaries
        if word in GERMAN_WORDS:
            valid_word_count += 1
            word_valid = True
            reason = "in GERMAN_WORDS dictionary"
        elif word in common_german_words:
            valid_word_count += 1
            word_valid = True
            reason = "in common German words list"
        else:
            # Check for German word formation patterns
            has_german_prefix = any(word.startswith(prefix) for prefix in german_prefixes)
            has_german_suffix = any(word.endswith(suffix) for suffix in german_suffixes)

            if has_german_prefix and has_german_suffix:
                valid_word_count += 0.9  # High confidence
                word_valid = True
                reason = "has German prefix and suffix"
            elif has_german_suffix:
                valid_word_count += 0.7  # Medium-high confidence
                word_valid = True
                reason = "has German suffix"
            elif has_german_prefix:
                valid_word_count += 0.5  # Medium confidence
                word_valid = True
                reason = "has German prefix"
            elif "ä" in word or "ö" in word or "ü" in word or "ß" in word:
                # Contains German-specific characters
                valid_word_count += 0.8
                word_valid = True
                reason = "contains German-specific characters"
            else:
                reason = "not recognized as German"

        word_analysis.append(f"'{word}': {'valid' if word_valid else 'invalid'} ({reason})")

    logger.debug(f"Word analysis: {', '.join(word_analysis)}")
    logger.debug(f"Valid word count: {valid_word_count}, Total word count: {total_word_count}")

    if total_word_count == 0:
        logger.debug("No valid words found for analysis")
        return False

    # For very short inputs (commands), be more lenient
    adjusted_threshold = threshold
    if total_word_count <= 5:
        adjusted_threshold = 0.1
        logger.debug(f"Short input detected, lowering threshold to {adjusted_threshold}")

    proportion = valid_word_count / total_word_count
    logger.debug(f"Proportion of valid words: {proportion:.2f}, Threshold: {adjusted_threshold}")

    is_sane = proportion >= adjusted_threshold
    logger.debug(f"Input {'is' if is_sane else 'is not'} considered sane German")

    return is_sane

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

