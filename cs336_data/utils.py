from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from fasttext import load_model
from typing import Any, Tuple
from detoxify import Detoxify
import re


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """
    Extract text from raw HTML bytes, detecting encoding if necessary.

    Parameters:
        html_bytes (bytes): Raw HTML content as bytes.

    Returns:
        str: Extracted plain text.
    """
    try:
        # First, try decoding with UTF-8 encoding
        html_str = html_bytes.decode('utf-8')
    except UnicodeDecodeError:
        # Detect encoding if UTF-8 fails
        detected_encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(detected_encoding, errors='replace')

    # Extract plain text using resiliparse
    text = extract_plain_text(html_str)

    return text


def identify_language(text: str, model_path: str = 'data/lid.176.bin') -> tuple[Any, float]:
    """
    Identify the main language of the given text and return the language code and confidence score.

    Parameters:
        text (str): The input Unicode string to classify.

    Returns:
        tuple: (language_code, confidence_score)
            language_code (str): ISO language code (e.g., 'en', 'zh').
            confidence_score (float): Confidence score between 0 and 1.
    """
    model = load_model(model_path)
    predictions = model.predict(text.replace('\n', ' '), k=1)  # Top 1 language
    language_label = predictions[0][0]
    confidence_score = predictions[1][0]

    # Extract ISO language code
    language_code = language_label.replace('__label__', '')

    return language_code, confidence_score


def mask_emails(text: str) -> Tuple[str, int]:
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    masked_text, count = re.subn(email_regex, "|||EMAIL_ADDRESS|||", text)
    return masked_text, count


def mask_phone_numbers(text: str) -> Tuple[str, int]:
    # Captures common US phone number patterns.
    phone_regex = r'(\+?1[\s.-]?)?\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
    masked_text, count = re.subn(phone_regex, "|||PHONE_NUMBER|||", text)
    return masked_text, count


def mask_ips(text: str) -> Tuple[str, int]:
    # Matches IPv4 addresses explicitly
    ipv4_regex = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    masked_text, count = re.subn(ipv4_regex, "|||IP_ADDRESS|||", text)
    return masked_text, count


def classify_nsfw(text: str, model_path: str = 'data/jigsaw_fasttext_bigrams_nsfw_final.bin') -> Tuple[str, float]:
    """
    Classifies the given text as NSFW or not, providing a confidence score.

    Returns:
        tuple: ('nsfw', confidence_score) or ('safe', confidence_score)
    """
    model = load_model(model_path)
    labels, scores = model.predict(text.replace('\n', ' '), k=1)
    label = labels[0].replace('__label__', '')
    return label, scores[0]


def classify_toxic_speech(text: str, model_path: str = 'data/jigsaw_fasttext_bigrams_hatespeech_final.bin') -> Tuple[str, float]:
    """
    Classifies the given text as toxic or non-toxic, providing a confidence score.

    Returns:
        tuple: ('toxic', confidence_score) or ('non-toxic', confidence_score)
    """
    model = load_model(model_path)
    labels, scores = model.predict(text.replace('\n', ' '), k=1)
    label = labels[0].replace('__label__', '')
    return label, scores[0]
