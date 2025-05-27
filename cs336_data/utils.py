from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from fasttext import load_model
from typing import Any, Tuple, Dict
import re
import os


_model_cache: Dict[str, Any] = {}


def _get_cached_model(model_path: str):
    """Load and cache FastText models to avoid repeated loading."""
    if model_path not in _model_cache:
        # Handle relative paths
        if not os.path.isabs(model_path):
            # Get the directory where this file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to assignment4-data and then to the model path
            # TODO: replace by a new find_data_file function?
            absolute_path = os.path.join(os.path.dirname(current_dir), model_path)
            if os.path.exists(absolute_path):
                model_path = absolute_path

        _model_cache[model_path] = load_model(model_path)
    return _model_cache[model_path]


def find_data_file(filename, search_paths=None):
    """Find a data file by checking multiple possible locations."""
    if search_paths is None:
        search_paths = [
            # Local paths
            f"data/{filename}",
            f"../data/{filename}",
            f"./data/{filename}",
            # Cluster paths
            f"/data/{filename}",
            f"/data/wiki/{filename}",
            f"/data/classifiers/{filename}",
            f"/data/CC/{filename}",
            # Current directory
            filename,
        ]

    for path in search_paths:
        if os.path.exists(path):
            return path

    return filename


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

    text = extract_plain_text(html_str)

    return text


def identify_language(text: str, model_path: str = 'data/lid.176.bin') -> tuple[Any, float]:
    """
    Identify the main language of the given text and return the language code and confidence score.

    Parameters:
        text (str): The input Unicode string to classify.
        model_path (str): Path to the FastText language model.

    Returns:
        tuple: (language_code, confidence_score)
            language_code (str): ISO language code (e.g., 'en', 'zh').
            confidence_score (float): Confidence score between 0 and 1.

    Raises:
        ValueError: If text is empty or None
        FileNotFoundError: If model file doesn't exist
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")

    # Handle relative paths for tests
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        # Try looking in parent directory's data folder
        parent_data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), model_path
        )
        if os.path.exists(parent_data_path):
            model_path = parent_data_path

    try:
        model = _get_cached_model(model_path)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load model from {model_path}: {e}")

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
    # Common US phone number patterns.
    phone_regex = r'(\+?1[\s.-]?)?\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
    masked_text, count = re.subn(phone_regex, "|||PHONE_NUMBER|||", text)
    return masked_text, count


def mask_ips(text: str) -> Tuple[str, int]:
    """
    Mask IPv4 and IPv6 addresses in the text.

    Returns:
        tuple: (masked_text, count of masked IPs)
    """

    # IPv4 pattern with validation (0-255 range)
    ipv4_pattern = (
        r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
        r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    )

    # Simplified IPv6 pattern
    ipv6_pattern = (
        r'\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b|'
        r'\b(?:[A-Fa-f0-9]{1,4}:)*:(?:[A-Fa-f0-9]{1,4}:)*[A-Fa-f0-9]{1,4}\b'
    )

    # Combined pattern
    ip_pattern = f'({ipv4_pattern})|({ipv6_pattern})'

    masked_text, count = re.subn(ip_pattern, "|||IP_ADDRESS|||", text)
    return masked_text, count


def classify_nsfw(
    text: str,
    model_path: str = 'data/jigsaw_fasttext_bigrams_nsfw_final.bin'
) -> Tuple[str, float]:
    """
    Classifies the given text as NSFW or not, providing a confidence score.

    Returns:
        tuple: ('nsfw', confidence_score) or ('safe', confidence_score)
    """
    model = _get_cached_model(model_path)
    labels, scores = model.predict(text.replace('\n', ' '), k=1)
    label = labels[0].replace('__label__', '')
    return label, scores[0]


def classify_toxic_speech(
    text: str,
    model_path: str = 'data/jigsaw_fasttext_bigrams_hatespeech_final.bin'
) -> Tuple[str, float]:
    """
    Classifies the given text as toxic or non-toxic, providing a confidence score.

    Returns:
        tuple: ('toxic', confidence_score) or ('non-toxic', confidence_score)
    """
    model = _get_cached_model(model_path)
    labels, scores = model.predict(text.replace('\n', ' '), k=1)
    label = labels[0].replace('__label__', '')
    return label, scores[0]
