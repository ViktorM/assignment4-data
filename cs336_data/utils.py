from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


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
