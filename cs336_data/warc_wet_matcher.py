"""
WARC/WET File Matcher

This module provides functions to match and compare documents between WARC
and WET files.
"""

import gzip
from warcio.archiveiterator import ArchiveIterator
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from typing import Tuple, List, Optional, Dict


def get_all_warc_uris(warc_path: str) -> List[Tuple[int, str]]:
    """
    Extract all URIs from WARC file with their positions.

    Args:
        warc_path: Path to WARC.gz file

    Returns:
        List of tuples (index, uri)
    """
    uris = []
    with gzip.open(warc_path, 'rb') as stream:
        for idx, record in enumerate(ArchiveIterator(stream)):
            if record.rec_type == 'response':
                uri = record.rec_headers.get_header('WARC-Target-URI')
                if uri:
                    uris.append((idx, uri))
    return uris


def get_all_wet_uris(wet_path: str) -> List[Tuple[int, str]]:
    """
    Extract all URIs from WET file with their positions.

    Args:
        wet_path: Path to WET.gz file

    Returns:
        List of tuples (index, uri)
    """
    uris = []
    current_idx = -1

    with gzip.open(wet_path, 'rt', encoding='utf-8') as file:
        for line in file:
            if line.startswith('WARC/1.0'):
                current_idx += 1
            elif line.startswith('WARC-Target-URI:'):
                uri = line.split('WARC-Target-URI: ')[1].strip()
                uris.append((current_idx, uri))
    return uris


def load_wet_record_by_uri(wet_path: str, target_uri: str) -> Optional[str]:
    """
    Load WET text content for a specific URI.

    Args:
        wet_path: Path to WET.gz file
        target_uri: URI to search for

    Returns:
        Text content if found, None otherwise
    """
    content = []
    found = False
    in_content = False

    with gzip.open(wet_path, 'rt', encoding='utf-8') as file:
        for line in file:
            if line.startswith('WARC/1.0'):
                if found:
                    break
                content = []
                in_content = False

            elif line.startswith('WARC-Target-URI:'):
                uri = line.split('WARC-Target-URI: ')[1].strip()
                if uri == target_uri:
                    found = True

            elif found and line.strip() == '':
                if not in_content:
                    in_content = True
                    continue

            elif found and in_content:
                content.append(line.rstrip())

    return '\n'.join(content).strip() if found else None


def load_html_by_uri_from_warc(
    warc_path: str, target_uri: str
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Load HTML content from WARC file for a specific URI.

    Args:
        warc_path: Path to WARC.gz file
        target_uri: URI to search for

    Returns:
        Tuple of (html_bytes, uri) if found, (None, None) otherwise
    """
    with gzip.open(warc_path, 'rb') as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type == 'response':
                record_uri = record.rec_headers.get_header('WARC-Target-URI')
                if record_uri == target_uri:
                    raw_html = record.content_stream().read()
                    return raw_html, record_uri
    return None, None


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """
    Extract text from raw HTML bytes, detecting encoding if necessary.

    Args:
        html_bytes: Raw HTML content as bytes

    Returns:
        Extracted plain text
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


def compare_extraction_methods(
    warc_path: str, wet_path: str, uri: str
) -> Dict[str, any]:
    """
    Compare text extraction from WARC (via resiliparse) vs WET file.

    Args:
        warc_path: Path to WARC.gz file
        wet_path: Path to WET.gz file
        uri: URI to compare

    Returns:
        Dictionary with comparison results
    """
    results = {
        'uri': uri,
        'warc_text': None,
        'wet_text': None,
        'warc_length': 0,
        'wet_length': 0,
        'length_diff': 0,
        'texts_match': False
    }

    # Extract from WARC
    html_bytes, _ = load_html_by_uri_from_warc(warc_path, uri)
    if html_bytes:
        warc_text = extract_text_from_html_bytes(html_bytes)
        results['warc_text'] = warc_text
        results['warc_length'] = len(warc_text)

    # Extract from WET
    wet_text = load_wet_record_by_uri(wet_path, uri)
    if wet_text:
        results['wet_text'] = wet_text
        results['wet_length'] = len(wet_text)

    # Compare
    if results['warc_text'] and results['wet_text']:
        results['length_diff'] = abs(
            results['warc_length'] - results['wet_length']
        )

        # Simple similarity check (normalize and compare)
        warc_normalized = (
            results['warc_text'].lower()
            .replace('\n', ' ').replace('  ', ' ').strip()
        )
        wet_normalized = (
            results['wet_text'].lower()
            .replace('\n', ' ').replace('  ', ' ').strip()
        )

        results['texts_match'] = warc_normalized == wet_normalized

    return results


def find_matching_documents(warc_path: str, wet_path: str) -> List[str]:
    """
    Find all matching URIs between WARC and WET files.

    Args:
        warc_path: Path to WARC.gz file
        wet_path: Path to WET.gz file

    Returns:
        List of matching URIs
    """
    warc_uris = get_all_warc_uris(warc_path)
    wet_uris = get_all_wet_uris(wet_path)

    warc_uri_set = {uri for _, uri in warc_uris}
    wet_uri_set = {uri for _, uri in wet_uris}

    return list(warc_uri_set.intersection(wet_uri_set))


def analyze_file_matching(warc_path: str, wet_path: str) -> Dict[str, any]:
    """
    Analyze the matching between WARC and WET files.

    Args:
        warc_path: Path to WARC.gz file
        wet_path: Path to WET.gz file

    Returns:
        Dictionary with analysis results
    """
    warc_uris = get_all_warc_uris(warc_path)
    wet_uris = get_all_wet_uris(wet_path)

    warc_uri_set = {uri for _, uri in warc_uris}
    wet_uri_set = {uri for _, uri in wet_uris}

    matching_uris = warc_uri_set.intersection(wet_uri_set)

    match_pct = (
        (len(matching_uris) / len(warc_uris) * 100) 
        if warc_uris else 0
    )

    return {
        'warc_count': len(warc_uris),
        'wet_count': len(wet_uris),
        'matching_count': len(matching_uris),
        'warc_only': len(warc_uri_set - wet_uri_set),
        'wet_only': len(wet_uri_set - warc_uri_set),
        'match_percentage': match_pct,
        'matching_uris': list(matching_uris)[:10]  # First 10 for preview
    }


if __name__ == "__main__":
    # Example usage
    warc_file = '../data/CC-MAIN-20180420081400-20180420101400-00000.warc.gz'
    wet_file = (
        '../data/CC-MAIN-20180420081400-20180420101400-00000.warc.wet.gz'
    )

    # Analyze matching
    print("Analyzing WARC/WET file matching...")
    analysis = analyze_file_matching(warc_file, wet_file)

    print(f"\nAnalysis Results:")
    print(f"WARC documents: {analysis['warc_count']}")
    print(f"WET documents: {analysis['wet_count']}")
    print(f"Matching documents: {analysis['matching_count']}")
    print(f"Match percentage: {analysis['match_percentage']:.2f}%")

    # Compare a few documents
    if analysis['matching_uris']:
        print("\nComparing first 3 matching documents...")
        for i, uri in enumerate(analysis['matching_uris'][:3]):
            print("\n" + "="*60)
            print(f"Document {i+1}: {uri}")
            comparison = compare_extraction_methods(warc_file, wet_file, uri)
            print(f"WARC text length: {comparison['warc_length']}")
            print(f"WET text length: {comparison['wet_length']}")
            print(f"Length difference: {comparison['length_diff']}")
            print(f"Texts match: {comparison['texts_match']}")

            print(f"WARC text: {comparison['warc_text']}")
            print(f"WET text: {comparison['wet_text']}")
