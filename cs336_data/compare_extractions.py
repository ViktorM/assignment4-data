import sys
sys.path.append('..')


from cs336_data.warc_wet_matcher import (
    find_matching_documents, 
    load_html_by_uri_from_warc,
    load_wet_record_by_uri,
    extract_text_from_html_bytes
)


def compare_text_extractions(warc_path, wet_path, num_samples=3):
    """Compare text extracted from WARC vs WET files."""

    # Find matching documents
    matching_uris = find_matching_documents(warc_path, wet_path)

    if not matching_uris:
        print("No matching documents found between WARC and WET files!")
        return

    print(f"Found {len(matching_uris)} matching documents")
    print(f"Comparing first {num_samples} documents...\n")

    for i, uri in enumerate(matching_uris[:num_samples]):
        print("="*80)
        print(f"Document {i+1}: {uri}")
        print("="*80)

        # Extract from WARC using your function
        html_bytes, _ = load_html_by_uri_from_warc(warc_path, uri)
        if html_bytes:
            warc_text = extract_text_from_html_bytes(html_bytes)
            print("\n--- Text extracted from WARC (via resiliparse) ---")
            print(warc_text[:500] + "..." if len(warc_text) > 500 else warc_text)
            print(f"\nTotal length: {len(warc_text)} characters")
        else:
            print("\nNo HTML found in WARC for this URI")
            continue

        # Get text from WET
        wet_text = load_wet_record_by_uri(wet_path, uri)
        if wet_text:
            print("\n--- Text from WET file ---")
            print(wet_text[:500] + "..." if len(wet_text) > 500 else wet_text)
            print(f"\nTotal length: {len(wet_text)} characters")
        else:
            print("\nNo text found in WET for this URI")

        # Quick comparison
        if warc_text and wet_text:
            print("\n--- Comparison ---")
            print(f"Length difference: {abs(len(warc_text) - len(wet_text))} chars")

            # Check first 200 chars (normalized)
            warc_start = warc_text[:200].lower().strip()
            wet_start = wet_text[:200].lower().strip()

            if warc_start == wet_start:
                print("Beginning of text is identical")
            else:
                print("Beginning of text differs")

            # Check for common issues
            if warc_text.count('\n') > wet_text.count('\n') * 2:
                print("WARC extraction has significantly more line breaks")
            if '|||EMAIL_ADDRESS|||' in wet_text or '|||PHONE_NUMBER|||' in wet_text:
                print("WET text has PII masking applied")

        print("\n")


if __name__ == "__main__":
    warc_file = '../data/CC-MAIN-20180420081400-20180420101400-00000.warc.gz'
    wet_file = '../data/CC-MAIN-20180420081400-20180420101400-00000.warc.wet.gz'

    compare_text_extractions(warc_file, wet_file, num_samples=5)
