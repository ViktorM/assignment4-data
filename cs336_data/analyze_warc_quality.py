# analyze_warc_quality.py
import random
import sys
sys.path.append('..')

from cs336_data.warc_wet_matcher import (
    find_matching_documents, 
    load_html_by_uri_from_warc
)
from cs336_data.utils import extract_text_from_html_bytes
from gopher_filters import gopher_quality_filter, analyze_quality_filters


def analyze_warc_with_filters(warc_path, wet_path, num_samples=20):
    """Extract text from WARC and analyze with quality filters."""

    matching_uris = find_matching_documents(warc_path, wet_path)

    if len(matching_uris) < num_samples:
        print(f"Only found {len(matching_uris)} matching documents")
        num_samples = len(matching_uris)

    # Random sample
    sampled_uris = random.sample(matching_uris, num_samples)

    texts = []
    names = []

    print("Extracting text from WARC files...")
    for i, uri in enumerate(sampled_uris):
        html_bytes, _ = load_html_by_uri_from_warc(warc_path, uri)
        if html_bytes:
            text = extract_text_from_html_bytes(html_bytes)
            texts.append(text)
            names.append(f"Doc {i+1}: {uri[:50]}...")

        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{num_samples} documents")

    print(f"\nAnalyzing {len(texts)} documents with quality filters...\n")
    results = analyze_quality_filters(texts, names)

    # Summary statistics
    passed_count = sum(1 for r in results if r['passed'])
    print("\nSummary:")
    print("  Total documents analyzed: {len(results)}")
    print(
        f"  Passed quality filters: {passed_count} "
        f"({passed_count/len(results)*100:.1f}%)"
    )

    # Analyze disagreements
    print("\n\nManual Review Notes:")
    print("Look for cases where filters might be too strict or lenient:")
    print("- Short but high-quality excerpts (failed due to word count)")
    print("- Navigation menus or forms (might pass filters but are low quality)")
    print("- Foreign language text mixed with English")
    print("- Technical documentation with code snippets")

    print("\nAnalysis Results:")


if __name__ == "__main__":
    warc_file = '../data/CC-MAIN-20180420081400-20180420101400-00000.warc.gz'
    wet_file = '../data/CC-MAIN-20180420081400-20180420101400-00000.warc.wet.gz'

    analyze_warc_with_filters(warc_file, wet_file, num_samples=20)
