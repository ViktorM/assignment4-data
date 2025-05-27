# gopher_filters.py
import re
import nltk
from typing import List


# Download punkt tokenizer if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def gopher_quality_filter(text: str) -> bool:
    """
    Apply Gopher quality filters to determine if text is suitable for LM training.

    Filters:
    1. Document contains 50-100,000 words
    2. Mean word length is 3-10 characters
    3. Less than 30% of lines end with ellipsis
    4. At least 80% of words contain alphabetic characters

    Args:
        text: Input text string

    Returns:
        bool: True if text passes all filters, False otherwise
    """
    if not text or not text.strip():
        return False

    # Tokenize into words
    words = nltk.word_tokenize(text)

    # Filter 1: Word count between 50 and 100,000
    word_count = len(words)
    if word_count < 50 or word_count > 100_000:
        return False

    # Filter 2: Mean word length between 3 and 10
    if words:
        mean_word_length = sum(len(word) for word in words) / len(words)
        if mean_word_length < 3 or mean_word_length > 10:
            return False
    else:
        return False

    # Filter 3: Less than 30% of lines ending with ellipsis
    lines = text.strip().split('\n')
    lines_with_ellipsis = sum(1 for line in lines if line.rstrip().endswith('...'))
    if lines and (lines_with_ellipsis / len(lines)) > 0.3:
        return False

    # Filter 4: At least 80% of words have alphabetic characters
    words_with_alpha = sum(1 for word in words if any(c.isalpha() for c in word))
    if words and (words_with_alpha / len(words)) < 0.8:
        return False

    return True


def analyze_quality_filters(text_samples: List[str], sample_names: List[str] = None):
    """
    Analyze quality filter results on a set of text samples.

    Args:
        text_samples: List of text strings to analyze
        sample_names: Optional list of names/IDs for the samples
    """
    if sample_names is None:
        sample_names = [f"Sample {i+1}" for i in range(len(text_samples))]

    results = []
    for name, text in zip(sample_names, text_samples):
        passed = gopher_quality_filter(text)

        # Get detailed stats for analysis
        words = nltk.word_tokenize(text)
        word_count = len(words)
        mean_word_length = sum(len(word) for word in words) / len(words) if words else 0

        lines = text.strip().split('\n')
        ellipsis_ratio = sum(1 for line in lines if line.rstrip().endswith('...')) / len(lines) if lines else 0

        alpha_ratio = sum(1 for word in words if any(c.isalpha() for c in word)) / len(words) if words else 0

        results.append({
            'name': name,
            'passed': passed,
            'word_count': word_count,
            'mean_word_length': mean_word_length,
            'ellipsis_ratio': ellipsis_ratio,
            'alpha_ratio': alpha_ratio,
            'preview': text[:200] + '...' if len(text) > 200 else text
        })

    # Print results
    print("Quality Filter Analysis")
    print("=" * 80)

    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Passed: {'✓' if result['passed'] else '✗'}")
        print(f"  Word count: {result['word_count']} {'✓' if 50 <= result['word_count'] <= 100000 else '✗'}")
        print(f"  Mean word length: {result['mean_word_length']:.2f} {'✓' if 3 <= result['mean_word_length'] <= 10 else '✗'}")
        print(f"  Ellipsis ratio: {result['ellipsis_ratio']:.2%} {'✓' if result['ellipsis_ratio'] < 0.3 else '✗'}")
        print(f"  Alpha ratio: {result['alpha_ratio']:.2%} {'✓' if result['alpha_ratio'] >= 0.8 else '✗'}")
        print(f"  Preview: {result['preview']}")

    return results


# Test script
if __name__ == "__main__":
    # Test with some example texts
    test_texts = [
        # Good quality text
        """This is a well-formatted article about technology and innovation. 
        It contains multiple paragraphs with proper sentences and punctuation.
        The content is informative and suitable for language model training.
        """ * 20,  # Repeat to get enough words

        # Too short
        "This is too short.",

        # Too many ellipses
        "This is a page...\nWith too many...\nEllipses everywhere...\n" * 10,

        # Too many non-alphabetic
        "123 456 789 $$$$ #### @@@@ !!!! 1234567890" * 10,
    ]

    analyze_quality_filters(test_texts)
