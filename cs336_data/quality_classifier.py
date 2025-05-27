import gzip
import random
import fasttext
from typing import Tuple, Optional
import requests
from cs336_data.utils import extract_text_from_html_bytes, find_data_file
from cs336_data.gopher_filters import gopher_quality_filter
from cs336_data.language import identify_language
import os

# Cache for loaded model
_quality_model_cache = {}


def load_quality_model(model_path: str = 'data/quality_classifier.bin'):
    """Load the quality classifier model with caching."""
    if model_path not in _quality_model_cache:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Quality classifier model not found at {model_path}. "
                "Please train the model first."
            )
        _quality_model_cache[model_path] = fasttext.load_model(model_path)
    return _quality_model_cache[model_path]


def download_url_content(url: str, timeout: int = 5) -> Optional[str]:
    """Download and extract text content from a URL."""
    try:
        response = requests.get(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()

        html_bytes = response.content
        text = extract_text_from_html_bytes(html_bytes)

        return text
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


def prepare_training_data(
    wiki_urls_file: str,
    num_positive: int = 10000,
    num_negative: int = 10000,
    output_file: str = 'quality_training_data.txt'
):
    """
    Prepare training data for quality classifier.

    Args:
        wiki_urls_file: Path to Wikipedia URLs file
        num_positive: Number of positive examples to collect
        num_negative: Number of negative examples to collect
        output_file: Output file for training data
    """
    print("Preparing training data for quality classifier...")

    # Load and sample Wikipedia URLs
    print(f"Loading URLs from {wiki_urls_file}...")
    wiki_urls = []

    with gzip.open(wiki_urls_file, 'rt') as f:
        for line in f:
            url = line.strip()
            # Filter out some common non-content URLs
            if url and not any(skip in url for skip in [
                'wikipedia.org', 'wikimedia.org', 'archive.org',
                '.pdf', '.jpg', '.png', '.gif'
            ]):
                wiki_urls.append(url)

    print(f"Loaded {len(wiki_urls)} URLs")

    # Sample URLs
    if len(wiki_urls) > num_positive:
        sampled_urls = random.sample(wiki_urls, num_positive * 2)  # Sample more to account for failures
    else:
        sampled_urls = wiki_urls

    # Collect positive examples with statistics
    positive_examples = []
    stats = {
        'attempted': 0,
        'successful': 0,
        'failed_download': 0,
        'failed_language': 0,
        'failed_quality': 0,
    }

    print(f"Downloading {num_positive} positive examples...")

    for i, url in enumerate(sampled_urls):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(sampled_urls)} - "
                  f"Success rate: {stats['successful']}/{stats['attempted']} "
                  f"({stats['successful']/max(1, stats['attempted'])*100:.1f}%)")

        stats['attempted'] += 1
        text = download_url_content(url)

        if not text:
            stats['failed_download'] += 1
            continue

        # Apply quality filters
        lang, _ = identify_language(text)
        if lang != 'en':
            stats['failed_language'] += 1
            continue

        if not gopher_quality_filter(text):
            stats['failed_quality'] += 1
            continue

        stats['successful'] += 1
        clean_text = ' '.join(text.split())[:1000]  # Limit length
        positive_examples.append(clean_text)

        if len(positive_examples) >= num_positive:
            break

    # Print final statistics
    print(f"\n=== Download Statistics ===")
    print(f"Total attempted: {stats['attempted']}")
    print(f"Successful: {stats['successful']} ({stats['successful']/stats['attempted']*100:.1f}%)")
    print(f"Failed download: {stats['failed_download']} ({stats['failed_download']/stats['attempted']*100:.1f}%)")
    print(f"Failed language filter: {stats['failed_language']} ({stats['failed_language']/stats['attempted']*100:.1f}%)")
    print(f"Failed quality filter: {stats['failed_quality']} ({stats['failed_quality']/stats['attempted']*100:.1f}%)")
    print(f"Collected {len(positive_examples)} positive examples")

    # For negative examples, we'll use synthetic low-quality text
    # TODO: switch to Common Crawl samples
    negative_examples = []
    print("Generating negative examples...")

    # Types of low-quality content
    for i in range(num_negative):
        example_type = i % 5

        if example_type == 0:
            # Spam/promotional content
            text = "Buy now! Best prices! Click here! " * 20
            text += "Limited offer! Act now! Free shipping!"
        elif example_type == 1:
            # Low content density
            text = "Home | About | Contact | Services | Blog\n" * 10
            text += "Copyright 2024. All rights reserved."
        elif example_type == 2:
            # Error pages
            text = "404 Error - Page Not Found\n"
            text += "The page you requested could not be found."
        elif example_type == 3:
            # Repetitive content
            word = random.choice(['test', 'lorem', 'data', 'page'])
            text = f"{word} " * 100
        else:
            # Mixed non-English
            text = "这是 test страница для тестирования مع بعض النص"
            text += " Some English mixed with other languages."

        negative_examples.append(text)

    print(f"Generated {len(negative_examples)} negative examples")

    # Write training data
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in positive_examples:
            f.write(f"__label__high {text}\n")
        for text in negative_examples:
            f.write(f"__label__low {text}\n")

    print(f"Training data saved to {output_file}")
    return output_file


def train_quality_classifier(
    training_file: str = 'quality_training_data.txt',
    model_path: str = 'data/quality_classifier.bin',
    **kwargs
):
    """
    Train a fastText quality classifier.

    Args:
        training_file: Path to training data file
        model_path: Output path for trained model
        **kwargs: Additional fastText parameters
    """
    print("Training quality classifier...")

    # Default parameters
    params = {
        'input': training_file,
        'epoch': kwargs.get('epoch', 25),
        'lr': kwargs.get('lr', 0.5),
        'wordNgrams': kwargs.get('wordNgrams', 2),
        'dim': kwargs.get('dim', 100),
        'loss': kwargs.get('loss', 'softmax')
    }

    model = fasttext.train_supervised(**params)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)

    print(f"Model saved to {model_path}")

    # Test on training data
    result = model.test(training_file)
    print(f"Training accuracy: {result[1]:.2%}")

    return model


def classify_quality(text: str, model_path: str = 'data/quality_classifier.bin') -> Tuple[str, float]:
    """
    Classify text quality and return label with confidence.

    Args:
        text: Input text to classify
        model_path: Path to trained model

    Returns:
        Tuple of (label, confidence) where label is 'high' or 'low'
    """
    if not text or not text.strip():
        return 'low', 1.0

    model = load_quality_model(model_path)

    # Clean text for classification
    clean_text = ' '.join(text.split())[:1000]  # Limit length

    labels, scores = model.predict(clean_text, k=2)

    top_label = labels[0].replace('__label__', '')
    confidence = scores[0]

    return top_label, confidence


def get_quality_score(text: str, model_path: str = 'data/quality_classifier.bin') -> float:
    """
    Get a numeric quality score for text (0-1, higher is better).

    Args:
        text: Input text to score
        model_path: Path to trained model

    Returns:
        Quality score between 0 and 1
    """
    label, confidence = classify_quality(text, model_path)

    if label == 'high':
        # High quality: score is confidence
        return confidence
    else:
        # Low quality: score is 1 - confidence
        return 1.0 - confidence


if __name__ == "__main__":

    wiki_urls_file = find_data_file("enwiki-20240420-extracted_urls.txt.gz", [
        "data/wiki/enwiki-20240420-extracted_urls.txt.gz",  # from project root
        "../data/wiki/enwiki-20240420-extracted_urls.txt.gz",  # from cs336_data dir
        "/data/wiki/enwiki-20240420-extracted_urls.txt.gz",  # cluster
    ])

    model_path = 'data/quality_classifier.bin'
    if not os.path.exists(model_path):
        print("Model not found. Training new model...")

        training_file = prepare_training_data(
            wiki_urls_file,
            num_positive=5000,
            num_negative=5000
        )

        train_quality_classifier(training_file, model_path)

    # Test the classifier
    test_texts = [
        # High quality example
        """Machine learning is a subset of artificial intelligence that
        enables systems to learn and improve from experience without being
        explicitly programmed. It focuses on developing computer programs
        that can access data and use it to learn for themselves.""",

        # Low quality example
        "Click here! Buy now! Best prices! Limited offer!"
    ]

    for tt in test_texts:
        lab, conf = classify_quality(tt)
        score = get_quality_score(tt)
        print(f"\nText: {tt[:50]}...")
        print(f"Label: {lab}, Confidence: {conf:.2f}")
        print(f"Quality score: {score:.2f}")
