import random
import os
from pathlib import Path
from collections import defaultdict
from typing import List


def exact_line_deduplication(
    input_files: List[os.PathLike], output_directory: os.PathLike
):
    """Remove duplicate lines across all documents in the corpus."""
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # First pass: count line frequencies across all files
    line_counts = defaultdict(int)

    for input_file in input_files:
        input_path = Path(input_file)
        with open(input_path, 'r') as f:
            for line in f:
                # Use hash of the line to save memory
                line_hash = hash(line.strip())
                if line.strip():  # Only count non-empty lines
                    line_counts[line_hash] += 1

    # Second pass: write files keeping only unique lines
    for input_file in input_files:
        input_path = Path(input_file)
        output_path = output_dir / input_path.name

        with open(input_path, 'r') as f:
            content = f.read()

        lines = content.splitlines(keepends=True)
        deduplicated_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:  # Keep empty lines
                deduplicated_lines.append(line)
            else:
                line_hash = hash(stripped)
                if line_counts[line_hash] == 1:  # Only keep unique lines
                    deduplicated_lines.append(line)

        with open(output_path, 'w') as f:
            output = ''.join(deduplicated_lines)
            if not content.endswith('\n') and output.endswith('\n'):
                output = output[:-1]
            f.write(output)


class MinHasher:
    """Simple MinHash implementation for document deduplication."""

    def __init__(self, num_hashes: int, seed: int = 42):
        self.num_hashes = num_hashes
        # Generate random hash functions
        random.seed(seed)
        self.hash_funcs = [
            (random.randint(1, 2**32), random.randint(1, 2**32))
            for _ in range(num_hashes)
        ]

    def get_shingles(self, text: str, n: int):
        """Get n-grams (shingles) from text."""
        words = text.split()
        if len(words) < n:
            return {text}  # Return the whole text as a single shingle
        return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))

    def hash_shingle(self, shingle: str, a: int, b: int):
        """Hash a shingle using linear hash function."""
        return (a * hash(shingle) + b) % (2**32)

    def minhash(self, text: str, ngrams: int):
        """Compute MinHash signature for text."""
        shingles = self.get_shingles(text, ngrams)
        if not shingles:
            return [0] * self.num_hashes

        signature = []
        for a, b in self.hash_funcs:
            min_hash = min(self.hash_shingle(shingle, a, b)
                           for shingle in shingles)
            signature.append(min_hash)
        return signature

    def jaccard_similarity(self, sig1: List[int], sig2: List[int]):
        """Estimate Jaccard similarity from MinHash signatures."""
        if len(sig1) != len(sig2):
            return 0.0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)


def minhash_deduplication(
    input_files: List[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    """
    Perform MinHash-based deduplication on input files.

    Args:
        input_files: List of input file paths
        num_hashes: Number of hash functions for MinHash
        num_bands: Number of bands for LSH
        ngrams: N-gram size for shingling
        jaccard_threshold: Jaccard similarity threshold for duplicates
        output_directory: Directory to write deduplicated files
    """
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    hasher = MinHasher(num_hashes)

    all_documents = []

    # Read each file as a single document
    for input_file in input_files:
        input_path = Path(input_file)

        with open(input_path, 'r', encoding='utf-8') as infile:
            text = infile.read().strip()
            if text:  # Skip empty files
                signature = hasher.minhash(text, ngrams)
                all_documents.append({
                    'text': text,
                    'signature': signature,
                    'file': input_path
                })

    # Simple LSH using bands (simplified implementation)
    rows_per_band = num_hashes // num_bands
    buckets = defaultdict(list)

    for i, doc in enumerate(all_documents):
        for band in range(num_bands):
            start = band * rows_per_band
            end = start + rows_per_band
            band_sig = tuple(doc['signature'][start:end])
            buckets[band_sig].append(i)

    # Find candidate pairs and check similarity
    duplicate_indices = set()

    for bucket in buckets.values():
        if len(bucket) > 1:
            # Check all pairs in this bucket
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    idx1, idx2 = bucket[i], bucket[j]
                    if idx1 not in duplicate_indices and idx2 not in duplicate_indices:
                        sim = hasher.jaccard_similarity(
                            all_documents[idx1]['signature'],
                            all_documents[idx2]['signature']
                        )
                        if sim >= jaccard_threshold:
                            # Mark the second document as duplicate
                            duplicate_indices.add(idx2)

    # Write non-duplicate files
    for i, doc in enumerate(all_documents):
        if i not in duplicate_indices:
            output_path = output_dir / doc['file'].name
            with open(output_path, 'w', encoding='utf-8') as f:
                # Read the original file to preserve the exact format
                with open(doc['file'], 'r', encoding='utf-8') as orig:
                    original_content = orig.read()
                f.write(original_content)
