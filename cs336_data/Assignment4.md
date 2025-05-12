# CS336 Assignment 4: Filtering Language Modeling Data

## 1. Assignment Overview

Implement:

* Convert Common Crawl HTML to text.
* Filter extracted text (harmful content, personal identifiable information, etc.).
* Deduplicate training data.

Run:

* Train language models on different datasets.

## 2. Filtering Common Crawl

### Problem (look\_at\_cc) - 4 points

**(a)**

* URL:
* Accessible?:
* Content description:

**(b)**

* Text that should've been filtered:
* Issues for training:
* Useful information:

**(c)**

* Application domain useful:
* Application domain not useful:

**(d)**

* Annotations of 25 documents:

### Problem (extract\_text) - 3 points

**(a)**

* Function implementation:

**(b)**

* Comparison to WET file extraction:

### Problem (language\_identification) - 6 points

**(a)**

* Language identification function implementation:

**(b)**

* Potential issues:
* Mitigation strategy:

**(c)**

* Classifier accuracy observations:
* English document fraction:
* Recommended threshold:

### Problem (mask\_pii) - 3 points

1\. **Mask Emails:**

* Implementation:

2\. **Mask Phone Numbers:**

* Implementation:

3\. **Mask IP Addresses:**

* Implementation:

4\. **Potential downstream issues and mitigation:**

5\. **False positives/negatives examples:**

### Problem (harmful\_content) - 6 points

1\. **NSFW Detection:**

* Implementation:

2\. **Toxic Speech Detection:**

* Implementation:

3\. **Potential downstream issues and mitigation:**

4\. **Classifier errors and suitable threshold:**

### Problem (gopher\_quality\_filters) - 3 points

**(a)**

* Quality filters implementation:

**(b)**

* Filter accuracy observations:

### Problem (quality\_classifier) - 15 points

**(a)**

* Quality classifier implementation:

**(b)**

* Labeling and confidence score implementation:

## 3. Deduplication

### Problem (exact\_deduplication) - 3 points

* Implementation:

### Problem (minhash\_deduplication) - 8 points

* Implementation details:

## 4. Leaderboard: Filter Data for Language Modeling

### Problem (filter\_data) - 6 points

**(a)**

* Filtering script:
* Filter steps breakdown:

**(b)**

* Runtime analysis:

### Problem (inspect\_filtered\_data) - 4 points

**(a)**

* Five random examples (with quality analysis):

**(b)**

* Five discarded examples (with justification):

**(c)**

* Iterations and improvements:

### Problem (tokenize\_data) - 2 points

* Tokenization script:
* Number of tokens:

### Problem (train\_model) - 2 points

* Best validation loss:
* Learning curve:
* Description of actions taken:
