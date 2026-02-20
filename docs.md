# Natural Language Processing (NLP) — Study Notes

---

## 1. Core Components

**NLU (Natural Language Understanding)** — Enables machines to *understand* human language, including intent, meaning, entities, and context.

**NLG (Natural Language Generation)** — Enables machines to *generate* human-like language from structured or unstructured data.

---

## 2. NLP Pipeline

| Step | Description |
|------|-------------|
| **1. Input Text** | Raw text or speech data (documents, chats, reviews, audio → text) |
| **2. Text Cleaning & Tokenization** | Remove noise, tokenize, build vocabulary |
| **3. Feature Extraction** | Convert text to numbers: BoW, TF-IDF, Word Embeddings |
| **4. Model Training** | Naive Bayes, SVM, RNN, LSTM, Transformers |
| **5. Output** | Classification, translation, summarization, generated text |

---

## 3. Phases of NLP

### Lexical Analysis
Word-level processing: tokenization, stop-word removal, stemming, lemmatization.

### Syntactic Analysis
Grammatical structure: POS tagging, dependency parsing, sentence structure validation.

### Semantic Analysis
Understanding meaning: word sense disambiguation, Named Entity Recognition (NER), meaning independent of syntax.

### Discourse Analysis
Cross-sentence understanding: context checking, co-reference resolution, logical flow.

### Pragmatic Analysis
Real-world intent: speaker intention, situational context, implicit meaning (sarcasm, indirect requests).

---

## 4. Tokenization

**Definition:** Breaking text into smaller units called *tokens* (words, sentences, or subwords).

**Example:** `"NLP is fun"` → `["NLP", "is", "fun"]`

### Why Tokenization?
- Computers cannot process raw text — tokens enable numerical conversion
- Simplifies text processing and manipulation
- Enables feature extraction for model learning
- Improves syntactic and semantic understanding

### Challenges Without Tokenization
- Meaning of text may change
- Punctuation can be misinterpreted
- Words can merge incorrectly

---

### Types of Tokenization

#### 1. Sentence Tokenization
Splits a document into individual sentences. Used in summarization, text analysis, sentiment analysis.

> **Input:** `This is NLP. It is interesting.`  
> **Output:** `["This is NLP.", "It is interesting."]`

#### 2. Word Tokenization
Divides text into individual words. Used in text classification, machine translation, information retrieval.

> **Input:** `NLP is interesting`  
> **Output:** `["NLP", "is", "interesting"]`

#### 3. Whitespace Tokenization
Splits on whitespace only. Fast and simple, but does **not** handle punctuation.

> **Input:** `NLP is interesting!`  
> **Output:** `["NLP", "is", "interesting!"]`

#### 4. Treebank Tokenization
Rule-based method from Penn Treebank. Separates contractions and special characters. Used in syntactic/grammatical analysis.

> **Input:** `Don't worry, it's fine.`  
> **Output:** `["Do", "n't", "worry", ",", "it", "'s", "fine", "."]`

#### 5. Regular Expression (Regex) Tokenization
Uses regex patterns for custom rules. Highly flexible for domain-specific text.

> **Input:** `Email me at test123@gmail.com`  
> **Output:** `["Email", "me", "at", "test123", "gmail", "com"]`

#### 6. Tweet Tokenization
Designed for social media. Handles hashtags, mentions, emojis, URLs.

> **Input:** `Loving NLP! 🚀 #AI @OpenAI`  
> **Output:** `["Loving", "NLP", "!", "🚀", "#AI", "@OpenAI"]`

**Real-world examples:** Chatbots, voice assistants, search engines.

---

## 5. Text Normalization

Raw text must be cleaned and standardized before processing. Two key techniques are **stemming** and **lemmatization**, both of which reduce words to their base form.

---

### Stemming

**Definition:** Removes suffixes (sometimes prefixes) to reduce a word to its *stem* — which may or may not be a valid dictionary word.

> **Example:** `playing` → `play`

**Uses:** Information retrieval, search engines, text classification.

#### Porter Stemmer
- Oldest and most widely used algorithm (Martin Porter, 1980)
- Rule-based: applies a sequence of suffix-removal rules
- `"ing"` is replaced with nothing

| Advantages | Disadvantages |
|------------|---------------|
| Fast, simple, widely supported | No understanding of meaning; may produce nonsense words |

#### Snowball Stemmer (Porter 2)
- Improved version of Porter Stemmer
- More consistent and accurate
- Supports multiple languages (English, German, Spanish, etc.)

| Word | Porter | Snowball |
|------|--------|----------|
| studies | study | study |
| fairness | fair | fair |
| generational | gener | generat |

| Advantages | Disadvantages |
|------------|---------------|
| More accurate, multilingual support | Not context-aware; output may not be a valid word |

---

### Lemmatization

**Definition:** Reduces words to their base/dictionary form (*lemma*) by considering vocabulary, morphology, and **part of speech (POS)**. Output is always a valid word.

| Word | POS | Lemma |
|------|-----|-------|
| running | verb | run |
| better | adj | good |
| cars | noun | car |
| studies | verb | study |

**Why POS matters:** The same word can have different lemmas depending on its grammatical role (e.g., "better" as adjective → "good"; "better" as verb → "better").

#### WordNet Lemmatizer
Uses WordNet (a lexical database of English) to find the correct lemma.

---

### Stemming vs. Lemmatization — When to Use Which?

| Task | Recommended |
|------|-------------|
| Search engines, IR | Stemming (speed matters) |
| Chatbots, machine translation | Lemmatization (accuracy matters) |
| Text classification | Either, depending on dataset |

---

## 6. Vectorization

**Definition:** Converting text into numerical format so machine learning models can process it.

---

### Bag of Words (BoW)

Represents text as word frequency counts, ignoring grammar and word order.

**Example corpus:**
```
I love NLP
I love ML
NLP loves ML
```

A vocabulary is built from all unique words, and each sentence is represented as a vector of word counts.

---

### TF-IDF (Term Frequency–Inverse Document Frequency)

Weights words by how important they are in a document relative to the entire corpus.

**Term Frequency:**
$$TF(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total words in } d}$$

> **Example:** `NLP is easy and NLP is powerful` → TF("NLP") = 2/6 ≈ 0.333

**Inverse Document Frequency:**
$$IDF(t) = \log\left(\frac{N}{df(t)}\right)$$

- *N* = total number of documents
- *df(t)* = number of documents containing term *t*

**Why normalize?** Long documents naturally have higher raw counts. IDF penalizes common words (like "the", "is") that appear in many documents and rewards rare, informative words.

**TF-IDF Score:** `TF × IDF`

---

### N-grams

| Type | Description | Example (input: "I love NLP") |
|------|-------------|-------------------------------|
| **Unigram** | Single words | `["I", "love", "NLP"]` |
| **Bigram** | Pairs of consecutive words | `["I love", "love NLP"]` |

N-grams capture some word order and context that BoW misses.

---

### Word Embeddings

**Definition:** Representing words as dense numeric vectors where similar words have similar vector representations. Instead of sparse 10,000-dimensional vectors (like BoW), embeddings use dense 100–300 dimensional vectors (e.g., `[0.24, -0.14, ...]`).

#### Word2Vec (Google, 2013)
A neural network-based embedding model with two architectures:

| Architecture | How it Works |
|--------------|--------------|
| **CBOW** (Continuous Bag of Words) | Predicts the target word from surrounding context words |
| **Skip-Gram** | Predicts surrounding context words from the target word |

---

## 7. NLP Tools

| Tool | Description | Strengths |
|------|-------------|-----------|
| **spaCy** | High-speed, production-ready NLP library | NER, dependency parsing |
| **NLTK** | Educational and research-oriented | Lexical analysis, wide task coverage |
| **Gensim** | Word/doc vectorization and topic modeling | Word2Vec, Doc2Vec, LDA |

---

## 8. Applications of NLP

Machine Translation, Chatbots & Virtual Assistants, Sentiment Analysis, Text Summarization, Speech Recognition, Spam Detection, Search Engines, Question Answering Systems.

---

## 9. Why NLP?

- Humans communicate primarily through language
- Enables natural interaction between humans and machines
- Automates analysis of large volumes of unstructured text
- Improves decision-making using textual data
- Forms the backbone of modern AI applications

---

## 10. Project: Sentiment Classifier

**Task:** Train a Logistic Regression / Naive Bayes / SVM sentiment classifier on TF-IDF features and print a classification report and confusion matrix.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Vectorize
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 2. Train
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 3. Evaluate
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

