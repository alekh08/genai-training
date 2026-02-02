# Natural Language Processing (NLP)

## Core Components

### 1. NLU (Natural Language Understanding)
Focuses on enabling machines to **understand** human language, including intent, meaning, entities, and context.

### 2. NLG (Natural Language Generation)
Focuses on enabling machines to **generate** human-like language from structured or unstructured data.

---

## Natural Language Processing Pipeline

1. **Input Text**  
   Raw text or speech data (documents, chats, reviews, audio → text).

2. **Text Cleaning & Tokenization**  
   - Remove noise (punctuation, HTML tags, special characters)  
   - Tokenization  
   - Vocabulary creation

3. **Feature Extraction**  
   Conversion of text into numerical representations such as:
   - Bag of Words (BoW)
   - TF-IDF
   - Word Embeddings

4. **Model Creation / Training**  
   Machine Learning or Deep Learning models like:
   - Naive Bayes
   - SVM
   - RNN, LSTM
   - Transformers

5. **Output**  
   Classification, prediction, translation, summarization, or generated text.

---

## Phases (Faces) of NLP

### 1. Lexical Analysis
Word-level processing:
- Tokenization  
- Stop-word removal  
- Stemming  
- Lemmatization  

### 2. Syntactic Analysis
Grammatical structure of sentences:
- Part-of-Speech (POS) tagging  
- Dependency parsing  
- Sentence structure validation  

### 3. Semantic Analysis
Understanding the **meaning** of text:
- Word sense disambiguation  
- Named Entity Recognition (NER)  
- Meaning representation independent of syntax  

### 4. Discourse Analysis
Understanding relationships across sentences:
- Context checking  
- Co-reference resolution  
- Maintaining logical flow of text  

### 5. Pragmatic Analysis
Understanding real-world meaning and intent:
- Speaker intention  
- Situational context  
- Implicit meaning (sarcasm, indirect requests)

---

## Tools Used in NLP

1. **spaCy**  
   - High-speed NLP library  
   - Production-ready pipelines  
   - Strong for NER and dependency parsing  

2. **NLTK**  
   - Educational and research-oriented library  
   - Widely used for lexical analysis  

3. **Gensim**  
   - Word-to-vector conversion  
   - Word2Vec, Doc2Vec  
   - Topic modeling (LDA)

---

## Applications of NLP

1. Machine Translation  
2. Chatbots and Virtual Assistants  
3. Sentiment Analysis  
4. Text Summarization  
5. Speech Recognition  
6. Spam Detection  
7. Search Engines  
8. Question Answering Systems  

---

## Why NLP?

- Humans communicate primarily through language  
- Enables natural interaction between humans and machines  
- Automates analysis of large volumes of unstructured text  
- Improves decision-making using textual data  
- Forms the backbone of modern AI applications


Tokenization -- it is the process of breaking text into smaller units called tokens 
tokens--sentences,subwords

- ex --- "NLP is fun" -----> "NLP", "is", "fun"

## Why tokenization?
computers cannot understand any text. Using this is it easier to get numerical value
- Simplifies text processing – Splitting text into tokens makes it easier to manipulate and analyze.

- Enables feature extraction – Tokens act as features for models, allowing them to learn patterns and relationships.

- Improves language understanding – Helps capture syntactic and semantic structures for accurate NLP tasks.


## Challenges without tokenization

- meaning of text may change
- punctuation can be misinterpreted
- without this words can merge incorrectly


## Types of tokenization

1. Sentence --- Document summarization, text analysis

- e.g. ---> This is NLP. It is interesting
## Types of Tokenization

Tokenization is the process of breaking text into smaller units called *tokens*. Depending on the use case, different tokenization strategies are applied.

---

### 1. Sentence Tokenization
- Splits a document into individual sentences.
- Commonly used in **document summarization**, **text analysis**, and **sentiment analysis**.
- Handles punctuation such as periods, question marks, and exclamation marks.

**Example:**


   - Input : This is NLP. It is interesting.
   - Output : ["This is NLP.", "It is interesting."]

---

### 2. Word Tokenization
- Divides text into individual words.
- Widely used in **text classification**, **machine translation**, and **information retrieval**.

**Example:**

   - Input : NLP is interesting
   - Output : ["NLP", "is", "interesting"]
---

### 3. Whitespace Tokenization
- Splits text based strictly on whitespace (spaces, tabs, newlines).
- Simple and fast, but does **not** handle punctuation properly.

**Example:**
   - Input : NLP is interesting!
   - Output : ["NLP", "is", "interesting!"]



---

### 4. Treebank Tokenization
- A rule-based tokenization method used in **Penn Treebank**.
- Separates punctuation, contractions, and special characters.
- Commonly used in syntactic and grammatical analysis.

**Example:**

   - Input : Don't worry, it's fine.
   - Output : ["Do", "n't", "worry", ",", "it", "'s", "fine", "."]

---

### 5. Regular Expression (Regex) Tokenization
- Uses **regular expressions** to define custom tokenization rules.
- Highly flexible and suitable for domain-specific text processing.

**Example:**
   - Input : Email me at test123@gmail.com

   - Output : ["Email", "me", "at", "test123", "gmail", "com"]

---

### 6. Tweet Tokenization
- Designed specifically for **social media text** like tweets.
- Handles hashtags, mentions, emojis, URLs, and abbreviations.

**Example:**

   - Input : Loving NLP! 🚀 #AI @OpenAI
   - Output : ["Loving", "NLP", "!", "🚀", "#AI", "@OpenAI"]