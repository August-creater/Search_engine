import math
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as EnglishStopWords
import re

# Preprocessing Function
def preprocess_text(doc):
    stop_words = EnglishStopWords # Load stopwords
    # Tokenizing the document
    tokens = re.findall(r'\w+', doc.lower())
    # Lowercasing, removing punctuation, and filtering stopwords
    cleaned_tokens = [
        word for word in tokens if word.lower() not in stop_words
    ]
    # Join cleaned tokens back into a single document
    return cleaned_tokens

# TF-IDF Calculation Functions
def term_frequency(word, words):
    return words.count(word) / len(words) if words else 0

def n_containing(word, doc_list):
    return sum(1 for doc in doc_list if word in doc)

def inverse_doc_freq(word, new_list):
    return math.log(1 +len(new_list) / (1 + n_containing(word, new_list))) + 1

def tfidf(word, words, corpus):
    return term_frequency(word, words) * inverse_doc_freq(word, corpus)
# Dataset
document1 = ("""Python is a 2000 made-for-TV horror movie directed by Richard
Limbaugh. The film features several cult favorite actors, including William
Zabka of The Karate Kid fame, Wil Wheaton, Casper Van Dien, Jenny McCarthy,
Keith Coogan, Robert Englund (best known for his role as Freddy Krueger in the
A Nightmare on Elm Street series of films), Dana Barron, David Bowe, and Sean
Whalen. The film concerns a genetically engineered snake, a python, that
escapes and unleashes itself on a small town. It includes the classic final
girl scenario evident in films like Friday the 13th. It was filmed in Los Angeles,
 California and Malibu, California. Python was followed by two sequels: Python
 II (2002) and Boa vs. Python (2004), both also made-for-TV films.""")

document2 = ("""Python is an interpreted, high-level, general-purpose programming
language. Its design philosophy emphasizes code readability with its use of
whitespace indentation. Its language constructs and object-oriented model encourage""")

better_list = [document1, document2]

# Preprocess the dataset
processed_list = [preprocess_text(doc) for doc in better_list]

# Calculate TF-IDF Scores
# Calculate TF-IDF Scores
for i, document in enumerate(processed_list):
    print("Most common words in doc {}".format(i + 1))
    score = {word: tfidf(word, document, processed_list) for word in set(document)}
    sorted_words = sorted(score.items(), key=lambda x: x[1], reverse=True)
    for word, s in sorted_words[:5]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(s, 5)))

    print("Least common words in doc {}".format(i + 1))
    sorted_words = sorted(score.items(), key=lambda x: x[1])
    for word, s in sorted_words[:5]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(s, 5)))