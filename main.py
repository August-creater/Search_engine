import math # For TF-IDF Calculation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as EnglishStopWords # For Removing Stopwords
import re # For Tokenization and Preprocessing URLs (removing punctuation, lowercasing, etc.)
from Web_indexer import index, get_links # using a web indexer that I created to get text/cleaned HTML within the scope of the website crawler

# Preprocessing Url that removes punctuation, lowercases, and filters stopwords to highlight keywords off the website
def preprocess_url(pre_url):
    try:
        pre_url = pre_url.encode('utf-8').decode('ascii', 'ignore') # encoding and decoding to remove non-ascii characters
    except AttributeError:
       pass
    # Tokenizing the website
    tokens = re.findall(r'\b[a-z]{2,}\b', pre_url.lower())
    filtered_tokens = [token for token in tokens if token not in EnglishStopWords]
    # Lowercasing, removing punctuation, and filtering stopwords
    return filtered_tokens

# Processed URLs
def processed_url(url_list):
    documents = []
    for url in url_list:
        try:
         raw_text = index(url)
         if raw_text: #using index.py to get text/cleaned HTML
            Tokenize_url = preprocess_url(raw_text)
            documents.append(Tokenize_url)
        except Exception as e:
            print(f"Error processing URL: {url}. Error: {e}")
        return documents
    return None


# TF-IDF Calculations
def term_frequency(word, words):
    return words.count(word) / len(words) if words else 0


class TFIDF:
    def __init__(self, corpus=None):
        self.corpus = corpus or []
        self.doc_count = len(self.corpus)

    def add_documents(self, documents):
        self.corpus.extend(documents)
        self.doc_count = len(self.corpus)

    def n_containing(self, word, corpus=None):
        corpus = corpus or self.corpus
        return sum(word in doc for doc in corpus)

    def inverse_doc_freq(self, word, corpus=None):
        corpus = corpus or self.corpus
        num_docs = len(corpus)
        num_containing_word = self.n_containing(word, corpus)
        return math.log(num_docs / (1 + num_containing_word))

    def tfidf(self, word, words, corpus=None):
        tf = term_frequency(word, words)
        idf = self.inverse_doc_freq(word, corpus)
        return tf * idf

    def get_document_scores(self, document, top_n=None):
        scores = {word: self.tfidf(word, document) for word in set(document)}
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_n] if top_n else sorted_scores

    @staticmethod # static method
    def analyze_document(document, top_n=5):
        tfidf_analyzer = TFIDF(document)
        for i, document in enumerate(documents):
            score = tfidf_analyzer.get_document_scores(document, top_n)
            sorted_words = sorted(score.items(), key=lambda x: x[1], reverse=True)
            for word, s in sorted_words[:top_n]:
                print(f"Document {i + 1}: {word} (TF-IDF: {s:.2f})")


# Main
if __name__ == "__main__":
    url = "https://www.delish.com/"
    try:
        url_list = list(get_links(url))[:2]
        processed_url(url_list)
    except Exception  and ImportError as e:
        print(f"Error indexing URL: {url}. Error: {e}")