import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
import heapq

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

def _create_frequency_table(text_string):
    """
    Creates a frequency table of words in the given text, excluding stopwords.
    """
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    freq_table = defaultdict(int)
    for word in words:
        word = word.lower()
        if word.isalpha() and word not in stop_words:
            freq_table[word] += 1
    return freq_table

def _score_sentences(sentences, freq_table):
    """
    Scores each sentence based on the frequency of words in the frequency table.
    """
    sentence_scores = defaultdict(int)
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in freq_table:
                sentence_scores[i] += freq_table[word]
    return sentence_scores

def _find_average_score(sentence_scores):
    """
    Calculates the average score of sentences.
    """
    sum_values = 0
    for entry in sentence_scores:
        sum_values += sentence_scores[entry]
    return (sum_values / len(sentence_scores)) if sentence_scores else 0

def summarize_extractive(text, top_n=5):
    """
    Generates an extractive summary of the given text.

    Args:
        text (str): The input text to summarize.
        top_n (int): The number of top sentences to include in the summary.

    Returns:
        str: The extractive summary.
    """
    sentences = sent_tokenize(text)
    if not sentences:
        return ""

    freq_table = _create_frequency_table(text)
    sentence_scores = _score_sentences(sentences, freq_table)

    # If no words are found or scored, return first sentence as fallback
    if not sentence_scores:
        return sentences[0]

    average_score = _find_average_score(sentence_scores)

    # Select sentences that are above 1.2 times the average score
    summary_sentences = []
    for i, sentence in enumerate(sentences):
        if i in sentence_scores and sentence_scores[i] > (1.2 * average_score):
            summary_sentences.append(sentence)

    # Fallback to top N sentences if threshold doesn't yield enough
    if len(summary_sentences) < top_n and sentence_scores:
        ranked_sentences = heapq.nlargest(top_n, sentence_scores, key=sentence_scores.get)
        summary_sentences = [sentences[j] for j in sorted(ranked_sentences)]

    return " ".join(summary_sentences)

if __name__ == "__main__":
    sample_text = """Artificial intelligence (AI) is rapidly transforming various aspects of our lives, from how we work and communicate to how we access information and make decisions. Machine learning, a subset of AI, has been particularly instrumental in this revolution. It involves training algorithms on vast amounts of data to identify patterns and make predictions or decisions without being explicitly programmed for each task. Deep learning, in turn, is a specialized branch of machine learning that uses neural networks with multiple layers (deep neural networks) to learn complex representations of data. This hierarchical learning allows deep learning models to achieve remarkable performance in areas like image recognition, natural language processing, and speech synthesis. The advancements in AI are driven by several factors, including the availability of large datasets, increased computational power, and innovative algorithmic research. However, challenges remain, such as ensuring fairness, transparency, and ethical considerations in AI systems. As AI continues to evolve, its impact on society will only grow, necessitating careful development and deployment strategies to harness its full potential while mitigating potential risks. The integration of AI into everyday applications, such as virtual assistants, recommendation systems, and autonomous vehicles, highlights its pervasive influence. Furthermore, AI is being applied in scientific research, healthcare, and environmental monitoring, promising breakthroughs in diverse fields. The future of AI is bright, but it requires a collaborative effort from researchers, engineers, policymakers, and the public to shape its trajectory responsibly."""
    
    print("\n--- Extractive Summary (NLTK-based) ---")
    ext_summary = summarize_extractive(sample_text, top_n=3)
    print(ext_summary)
