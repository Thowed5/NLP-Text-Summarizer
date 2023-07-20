import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os

# --- 1. Configuration ---
# Define a list of common summarization models to try
# Smaller models are chosen for sandbox compatibility and faster execution
SUMMARIZER_MODELS = [
    "sshleifer/distilbart-cnn-6-6", # Good balance of speed and quality
    "t5-small",
    "facebook/bart-large-cnn", # Larger, higher quality
]

# --- 2. Text Input ---
def get_sample_text():
    # A longer sample text for summarization
    return """Artificial intelligence (AI) is rapidly transforming various aspects of our lives, from how we work and communicate to how we access information and make decisions. Machine learning, a subset of AI, has been particularly instrumental in this revolution. It involves training algorithms on vast amounts of data to identify patterns and make predictions or decisions without being explicitly programmed for each task. Deep learning, in turn, is a specialized branch of machine learning that uses neural networks with multiple layers (deep neural networks) to learn complex representations of data. This hierarchical learning allows deep learning models to achieve remarkable performance in areas like image recognition, natural language processing, and speech synthesis. The advancements in AI are driven by several factors, including the availability of large datasets, increased computational power, and innovative algorithmic research. However, challenges remain, such as ensuring fairness, transparency, and ethical considerations in AI systems. As AI continues to evolve, its impact on society will only grow, necessitating careful development and deployment strategies to harness its full potential while mitigating potential risks. The integration of AI into everyday applications, such as virtual assistants, recommendation systems, and autonomous vehicles, highlights its pervasive influence. Furthermore, AI is being applied in scientific research, healthcare, and environmental monitoring, promising breakthroughs in diverse fields. The future of AI is bright, but it requires a collaborative effort from researchers, engineers, policymakers, and the public to shape its trajectory responsibly."""

# --- 3. Summarization Function ---
def perform_summarization(text, model_name):
    print(f"\n--- Performing summarization with model: {model_name} ---")
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Create summarization pipeline
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

        # Generate summary
        # max_length and min_length can be adjusted based on desired summary length
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        print("Original Text Length:", len(text.split()))
        print("Summary Length:", len(summary[0]["summary_text"].split()))
        print("Summary:", summary[0]["summary_text"])
        return summary[0]["summary_text"]
    except Exception as e:
        print(f"Error with model {model_name}: {e}")
        return None

# --- 4. Main Execution ---
if __name__ == "__main__":
    sample_text = get_sample_text()
    summaries = []

    for model_name in SUMMARIZER_MODELS:
        summary = perform_summarization(sample_text, model_name)
        if summary:
            summaries.append((model_name, summary))

    if not summaries:
        print("No summaries could be generated. Please check your internet connection or try different models.")
    else:
        print("\n--- All Summaries Generated ---")
        for model_name, summary_text in summaries:
            print(f"Model: {model_name}\nSummary: {summary_text}\n")

    # Create requirements.txt
    with open("requirements.txt", "w") as f:
        f.write("torch\n")
        f.write("transformers\n")
        f.write("sentencepiece\n") # Often a dependency for transformer models
        f.write("numpy\n")
    print("requirements.txt created.")


# --- 5. Example Usage for Extractive Summarization (using NLTK) ---
# This is a simple example and can be expanded with more sophisticated algorithms
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
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    freq_table = defaultdict(int)
    for word in words:
        word = word.lower()
        if word not in stop_words:
            freq_table[word] += 1
    return freq_table

def _score_sentences(sentences, freq_table):
    sentence_scores = defaultdict(int)
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in freq_table:
                sentence_scores[i] += freq_table[word]
    return sentence_scores

def _find_average_score(sentence_scores):
    sum_values = 0
    for entry in sentence_scores:
        sum_values += sentence_scores[entry]
    return (sum_values / len(sentence_scores)) if sentence_scores else 0

def generate_extractive_summary(text, top_n=5):
    sentences = sent_tokenize(text)
    freq_table = _create_frequency_table(text)
    sentence_scores = _score_sentences(sentences, freq_table)
    average_score = _find_average_score(sentence_scores)

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
    # ... (previous main execution block)

    print("\n--- Performing Extractive Summarization (NLTK-based) ---")
    extractive_summary = generate_extractive_summary(sample_text, top_n=3)
    print("Extractive Summary:", extractive_summary)

    with open("requirements.txt", "a") as f:
        f.write("nltk\n")
    print("Updated requirements.txt with nltk.")
