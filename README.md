# NLP Text Summarizer

## Project Overview

This repository hosts an advanced Natural Language Processing (NLP) project focused on text summarization. It implements both abstractive and extractive summarization techniques, leveraging state-of-the-art transformer models to generate concise and coherent summaries from longer texts. The project aims to provide a robust solution for various summarization tasks, from news articles to scientific papers.

## Features

*   **Abstractive Summarization:** Generates new sentences to form a summary, capturing the core meaning of the original text.
*   **Extractive Summarization:** Identifies and extracts the most important sentences directly from the source document.
*   **Transformer Models:** Utilizes pre-trained models like BERT, GPT, and T5 for high-quality summarization.
*   **Customizable Pipelines:** Flexible architecture allowing easy integration of different models and datasets.
*   **Evaluation Metrics:** Includes ROUGE scores and other metrics for comprehensive model evaluation.

## Technologies Used

*   **Python:** Core programming language.
*   **Hugging Face Transformers:** For accessing and utilizing pre-trained NLP models.
*   **PyTorch/TensorFlow:** Deep learning frameworks for model implementation.
*   **NLTK/SpaCy:** For text preprocessing and linguistic analysis.
*   **Scikit-learn:** For traditional machine learning components and evaluation.

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. Install the necessary libraries using pip:

```bash
pip install transformers torch numpy nltk spacy scikit-learn
python -m spacy download en_core_web_sm
```

### Installation

1.  Clone the repository:

    ```bash
git clone https://github.com/Thowed5/NLP-Text-Summarizer.git
cd NLP-Text-Summarizer
    ```

2.  (Optional) Set up a virtual environment:

    ```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

### Usage

To run extractive summarization:

```bash
python summarize_extractive.py --text_file path/to/document.txt --length 3
```

To run abstractive summarization (requires a pre-trained model):

```bash
python summarize_abstractive.py --text_file path/to/document.txt --model_name t5-small
```

## Project Structure

```
. 
├── data/                 # Sample input texts and datasets
├── models/               # Fine-tuned summarization models
├── src/                  # Source code for summarization logic
│   ├── __init__.py
│   ├── extractive.py     # Extractive summarization implementation
│   ├── abstractive.py    # Abstractive summarization implementation
│   └── utils.py          # Utility functions
├── summarize_extractive.py # Script for extractive summarization
├── summarize_abstractive.py # Script for abstractive summarization
├── README.md             # Project README file
└── requirements.txt      # Python dependencies
```

## Contributing

Contributions are highly encouraged! Please open issues for bugs or feature requests, and submit pull requests for improvements.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
Adding additional examples to the summarization script.
