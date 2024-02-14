import re
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import nltk

# Download the punkt sentence tokenizer resources if you haven't already.
nltk.download('punkt')

def html_to_sentences(html):
    # Parse the HTML using Beautiful Soup
    soup = BeautifulSoup(html, 'html.parser')

    # Extract text from the parsed HTML
    text = soup.get_text(separator=' ')

    # Use nltk to split text into sentences
    sentences = sent_tokenize(text)

    # Alternatively, you can use a regex to split text into sentences if you don't want to use nltk.
    # Note that regex is not as reliable as nltk for sentence tokenization.
    # sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    cleaned_sentences = []
    for sentence in sentences:
        # Remove leading and trailing whitespace
        sentence = sentence.strip()
        # Remove extra whitespace
        sentence = re.sub(r'\s+', ' ', sentence)

        # Remove non-ascii characters
        sentence = sentence.encode('ascii', 'ignore').decode('ascii')
        if sentence:
            cleaned_sentences.append(sentence)

    return cleaned_sentences


if __name__ == "__main__":
    with open("../../../output/6e3bce93-76c7-4691-8a1e-45c52fc44a46.html", "r") as f:
        html = f.read()

    sentences = html_to_sentences(html)
    print(sentences)
