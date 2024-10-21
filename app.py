from flask import Flask, request, render_template
import pandas as pd
from text_cleaner import clean_article
from term_doc_mat import term_doc_mat
import re

app = Flask(__name__)


def highlight_matches(text, words):
    """Highlight matching words by wrapping them in a span with a highlight class."""
    pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in words) + r')\b', re.IGNORECASE)
    highlighted_text = pattern.sub(r'<span class="highlight">\1</span>', text)
    return highlighted_text


def search(query, term_document_matrix, original_articles):
    # Clean the input query
    cleaned_query = clean_article(query)

    # Split the cleaned query into words
    query_words = cleaned_query.split()

    # Check if the query words exist in the term-document matrix
    matching_terms = term_document_matrix.loc[term_document_matrix.index.isin(query_words)]

    if matching_terms.empty:
        return []

    # Calculate the total score for each document based on matching term frequencies
    doc_scores = matching_terms.drop(columns=['total_count']).sum()

    # Filter out documents with a score of zero
    doc_scores = doc_scores[doc_scores > 0]

    # Rank the documents based on the scores
    ranked_docs = doc_scores.sort_values(ascending=False)

    # Get the top 10 search results
    top_10_results = ranked_docs.head(10)

    # Prepare the search results with highlighted matches
    results = []
    for doc, score in top_10_results.items():
        doc_index = int(doc.split(' ')[-1]) - 1  # Extract the document index from the document name
        original_text = original_articles[doc_index]

        # Highlight the matching terms in the original document text
        highlighted_text = highlight_matches(original_text, query_words)

        # Store the result
        results.append({
            'document': doc,
            'score': score,
            'highlighted_text': highlighted_text[:500] + '...'
            # Show a snippet of the original text (first 500 characters)
        })

    # Return the top 10 results
    return results


# Load the articles dataset and prepare term-document matrix
file_path = './Articles.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')
articles_list = df['Article'].tolist()
cleaned_articles_list = [clean_article(article) for article in articles_list]
term_document_matrix = term_doc_mat(cleaned_articles_list)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form['query']
        search_results = search(query, term_document_matrix, articles_list)
        return render_template('index.html', query=query, results=search_results)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

