from flask import Flask, request, render_template
import pandas as pd
from text_cleaner import clean_article
from term_doc_mat import term_doc_mat
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np

from tf_idf_mat import tfidf_matrix

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
    top_50_results = ranked_docs.head(50)
    ranked_docs_with_ranks = {f"{doc_index}": rank for rank, (doc_index, _) in
                              enumerate(top_50_results.items(), start=1)}

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
    return results,ranked_docs_with_ranks


def search_tfidf(query, tfidf_df, original_articles):
    cleaned_query = clean_article(query)
    query_words = cleaned_query.split()
    matching_terms = tfidf_df.loc[:, tfidf_df.columns.isin(query_words)]

    if matching_terms.empty:
        return []

    doc_scores = matching_terms.sum(axis=1)
    doc_scores = doc_scores[doc_scores > 0]
    ranked_docs = doc_scores.sort_values(ascending=False)
    top_10_results = ranked_docs.head(10)
    top_50_results = ranked_docs.head(50)
    ranked_docs_with_ranks = {f"Doc {doc_index}": rank for rank, (doc_index, _) in
                              enumerate(top_50_results.items(), start=1)}

    results = []
    for doc_index, score in top_10_results.items():
        original_text = original_articles[doc_index]
        highlighted_text = highlight_matches(original_text, query_words)
        results.append({
            'document': f"Doc {doc_index + 1}",
            'score': score,
            'highlighted_text': highlighted_text[:500] + '...'
        })
    return results, ranked_docs_with_ranks


def calculate_displacement(old_rank, new_rank):
    displacement = {}

    # Combine old and new ranks into one set to handle all documents
    all_docs = set(old_rank.keys()).union(set(new_rank.keys()))

    for doc in all_docs:
        old_position = old_rank.get(doc, None)
        new_position = new_rank.get(doc, None)

        if old_position is not None and new_position is not None:
            displacement[doc] = new_position - old_position
        elif old_position is None and new_position is not None:
            displacement[doc] = f'New document, position: {new_position}'
        elif old_position is not None and new_position is None:
            displacement[doc] = f'Removed document, previous position: {old_position}'

    return displacement


# Load the articles dataset and prepare term-document matrix
file_path = './Articles.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')
articles_list = df['Article'].tolist()[:500]
cleaned_articles_list = [clean_article(article) for article in articles_list]
term_document_matrix = term_doc_mat(cleaned_articles_list)

# Calculate TF-IDF term-document matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cleaned_articles_list)
terms = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=terms)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form['query']
        search_results,top_rank = search_tfidf(query,tfidf_df,articles_list)
        _,old_top_rank= search(query, term_document_matrix, articles_list)
        print(old_top_rank)
        print(top_rank)
        displacement = calculate_displacement(old_top_rank,top_rank)
        for key in displacement:
            print(key,displacement[key])
        return render_template('index.html', query=query, results=search_results)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

