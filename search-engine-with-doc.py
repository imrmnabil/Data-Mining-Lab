import pandas as pd
from text_cleaner import clean_article
from term_doc_mat import term_doc_mat
import re


def highlight_matches(text, words):
    """Highlight matching words by quoting them in the original text."""
    pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in words) + r')\b', re.IGNORECASE)
    highlighted_text = pattern.sub(r'"\1"', text)
    return highlighted_text


def search(query, term_document_matrix, original_articles):
    # Clean the input query
    cleaned_query = clean_article(query)

    # Split the cleaned query into words
    query_words = cleaned_query.split()

    # Check if the query words exist in the term-document matrix
    matching_terms = term_document_matrix.loc[term_document_matrix.index.isin(query_words)]

    if matching_terms.empty:
        print("No matching results found.")
        return []

    # Calculate the total score for each document based on matching term frequencies
    doc_scores = matching_terms.drop(columns=['total_count']).sum()

    # Rank the documents based on the scores
    ranked_docs = doc_scores.sort_values(ascending=False)

    # Get the top 10 search results
    top_10_results = ranked_docs.head(10)

    # Display the search results with highlighted matches
    print("\nTop 10 Search Results:")
    results = []
    for doc, score in top_10_results.items():
        doc_index = int(doc.split(' ')[-1]) - 1  # Extract the document index from the document name
        original_text = original_articles[doc_index]

        # Highlight the matching terms in the original document text
        highlighted_text = highlight_matches(original_text, query_words)

        # Display the document with the highlighted matches
        print(f"\n{doc}: Score {score}")
        print(f"Excerpt: {highlighted_text[:500]}...")  # Show a snippet of the original text (first 500 characters)

        # Store the result
        results.append((doc, score, highlighted_text))

    # Return the top 10 results as a list of tuples (document name, score, highlighted text)
    return results


if __name__ == '__main__':
    # Load the articles dataset
    file_path = './Articles.csv'
    df = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Convert articles to a list
    articles_list = df['Article'].tolist()

    # Clean the articles
    cleaned_articles_list = [clean_article(article) for article in articles_list]

    # Create the term-document matrix
    term_document_matrix = term_doc_mat(cleaned_articles_list)

    # Perform search
    query = input("Enter your search query: ")
    search(query, term_document_matrix, articles_list)
