import pandas as pd
from text_cleaner import clean_article
from term_doc_mat import term_doc_mat

def search(query, term_document_matrix):
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

    # Display the search results
    print("Top 10 Search Results:")
    for doc, score in top_10_results.items():
        print(f"{doc}: Score {score}")

    # Return the top 10 results as a list of tuples (document name, score)
    return list(top_10_results.items())

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
    search(query, term_document_matrix)
