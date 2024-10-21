import pandas as pd
from text_cleaner import clean_article
from term_doc_mat import term_doc_mat

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

    # Display the search results with snippets
    print("Top 10 Search Results:")
    results = []
    for doc, score in top_10_results.items():
        # Extract the document index from the document name (e.g., "Doc 1" -> index 0)
        doc_index = int(doc.split()[1]) - 1

        # Get the original text of the matching article
        original_text = original_articles[doc_index]

        # Find a snippet of the text containing any of the query words
        snippet = get_snippet(original_text, query_words)

        # Display the result with a snippet
        print(f"{doc}: Score {score}\nSnippet: {snippet}\n")
        results.append((doc, score, snippet))

    # Return the top 10 results as a list of tuples (document name, score, snippet)
    return results

def get_snippet(text, query_words, window_size=30):
    # Convert text to lowercase to match case-insensitively
    lower_text = text.lower()

    # Find the first occurrence of any query word in the text
    for word in query_words:
        index = lower_text.find(word)
        if index != -1:
            # Get a snippet around the found word
            start = max(0, index - window_size)
            end = min(len(text), index + window_size)
            return text[start:end].strip()

    # If no query words are found, return the first part of the text as a fallback
    return text[:window_size * 2].strip()

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
