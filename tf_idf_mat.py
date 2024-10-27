import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_matrix(cleaned_articles_list):
    # Initialize the TfidfVectorizer
    tfidf_vect = TfidfVectorizer()

    # Fit and transform the cleaned articles
    tfidf_matrix = tfidf_vect.fit_transform(cleaned_articles_list)

    # Get the size of the cleaned articles list
    size = len(cleaned_articles_list)

    # Create a DataFrame from the TF-IDF matrix
    tfidf_df = pd.DataFrame(tfidf_matrix.todense()).iloc[:size]
    tfidf_df.columns = tfidf_vect.get_feature_names_out()

    # Transpose the DataFrame
    tfidf_df = tfidf_df.T

    # Rename the columns to indicate document number
    tfidf_df.columns = ['Doc ' + str(i) for i in range(1, size + 1)]

    return tfidf_df