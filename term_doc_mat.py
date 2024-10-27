import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def term_doc_mat(cleaned_articles_list):
    vect = CountVectorizer()
    vects = vect.fit_transform(cleaned_articles_list)

    size = len(cleaned_articles_list)

    td = pd.DataFrame(vects.todense()).iloc[:size]
    td.columns = vect.get_feature_names_out()
    term_document_matrix = td.T
    term_document_matrix.columns = ['Doc ' + str(i) for i in range(1, size+1)]
    term_document_matrix['total_count'] = term_document_matrix.sum(axis=1)
    return term_document_matrix