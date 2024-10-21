import pandas as pd
from text_cleaner import clean_article
from term_doc_mat import term_doc_mat
file_path = './Articles.csv'

df = pd.read_csv(file_path, encoding='ISO-8859-1')


articles_list = df['Article'].tolist()

cleaned_articles_list = [clean_article(article) for article in articles_list]

# print(cleaned_articles_list[:1])

term_document_matrix = term_doc_mat(cleaned_articles_list)


partial_term_document_matrix = term_document_matrix.sort_values(by ='total_count',ascending=False)[:100]

