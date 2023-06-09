from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import glob
import os
import pickle
import DataProcess
# List of documents
docs = {}


def read_docs_from_pickle(pickle_file_path):
    with open(pickle_file_path, 'rb') as file:
        data = pickle.load(file)
        for doc_id, doc_text in data.items():
            docs[doc_id] = doc_text


# تحميل المستندات من ملف pickle
pickle_file_path = r"C:/Users/USER/PycharmProjects/DataSet1/docs.pickle"  # استبدل بمسار ملف pickle الخاص بك
read_docs_from_pickle(pickle_file_path)

# read tfidf and vectorizer
with open('C:/Users/USER/PycharmProjects/DataSet1/vectorizer.pickle', 'rb') as file:
    vectorizer = pickle.load(file)

with open('C:/Users/USER/PycharmProjects/DataSet1/tfidf_matrix.pickle', 'rb') as file:
    tfidf_matrix = pickle.load(file)

# Read relevance judgments (qrels)
qrels_file_paths = glob.glob(r"C:/Users/USER/PycharmProjects/DataSet1/qrels.txt")
qrels = {}

for qrels_file_path in qrels_file_paths:
    with open(qrels_file_path, 'r') as qrels_file:
        for line in qrels_file:
            line_parts = line.strip().split(' ')
            if len(line_parts) != 4:
                print("Invalid line format:", line)
                continue

            query_id, _, doc_id, relevance = line_parts
            query_id = int(query_id)
            doc_idd = doc_id.split('_')[0]  # Extract the document ID without the suffix

            if query_id in qrels:
                qrels[query_id][doc_idd] = relevance
            else:
                qrels[query_id] = {doc_idd: relevance}

# Read queries from file
queries_file_path = r"C:/Users/USER/PycharmProjects/DataSet1/queries.txt"
queries = {}

with open(queries_file_path, 'r') as queries_file:
    for line in queries_file:
        line_parts = line.strip().split(None, 1)
        if len(line_parts) < 2:
            continue  # Skip lines that don't have the expected format
        query_id = int(line_parts[0])
        query = line_parts[1]
        query1 = DataProcess.process(query)
        queries[query_id] = query1

# Create relevant_num_file to store the results
# ...

# Create relevant_num_file to store the results
relevant_num_file_path = r"C:/Users/USER/PycharmProjects/DataSet1/relevant_newleen.txt"
with open(relevant_num_file_path, 'w') as relevant_num_file:
    mrr_values = []
    for query_id, query1 in queries.items():
        print("Query ID:", query_id)
        print("Query:", query1)

        if query_id not in qrels:
            print("No relevance judgments found for this query.")
            continue

        # Tokenize and transform the query
        query_vector = vectorizer.transform([query1]).toarray().flatten()

        # Calculate similarity using inverted index
        similarity_scores = cosine_similarity(query_vector.reshape(1, -1), tfidf_matrix)

        # Sort the documents by similarity score
        sorted_indices = np.argsort(similarity_scores, axis=1)[0, ::-1]

        # Print the similar documents
        threshold = 0

        num_relevant = 0
        num_returned = 0
        i = 1
        reciprocal_ranks = []
        relevant_documents = []

        for idx in sorted_indices:
            similarity_score = similarity_scores[0, idx]
            doc_id = list(docs.keys())[idx]  # Retrieve the document ID using the index
            doc_idd = doc_id.split('_')[0]  # Extract the document ID without the suffix

            if similarity_score >= threshold and doc_idd in qrels.get(query_id, {}):
                num_relevant += 1
                reciprocal_ranks.append(1.0 / i)
                i = i + 1
                relevant_documents.append(
                    f"Similarity Score: {similarity_score}, Document ID: {doc_id}, Document: {docs[doc_id]}")

            num_returned += 1
            # print(f"Document ID: {doc_id}, Similarity Score: {similarity_score}")
            print(doc_id)

            if num_returned >= 20:  # Stop after retrieving 10 documents
                break
        precision = num_relevant / num_returned if num_returned > 0 else 0.0
        # print("Precision:", num_relevant)
        # print("")
        Total = len(sorted_indices)

        # Write relevant num and documents in the file
        relevant_num_file.write(f"Query ID: {query_id}\n")
        relevant_num_file.write(f"Query: {query1}\n")
        relevant_num_file.write(f"Relevant Documents: {num_relevant}\n")
        relevant_num_file.write(f"Total Returned Documents: {num_returned}\n")
        relevant_num_file.write(f"reciprocal_ranks: {reciprocal_ranks}\n")
        relevant_num_file.write(f"Precision: {precision}\n")
        relevant_num_file.write(f"Total: {Total}\n")
        relevant_num_file.write(f"mrr: {mrr}\n")
        relevant_num_file.write("\n".join(relevant_documents))
        relevant_num_file.write("\n\n")
        if len(reciprocal_ranks) != 0:
            mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
            mrr_values.append(mrr)
            # print("MRR:", mrr)

        if len(mrr_values) != 0:
            mean_mrr = sum(mrr_values) / len(mrr_values)
        print("Mean Reciprocal Rank (MRR):", mean_mrr)