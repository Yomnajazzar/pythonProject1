from autocorrect import Speller
import pandas as pd
import os
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import wordnet
import datefinder
import re
from typing import Any
from datetime import datetime
from dateutil.parser import parse
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify,render_template
import glob
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fast_autocomplete import AutoComplete
import pickle
import ipywidgets as widgets
from IPython.display import display
from spellchecker import SpellChecker
from nltk.corpus import wordnet
import DataProcess;
# Initialize the spellchecker
spellchecker = nltk.corpus.words.words()

#auto Complete method
def read_file(file_path):
    queries = []
    with open(file_path, 'r') as file:
        file_queries = file.readlines()
        for line in file_queries:
            query = line.split("\t")[1]  # يتم الحصول على الجزء الثاني من السطر (نص الاستعلام)
            queries.append(query.strip())  # يتم إضافة الاستعلام إلى القائمة

    return queries

file_path = "C:/Users/USER/PycharmProjects/DataSet2/queries.txt"
queries = read_file(file_path)

def complete(query, folder_path):
    print("complete",query)
    queries = read_file(folder_path)
    words = {}
    for value in queries:
        value = value.strip()  # Remove leading/trailing whitespace
        new_key_values_dict = {value: {}}
        words.update(new_key_values_dict)

    autocomplete = AutoComplete(words=words)
    suggestions = autocomplete.search(query, max_cost=10, size=10)

    return suggestions

def suggest_non_start_words(query, folder_path):
    queries = read_file(folder_path)
    suggestions = []
    for value in queries:
        value = value.strip()  # Remove leading/trailing whitespace
        if query in value and not value.startswith(query):
            suggestions.append(value)

    return suggestions

def suggest_spelling_corrections(query):
    tokens = nltk.word_tokenize(query)
    spellchecker = SpellChecker()
    corrections = []
    for token in tokens:
        if token.lower() not in spellchecker:
            correction = spellchecker.correction(token)
            corrections.append(correction)

    return corrections

def expand_query(query):
    synonyms = set()
    antonyms = set()
    hypernyms = set()

    # Find synonyms المرادفات
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())

    # Find antonyms المضادات
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonyms.add(lemma.antonyms()[0].name())

    # Find hypernyms
    for word in query.split():
        for syn in wordnet.synsets(word):
            for hyper in syn.hypernyms():
                for lemma in hyper.lemmas():
                    hypernyms.add(lemma.name())

    expanded_query = query.split() + list(synonyms) + list(antonyms) + list(hypernyms)

    return expanded_query

def on_text_changed(change):
    entered_text = change
    if entered_text:
        print("entered_text",entered_text)
        print(file_path)
        suggestions = complete(entered_text, file_path)
        non_start_words_suggestions = suggest_non_start_words(entered_text, file_path)
        spelling_corrections = suggest_spelling_corrections(entered_text)
        expanded_query = expand_query(entered_text)
        all_suggestions = suggestions + non_start_words_suggestions + spelling_corrections + expanded_query
        print("here")
        print(all_suggestions)
        return all_suggestions
    else:
        print([])
        return []

#Read All Data From File
print("************************************** read Data File 2**********************")

with open('C:/Users/USER/PycharmProjects/DataSet2/vectorizer.pickle', 'rb') as file:
    vectorizer2 = pickle.load(file)
with open('C:/Users/USER/PycharmProjects/DataSet2/tfidf_matrix.pickle', 'rb') as file:
    tfidf_matrix2 = pickle.load(file)
with open('C:/Users/USER/PycharmProjects/DataSet2/inverted_index.pickle', 'rb') as file:
    inverted_index2 = pickle.load(file)
with open('C:/Users/USER/PycharmProjects/DataSet2/docs.pickle', 'rb') as file:
    docs2 = pickle.load(file)
print("**************************************read read Data File 1**********************")
with open('C:/Users/USER/PycharmProjects/DataSet1/vectorizer.pickle', 'rb') as file:
    vectorizer1= pickle.load(file)
with open('C:/Users/USER/PycharmProjects/DataSet1/tfidf_matrix.pickle', 'rb') as file:
    tfidf_matrix1 = pickle.load(file)
with open('C:/Users/USER/PycharmProjects/DataSet1/inverted_index.pickle', 'rb') as file:
    inverted_index1 = pickle.load(file)
with open('C:/Users/USER/PycharmProjects/DataSet1/docs.pickle', 'rb') as file:
    docs1 = pickle.load(file)

files2 = glob.glob("C:\\Users\\USER\\PycharmProjects\\documents2\\*")
files1 = glob.glob("C:\\Users\\USER\\PycharmProjects\\documents1\\*")

app = Flask(__name__)

@app.route('/complete', methods=['GET'])
def complete2():
    query = request.args.get('query')
    print(query)
    resultsQyery = on_text_changed(query)
    print("results ", resultsQyery)
    result = [];
    # for v in resultsQyery:
    #     result.append(v)
    #     result.append(";")

    result=resultsQyery[:10]
    return jsonify(results=result)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    dataSet = request.args.get('dataSet')
    print("dataSet",dataSet)
    files=[]
    if dataSet=="1":
        print("here1")
        files=files1
        tfidf_matrix=tfidf_matrix1
        vectorizer=vectorizer1
        docs=docs1
    else:
        print("here2")
        files = files2
        tfidf_matrix = tfidf_matrix2
        vectorizer = vectorizer2
        docs = docs2

    query_string=DataProcess.process(query);
    print(query_string)
    query_vector = vectorizer.transform([query_string])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    # print(similarity_scores)
    # Sort the documents by similarity score
    sorted_indices = np.argsort(similarity_scores, axis=1)[0, ::-1]
    # print(sorted_indices)
    result=[]
    count = 0
    threshold=0.01
    for idx in sorted_indices:
        if count == 5:
            break
        count +=1
        similarity_score = similarity_scores[0, idx]
        if similarity_score >= threshold:
            doc_id = list(docs.keys())[idx]  # Retrieve the document ID using the index
            print(doc_id)
            with open(files[idx], 'r', encoding='utf-8') as f:
                file_content = f.read()
            # print(file_content)
            result.append(file_content)
        else:
            break
    return jsonify(results=result)

@app.route('/')
def index():
    return render_template('search2.html')

if __name__ == '__main__':
    app.run()
