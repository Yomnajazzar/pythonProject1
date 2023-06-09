import json
import pickle
from collections import defaultdict
import collections.abc

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
from nltk.corpus import wordnet
import glob

from sklearn.feature_extraction.text import TfidfVectorizer

abbreviations = {
    'Dr.': 'Doctor',
    'Mr.': 'Mister',
    'Mrs.': 'Misess',
    'Ms.': 'Misess',
    'Jr.': 'Junior',
    'Sr.': 'Senior',
    'U.S': 'UNITED STATES',
    'U-S': 'UNITED STATES',
    'U_K': 'UNITED KINGDOM',
    'U_S': 'UNITED STATES',
    'U.K': 'UNITED KINGDOM',
    'U.S': 'UNITED STATES',
    'VIETNAM': 'VIET NAM',
    'VIET NAM': 'VIET NAM',
    'U-N': 'NITED NATIONS',
    'U_N': 'NITED NATIONS',
    'U.N': 'NITED NATIONS',
    'UK': 'UNITED KINGDOM',
    'US': 'UNITED STATES',
    'U-K': 'UNITED KINGDOM',
    'mar': 'March',
    'march': 'March',
    'jan': 'January',
    'anuary': 'January',
    'feb': 'February',
    'february': 'February',
    'apr': 'April',
    'april': 'April',
    'jun': 'June',
    'june': 'June',
    'jul': 'July',
    'july': 'July',
    'dec': 'December',
    'december': 'December',
    'nov': 'November',
    'november': 'November',
    'oct': 'October',
    'october': 'October',
    'sep': 'September',
    'september': 'September',
    'aug': 'August',
    'august': 'August',
}

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def stem_words(txt):
    stems = [stemmer.stem(word) for word in txt]
    return stems


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def handle(text):
    REGEX_1 = r"(([12]\d|30|31|0?[1-9])[/-](0?[1-9]|1[0-2])[/.-](\d{4}|\d{2}))"
    REGEX_2 = r"((0?[1-9]|1[0-2])[/-]([12]\d|30|31|0?[1-9])[/.-](\d{4}|\d{2}))"
    REGEX_3 = r"((\d{4}|\d{2})[/-](0?[1-9]|1[0-2])[/-]([12]\d|30|31|0?[1-9]))"
    REGEX_4 = r"((January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec) ([12]\d|30|31|0?[1-9]),? (\d{4}|\d{2}))"
    REGEX_5 = r"(([12]\d|30|31|0?[1-9]) (January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec),? (\d{4}|\d{2}))"
    REGEX_6 = r"((\d{4}|\d{2}) ,?(January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec) ([12]\d|30|31|0?[1-9]))"
    REGEX_7 = r"((January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec),? (\d{4}|\d{2}))"

    COMBINATION_REGEX = "(" + REGEX_1 + "|" + REGEX_2 + "|" + REGEX_3 + "|" + \
                       REGEX_4 + "|" + REGEX_5 + "|" + REGEX_6 + ")"

    for key, value in abbreviations.items():
        text = re.sub(r'\b{}\b'.format(re.escape(key)), value, text, flags=re.IGNORECASE)

    all_dates = re.findall(COMBINATION_REGEX, text)

    for s in all_dates:
        try:
            date = datetime.strptime(s[0], "%d %B %Y")
        except ValueError:
            continue  # Skip invalid dates

        new_date = date.strftime("%d-%m-%Y")
        text = text.replace(s[0], new_date)

    # إزالة علامات الترقيم ما عدا الشرطة "-"
    text = re.sub(r'[^-\w\s]', '', text)

    return text


folder_path = "C:/Users/USER/PycharmProjects/documents1"
inverted_index = {}
file_paths = glob.glob(folder_path + "/*.txt")
docs = {}
for file_path in file_paths:
    with open(file_path, 'r') as file:
        text = file.read()
        file_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(file_name)[0]
        print(f"File: {file_name_without_ext}")
        unified_text = handle(text)
        sentences = sent_tokenize(unified_text)
        tokens = word_tokenize(unified_text)
        tokens = [w.lower() for w in tokens]
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
        stem_word = stem_words(filtered_tokens)
        listToStr = ' '.join([str(elem) for elem in stem_word])

        pos_tagged = nltk.pos_tag(nltk.word_tokenize(listToStr))
        wordnet_tagged = [(word, pos_tagger(tag)) for word, tag in pos_tagged]
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                lemmatized_sentence.append(word)
            else:
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))

        # update inverted index for each term in document
        for term in lemmatized_sentence:
            if term not in inverted_index:
                inverted_index[term] = [file_name_without_ext]
            else:
                inverted_index[term].append(file_name_without_ext)


        #store data after process
        lemmatized_sentence = " ".join(lemmatized_sentence)
        # print("lemmatized_sentence:", lemmatized_sentence)
        docs[file_name_without_ext] = lemmatized_sentence
        # with open("C:/Users/USER/PycharmProjects/DataSet1/document_process/"+file_name, "w") as output_file:
        #     output_file.write(lemmatized_sentence)

print("finish process")

print(docs.values())
# store vectorizer and TF-IDF matrix to files

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs.values())


# print(tfidf_matrix)
# print(docs)
with open('C:/Users/USER/PycharmProjects/DataSet1/vectorizer.pickle', 'wb') as file:
    pickle.dump(vectorizer, file)

with open('C:/Users/USER/PycharmProjects/DataSet1/tfidf_matrix.pickle', 'wb') as file:
    pickle.dump(tfidf_matrix, file)



#read tfidf and vectorizer
# with open('C:\Users\USER\PycharmProjects\DataSet1\_vectorizer2.pickle', 'rb') as file:
#     vectorizer = pickle.load(file)
# with open('C:/Users/USER/PycharmProjects/DataSet1/tfidf_matrix.pickle', 'rb') as file:
#     tfidf_matrix = pickle.load(file)
#


# store inverted index to file
with open('C:/Users/USER/PycharmProjects/DataSet1/inverted_index.pickle', 'wb') as file:
    pickle.dump(inverted_index, file)

# store docs to file
with open('C:/Users/USER/PycharmProjects/DataSet1/docs.pickle', 'wb') as file:
    pickle.dump(docs, file)

# load inverted index from file
# with open('C:/Users/USER/PycharmProjects/DataSet1/inverted_index.pickle', 'rb') as file:
#     inverted_index = pickle.load(file)
# print(inverted_index)

# load docs  from file
# with open('C:/Users/USER/PycharmProjects/DataSet1/docs.pickle', 'rb') as file:
#     docs = pickle.load(file)
# print(docs)