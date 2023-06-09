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
from autocorrect import Speller

abbreviations = {
    'Dr.': 'Doctor ',
    'Mr.': 'Mister ',
    'Mrs.': 'Misess ',
    'Ms.': 'Misess ',
    'Jr.': 'Junior ',
    'Sr.': 'Senior ',
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
    'jan': 'January',
    'feb': 'February',
    'apr': 'April',
    'jun': 'June',
    'jul': 'July',
    'dec': 'December',
    'nov': 'November',
    'oct': 'October',
    'sep': 'September',
    'aug': 'August',
}


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

    #  text="hoq are Dr.maya yuo pleased me in feb ,? "
    for key, value in abbreviations.items():
        text = re.sub(r'\b{}\b'.format(re.escape(key)), value, text, flags=re.IGNORECASE)
        all_dates = re.findall(COMBINATION_REGEX, text)

    for s in all_dates:
        new_date = parse(s[0]).strftime("%d-%m-%Y")
        text = text.replace(s[0], new_date)

    # إزالة علامات الترقيم ما عدا الشرطة "-"
    text = re.sub(r'[^-\w\s]', '', text)
    return text


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


def process(query):
    spell = Speller(lang='en')

    # تقوم بتصحيح الأخطاء الإملائية في الجملة
    corrected_sentence = " ".join([spell(word) for word in query.split()])

    unified_text = handle(corrected_sentence)

    tokens = word_tokenize(unified_text)
    tokens = [w.lower() for w in tokens]

    stop_words = set(stopwords.words('english'))
    words = [w for w in tokens if w not in stop_words]

    stem_word = stem_words(words)

    # lemmas = [lemmatizer.lemmatize(word, pos='v') for word in stem_word]
    # print("lemmatizing:", lemmas)

    # lemmatized_sentence = " ".join(lemmas)

    # return lemmatized_sentence

    listToStr = ' '.join([str(elem) for elem in stem_word])
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(listToStr))

    # print("pos_tagged:", pos_tagged)



    wordnet_tagged = [(word, pos_tagger(tag)) for word, tag in pos_tagged]
    # print("wordnet_tagged:", wordnet_tagged)


    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    lemmatized_sentence = " ".join(lemmatized_sentence)
    return lemmatized_sentence

# print("lemmatized_sentence:", lemmatized_sentence)

query = "hoq are  yuo visited me in feb ? "
result = process(query)
print("Processed query:", result)