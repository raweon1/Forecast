from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk.stem.cistem import Cistem
from HanTa import HanoverTagger as ht
from transformers import AutoModel, AutoTokenizer
import re
import nltk
from nltk.tokenize import word_tokenize
from language_detector import detect_language

import pkg_resources
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
if sym_spell.word_count:
    pass
else:
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


###################################
#### sentence level preprocess ####
###################################

# lowercase + base filter
# some basic normalization
def f_base(s):
    """
    :param s: string to be processed
    :return: processed string: see comments in the source code for more info
    """
    # normalization 1: xxxThis is a --> xxx. This is a (missing delimiter)
    s = re.sub(r'([a-z])([A-Z])', r'\1\. \2', s)  # before lower case
    # normalization 2: lower case
    s = s.lower()
    # normalization 3: "&gt", "&lt"
    s = re.sub(r'&gt|&lt', ' ', s)
    # normalization 4: letter repetition (if more than 2)
    s = re.sub(r'([a-z])\1{2,}', r'\1', s)
    # normalization 5: non-word repetition (if more than 1)
    s = re.sub(r'([\W+])\1{1,}', r'\1', s)
    # normalization 6: string * as delimiter
    s = re.sub(r'\*|\W\*|\*\W', '. ', s)
    # normalization 7: stuff in parenthesis, assumed to be less informal
    s = re.sub(r'\(.*?\)', '. ', s)
    # normalization 8: xxx[?!]. -- > xxx.
    s = re.sub(r'\W+?\.', '.', s)
    # normalization 9: [.?!] --> [.?!] xxx
    s = re.sub(r'(\.|\?|!)(\w)', r'\1 \2', s)
    # normalization 10: ' ing ', noise text
    s = re.sub(r' ing ', ' ', s)
    # normalization 11: noise text
    s = re.sub(r'product received for free[.| ]', ' ', s)
    # normalization 12: phrase repetition
    s = re.sub(r'(.{2,}?)\1{1,}', r'\1', s)

    return s.strip()


# language detection
def f_lan(s):
    """
    :param s: string to be processed
    :return: boolean (s is English)
    """

    # some reviews are actually english but biased toward french
    return detect_language(s) in {'English', 'French', 'Spanish', 'Chinese', 'German'}


###############################
#### word level preprocess ####
###############################

# filtering out punctuations and numbers
def f_punct(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with punct and number filter out
    """
    return [word for word in w_list if word.isalpha()]


# selecting nouns
def f_noun(w_list, lang=None):
    """
    :param w_list: word list to be processed
    :return: w_list with only nouns selected
    """
    if lang == "german":
        tagger = ht.HanoverTagger('morphmodel_ger.pgz')
        return [word for (word, lemma, pos) in tagger.tag_sent(w_list, casesensitive=False) if pos == 'NN']
    return [word for (word, pos) in nltk.pos_tag(w_list) if pos[:2] == 'NN']


# typo correction
def f_typo(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with typo fixed by symspell. words with no match up will be dropped
    """
    w_list_fixed = []
    for word in w_list:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=3)
        if suggestions:
            w_list_fixed.append(suggestions[0].term)
        else:
           pass
            # do word segmentation, deprecated for inefficiency
            # w_seg = sym_spell.word_segmentation(phrase=word)
            # w_list_fixed.extend(w_seg.corrected_string.split())
    return w_list_fixed


# stemming if doing word-wise
p_stemmer = PorterStemmer()


def f_stem(w_list, lang=None):
    """
    :param w_list: word list to be processed
    :return: w_list with stemming
    """
    if lang == "german":
        tagger = ht.HanoverTagger('morphmodel_ger.pgz')
        return [lemma for (word, lemma, pos) in tagger.tag_sent(w_list)]
    return [p_stemmer.stem(word) for word in w_list]


# filtering out stop words
# create English stop words list

stop_words = (list(
    set(get_stop_words('en'))
    | set(get_stop_words('es'))
    | set(get_stop_words('de'))
    | set(get_stop_words('it'))
    | set(get_stop_words('ca'))
    # |set(get_stop_words('cy'))
    | set(get_stop_words('pt'))
    # |set(get_stop_words('tl'))
    | set(get_stop_words('pl'))
    # |set(get_stop_words('et'))
    | set(get_stop_words('da'))
    | set(get_stop_words('ru'))
    # |set(get_stop_words('so'))
    | set(get_stop_words('sv'))
    | set(get_stop_words('sk'))
    # |set(get_stop_words('cs'))
    | set(get_stop_words('nl'))
    # |set(get_stop_words('sl'))
    # |set(get_stop_words('no'))
    # |set(get_stop_words('zh-cn'))
))


def f_stopw(w_list):
    """
    filtering out stop words
    """
    return [word for word in w_list if word not in stop_words]


def preprocess_sent(rw):
    """
    Get sentence level preprocessed data from raw review texts
    :param rw: review to be processed
    :return: sentence level pre-processed review
    """
    s = f_base(rw)
    if not f_lan(s):
        return None
    return s


def preprocess_word(s):
    """
    Get word level preprocessed data from preprocessed sentences
    including: remove punctuation, select noun, fix typo, stem, stop_words
    :param s: sentence to be processed
    :return: word level pre-processed review
    """
    if not s:
        return None
    w_list = word_tokenize(s, language='german')
    w_list = f_punct(w_list)
    w_list = f_noun(w_list, lang="german")
    if len(w_list) == 0:
        return None
    # w_list = f_typo(w_list)
    w_list = f_stem(w_list, lang="german")
    w_list = f_stopw(w_list)

    return w_list


def preprocess_word2(s):
    if not s:
        return None
    w_list = word_tokenize(s, language='german')
    w_list = f_punct(w_list)
    w_list = f_stem(w_list, lang="german")
    w_list = f_stopw(w_list)
    return w_list
