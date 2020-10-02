import numpy as np
from collections import defaultdict
from functools import partial, reduce
from sklearn.cluster import DBSCAN


class Synonym:
    def __init__(self, words, unique_words_count, embeddings, eps=0.3, min_samples=2, verbose=False):
        self.words = words
        self.words_index = dict(zip(self.words, range(len(self.words))))
        self.unique_words_count = unique_words_count
        self.embeddings = embeddings
        self.eps = eps
        self.min_samples = min_samples
        self.verbose = verbose
        self.syn_map = None
        self.syn_inverse_map = None
        self.n_syn = None
        self.create_synonym(self.embeddings, self.verbose)

    def create_synonym(self, embeddings, verbose=False):
        if verbose:
            print("Creating synonyms from clustering")
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(embeddings)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        if verbose:
            print('Estimated number of clusters: %d' % n_clusters_)
            print('Estimated number of noise points: %d' % n_noise_)

        count = 0
        syn = {}
        syn_i = {}
        for word_idx in np.where(labels == -1)[0]:
            syn[word_idx] = count
            syn_i[count] = word_idx
            count += 1
        for cluster in [np.where(labels == i)[0] for i in range(n_clusters_)]:
            for word_idx in cluster:
                syn[word_idx] = count
            syn_i[count] = cluster[self.unique_words_count[cluster].argmax()]
            count += 1
        self.syn_map = syn
        self.syn_inverse_map = syn_i
        self.n_syn = count

    def get_word(self, index):
        return self.words[self.syn_inverse_map[index]]

    def get_syn(self, word):
        return self.syn_map[self.words_index[word]]


class ConditionalProbability:
    def __init__(self, sentences, token_lists, syn):
        self.sentences = sentences
        self.token_lists = token_lists
        self.synonym = syn
        self.syn_token_list = self.syn_token_list()
        self.token_sentence_map = self.token_sentence_map()

    def syn_token_list(self):
        '''
        transforms the tokens in a token_list to the corresponding synonyms
        :return:
        '''
        syn_token_list = []
        for token_list in self.token_lists:
            syn_tokens = np.unique([self.synonym.get_syn(word) for word in token_list]).tolist()
            syn_token_list.append(syn_tokens)
        return syn_token_list

    def token_sentence_map(self):
        '''
        maps sentences to tokens, i.e. for each token we get a list of sentences which contain that token
        :return:
        '''
        syn_token_to_sentence = defaultdict(list)
        for i, syn_tokens in enumerate(self.syn_token_list):
            for token in syn_tokens:
                syn_token_to_sentence[token].append(i)
        return syn_token_to_sentence

    def sentences_from_tokens(self, tokens, literal=False):
        '''
        returns a list of sentences which contain all tokens specified in tokens
        :param tokens:
        :param literal:
        :return:
        '''
        if literal:
            tokens = [self.synonym.get_syn(token) for token in tokens]
        return reduce(partial(np.intersect1d, assume_unique=True), [self.token_sentence_map[i] for i in tokens])

    def cond_prob(self, event_token, condition_tokens, literal=False):
        '''
        returns the conditional probability of a sentence to contain event_token if that sentence
        also contains all tokens specified in condition_tokens
        :param event_token:
        :param condition_tokens:
        :param literal:
        :return:
        '''
        if literal:
            event_token = self.synonym.get_syn(event_token)
        sen_with_token = self.sentences_from_tokens(condition_tokens, literal)
        count = len(sen_with_token)
        if count == 0:
            return 0
        sen_with_token = np.intersect1d(sen_with_token, self.token_sentence_map[event_token], assume_unique=True)
        return len(sen_with_token) / count

    def prob(self, event_token, literal=False):
        if literal:
            event_token = self.synonym.get_syn(event_token)
        return len(self.token_sentence_map[event_token]) / len(self.syn_token_list)

    def cond_prob_all_tokens(self, condition_tokens, return_probabilities=False, return_absolut=False, literal=False):
        if literal:
            condition_tokens = [self.synonym.get_syn(token) for token in condition_tokens]
        sen_with_token = self.sentences_from_tokens(condition_tokens)
        if len(sen_with_token) == 0:
            return np.zeros(self.synonym.n_syn).tolist()
        cond_probs = np.zeros(self.synonym.n_syn)
        cond_abs = np.zeros(self.synonym.n_syn)
        for i in range(self.synonym.n_syn):
            if i not in condition_tokens:
                all_events_count = len(np.intersect1d(sen_with_token, self.token_sentence_map[i], assume_unique=True))
                cond_probs[i] = all_events_count / len(sen_with_token)
                cond_abs[i] = all_events_count
            else:
                cond_probs[i] = -1
        mask = [True, return_probabilities, return_absolut]
        result = np.array([np.argsort(cond_probs)[::-1], None, None])
        if return_probabilities:
            result[1] = np.sort(cond_probs)[::-1]
        if return_absolut:
            result[2] = cond_abs[result[0]]
        return result[mask]


class Node:
    def __init__(self, value, parent=None):
        self.value = value
        self.parent = parent
        self.children = []

    def add_child(self, value):
        node = Node(value, self)
        self.children.append(node)
        return self

    def get_child(self, value):
        for child in self.children:
            if child.value == value:
                return child
        return None

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0

    def __str__(self):
        return "%d" % self.value

    def __repr__(self):
        return str(self)

    def name(self):
        return str(self)

    def cum_name(self):
        node = self.parent
        cum_name = [self.value]
        while node is not None:
            cum_name.insert(0, node.value)
            node = node.parent
        return cum_name

    def get_leaves(self):
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.children:
            leaves += child.get_leaves()
        return leaves

    def get_newick(self):
        if self.is_leaf():
            return self.name()
        name_children = ",".join([child.get_newick() for child in self.children])
        return "(%s)%s" % (name_children, self.name())

    def get_newick_alt(self, func):
        if self.is_leaf():
            return func(self.value)
        name_children = ",".join([child.get_newick_alt(func) for child in self.children])
        return "(%s)%s" % (name_children, func(self.value))

