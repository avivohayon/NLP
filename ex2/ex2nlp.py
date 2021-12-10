import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
#
nltk.download('brown')
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import Counter

DELTA = 1
CUT_OFF = 5

# utility functions
def find_unknown_words(test, vocabulary):
    """
    param: test: the tagged test set
           vocabulary: the train set vocabulary set of words (aka known words)
   return: a set of all words which didnt appeared in the training set (aka unknown words)
    """
    unknown_words = set()
    for sent in test:   # find the must common tag for each word
        for tagged_word in sent:
            if tagged_word[0] in vocabulary:
                continue
            else:
                unknown_words.add(tagged_word[0])
    # print(test_unknown_words)
    return unknown_words


def clean_tage(tagged_sents):
    """
    param: the text which each sentence is a tuple (word, complex_tag)
    return: same text but only with the prefix the complex tags
    """
    clean_sentences = []
    for tagged_sent in tagged_sents:
        clean_words_in_sent = []
        for tagged_word in tagged_sent:
            clean_words_in_sent.append((tagged_word[0], tagged_word[1].split('+')[0].split('-')[0]))
        clean_sentences.append(clean_words_in_sent)
    return clean_sentences


def get_tages(tagged_text):
    """
    param: tagged text
    return: tuple[0]: tagged_sequence - list of all the tags in their order or appearance in the text
            tuple[1]: tagged_appreas - Counter dict where its keys is the tags from the text and value the number of time
            that tags appeared in the corpus
    """
    tagged_sequence = []
    for sent in tagged_text:
        for tagged_word in sent:
            tagged_sequence.append(tagged_word[1])

    tagged_appreas = Counter(tagged_sequence)

    return tagged_sequence, tagged_appreas



def number_of_wrods_per_tags(sentences):
    """
    compute the number of appearances of a word with a tag #(x|y) were x = word, y = tag
    """
    result = Counter([x for sublist in sentences for x in sublist])   # count the number (word, tag) appears
    return result





def create_tags_per_each_word_dict(tagged_text):
    """
    param: tagged text: the text which each sentence is a tuple (word, tag)
    return tag_per_word_dict: dict where its keys: tags, value: a dict of all the words with the key as tags
    """
    tag_per_word_dict = dict()
    for sent in tagged_text:
        for tagged_word in sent:
            clean_tag = tagged_word[1]
            if tagged_word[0] not in tag_per_word_dict.keys():
                tag_per_word_dict[tagged_word[0]] = dict()
                tag_per_word_dict[tagged_word[0]][clean_tag] = 1
            else:
                tags_dict = tag_per_word_dict[tagged_word[0]]
                if clean_tag not in tags_dict.keys():
                    tags_dict[clean_tag] = 1
                else:
                    tags_dict[clean_tag] += 1

    return tag_per_word_dict

def create_tag_bigram_counter_dict(tagged_text):
    """
    param: the text which each sentence is a tuple (word, tag)
    return: word_per_tag_dict: dict where its keys: tag number i , value: dict represents a bigram tag counter, where
    each key of the dict is the next tag number i + 1 and values are the amount of time tag number i + 1 appeared after the
    tage number i of the key value of from the tag_bigram_counter_dict. i.e {"START": {"AT: 2, "CC": 17 ...}}
    """
    tag_bigram_counter_dict = dict()
    for sent in tagged_text:
        for i in range(len(sent)):
            if i == 0:
                tag_i = "START"
                tag_i_plus_one = sent[i][1]
            elif i == len(sent) - 1:
                tag_i = sent[i][1]
                tag_i_plus_one = "STOP"
            else:
                tag_i = sent[i][1]
                tag_i_plus_one = sent[i + 1][1]
            if tag_i not in tag_bigram_counter_dict.keys():
                tag_bigram_counter_dict[tag_i] = dict()
                tag_bigram_counter_dict[tag_i][tag_i_plus_one] = 1
            else:
                bigram_tag_dict = tag_bigram_counter_dict[tag_i]
                if tag_i_plus_one not in bigram_tag_dict.keys():
                    bigram_tag_dict[tag_i_plus_one] = 1
                else:
                    bigram_tag_dict[tag_i_plus_one] += 1
    return tag_bigram_counter_dict


def create_words_per_tag_dict(tagged_text):
    """
    param: the text which each sentence is a tuple (word, tag)
    return: word_per_tag_dict: dict where its keys: word, value: a dict of all the tags with the key as words
    """
    word_per_tag_dict = dict()
    for sent in tagged_text:
        for tagged_words in sent:
            clean_tag = tagged_words[1]
            if clean_tag not in word_per_tag_dict.keys():
                word_per_tag_dict[clean_tag] = dict()
                word_per_tag_dict[clean_tag][tagged_words[0]] = 1
            else:
                cur_new_words_dict = word_per_tag_dict[clean_tag]
                if tagged_words[0] not in cur_new_words_dict.keys():
                    cur_new_words_dict[tagged_words[0]] = 1
                else:
                    cur_new_words_dict[tagged_words[0]] += 1
    return word_per_tag_dict




def update_vocabulary_known_words(train_set):
    """
    param: the tagged train set
    return: the train corpus vocabulary
    """
    d = number_of_wrods_per_tags(train_set)   # count the number (word, tag) appears
    vocabulary_set = {tagged_word[0] for tagged_word in d}
    return vocabulary_set

def find_rare_used_words(test_set, word_per_tag_dict):
    """
    param: test_set - tagged test set
           word_per_tag_dict - the words_per_tag_dicts from the "create_words_per_tag_dict: func
    return: rare_used_words_list - a list of the rare used words which appeared less then a global CUT_OFF value
    """
    rare_used_words_list = [word for word, tag_dict in word_per_tag_dict.items() if sum(tag_dict.values()) < CUT_OFF]
    test_unknown_words = find_unknown_words(test_set, word_per_tag_dict.keys())
    for sent in test_set:
        for word_tag in sent:
            cur_word = word_tag[0]
            if cur_word in test_unknown_words:
                rare_used_words_list.append(cur_word)
    return rare_used_words_list
###############################################################################

# question 3bi
def MLE_baseline(train, test, vocabulary):
    """
    param: train - tagged train set
           test - tagged test set
           vocabulary - all the known words form the train set
    return: best match : a dict of most likely tag for each known words and the number of time the tag appeared in the
            train set. unknown words get a default tagging "NN"
    """
    best_match, words_appeaed_set = dict(), set()
    d = number_of_wrods_per_tags(train)   # count the number (word, tag) appears
    for key, val in d.items():   # find the must common tag for each word
        if key[0] in words_appeaed_set:
            continue
        else:
            words_appeaed_set.add(key[0])  # create the word train_vocabulary from the traning set
            best_match[key[0]] = key[1]

    # find all the unknown words and default classified them as 'NN'
    test_unknown_words = find_unknown_words(test, vocabulary)
    for unknow_word in test_unknown_words:
        best_match[unknow_word] = 'NN'
    return best_match

# question 3bii
def print_and_calculate_mle_error_rate(test, mle_dict, vocabulary):
    """
    :param test: test set of sentences and ther tagging
    :param mle_dict: the dict represent each word for the test sentence and the most likely tag it have
    :return: print the error rate correspond to the test set divide by known words (form the test set), unknown words
     (words which didnt appeared in the test set)
    and the total error rate
    """
    train_num_of_words = 0
    test_num_of_words = 0
    known_word_tag_match = 0
    unknown_word_tag_match = 0
    for cur_sent in test:
        for word, tag in cur_sent:
            if word in vocabulary:
                train_num_of_words += 1
                if mle_dict[word] == tag:
                    known_word_tag_match += 1
            else:
                test_num_of_words += 1
                if mle_dict[word] == tag:
                    unknown_word_tag_match += 1

    known_error_rate = 1 - known_word_tag_match / train_num_of_words
    unknown_error_rate = 1 - unknown_word_tag_match / test_num_of_words
    total_error_rate = 1 - (known_word_tag_match + unknown_word_tag_match) / (test_num_of_words + train_num_of_words)
    print("known error rate : ", known_error_rate, " unknown error rate : ", unknown_error_rate,
          " total error rate : ", total_error_rate)



def qml_func(yi, yi_minus_1, tag_tag_dict):
    """
    param: yi - tag number i
           yi_minus_1 - tag number i - 1
           tag_tag_dict
    return: the transition probability value
    """
    if yi_minus_1 in tag_tag_dict.keys():
        if yi in tag_tag_dict[yi_minus_1].keys():

            return tag_tag_dict[yi_minus_1][yi] / sum(tag_tag_dict[yi_minus_1].values())
        else:
            return 0
    return 0


def eml_func(xi, yi, tag_word_dict, words_tag_dict):
    """
    param: xi - word number i
           yi - the tag for the word xi
           tag_word_dict - the tag per words dict
           words_tag_dict - the words per tag dict
    return: the emission probability value
    """
    if yi not in tag_word_dict.keys():
        return 0
    count_x_given_y = 0
    if xi in words_tag_dict.keys():
        if yi in words_tag_dict[xi].keys():
           count_x_given_y = tag_word_dict[yi][xi]
        return count_x_given_y / sum((tag_word_dict[yi].values()))
    if yi == "NN":
        return 1
    return 0


def eml_func_laplace(xi, yi, tag_wrod_dict, words_tag_dict):
    """
    param: xi - word number i
           yi - the tag for the word xi
           tag_word_dict - the tag per words dict
           words_tag_dict - the words per tag dict
    return: the emission probability value with the laplace smoothing (default with delta == 1 but can be changed
    by updating the global DELTA value)
    """
    delta = DELTA
    vocabulary = words_tag_dict.keys()
    if yi not in tag_wrod_dict.keys():
        return 0
    if xi in words_tag_dict.keys():
        if yi in words_tag_dict[xi].keys():
            return (tag_wrod_dict[yi][xi] + delta) / (sum(tag_wrod_dict[yi].values()) + delta * len(vocabulary))
    return (delta) / (sum(tag_wrod_dict[yi].values()) + delta * len(vocabulary))

# question 3c
def viterbi(sent, transition_tag_tag_dict, emission_tag_wrod_dict, set_of_all_tags, word_tag_dict, transition_func, emission_func):
    """
    VITERBI ALGORITHM
    param: sent - the current sentence for the viterbi algo
           transition_tag_tag_dict - the dict used by the transition function
           emission_tag_wrod_dict - the dict used by the emission function
           set_of_all_tags - the vocabulary set
           word_tag_dict -  the dict of all the words and their tags
           transition_func- the transition function used by the algo
           emission_func - the emission function used by the algo
    return the predicted tag sequence by the viterbi algo
    """
    # initialize state
    phi_matrix, back_pointer = {}, {}
    phi_matrix[0] = dict()
    phi_matrix[0]["START"] = 1
    n = len(sent)
    vocabulary_set_s = set_of_all_tags

    for k in range(1, n + 1):
        phi_matrix[k], back_pointer[k] = dict(), dict()
        for v in vocabulary_set_s:
            xk_minus_1 = sent[k - 1]
            if k - 1 == 0:
                # the base case calculation
                phi_val = phi_matrix[0]['START'] * transition_func(v, 'START', transition_tag_tag_dict) * \
                          emission_func(xk_minus_1, v, emission_tag_wrod_dict, word_tag_dict)
                phi_matrix[k][v] = phi_val
                back_pointer[k][v] = 'START'
            else:  # for all  k > 2 , find the max w from Sk-2 of {phi(k - 1, w, u) * q(v|w, u) * e(xk| v)}
                cur_max_tag_w, cur_max_prob_for_w = None, -1
                for w in vocabulary_set_s:
                    # calculate {phi(k - 1, w, u) * q(v|w, u) * e(xk| v)}
                    prob_for_cur_tag = phi_matrix[k - 1][w] * transition_func(v, w, transition_tag_tag_dict) * \
                              emission_func(xk_minus_1, v, emission_tag_wrod_dict, word_tag_dict)
                    # find the tag who gives the max probability for this step
                    if cur_max_prob_for_w < prob_for_cur_tag and prob_for_cur_tag > 0:
                        cur_max_prob_for_w = prob_for_cur_tag
                        cur_max_tag_w = w
                # if its an unknown word, tag it as "NN" by default
                if cur_max_tag_w is None:
                    cur_max_prob_for_w = 0
                    cur_max_tag_w = 'NN'
                phi_matrix[k][v] = cur_max_prob_for_w
                back_pointer[k][v] = cur_max_tag_w
                # S
    tag_prediction = get_viterbi_tag_sequence(vocabulary_set_s, sent, phi_matrix, back_pointer, transition_func,
                                              transition_tag_tag_dict)
    return tag_prediction

def get_viterbi_tag_sequence(vocabulary, sent, phi_matrix, back_pointer, transition_func, transition_tag_tag_dict):
    """
    param: vocabulary - the set S of all the words in the corpus
           sent - the cur sentence the viterbi used on
           phi_matrix - the phi matrix the viterbi algo dynamically filled
           back_pointer - the back pointer from the viterbi algo
           transition_func- the transition function used by the algo
           transition_tag_tag_dict - the dict used by the transition function
   return -  the predicted sequence of words
    """
    n = len(sent)
    vocabulary_set_s = vocabulary
    viterbi_tag_sequence = list()
    cur_max_tag_y, cur_max_prob_for_y = None, -1
    for v in vocabulary_set_s:
        probability_for_cur_tag_v = phi_matrix[n][v] * transition_func("STOP", v, transition_tag_tag_dict)
        if cur_max_prob_for_y < probability_for_cur_tag_v:
            cur_max_prob_for_y = probability_for_cur_tag_v
            cur_max_tag_y = v
    last_tag = cur_max_tag_y
    viterbi_tag_sequence.append(cur_max_tag_y)
    for k in range(n - 1, 0, -1):
        cur_tag = back_pointer[k + 1][last_tag]
        viterbi_tag_sequence.append(cur_tag)
        last_tag = cur_tag

    viterbi_tag_sequence.reverse()
    return viterbi_tag_sequence

def print_viterbi_error_rate(test_sents, transition_tag_tag_dict, emission_tag_wrod_dict, word_tag_dict, transition_fun, emission_func):
    """
    param: test_sents - the current sentence for the viterbi algo
           transition_tag_tag_dict - the dict used by the transition function
           emission_tag_wrod_dict - the dict used by the emission function
           word_tag_dict -  the dict of all the words and their tags
           transition_func- the transition function used by the algo
           emission_func - the emission function used by the algo
    prints the error rate result
    """
    train_num_of_words = 0
    test_num_of_words = 0
    known_word_tag_match = 0
    unknown_word_tag_match = 0
    list_of_all_tags = list(emission_tag_wrod_dict.keys())
    vocabulary = word_tag_dict.keys()
    for sent in test_sents:
        only_words_sent, real_tags = list(), list()
        for tagged_word in sent:
            only_words_sent.append(tagged_word[0])
            real_tags.append(tagged_word[1])
        n = len(only_words_sent)
        viterbi_tags_prediction = viterbi(only_words_sent, transition_tag_tag_dict, emission_tag_wrod_dict, list_of_all_tags, word_tag_dict, transition_fun, emission_func)
        for i in range(n):
            if only_words_sent[i] not in vocabulary:
                test_num_of_words += 1
                if viterbi_tags_prediction[i] == real_tags[i]:
                    unknown_word_tag_match += 1
            else:
                train_num_of_words += 1
                if viterbi_tags_prediction[i] == real_tags[i]:
                    known_word_tag_match += 1
    known_words_error_rate = (1 - (known_word_tag_match / train_num_of_words))
    unknown_word_error_rate = (1 - (unknown_word_tag_match / test_num_of_words))
    total_error_rate = (
                1 - ((known_word_tag_match + unknown_word_tag_match) / (test_num_of_words + train_num_of_words)))
    print("known words error rate: : ", known_words_error_rate,
          "unknown words error rate : ", unknown_word_error_rate,
          " total error rate : ", total_error_rate)

# question 3d
def print_viterbi_error_rate_with_laplace(test_sents, transition_tag_tag_dict, emission_tag_wrod_dict, word_tag_dict, delta):
    """
        param: test_sents - the current sentence for the viterbi algo
           transition_tag_tag_dict - the dict used by the transition function
           emission_tag_wrod_dict - the dict used by the emission function
           word_tag_dict -  the dict of all the words and their tags
           transition_func- the transition function used by the algo
           emission_func - the emission function used by the algo
           delta - the delta constant for the laplace smoothing, by default delta == 1
    prints the error rate result
    """
    global DELTA
    DELTA = delta
    return print_viterbi_error_rate(test_sents, transition_tag_tag_dict, emission_tag_wrod_dict, word_tag_dict, qml_func, eml_func_laplace)


# question 3e
def pw_generator(tagged_word):
    """
    param :tagged word
    return: the pseduo words replacement
    """
    word, tag = tagged_word[0], tagged_word[1]
    if len(word) == 1:
        word = "char_pw"
    elif word.isupper():
        word = "all_cap_pw"
    elif word[0].isupper():
        word = "start_with_cap_pw"
    elif word.endswith("ed"):
        word = "past_pw"
    elif word.endswith("ing"):
        word = "ing_tense_pw"
    elif word.endswith(("er", "ish", "ship", "ee", "tion", "sion",  "ness", "ism", "dom", "ence")):
        word = "noun_pw"
    elif word.endswith(("ate", "en")):
        word = "verb_pw"
    elif word.endswith(("able", "en", "ish", "ive",  "ous", "ible", "ous")):
        word = "adjective_pw"
    elif word.endswith(("ly", "ble", "wise")):
        word = "adverb_pw"
    elif word.endswith("est"):
        word = "the_most_pw"
    elif word.endswith("er"):
        word = "er_pw"
    elif word.endswith("s"):
        word = "many_pw"
    elif word.endswith("een"):
        word = "number_teen_pw"
    elif word[:len(word) - 1].isdigit() and word.endswith("%"):
        word = "percentage_pw"
    elif word.isnumeric() and word.startswith(("1", "20")) and len(word) == 4:
        word = "year_pw"
    elif word.isnumeric():
        word = "number_pw"
    elif ":" in word:
        word = "time_pw"
    elif word.endswith(("st", "rd", "st")):
        word = "day_in_the_month_pw"
    elif "," in word:
        word = "large_number_pw"
    elif word.endswith("$"):
        word = "price_tag_pw"
    elif "-" in word:
        word = "dash-pw"
    elif word.endswith("\\'s"):
        word = "belong_to_pw"
    return tuple([word, tag])


def replace_words_with_pseudo_words(text, rare_used_words_list,  word_per_tag_dict):
    """
    param: text - the tagged corpus text
           rare_used_words_list - a list of all the rare used words from the corpus
           word_per_tag_dict - dict of all the words for the vocabulary
    return: a new text with all the pseduo words we replaced
    """
    new_text = list()
    vocabulary = word_per_tag_dict.keys()
    for sent in text:
        pw_sent_list = list()
        for tagged_word in sent:
            cur_word = tagged_word[0]
            if cur_word not in vocabulary or cur_word in rare_used_words_list:
                converted_pw = pw_generator(tagged_word)
                pw_sent_list.append(converted_pw)
            else:
                pw_sent_list.append(tagged_word)
        new_text.append(pw_sent_list)
    return new_text


def print_viterbi_error_rate_with_laplace_and_pw(test_sents, transition_tag_tag_dict, emission_tag_wrod_dict, word_tag_dict, transition_fun, emission_func):

    train_num_of_words = 0
    test_num_of_words = 0
    known_word_tag_match = 0
    unknown_word_tag_match = 0
    list_of_all_tags = list(emission_tag_wrod_dict.keys())
    vocabulary = word_tag_dict.keys()
    prediction_list, ture_tag_list = [], []

    for sent in test_sents:
        only_words_sent, real_tags = list(), list()
        for tagged_word in sent:
            only_words_sent.append(tagged_word[0])
            real_tags.append(tagged_word[1])
        n = len(only_words_sent)
        viterbi_tags_prediction = viterbi(only_words_sent, transition_tag_tag_dict, emission_tag_wrod_dict, list_of_all_tags, word_tag_dict, transition_fun, emission_func)
        ture_tag_list += real_tags
        prediction_list += viterbi_tags_prediction

        for i in range(n):
            if only_words_sent[i] not in vocabulary:
                test_num_of_words += 1
                if viterbi_tags_prediction[i] == real_tags[i]:
                    unknown_word_tag_match += 1
            else:
                train_num_of_words += 1
                if viterbi_tags_prediction[i] == real_tags[i]:
                    known_word_tag_match += 1
    confu_matrix = confusion_matrix(ture_tag_list, prediction_list)
    known_words_error_rate = (1 - (known_word_tag_match / train_num_of_words))
    unknown_word_error_rate = (1 - (unknown_word_tag_match / test_num_of_words))
    total_error_rate = (1 - ((known_word_tag_match + unknown_word_tag_match) / (test_num_of_words + train_num_of_words)))
    print("known words error rate: : " ,known_words_error_rate,
          "unknown words error rate : " , unknown_word_error_rate,
          " total error rate : "  , total_error_rate)
    return confu_matrix

def main():
    train_sentences = brown.tagged_sents(categories=['news'])
    clean_sentences = clean_tage(train_sentences)
    train, test = train_test_split(np.array(clean_sentences), test_size = 0.1, shuffle=False)

    #  3b
    # MLE base line error rate -
    # known: 0.07516597276921344, unknown is: 0.75043630017452, total is: 0.15229741851888767
    vocabulary = update_vocabulary_known_words(train)
    best_match = MLE_baseline(train, test, vocabulary)
    print_and_calculate_mle_error_rate(test, best_match, vocabulary)

    transition_tag_tag_dict = create_tag_bigram_counter_dict(train)
    word_tag_dict = create_tags_per_each_word_dict(train)
    emission_tag_wrod_dict = create_words_per_tag_dict(train)

    # 3c
    # viterbi error rate -
    # known words: 0.16023404973556876, unknown words 0.75043630017452, total error: 0.2276487590949865
    print_viterbi_error_rate(test, transition_tag_tag_dict, emission_tag_wrod_dict, word_tag_dict, qml_func, eml_func)
#     # 3d
#     #  viterbi add one error rate -
#     #  known_words: 0.14616856081917406, unknown_words: 0.7146596858638743,total error:  0.21110335891557863
    print_viterbi_error_rate_with_laplace(test, transition_tag_tag_dict, emission_tag_wrod_dict, word_tag_dict, 1)
#
    # 3e1
    rare_used_words_list = find_rare_used_words(test, word_tag_dict)
    train_converted_pw = replace_words_with_pseudo_words(train, rare_used_words_list, word_tag_dict)
    test_converted_pw = replace_words_with_pseudo_words(test, rare_used_words_list, word_tag_dict)
    pw_transition_tag_tag_dict =create_tag_bigram_counter_dict(train_converted_pw)
    pw_word_per_tag_dict = create_tags_per_each_word_dict(train_converted_pw)
    pw_emission_tag_per_word = create_words_per_tag_dict(train_converted_pw)
#
#     # 3e2 viterbi error rate with pw
#     # viterbi with pw error rate -
#     # known words: 0.16252545824847253, unknown words: 0.44131455399061037, total error: 0.1684441343566232
    print_viterbi_error_rate(test_converted_pw, pw_transition_tag_tag_dict, pw_emission_tag_per_word, pw_word_per_tag_dict, qml_func, eml_func)
#
#     # 3e3
#     # viterbi with pw and add one error rate -
#     # known words: 0.14287169042769854, unknown words: 0.5586854460093897, total error: 0.15169939200637894
    confn_matrix = print_viterbi_error_rate_with_laplace_and_pw(test_converted_pw, pw_transition_tag_tag_dict, pw_emission_tag_per_word, pw_word_per_tag_dict, qml_func, eml_func_laplace)[3]
    df = pd.DataFrame(confn_matrix)
    df.to_excel("NLP_ex2_confusion_matrix.xlsx", index=False)

#
if __name__ == '__main__':
    main()


#
#     # print("jijijijiji")
#     # print(d)
#
#




