from __future__ import division
from codecs import open
from collections import Counter
import numpy as np

def read_documents(doc_file):
    docs = []
    labels = [] #to label if the review is pos or neg
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
    return docs, labels

def train_nb(documents, labels):
    pos_words = set() #get all distinct words with pos review
    neg_words = set() 
    num_pos = sum(1 for label in labels if label=='pos') #count pos review are there
    prior_pos = np.log(num_pos/len(labels)) #probability of pos review
    num_neg = sum(1 for label in labels if label=='neg')
    prior_neg = np.log(num_neg/len(labels))

    freq_pos = Counter()
    freq_neg = Counter()

    for doc, label in zip(documents, labels): #to take [] in documents to pair with the label in labels
        if label=='pos': 
            pos_words.update(doc)
            freq_pos.update(doc) #gives the word with the no. of times the word appearing in all pos reviews
        if label=='neg': 
            neg_words.update(doc)
            freq_neg.update(doc)

    total_vocab = pos_words.union(neg_words)

    model = {
        'prior_pos': prior_pos,
        'prior_neg': prior_neg,
        'freq_pos': freq_pos,
        'freq_neg': freq_neg,
        'total_pos_words': sum(freq_pos.values()),
        'total_neg_words': sum(freq_neg.values()),
        'total_words': len(total_vocab)
    }
    return model

def score_doc_label(document, label, parameters):
    if label == 'pos':
        score = parameters['prior_pos']
        word_stat = parameters['freq_pos']
        all_words = parameters['total_pos_words']
    if label == 'neg':
        score = parameters['prior_neg']
        word_stat = parameters['freq_neg']
        all_words = parameters['total_neg_words']
    vocab_size = parameters['total_words']
    for word in document:
        word_count = word_stat.get(word, 0)
        score += np.log((1 + word_count)/(all_words + vocab_size))
    return score

def classify_nb(doc, parameters):
    prob_pos = score_doc_label(doc, 'pos', parameters)
    prob_neg = score_doc_label(doc, 'neg', parameters)
    
    if prob_pos > prob_neg: guess = 'pos'
    else: guess = 'neg'

    return guess

def classify_documents(docs, parameters):
    guess_result = []
    for doc in docs:
        prob_pos = score_doc_label(doc, 'pos', parameters)
        prob_neg = score_doc_label(doc, 'neg', parameters)
        
        if prob_pos > prob_neg: guess = 'pos'
        else: guess = 'neg'

        guess_result.append(guess)

    return guess_result

def accuracy(true_labels, guessed_labels):
    num_correct_guess = 0
    num_total_docs = len(true_labels)

    for i in range(num_total_docs):
        if true_labels[i] == guessed_labels[i]: num_correct_guess += 1

    correct_ratio = num_correct_guess/num_total_docs
    return correct_ratio

def error_analysis(docs, true_labels, guessed_labels, parameters):
    wrongly_classified = []
    count = 0
    index_wrong_docs = []
    for i in range(len(docs)):
        if count == 3: break
        if true_labels[i] != guessed_labels[i]:
            wrongly_classified.append(docs[i])
            count += 1
            index_wrong_docs.append(i)
    return wrongly_classified, index_wrong_docs


all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')
split_point = int(0.80*len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

train_param = train_nb(train_docs, train_labels)

# #sanity check 1: test 'great' --> check that for 'great', P(pos) > P(neg) 
# print("Sanity Check 1")
# pos_score_great = score_doc_label(['great'], 'pos', train_param)
# neg_score_great = score_doc_label(['great'], 'neg', train_param)
# print("For the word 'great':")
# print(f"P(pos): {np.exp(pos_score_great)}")
# print(f"P(neg): {np.exp(neg_score_great)}")

# #sanity check 1: test 'bad' --> check that for 'bad', P(neg) > P(pos) 
# pos_score_bad = score_doc_label(['bad'], 'pos', train_param)
# neg_score_bad = score_doc_label(['bad'], 'neg', train_param)
# print("For the word 'bad':")
# print(f"P(pos): {np.exp(pos_score_bad)}")
# print(f"P(neg): {np.exp(neg_score_bad)}")

# print()

# #sanity check 2: make sure that score_doc_label will not crash
# print("Sanity Check 2")
# unseen_doc = ['a', 'top-quality', 'performance']
# pos_score__unseen = score_doc_label(unseen_doc, 'pos', train_param)
# neg_score_unseen = score_doc_label(unseen_doc, 'neg', train_param)
# print("For the unseen doc:")
# print(f"P(pos): {np.exp(pos_score__unseen)}")
# print(f"P(neg): {np.exp(neg_score_unseen)}")

# #test classify_nb using first 3 samples from eval_docs
# for i in range(3):
#     test_doc = eval_docs[i]
#     true_label = eval_labels[i]
#     predicted_label = classify_nb(test_doc, train_param)
#     print(f"Sample {i+1}")
#     print(f"True Label: {true_label}")
#     print(f"Predicted Label: {predicted_label}")
#     print()

predict_labels_eval_docs = classify_documents(eval_docs, train_param)

prediction_accuracy = accuracy(eval_labels, predict_labels_eval_docs)
# print(f"Accuracy when evaluating the classifier on the test set: {(prediction_accuracy*100):.1f}%")

#Error analysis   
wrong_result, index_wrong_result = error_analysis(eval_docs, eval_labels, predict_labels_eval_docs, train_param)
print("These are the first 3 documents that were wrongly classified:\n")
for doc, index in zip(wrong_result, index_wrong_result):
    sentence = " ".join(doc)
    true_label = eval_labels[index]
    guessed_label = predict_labels_eval_docs[index]
    
    print(f"Review Text: {sentence}")
    print(f"-> True Label: {true_label}  |  Model's Guess: {guessed_label}")
    print("-" * 60)