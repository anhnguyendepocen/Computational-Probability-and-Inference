import sys
import os.path
import numpy as np
from collections import Counter
from collections import defaultdict

import util

USAGE = "%s <test data folder> <spam folder> <ham folder>"

def get_counts(file_list):#, single=False):
    """
    Computes counts for each word that occurs in the files in file_list.

    Inputs
    ------
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the number of files the
    key occurred in.
    """
    cnt = Counter()

    for file in file_list:
        words = set(util.get_words_in_file(file))
        for word in words:
            cnt[word] += 1
    # print(cnt)
    return cnt

def get_log_probabilities(file_list):
    """
    Computes log-frequencies for each word that occurs in the files in 
    file_list.

    Input
    -----
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the log of the smoothed
    estimate of the fraction of files the key occurred in.
    """

    num_files = len([f for file in file_list
                       for f in file])
    words = get_counts(file_list)
    val = -np.log(num_files)
    log_counts = defaultdict(lambda: val)
    for k,v in words.items():
        log_counts[k] = np.log((v+1)/(num_files+2))

    return log_counts


def learn_distributions(file_lists_by_category):
    """
    Input
    -----
    A two-element list. The first element is a list of spam files, 
    and the second element is a list of ham (non-spam) files.

    Output
    ------
    (log_probabilities_by_category, log_prior)

    log_probabilities_by_category : A list whose first element is a smoothed
                                    estimate for log P(y=w_j|c=spam) (as a dict,
                                    just as in get_log_probabilities above), and
                                    whose second element is the same for c=ham.

    log_prior_by_category : A list of estimates for the log-probabilities for
                            each class:
                            [est. for log P(c=spam), est. for log P(c=ham)]
    """
    log_probabilities_by_category = [None]*2
    spam = file_lists_by_category[0]
    ham = file_lists_by_category[1]
    lspam = len(spam)
    lham = len(ham)
    p_spam = np.log(lspam / (lspam + lham))
    p_ham = np.log(lham / (lspam + lham))

    spam_words = get_log_probabilities(spam)
    ham_words = get_log_probabilities(ham)

    log_probabilities_by_category[0] = spam_words
    log_probabilities_by_category[1] = ham_words
    log_prior_by_category = [p_spam, p_ham]

    # print(log_probabilities_by_category)
    return (log_probabilities_by_category, log_prior_by_category)

def classify_email(email_filename,
                   log_probabilities_by_category,
                   log_prior_by_category):
    """
    Uses Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    email_filename : name of the file containing the email to be classified

    log_probabilities_by_category : See output of learn_distributions

    log_prior_by_category : See output of learn_distributions

    Output
    ------
    One of the labels in names.
    """
    # **********************
    email_words = set(util.get_words_in_file(email_filename))
    N_categories  = len(log_probabilities_by_category)

    # get the union of all words encountered during training
    all_words = []
    for i in range(N_categories):
        all_words += log_probabilities_by_category[i].keys()
    all_words = list(set(all_words))

    log_likelihoods = []
    for i in range(N_categories):
        total = 0
        all_word_log_probs = log_probabilities_by_category[i]
        for w in all_words:
            log_prob = all_word_log_probs[w]
            test = (w in email_words)
            if w in email_words:
                total += log_prob
            else:
                total += np.log(1 - np.exp(log_prob))
        log_likelihoods.append(total)
    posterior = np.array(log_likelihoods) + np.array(log_prior_by_category)
    winner = np.argmax(posterior)
    if winner == 0:
        return "spam"
    else:
        return "ham"


def classify_emails(spam_files, ham_files, test_files):
    # DO NOT MODIFY -- used by the autograder
    log_probabilities_by_category, log_prior = \
        learn_distributions([spam_files, ham_files])
    estimated_labels = []
    for test_file in test_files:
        estimated_label = \
            classify_email(test_file, log_probabilities_by_category, log_prior)
        estimated_labels.append(estimated_label)
    return estimated_labels



def main():
    ### Read arguments
    if len(sys.argv) != 4:
        print(USAGE % sys.argv[0])
    testing_folder = 'data/testing/'
    (spam_folder, ham_folder) = ['data/spam/', 'data/ham/']#sys.argv[2:4]

    ### Learn the distributions
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
    (log_probabilities_by_category, log_priors_by_category) = \
            learn_distributions(file_lists)

    # learn_distributions([['autograder_spam1', 'autograder_spam2', 'autograder_spam3'], ['autograder_ham1', 'autograder_ham2']])
    # learn_distributions(file_lists)
    # get_log_probabilities(file_lists)

    # Here, columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    performance_measures = np.zeros([2,2])

    ### Classify and measure performance
    for filename in (util.get_files_in_folder(testing_folder)):
        ## Classify
        label = classify_email(filename,
                               log_probabilities_by_category,
                               log_priors_by_category)
        ## Measure performance
        # Use the filename to determine the true label
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[true_index, guessed_index] += 1


        # (Un)comment this line to see which files your classifier
        # gets right/wrong:
        print("%s : %s" %(label, filename))

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],
                      totals[0],
                      correct[1],
                      totals[1]))

if __name__ == '__main__':
    main()
