#Creating a naive bayes classifier to distinguish between hockey emails and baseball emails

from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

#Selecting baseball and hockey emails to be used in training the model
train_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset = "train", shuffle = True, random_state = 108)

#Exploring the data
print(train_emails.target_names)
print(train_emails.data[5])
print(train_emails.target[5])
#The labels themselves are numbers, but those numbers correspond to the label names found at emails.target_names

#Selecting the emails to be used for testing the model
test_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset = "test", shuffle = True, random_state = 10)

#Transfroming the emails into lists of word counts (before the naive bayes classification)
counter = CountVectorizer()
counter.fit(test_emails.data + train_emails.data)
#Getting the counts of words in the training set
train_counts = counter.transform(train_emails.data)
#Getting the counts of words in the test set
test_counts = counter.transform(test_emails.data)

#Creating a naive bayes classifier
classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)
#Getting the accuracy of the classifier using the test data
print(classifier.score(test_counts, test_emails.target))

#Testing other datasets (hockey and tech) to see how well the classifier does in distinguishing them
train_emails_2 = fetch_20newsgroups(categories = ['comp.sys.ibm.pc.hardware','rec.sport.hockey'], subset = "train", shuffle = True, random_state = 108)
test_emails_2 = fetch_20newsgroups(categories = ['comp.sys.ibm.pc.hardware','rec.sport.hockey'], subset = "test", shuffle = True, random_state = 108)

counter_2 = CountVectorizer()
counter_2.fit(test_emails_2.data + train_emails_2.data)
#Getting the counts of words in the training set
train_counts_2 = counter.transform(train_emails_2.data)
#Getting the counts of words in the test set
test_counts_2 = counter.transform(test_emails_2.data)

#Creating a naive bayes classifier
classifier_2 = MultinomialNB()
classifier_2.fit(train_counts_2, train_emails_2.target)
#Getting the accuracy of the classifier using the test data
print(classifier_2.score(test_counts_2, test_emails_2.target))

#Testing three other datasets (medicine, space and religion) to see how well the classifier does in distinguishing them
train_emails_3 = fetch_20newsgroups(categories = ['sci.med', 'sci.space', 'soc.religion.christian'], subset = "train", shuffle = True, random_state = 108)
test_emails_3 = fetch_20newsgroups(categories = ['sci.med', 'sci.space', 'soc.religion.christian'], subset = "test", shuffle = True, random_state = 108)

counter_3 = CountVectorizer()
counter_3.fit(test_emails_3.data + train_emails_3.data)
#Getting the counts of words in the training set
train_counts_3 = counter.transform(train_emails_3.data)
#Getting the counts of words in the test set
test_counts_3 = counter.transform(test_emails_3.data)

#Creating a naive bayes classifier
classifier_3 = MultinomialNB()
classifier_3.fit(train_counts_3, train_emails_3.target)
#Getting the accuracy of the classifier using the test data
print(classifier_3.score(test_counts_3, test_emails_3.target))