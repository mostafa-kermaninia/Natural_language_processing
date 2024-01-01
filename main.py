import pandas as pd
from hazm import *
import numpy as np

trainFileAddress = "books_train.csv"
testFileAddress = "books_test.csv"
ALPHA = 1/500000


def preprocess_a_cell(txt):
    punctuationMarks = [':', '،', '?', '؟', '.', ';',
                        '(', ')', '!', '+', '-', '*', '/',
                        '=', '\\', '»', '«', '"', '_', 'ـ', '...']
    stopWords = pd.read_csv("sw.csv")

    wordsDf = pd.DataFrame(word_tokenize(txt), columns=["word"])
    # V2
    # wordsDf = pd.DataFrame(word_tokenize(Normalizer().normalize(txt)), columns=["word"])
    wordsDf["word"] = wordsDf[wordsDf["word"].apply(
        lambda w: w not in punctuationMarks and
        w not in stopWords["word"].values and
        w.isdigit() == False)]
    # V2
    # wordsDf = pd.DataFrame(word_tokenize(txt), columns=["word"])
    # wordsDf["word"] = wordsDf[wordsDf["word"].apply(
    #     lambda w: w.isalpha() and
    #     w not in stopWords["word"].values )]
    wordsDf = wordsDf.dropna()

    # for adding bonus parts
    # wordsDf["word"]=wordsDf["word"].apply(Lemmatizer().lemmatize)
    # wordsDf["word"] = wordsDf["word"].apply(Stemmer().stem)
    return wordsDf


def preprocess_a_test_cell(txt):
    wordsDf = pd.DataFrame(word_tokenize(txt), columns=["word"])
    # V2
    # wordsDf = pd.DataFrame(word_tokenize(Normalizer().normalize(txt)), columns=["word"])
    # for adding bonus parts
    # wordsDf["word"]= wordsDf["word"].apply(Lemmatizer().lemmatize)
    # wordsDf["word"] = wordsDf["word"].apply(Stemmer().stem)
    return wordsDf


def preprocess(trainFileAddress):
    df = pd.read_csv(trainFileAddress)

    df["description"] = df["description"].apply(preprocess_a_cell)
    df["title"] = df["title"].apply(preprocess_a_cell)

    return df


def build_BOW(booksDf):
    BOW = {'جامعه‌شناسی': {}, 'کلیات اسلام': {}, 'داستان کودک و نوجوانان': {},
           'داستان کوتاه': {}, 'مدیریت و کسب و کار': {}, 'رمان': {}}

    for bookIndex in booksDf.index:
        curCategory = booksDf['categories'][bookIndex]
        for word in booksDf['description'][bookIndex]['word']:
            if word not in BOW[curCategory].keys():
                BOW[curCategory][word] = 0
            BOW[curCategory][word] += 1
        for word in booksDf['title'][bookIndex]['word']:
            if word not in BOW[curCategory].keys():
                BOW[curCategory][word] = 0
            BOW[curCategory][word] += 1

    return BOW


def countAllWords(categoryDict):
    countOfAllWords = 0
    for word in categoryDict.keys():
        countOfAllWords += categoryDict[word]
    return countOfAllWords


def convert_BOW_to_sample_probs_BOW(BOW):
    for category in BOW.keys():
        countOfAllWords = countAllWords(BOW[category])

        for word in BOW[category].keys():
            # change value of words to conditional prob of them
            BOW[category][word] = BOW[category][word]/countOfAllWords

    return BOW


def convert_BOW_to_additive_smooth_probs_BOW(BOW):
    for category in BOW.keys():
        countOfDifferentWords = len(BOW[category])

        countOfAllWords = countAllWords(BOW[category])
        for word in BOW[category].keys():
            # change value of words to conditional prob of them
            BOW[category][word] = (BOW[category][word] + ALPHA) / \
                (countOfAllWords + (ALPHA * countOfDifferentWords))

    return BOW


def calculateCategorylikelihood(bookTitle, bookDescription, categoryDict):
    likelyhood = np.log(1/6)
    wordsCount = countAllWords(categoryDict)
    for word in bookDescription:
        if word in categoryDict.keys():
            likelyhood += np.log(categoryDict[word])
        else:
            likelyhood += np.log(ALPHA/(wordsCount +
                                 (ALPHA * len(categoryDict))))
    for word in bookTitle:
        if word in categoryDict.keys():
            likelyhood += np.log(categoryDict[word])
        else:
            likelyhood += np.log(ALPHA/(wordsCount +
                                 (ALPHA * len(categoryDict))))
    return likelyhood


def returnModelGuess(categoryProbs):
    probs = list(categoryProbs.values())
    categories = list(categoryProbs.keys())
    return categories[probs.index(max(probs))]


def test(BOW, testFileAddress):
    testDf = pd.read_csv(testFileAddress)
    testDf["description"] = testDf["description"].apply(preprocess_a_test_cell)
    testDf["title"] = testDf["title"].apply(preprocess_a_test_cell)
    trueTests = 0
    for bookIndex in testDf.index:
        categoryProbs = {'جامعه‌شناسی': 0, 'کلیات اسلام': 0, 'داستان کودک و نوجوانان': 0,
                         'داستان کوتاه': 0, 'مدیریت و کسب و کار': 0, 'رمان': 0}
        bookTitle = testDf['title'][bookIndex]['word']
        bookDescription = testDf['description'][bookIndex]['word']
        for category in categoryProbs.keys():
            categoryProbs[category] = calculateCategorylikelihood(
                bookTitle, bookDescription, BOW[category])
        # print(categoryProbs)
        if returnModelGuess(categoryProbs) == testDf['categories'][bookIndex]:
            trueTests += 1
    print()
    print("تعداد کل تست ها : ",  len(testDf))
    print("تعداد تست های درست پاسخ داده شده : ", trueTests)
    print("درصد بازدهی این مدل : ", trueTests/len(testDf) * 100)


# main
booksDF = preprocess(trainFileAddress)
BOW = build_BOW(booksDF)
# convert_BOW_to_sample_probs_BOW(BOW)
convert_BOW_to_additive_smooth_probs_BOW(BOW)
test(BOW, testFileAddress)
