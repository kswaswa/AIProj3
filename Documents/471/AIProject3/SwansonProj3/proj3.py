#to descriptor, load image with python image loader jpeg = image.open, get pixels, flatten list, put that as descriptor, normalize it

import cv2
import numpy as np
import sklearn
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
import pickle
from PIL import Image
import os


usrImg = 0

labels = []
descriptors = []

validationSet = []


def append(name, strName):
    global descriptors
    global labels
    temp = []
    labels.append(strName)
    for i in range(len(name)):
        #python image library
        im = Image.open(name[i])
        gd = im.getdata()
        sum = 0
        for pixel in gd:
            sum += (pixel[0] + pixel[1] + pixel[2])
        #sum of RBG value of each pixel (flattened 2d array)
        temp.append(sum)
    descriptors.append(temp)

def getPics():
    global validationSet

    smileys = []
    d = '/Users/katie/Documents/Data/01'
    for f in os.listdir(d):
        smileys.append(os.path.join(d, f))
    hats = []
    e = '/Users/katie/Documents/Data/02'
    for f in os.listdir(e):
        hats.append(os.path.join(e, f))
    hashtags = []
    g = '/Users/katie/Documents/Data/03'
    for f in os.listdir(g):
        hashtags.append(os.path.join(g, f))
    hearts = []
    h = '/Users/katie/Documents/Data/04'
    for f in os.listdir(h):
        hearts.append(os.path.join(h, f))
    dollars = []
    i = '/Users/katie/Documents/Data/05'
    for f in os.listdir(i):
        dollars.append(os.path.join(i, f))

    validationSet = [smileys.pop(), hats.pop(), hashtags.pop(), hearts.pop(), dollars.pop(), smileys.pop(),hats.pop(), hashtags.pop(), hearts.pop(), dollars.pop()]

    names = [smileys, hats, hashtags, hearts, dollars]
    strNames = ['Smiley', 'Hat', 'Hashtag', 'Heart', 'Dollar']

    for i in range(len(names)):
        append(names[i], strNames[i])


def pickle():
    global labels
    global descriptors
    getPics()

    clf = svm.SVC()
    clf.fit(labels, descriptors)
    s = pickle.dumps(clf)
    clf2 = pickle.loads(s)
    clf2 = predict(usrImg)
    print("Prediction of user image is: ", clf2)

def printScore(labels, descriptors):
    model = svm.SVC(kernel='linear',C=1)
    scores = sklearn.cross_validation.cross_val_score(estimator=model, X=descriptors, 
y=labels, cv=5, scoring='f1_weighted') 
    sum = 0
    for i in range(len(scores)):
        sum += scores[i]

    score = sum / len(scores)
    print("Average accuracy is: ", score, "%.")


def normalize(descriptors):
    min = 100000000000
    for i in descriptors:
        for j in i:
            if j < min:
                min = j
    max = 0
    for i in descriptors:
        for j in i:
            if j > max:
                max = j

    normalized = []
    for i in descriptors:
        temp = []
        for j in i:
            j = j - min
            j = j / float(max-min)
            temp.append(j)
        normalized.append(temp)

    return normalized

def main():
    global descriptors
    global labels
    getPics()
    descriptors = normalize(descriptors)
    printScore(labels, descriptors)
    usrImg = input('Enter your image name: ')
    usrImg = cv2.imread('/'+usrImg, 0)

main()




