import matplotlib
#matplotlib.use('TkAgg')
import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import math
from tools import load_attention_weights,load_contents
from tools import loadTextRepresentations,loadTextLabels

epochs = 10
global_colors = ['darkgreen',"crimson",'dodgerblue','darkturquoise','dimgray','darkviolet','darkorange']
fig_path = '../figs/'

#show batch contents and weights
def showAttentionWeightsForContent(contents,attweights,classifierType):
    #transform numpy to list
    plt.cla()
    plt.rcParams['figure.figsize'] = (16.0, 4.0)
    #fig = plt.figure(figsize=(8, 6))
    fig = plt.figure()
    fig.suptitle(' Visualization of Attention Weights', fontsize=14, fontweight='bold')

    #ax = fig.add_subplot(111)
    unit = 0.0082
    x,y = 0.05,0.90
    for content,attweight in zip(contents,attweights):
        #drop the useless weigths
        attweight = attweight[:len(content)]
        #min max scaler
        attweight = [(weight-min(attweight))/(max(attweight)-min(attweight)) for weight in attweight]
        count =0
        for word,weight in zip(content,attweight):
            plt.text(x, y, word, style='italic',fontsize=12,
                    bbox={'facecolor': 'red', 'alpha': weight, 'pad': 2})
            x+=len(word)*unit+0.01
            count +=1
            #every 10 words,change the line
            if count%14==0:
                x = 0.1
                y -=0.09

            print (x,len(word),'---',word,' --- ',weight)
            print(word)
        print(' ')
        #reset coordinates
        x = 0.05
        y -= 0.14
    plt.savefig('./figs/'+'comp_positive_'+classifierType+'_weights_visualization.png', bbox_inches='tight')
    plt.show()

def visualize_Attention(datasetType,classifierType):
    reviews = load_contents(datasetType,classifierType)
    weights = load_attention_weights(datasetType,classifierType)
    start= 10
    review_idx = range(start,start+6)

    visualize_reviews = []
    visualize_weights = []
    for idx in review_idx:
        visualize_reviews.append(reviews[idx])
        visualize_weights.append(weights[idx])

    showAttentionWeightsForContent(visualize_reviews,visualize_weights,classifierType)

def find_review_idx(single_review,reviews):
    for idx,review in enumerate(reviews):
        if compare_list(single_review,review):
            return idx
    return -1

def compare_list(list1,list2):
    if len(list1)==len(list2):
        for i in range(len(list1)):
            if list1[i]==list2[i]:
                pass
            else:
                return False
    else:
        return False
    return True

def visualize_compare_Attention(datasetType, classifierType1, classifierType2):
    type1_reviews = load_contents(datasetType, classifierType1)
    type1_weights = load_attention_weights(datasetType, classifierType1)

    type2_reviews = load_contents(datasetType, classifierType2)
    type2_weights = load_attention_weights(datasetType, classifierType2)
    i=30
    review_idx = [i,i+2,i+3,i+4]

    #CR
    review_idx = [3235,2133,30,22]
    #review_idx = [3253,3543,4062,3235]


    type1_visualize_reviews = []
    type1_visualize_weights = []
    for idx in review_idx:
        type1_visualize_reviews.append(type1_reviews[idx])
        type1_visualize_weights.append(type1_weights[idx])

    showAttentionWeightsForContent(type1_visualize_reviews, type1_visualize_weights,classifierType1)

    #locate the corresponding review,weights in type2
    type2_visualize_reviews = []
    type2_visualize_weights = []
    for review in type1_visualize_reviews:
        correspond_idx = find_review_idx(review,type2_reviews)
        print(correspond_idx)
        if correspond_idx==-1:
            print('###Error not found review',review)
        type2_visualize_reviews.append(type2_reviews[correspond_idx])
        type2_visualize_weights.append(type2_weights[correspond_idx])
    showAttentionWeightsForContent(type2_visualize_reviews, type2_visualize_weights,classifierType2)


if __name__ == '__main__':
    visualize_compare_Attention('TREC', 'LSTM_IAtt', 'LSTMAtt')
    #visualize_compare_Attention('CR', 'LSTM_IAtt', 'LSTMAtt') 
