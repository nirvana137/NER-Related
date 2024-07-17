# -*- coding: utf-8 -*-

#import pickle
#import torch
#from transformers import BertForQuestionAnswering
#from transformers import BertTokenizer
#pip install flask_cors

from flask_cors import CORS

#app = Flask(__name__)
CORS(app)
cors = CORS(app, resource={
    r"/*":{
        "origins":"*"
    }
})

from flask import current_app, flash, jsonify, make_response, redirect, request, url_for
import re
import json
#from ner_train import train
import spacy
import random
import json
from spacy.language import Language
from spacy.training import Example
from keybert import KeyBERT
#import numpy as np

#model = pickle.load(open('QA_model.pickle','rb'))

#tokenizer = pickle.load(open('tokenizer.pickle','rb'))

#!pip install flask-ngrok

#from flask_ngrok import run_with_ngrok
from flask import Flask, request
import requests

app = Flask(__name__)

#run_with_ngrok(app)
#app.config['SECRET_KEY'] = 'GDtfDCFYjD'
@app.route('/predict',methods=['POST'])
def Customer_behavior():
    request_data = request.get_json(force=True)
    #question = request_data['question']
    answer_text = request_data['answer_text']
    #print(question)
    print(answer_text) 
    #prediction = classifier_colab.predict(scaler_colab.transform(np.array([[age,salary]])))
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(answer_text)
    match=kw_model.extract_keywords(answer_text, keyphrase_ngram_range=(1, 1), stop_words=None)

    nlp = spacy.load("course.model")
    #train()
    pattern = r"(\W|^)(5G|artificial intelligence|AR/VR)(\W|$)"
    doc1 = "So in AR/VR this section 5G, we'll look at what is artificial intelligence. So artificial intelligence really at the core is about machines mimicking the behavior of the human mind, very much looking at a data-driven approach for this, and really about the key features of artificial intelligence, which is very much around perception, around reasoning, and around control. Now, within the whole scope of artificial intelligence, there are subdomains which include machine learning. And within machine learning, there's a further subdomain entitled deep learning. So if we look at the difference between machine learning versus deep learning, we'll use an example here of facial recognition. So if I have a problem where I'm trying to recognize a person's face, what I'll first of all do with classical machine learning is identify key features of that face I'm looking to recognize. I'll then passed that through an algorithm of choice to come up with the correct answer that I'm looking for. So some of the challenges around using classical machine learning is, first of all, identifying which features are key to that recognition process. And secondly is how you actually identify those features in the first place and getting the data necessary to run through the algorithm. In a deep learning approach, what then happens instead is we take the N by N pixel view of the image and, with sufficient data and labeling and hyperparameters, pass this through the deep neural network to provide a percentage outcome based upon the facial recognition imaging process. Now, this is called a \"deep neural network\" because these layers within the network are very similar aligned to the neurals within the brain and the activity within the brain. And secondly, it's deep because there's multiple layers of these deep neural networks. And as part of this, go through a training process to identify the weights, which are the connectors between these different neurals within the deep neural network. And then that model is then used from the training in an inference process to actually determine the weighting and the percentage outcome of images that are then passed through the trained neural network. So if we look at this in example, here we have a training session set up where we're looking to identify one of three different images, whether it's a human, a bicycle, or a strawberry. So first of all, the images are first of all sent through the neural network as part of the training process. If an error occurs, there's a backward path which provides some correction to the weights within the neural network. And this is how the neural network continuously trains based upon a number of images being sent through, the expected outcome, the labeling of the data on this forward and backward process. Now, once the neural network has been trained, this model then becomes frozen and then is used in the inference process. The inference process is where you're actually then scoring images that you have using that trained neural network to determine the appropriate outcome of that training to identify the images that you're looking for. And why is this now becoming so interesting and so important? So over the last few years, there's been multiple breakthroughs in deep learning process, deep learning activities, and research so that now, for a large number of cases, we're actually at a better level of recognition of images and speech than human using machine-driven deep learning NOTICES AND DISCLAIMERS"
    matches = re.findall(pattern, answer_text)
    #print (matches)
    #matches.type
    #nlp.to_disk("course.model")
    # Testing the NER
    doc = nlp(answer_text)
    #print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

    matches1 = [(ent.text, ent.label_) for ent in doc.ents]

    answer = matches +matches1
    print (answer)
    #answer = ' '.join(tokens[answer_start:answer_end+1])

    #print('Answer: "' + answer + '"')
    return jsonify("1-Regex & spacy Answer --{0}{1} 2- bert Answer {2}".format(matches,matches1,match))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8888, debug=True)
