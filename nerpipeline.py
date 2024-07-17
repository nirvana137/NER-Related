# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:39:07 2022

@author: ayushi
"""

import json
labeled_data = []
with open(r"Tag.jsonl", "r") as read_file:
    for line in read_file:
        data = json.loads(line)
        labeled_data.append(data)
    print(labeled_data)
    
TRAINING_DATA = []
for entry in labeled_data:
    entities = []
    for e in entry['label']:
        entities.append((e[0], e[1],e[2]))
    spacy_entry = (entry['data'], {"entities": entities})
    TRAINING_DATA.append(spacy_entry)
#print(TRAINING_DATA)

import spacy
import random
import json
from spacy.language import Language
from spacy.training import Example

nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")
ner.add_label("5G")
ner.add_label("AR/VR")

# Start the training
nlp.begin_training()

for itn in range(40):
    # Shuffle the training data
    random.shuffle(TRAINING_DATA)
    losses = {}
# Batch the examples and iterate over them
    for batch in spacy.util.minibatch(TRAINING_DATA, size=2):
        #texts = [text for text, entities in batch]
        #annotations = [entities for text, entities in batch]
        #example = Example.from_dict(texts, annotations)
# Update the model
        #nlp.update((texts, annotations), losses=losses, drop=0.3)
        #nlp.update([examples], losses=losses, drop=0.3)
    #print(losses)
        for text, annotations in batch:
         doc = nlp.make_doc(text)
         example = Example.from_dict(doc, annotations)
         nlp.update([example], drop=0.5, losses=losses)
         print("Losses",losses)
    
    
    #for itn in range(100):
     # shuffle examples before training
     #random.shuffle(TRAIN_DATA)
     # batch up the examples using spaCy's minibatch
     #batches = minibatch(TRAIN_DATA, size=sizes)
     # dictionary to store losses
     #losses = {}
     #for batch in batches:

nlp.to_disk("course.model")
# Testing the NER
doc = nlp("So in this section, we'll look at what is artificial intelligence. So artificial intelligence really at the core is about machines mimicking the behavior of the human mind, very much looking at a data-driven approach for this, and really about the key features of artificial intelligence, which is very much around perception, around reasoning, and around control. Now, within the whole scope of artificial intelligence, there are subdomains which include machine learning. And within machine learning, there's a further subdomain entitled deep learning. So if we look at the difference between machine learning versus deep learning, we'll use an example here of facial recognition. So if I have a problem where I'm trying to recognize a person's face, what I'll first of all do with classical machine learning is identify key features of that face I'm looking to recognize. I'll then passed that through an algorithm of choice to come up with the correct answer that I'm looking for. So some of the challenges around using classical machine learning is, first of all, identifying which features are key to that recognition process. And secondly is how you actually identify those features in the first place and getting the data necessary to run through the algorithm. In a deep learning approach, what then happens instead is we take the N by N pixel view of the image and, with sufficient data and labeling and hyperparameters, pass this through the deep neural network to provide a percentage outcome based upon the facial recognition imaging process. Now, this is called a \"deep neural network\" because these layers within the network are very similar aligned to the neurals within the brain and the activity within the brain. And secondly, it's deep because there's multiple layers of these deep neural networks. And as part of this, go through a training process to identify the weights, which are the connectors between these different neurals within the deep neural network. And then that model is then used from the training in an inference process to actually determine the weighting and the percentage outcome of images that are then passed through the trained neural network. So if we look at this in example, here we have a training session set up where we're looking to identify one of three different images, whether it's a human, a bicycle, or a strawberry. So first of all, the images are first of all sent through the neural network as part of the training process. If an error occurs, there's a backward path which provides some correction to the weights within the neural network. And this is how the neural network continuously trains based upon a number of images being sent through, the expected outcome, the labeling of the data on this forward and backward process. Now, once the neural network has been trained, this model then becomes frozen and then is used in the inference process. The inference process is where you're actually then scoring images that you have using that trained neural network to determine the appropriate outcome of that training to identify the images that you're looking for. And why is this now becoming so interesting and so important? So over the last few years, there's been multiple breakthroughs in deep learning process, deep learning activities, and research so that now, for a large number of cases, we're actually at a better level of recognition of images and speech than human using machine-driven deep learning NOTICES AND DISCLAIMERS")
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
