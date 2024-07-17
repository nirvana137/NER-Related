#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#pip install wget


# In[ ]:


import webvtt
import requests


# In[ ]:


#To download vtt files

import wget

url = 'https://networkbuilders.intel.com/components/com_vcapttools/files/1601074415-doupe-part2-04-v2.vtt'

wget.download(url, 'C:/Users/onsumaye/Desktop/Projects/test2_vtt/629.vtt')


# In[ ]:


# To convert VTT to TXT

# import required module
import os
from tqdm import tqdm

# assign directory
directory = "C:/Users/onsumaye/Desktop/Projects/NER_test/test2/test2_vtt/"
path =      "C:/Users/onsumaye/Desktop/Projects/NER_test/test2/test2_txt/"
 
# iterate over files in
# that directory
for filename in tqdm(os.listdir(directory)):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(f)
        print(filename)
        
        vtt = webvtt.read(f)
        transcript = ""
        #print(vtt)

        lines = []
        for line in vtt:
            lines.extend(line.text.strip().splitlines())

        previous = None
        for line in lines:
            if line == previous:
                   continue
            transcript += " " + line
            previous = line
            
        #print(transcript) 
        new = filename.split(".vtt")[0] + ".txt"
        #print(new)
      
       #for new in os.listdir(path):
        f = open(os.path.join(path, new) , 'w')
        f.write(transcript)
        f.close()
        


# In[ ]:




