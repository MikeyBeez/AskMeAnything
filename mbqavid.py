#!/usr/bin/env python
# coding: utf-8

import numpy as np
import requests
import html2text
from googlesearch import search
import json
import re
from simpletransformers.question_answering import QuestionAnsweringModel
from IPython.display import display
from IPython.html import widgets
from bs4 import BeautifulSoup
from markdown import markdown


model = QuestionAnsweringModel('distilbert', 'distilbert-base-uncased-distilled-squad')

question_data = {
    'qas': 
    [{'question': 'What color is the sky',
       'id': 0,
        'answers': [{'text': ' ', 'answer_start': 0}],
        'is_impossible': False}],
        'context': 'the sky is blue'
    }

prediction = model.predict([question_data])

print(prediction)

