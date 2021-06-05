{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8e6c0d05-f678-4933-980c-f3216ed324a0",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/bard/miniconda3/envs/Julie-Julie/lib/python3.6/site-packages/IPython/html.py:14: ShimWarning: The `IPython.html` package has been deprecated since IPython 4.0. You should import from `notebook` instead. `IPython.html.widgets` has moved to `ipywidgets`.\n",
      "  \"`IPython.html.widgets` has moved to `ipywidgets`.\", ShimWarning)\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import requests\n",
    "import html2text\n",
    "from googlesearch import search\n",
    "import json\n",
    "import re\n",
    "from simpletransformers.question_answering import QuestionAnsweringModel\n",
    "from IPython.display import display\n",
    "from IPython.html import widgets\n",
    "from bs4 import BeautifulSoup\n",
    "from markdown import markdown"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ab9b7904-a473-4fed-aae6-7da8f969851b",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = QuestionAnsweringModel('distilbert', 'distilbert-base-uncased-distilled-squad')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d30d0437-838f-4e34-98c9-f0a685acbb34",
   "metadata": {},
   "outputs": [],
   "source": [
    "question_data = {\n",
    "    'qas': \n",
    "    [{'question': 'What color is the sky',\n",
    "       'id': 0,\n",
    "        'answers': [{'text': ' ', 'answer_start': 0}],\n",
    "        'is_impossible': False}],\n",
    "        'context': 'the sky is blue'\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "bcd1bc6e-ec78-4810-b9ed-d0eeb95d8ede",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'qas': [{'question': 'What color is the sky',\n",
       "   'id': 0,\n",
       "   'answers': [{'text': ' ', 'answer_start': 0}],\n",
       "   'is_impossible': False}],\n",
       " 'context': 'the sky is blue'}"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "question_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "a431f4b2-8448-4d08-b521-0866d6831fcc",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "convert squad examples to features: 100%|██████████| 1/1 [00:00<00:00, 1679.06it/s]\n",
      "add example index and unique id: 100%|██████████| 1/1 [00:00<00:00, 25420.02it/s]\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "26e1dc0ca1d047479dd9c1636c635afd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Running Prediction:   0%|          | 0/1 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "prediction = model.predict([question_data])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "d8673973-f983-406b-8c55-9b3664cd75a3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "([{'id': 0, 'answer': ['the sky is blue', 'blue', 'sky is blue', 'the sky', 'is blue', 'sky', 'the', '', 'the sky is', 'sky is']}], [{'id': 0, 'probability': [0.6718919034393995, 0.2903915782771355, 0.025824106266482086, 0.009136212452780911, 0.0017607881690078252, 0.0003511495227819054, 0.0003260312949665356, 0.0002628223258765999, 5.322314582283977e-05, 2.0456269327398623e-06]}])\n"
     ]
    }
   ],
   "source": [
    "print(prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "55b97141-ad0c-488c-9005-653542fe3daf",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
