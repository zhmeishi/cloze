# Story Cloze Test

## About

This is a course project for the course Natural Language Understanding at ETH.

## Dependencies

This code is written in python. To use it you will need:

* Python 3.6
* A recent version of [NumPy](http://www.numpy.org/)
* [scikit-learn](http://scikit-learn.org/stable/index.html)
* [NLTK 3](http://www.nltk.org/)
* [Tensorflow](https://www.tensorflow.org/) 
* [pandas](https://pandas.pydata.org/) 

## Getting started

You will first need to download the word embeddings and preprocess data. The embedding files are quite large (>5GB) so make sure there is enough space available.

    ./get_data.sh

Now, set up virtual environment.

    conda create -n myenv pip python=3.6
    source activate myenv
    pip install requirements.txt
    pip install --upgrade pip
    pip install -r requirements.txt

Then, train the model (need test.csv). Training needs near 3 minutes.

    ./train.sh


## References

* A Simple and Effective Approach to the Story Cloze Test [link](http://aclweb.org/anthology/N18-2015)
* Story Comprehension for Predicting What Happens Next [link](https://pdfs.semanticscholar.org/613b/616c06d77ed1994d788ed1ebc47407b762bc.pdf)
* Two Discourse Driven Language Models for Semantics [link](http://www.aclweb.org/anthology/P16-1028)
* Skip-Thought Vectors [link](https://arxiv.org/pdf/1506.06726.pdf)
