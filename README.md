
# Movie Recommendation System 

### There are basically three types of recommender systems:

### Demographic Filtering :
They offer generalized recommendations to every user, based on movie popularity and/or genre. The System recommends the same movies to users with similar demographic features. Since each user is different , this approach is considered to be too simple. The basic idea behind this system is that movies that are more popular and critically acclaimed will have a higher probability of being liked by the average audience.
### Content Based Filtering : 
They suggest similar items based on a particular item. This system uses item metadata, such as genre, director, description, actors, etc. for movies, to make these recommendations. The general idea behind these recommender systems is that if a person liked a particular item, he or she will also like an item that is similar to it.
linkcode 
### Collaborative Filtering : 
This system matches persons with similar interests and provides recommendations based on this matching. Collaborative filters do not require item metadata like its content-based counterparts.

### Data sets : 
* TMDB 5000 from kagel 
* https://drive.google.com/file/d/150IG-0zbneNTy1KHop9IeY5Heg9yyODC/view?usp=sharing

### Install some required libraries such as : 
* !pip3 install numpy 
* !pip3 install scikit-surprise

### Import required libraries : 

* import pandas as pd
* from surprise import Reader
* from surprise import Dataset
* from surprise.model_selection import cross_validate
* from surprise import NormalPredictor
* from surprise import KNNBasic
* from surprise import KNNWithMeans
* from surprise import KNNWithZScore
* from surprise import KNNBaseline
* from surprise import SVD
* from surprise import BaselineOnly
* from surprise import SVDpp
* from surprise import NMF
* from surprise import SlopeOne
* from surprise import CoClustering
* from surprise.accuracy import rmse
* from surprise import accuracy
* from surprise.model_selection import train_test_split
* from surprise.model_selection import GridSearchCV
* from datetime import datetime
* from surprise import dump
* import os
* from pandas.core.dtypes.missing import isna
* from collections import defaultdict
* import numpy as np
* import matplotlib.pyplot as plt
* from sklearn.feature_extraction.text import TfidfVectorizer
* from sklearn.metrics.pairwise import linear_kernel
* from ast import literal_eval
* from sklearn.feature_extraction.text import CountVectorizer
* from sklearn.metrics.pairwise import cosine_similarity
* from surprise.model_selection.split import LeaveOneOut
* from plotly.offline import init_notebook_mode, plot, iplot
* import plotly.graph_objs as go 

### weighted rating : 
We'll be using IMDB's weighted rating (wr) which is given as :

* v is the number of votes for the movie;
* m is the minimum votes required to be listed in the chart;
* R is the average rating of the movie; And
* C is the mean vote across the whole report
### Feature Engineering  : 

TF-iDF and Count vectorizer used for Content-Based Filtering with cosine similarityty 

### cosine similarityty : 

we applay cosine simi on the content base recommender after applaying TF-iDF on the overview of the data 
and then with our meta data after applaying count vectorizer.

### Cross-Validation for seconed dataset for Collaborative Filtering :

The MovieLens dataset had either pre-computed cross-folds or scripts to carry out this calculation in earlier iterations. Since most contemporary toolkits already include this as a built-in feature, we no longer bundle either of these features with the dataset. Check out [LensKit](http://lenskit.org) for resources like tools, documentation, and open-source code samples if you want to learn about common methods for cross-fold computation in the context of recommender system evaluation.

### load some files "SVDppModel.pickle" and "modelSVDppLoo_1M.pickle"
 
these files are saved models from the training.

"SVDp Model.pickle" is champion Model (SVDpp) that saved after training.
"modelSVDppLoo_1M.pickle" is champion Model (SVDpp) that trained on Leave-One-Out cross-validation and 
it trained on One million (1,000,000) records of 25M (ratings.csv) in Error Analysis.

notes: this code was implemented on google colab.

## To connect our code with dialog flow chatbot we have used NGROK to host our application there.
### Libraries that we have used for this purpose : 
* numpy
* scikit-surprise
* fastapi
* pydentic
* pyngrAok
* unicorn
* fastapi nest-asyncio pyngrok uvicorn
* inspect
* itertools
* typing
* urllib.request
* matplotlib.pyplo
