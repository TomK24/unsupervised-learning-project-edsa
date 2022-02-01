"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
# import pickle
import pickle5 as pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)

with open('resources/data/X.pickle', 'rb') as handle:
    X = pickle.load(handle)

with open('resources/data/user_mapper.pickle', 'rb') as handle:
    user_mapper = pickle.load(handle)

with open('resources/data/movie_mapper.pickle', 'rb') as handle:
    movie_mapper = pickle.load(handle)

with open('resources/data/user_inv_mapper.pickle', 'rb') as handle:
    user_inv_mapper = pickle.load(handle)

with open('resources/data/movie_inv_mapper.pickle', 'rb') as handle:
    movie_inv_mapper = pickle.load(handle)    

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    print('collab_model function is being called')
    metric='cosine'
    show_distance=False
    movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))
    neighbour_ids = []
    movie_id = []
    recommended_movies = []
    top_n+=1
    print('reached line 88')
    for i in movie_list:
        for key, value in movie_titles.items():
            if value == i:
                movie_id.append(key)
    print('reached line 93')
    for i in movie_id:
        movie_ind = movie_mapper[i]
        movie_vec = X[movie_ind]
        
        kNN = NearestNeighbors(n_neighbors=top_n, algorithm="brute", metric=metric)
        kNN.fit(X)
        movie_vec = movie_vec.reshape(1,-1)
        neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)

        for i in range(0,top_n):
            n = neighbour.item(i)
            neighbour_ids.append(movie_inv_mapper[n])
        neighbour_ids.pop(0)
    print('reached line 107')
    
    
    final = []
    for i in neighbour_ids:
        if i not in final:
            final.append(i)
    
    similar_ids = final[:10]
    print('reached line 116')
   
    
    for i in similar_ids:
        recommended_movies.append(movie_titles[i])
    print('Reached final return statement')
    print(recommended_movies)
    return recommended_movies
