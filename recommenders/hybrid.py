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
from scipy.sparse import csr_matrix

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',')
# ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df = pd.read_csv('resources/data/train.csv')
# ratings_df.drop(['timestamp'], axis=1,inplace=True)

PCA_array = np.load('resources/data/PCA1000features.npy')
titles = pd.read_csv('resources/data/titles_df.csv')
PCA_df = pd.DataFrame(PCA_array)
PCA_df.index = titles['title']
del PCA_array, titles

def hybrid_part1(movie_list,top_n=20):
    recommended_movies = []
    sims1 = cosine_similarity(PCA_df.loc[movie_list[0]].values.reshape(1, -1), PCA_df)
    sims2 = cosine_similarity(PCA_df.loc[movie_list[1]].values.reshape(1, -1), PCA_df)
    sims3 = cosine_similarity(PCA_df.loc[movie_list[2]].values.reshape(1, -1), PCA_df)
    avg_sims = (sims1 + sims2 +sims3) / 3 
    sims_df = pd.DataFrame(avg_sims.T, index=PCA_df.index,columns=['similarity_score'])
    del sims1,sims2,sims3, avg_sims
    sims_df_sorted = sims_df.sort_values(by='similarity_score', ascending=False)
    del sims_df
    sims_df_sorted = sims_df_sorted.head(top_n*10)
    final = sims_df_sorted[~sims_df_sorted.index.duplicated(keep='first')]
    recommended_movies = final.head(top_n)
    ids = movies_df[['movieId', 'title']]
    recommended_movies = ids.merge(recommended_movies,  how='right', on = 'title')
    return recommended_movies

def hybrid_part2(df, similar):
      
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())
      
    # Map Ids to indices
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))
      
    # Map indices to IDs
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))
      
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]
    # print(df.shape)
    # print(len(movie_index))
    # print(len(user_index))
    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
       
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

def hybrid_part3(movie_list, top_n, X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper):
    metric='cosine'
    show_distance=False
    movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))
    neighbour_ids = []
    movie_id = []
    recommended_movies = []
    top_n+=1
    
    for i in movie_list:
        for key, value in movie_titles.items():
            if value == i:
                movie_id.append(key)
    
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
    final = []
    for i in neighbour_ids:
        if i not in final:
            final.append(i)
    
    similar_ids = final[:100]
    
    for i in similar_ids:
        if not movie_titles[i] in movie_list:
            recommended_movies.append(movie_titles[i])
    
    return recommended_movies[:10]

def hybrid_main(movie_list):
    # print('hybrid main called')
    content_based = hybrid_part1(movie_list)
    # print('part 1 worked')
    ratings_subset = ratings_df[ratings_df.movieId.isin(content_based.movieId)]
    X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = hybrid_part2(ratings_subset, content_based)
    # print('part 2 worked')
    final = hybrid_part3(movie_list, 10, X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper)
    # print('part 3 worked')
    return final
