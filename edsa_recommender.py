"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model
from recommenders.hybrid import hybrid_main
# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview", "Exploratory Data Analysis", "Hybrid Recommender"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('First Option',title_list)
        movie_2 = st.selectbox('Second Option',title_list)
        movie_3 = st.selectbox('Third Option',title_list)
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("How do these models actually worK?")
        model_selection = st.selectbox('Choose a model',['Content-based recommender', 'Collaborative recommender', 'Hybrid recommender'])
        if model_selection == 'Content-based recommender':
            st.image('resources/imgs/content-based-recommender.png',use_column_width=True)
            st.write('''Our content-based recommender generates recommendations based on similarity of properties. The properties of each movie in the dataset (genre, actors, directors, keywords, genome tags etc) are used here and it is assumed that the collection of properties of movies a user likes can be used to predict what other movies a user is likely to enjoy. For example, a user who's favourite movies are Pulp Fiction, Fight Club and Kill Bill vol. 2 is probably likely to enjoy other movies directed by Quintin Tarantino, as well as many movies with actors like Uma Thurman, Brad Pitt and Samuel L. Jackson in them. ''')
            st.write('''One challenge with this approach is dataset size. A dataset with thousands of movies will likely have thousands of different properties that all need to be tracked for each movie. In order to make a model that delivers results quickly and without using excessive amounts of memory, it is important to determine what properties are most useful when we want to distinguish between movies, and which properties are not.''')
            st.image('resources/imgs/Cumulative_explained_variance.png', use_column_width=False)
            st.write('Take for instance the graph above. What\'s important to know about the axes is that the y-axis is a measure of the % of information in the dataset that remains compared to the original. The X-axis is (crudely) a count of how many properties we are keeping in the dataset, ordered from most important to least important. ')
            st.write('Note how the % of information contained in the dataset rises incredibly rapidly at first, this means that the first few hundred properties contain most of the information in the dataset, this allowed us to significantly reduce the size of our dataset because we could retain 1000 properties and discard the other ~5500 while still retaining ~90% of the information in the dataset. This drastically increased the speed of our content-based recommender!')
        if model_selection == 'Collaborative recommender':
            st.image('resources/imgs/Utility-matrix.png',use_column_width=True)
            st.write('For our collaborative-based recommender, we used the similarity measured between users to make recommendations. We use an approach which clusters based on the idea that similar people (based on the data) generally tend to like similar movies. It predicts which movies a user will like based on the movie preferences of other similar users.')
        if model_selection == 'Hybrid recommender':
            st.image('resources/imgs/hybrid.png',use_column_width=False)
            st.write('Our hybrid recommender is actually deceptively simple. It starts by running the content-based recommender, but instead of just returning a list of most similar movies, it passes these movies to our collaborative recommender model, which then constructs a user-movie-rating utility matrix just containing those movies that are most similar. K means clustering is then performed on this subset of rating data to determine the most similar movies based on both movie metadata and user rating data.')
    if page_selection == 'Exploratory data analysis':
        st.title("Exploratory Data analysis")
        # st.image('resources/imgs/Image_header.png',use_column_width=True)
        st.write('To be implemented')
    if page_selection == 'Hybrid Recommender':
        st.title("Hybrid Recommender")
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.

    if page_selection == "Hybrid Recommender":
        # Header contents
        st.write('# Hybrid Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        # sys = st.radio("Select an algorithm",
        #                ('Hybrid Filter!',
        #                 'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('First Option',title_list)
        movie_2 = st.selectbox('Second Option',title_list)
        movie_3 = st.selectbox('Third Option',title_list)
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        # if sys == 'Hybrid Filter!':
        if st.button("Recommend"):
            try:
                with st.spinner('Crunching the numbers...'):
                    top_recommendations = hybrid_main(movie_list=fav_movies)
                st.title("We think you'll like:")
                for i,j in enumerate(top_recommendations):
                    st.subheader(str(i+1)+'. '+j)
            except:
                st.error("Oops! Looks like this algorithm does't work.\
                            We'll need to fix it!")


if __name__ == '__main__':
    main()
