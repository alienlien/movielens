#!/usr/bin/env python3
#
# This is used to train a recommendation system based on
# collaborative filtering.
import os.path
import pandas as pd
import numpy as np
from surprise import Reader
from surprise import Dataset
from surprise import SVD
from surprise import accuracy
from surprise import dump

# The dir and file to store data.
DATA_DIR = './ml-latest-small'
MOVIE_CSV_FILE = os.path.join(DATA_DIR, 'movies.csv')
RATINGS_CSV_FILE = os.path.join(DATA_DIR, 'ratings.csv')
TAGS_CSV_FILE = os.path.join(DATA_DIR, 'tags.csv')
LINKS_CSV_FILE = os.path.join(DATA_DIR, 'links.csv')
REC_MODEL_FILE = os.path.join(DATA_DIR, 'rec.model')

# Load movies data. Note that we user empty string for the movies
# without genres.
movies = pd.read_csv(MOVIE_CSV_FILE, sep=',')
movies['genres'] = np.where(movies['genres'] == '(no genres listed)', '',
                            movies['genres'])

# Load rating data.
ratings = pd.read_csv(RATINGS_CSV_FILE, sep=',')

if __name__ == '__main__':

    df = ratings.join(
        movies[['movieId', 'title']].set_index('movieId'),
        on='movieId').reset_index()

    print('Begin to read data')
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    data.split(n_folds=3)

    print('Begin to train data')
    alg = SVD()
    counter = 0
    for train_set, test_set in data.folds():
        print('Training {}th data'.format(counter))
        alg.train(train_set)
        print('Evaluating {}th data'.format(counter))
        accuracy.rmse(alg.test(test_set), verbose=True)
        counter += 1

    dump.dump(REC_MODEL_FILE, algo=alg)
