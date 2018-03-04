import pandas as pd

data = pd.read_csv('movie_metadata.csv', low_memory=False)

C = data['imdb_score'].mean()
m = data['num_voted_users'].quantile(0.75)

q_movies = data.copy().loc[data['num_voted_users'] >= m]

def weighted_ratings(x, m=m, C=C):
    v = x['num_voted_users']
    R = x['imdb_score']
    
    return (v/(v+m) * R) + (m/(m+v) * C)
    
q_movies['score'] = q_movies.apply(weighted_ratings, axis=1)

q_movies = q_movies.sort_values('score', ascending=False)

top_m = q_movies[['movie_title', 'num_voted_users', 'imdb_score', 'score']].head(15).sort_values('score',
                                                                                    ascending=False)

