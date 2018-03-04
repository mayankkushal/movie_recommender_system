import pandas as pd

data = pd.read_csv('movie_metadata.csv', low_memory=False)

def get_recommendations(title, cosine_sim=cosine_sim2):
    idx = indices[title]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:10]
    
    movie_indices = [i[0] for i in sim_scores]
    
    return data['movie_title'].iloc[movie_indices]
    

# split genres and plot_keywords
features = ['genres', 'plot_keywords']

for feature in features:
    data[feature] = data[feature].str.split('|')

# remove all the spaces and convert ot lower case
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ""

features = ['genres', 'plot_keywords', 'actor_1_name', 'actor_2_name', 'actor_3_name',
            'director_name']
for feature in features:
    data[feature] = data[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['plot_keywords']) + ' ' \
    + ' '.join(x['actor_1_name'])+ ' ' + ' '.join(x['actor_2_name'])+ ' ' \
    + ' '.join(x['actor_3_name'])+ ' ' + ' '.join(x['director_name'])+ ' ' \
    + ' '.join(x['genres'])
    
data['soup'] = data.apply(create_soup, axis=1)

from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(data['soup'])

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

data = data.reset_index()

indices = pd.Series(data.index, index=data['movie_title'])

get_recommendations('Harry Potter and the Half-Blood Prince ', cosine_sim2)