import pandas as pd;

def main():
    ratings = pd.read_csv('../../data/raw/ratings.dat', sep="::", usecols=[0, 1, 2, 3],
                          names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')
    movies = pd.read_csv('../../data/raw/movies.dat', sep="::", usecols=[0, 1, 2], names=['movieId', 'title', 'genres'],
                         engine='python')

    #Export only the columns used for training
    final_ratings_data = ratings[["userId","movieId","rating"]]
    final_ratings_data.to_csv('../../data/processed/ratings.csv', index=False)

    final_movies_data = movies[["movieId","title"]]
    final_movies_data.to_csv('../../data/processed/movies.csv', index=False)

main()
