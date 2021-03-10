import pandas as pd;

def main():
    ratings = pd.read_csv('../../data/20m_raw/ratings.csv')
    movies = pd.read_csv('../../data/20m_raw/movies.csv')

    #Export only the columns used for training
    final_ratings_data = ratings[["userId","movieId","rating"]]
    final_ratings_data.to_csv('../../data/20m_processed/ratings.csv', index=False)

    final_movies_data = movies[["movieId","title"]]
    final_movies_data.to_csv('../../data/20m_processed/movies.csv', index=False)

main()
