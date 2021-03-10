import pandas as pd;
import numpy as np;
import logging

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from gensim.models import Word2Vec
logging.basicConfig(level=logging.INFO)


class Movies2Vec:
    def __init__(self):
        # self.ratings = pd.read_csv('data/ratings.csv', header=0)
        self.ratings = pd.read_csv('../../data/20m_processed/ratings.csv')
        self.ratings['movieId'] = self.ratings['movieId'].astype(str)
        self.ratings = self.ratings
        print("Data reading finished")

    def create_and_evaluate(self):
        self.only_pos_scores = []
        self.pos_neg_scores = []
        self.k_values = [1, 2, 3]
        for i in range(len(self.k_values)):
            self.only_pos_scores.append({'precision': [], 'recall': [], 'f1': []})
            self.pos_neg_scores.append({'precision': [], 'recall': [], 'f1': []})

        # Use 5-Fold Cross-validation for model evaluation
        # kf = KFold(n_splits=5, shuffle=True)
        iter = 1

        self.train_data, self.test_data = train_test_split(self.ratings,
                                                         test_size=0.10)
        print("Train test split finished")

        self.train_sentences = self.get_sentences(self.train_data)
        print("Got sentences")
        print(len(self.train_sentences))

        self.create_model(iter)
        self.evaluate_model(iter)

        # for train, test in kf.split(self.ratings):
        #     # self.train_data = pd.DataFrame([self.ratings.iloc[x] for x in train],
        #     #                                columns=["userId", "movieId", "rating"])
        #     self.train_data = self.ratings.iloc[train]
        #     print("Got train data")
        #     self.test_data = self.ratings.iloc[test]
        #     print("Got test data")
        #     self.train_sentences = self.get_sentences(self.train_data)
        #     print("Got sentences")
        #     print(len(self.train_sentences))
        #
        #     self.create_model(iter)
        #     self.evaluate_model(iter)
        #
        #     iter += 1

        for i in range(len(self.k_values)):
            print("Mean scores (using only positive examples at testing) at K@{}".format(self.k_values[i]))
            print("Precision: {:.2%}".format(np.mean(self.only_pos_scores[i]['precision'])))
            print("Recall: {:.2%}".format(np.mean(self.only_pos_scores[i]['recall'])))
            print("F1: {:.2%}\n".format(np.mean(self.only_pos_scores[i]['f1'])))

            print("Mean scores (using both positive and negative examples at testing) at K@{}".format(self.k_values[i]))
            print("Precision: {:.2%}".format(np.mean(self.pos_neg_scores[i]['precision'])))
            print("Recall: {:.2%}".format(np.mean(self.pos_neg_scores[i]['recall'])))
            print("F1: {:.2%}\n".format(np.mean(self.pos_neg_scores[i]['f1'])))

    def get_sentences(self, ratings):
        positive_ratings = ratings[ratings['rating'] >= 3.5]
        negative_ratings = ratings[ratings['rating'] < 3.5]
        positive_ratings_by_user = positive_ratings.groupby(['userId'])
        negative_ratings_by_user = negative_ratings.groupby(['userId'])

        positive_sentences = [positive_ratings_by_user.get_group(positive_group)['movieId'].tolist()
                              for positive_group in positive_ratings_by_user.groups]
        negative_sentences = [negative_ratings_by_user.get_group(negative_group)['movieId'].tolist()
                              for negative_group in negative_ratings_by_user.groups]

        return (positive_sentences + negative_sentences)

    def create_model(self, iter):
        # self.model = Word2Vec(self.train_sentences,
        #                       size=128,
        #                       window=999,
        #                       min_count=3,
        #                       workers=8,
        #                       iter=30,
        #                       sg=1,
        #                       hs=0,
        #                       sample=5e-4,
        #                       negative=30)
        #
        # # Save the model for each try
        # model_name = "modelusq_" + str(iter)
        # self.model.save(model_name)
        self.model = Word2Vec.load('item2vec_word2vecSg_20180328')


    def evaluate_model(self, iter):
        liked_train = self.get_movies_list_by_user(self.train_data, is_liked=True)
        disliked_train = self.get_movies_list_by_user(self.train_data, is_liked=False)

        liked_test = self.get_movies_list_by_user(self.test_data, is_liked=True)

        self.get_scores(liked_train, disliked_train, liked_test, iter)

    def get_movies_list_by_user(self, data, is_liked):
        positive_ratings = data[data['rating'] >= 3]
        negative_ratings = data[data['rating'] < 3]
        ratings = positive_ratings if is_liked else negative_ratings

        ratings = ratings[ratings['movieId'].isin(self.model.wv.vocab.keys())]

        movies_by_user = {k: list(v) for k, v in ratings.groupby('userId')['movieId']}

        return movies_by_user

    def get_scores(self, liked_train, disliked_train, liked_test, iter):
        total_liked = np.zeros(len(self.k_values))
        total_correct = np.zeros(len(self.k_values))
        total_correct_with_negative = np.zeros(len(self.k_values))
        total_no_of_predictions = np.zeros(len(self.k_values));
        common_users = set(liked_test.keys()).intersection(set(liked_train.keys()))

        x = 0;
        for user_id in common_users:
            x += 1
            test_movies = liked_test[user_id]
            #             common_test_movies = [movie for movie in test_movies if movie in self.model.wv.vocab]

            disliked_movies = []
            if user_id in disliked_train:
                disliked_movies = disliked_train[user_id]
            #             common_disliked_movies = [movie for movie in disliked_movies if movie in self.model.wv.vocab]

            for i in range(len(self.k_values)):

                topn = self.k_values[i] * len(test_movies)
                predictions_with_pos_neg = self.model.wv.most_similar_cosmul(positive=liked_train[user_id],
                                                                             negative=disliked_movies,
                                                                             topn=topn)

                predictions_with_pos = self.model.wv.most_similar_cosmul(positive=liked_train[user_id],
                                                                         topn=topn)

                for predicted_movie, score in predictions_with_pos_neg:
                    if predicted_movie in test_movies:
                        total_correct_with_negative[i] += 1.0

                for predicted_movie, score in predictions_with_pos:
                    if predicted_movie in test_movies:
                        total_correct[i] += 1.0
                total_liked[i] += len(test_movies)
                total_no_of_predictions[i] += topn

        #         print("total_correct", total_correct)
        #         print("total_no_of_predictions", total_no_of_predictions)
        #         print("total_correct", total_liked)

        for i in range(len(self.k_values)):
            self.only_pos_scores[i]['precision'].append(total_correct[i] / total_no_of_predictions[i])
            self.only_pos_scores[i]['recall'].append(total_correct[i] / total_liked[i])
            self.only_pos_scores[i]['f1'].append(
                2 / ((total_no_of_predictions[i] / total_correct[i]) + (total_liked[i] / total_correct[i])))

            self.pos_neg_scores[i]['precision'].append(total_correct_with_negative[i] / total_no_of_predictions[i])
            self.pos_neg_scores[i]['recall'].append(total_correct_with_negative[i] / total_liked[i])
            self.pos_neg_scores[i]['f1'].append(2 / ((total_no_of_predictions[i] / total_correct_with_negative[i]) + (
                        total_liked[i] / total_correct_with_negative[i])))

            print("Scores (using only positive examples at prediction) at Fold {}, K @{}"
                  .format(iter, self.k_values[i]))
            print("Precision: {:.2%}; Recall: {:.2%}, F1: {:.2%}\n"
                  .format(self.only_pos_scores[i]['precision'][-1], self.only_pos_scores[i]['recall'][-1],
                          self.only_pos_scores[i]['f1'][-1]))

            print("Scores (using both positive and negative examples at prediction) at Fold {}, K @{}"
                  .format(iter, self.k_values[i]))
            print("Precision: {:.2%}; Recall: {:.2%}, F1: {:.2%}\n"
                  .format(self.pos_neg_scores[i]['precision'][-1], self.pos_neg_scores[i]['recall'][-1],
                          self.pos_neg_scores[i]['f1'][-1]))

        # score['precision_with_negative'] = total_correct_with_negative / total_no_of_predictions
        # score['recall_with_negative'] = total_correct_with_negative / total_liked
        # score['f1_with_negative'] = 2 / ((1 / score['precision_with_negative']) + (1 / score['recall_with_negative']))


model = Movies2Vec()
model.create_and_evaluate()