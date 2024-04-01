from flask import Flask, request
from flask_cors import CORS
import pandas as pd
from surprise import Reader, Dataset
from surprise import SVDpp
from collections import defaultdict

app = Flask(__name__)
CORS(app)

books = pd.read_csv('static/books10k.csv')
ratings = pd.read_csv('static/ratings10k.csv')
df = pd.merge(books, ratings, on='book_id', how='inner')
min_book_ratings = 100
min_user_ratings = 150
filter_books = df['book_id'].value_counts() > min_book_ratings
filter_books = filter_books[filter_books].index.tolist()
filter_users = df['user_id'].value_counts() > min_user_ratings
filter_users = filter_users[filter_users].index.tolist()
df = df[(df['book_id'].isin(filter_books)) & (df['user_id'].isin(filter_users))]

cold_start_ids = []


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/webhook', methods=['POST'])
def web_hook():
    req = request.get_json()
    tag = req["fulfillmentInfo"]["tag"]
    res = {
        "fulfillment_response": {
            "messages": [{"text": {"text": [tag]}}]
        }
    }
    return res


@app.route('/top5', methods=['GET', 'POST'])
def top5():
    top5 = books.nlargest(5, 'work_ratings_count')
    recommendation_str = "Please provide your ratings (1-5) for the following books to avoid cold start:\n"
    cold_start_ids.clear()
    for _, row in top5.iterrows():
        recommendation_str += f"{row['title']} by {row['authors']}; \n"
        cold_start_ids.append(row['book_id'])
    res = {
        "fulfillment_response": {
            "messages": [{"text": {"text": [recommendation_str]}}]
        }
    }
    return res


@app.route('/cf', methods=['POST'])
def collaborative_filtering():
    if len(cold_start_ids) == 0:
        top5()

    req = request.get_json()
    ratings_str = req["intentInfo"]["parameters"]["user_ratings"]['originalValue']

    ratings_list_str = ratings_str.split(',')
    cold_start_ratings = []

    for i in range(len(cold_start_ids)):
        cold_start_ratings.append((cold_start_ids[i], int(ratings_list_str[i])))
        i += 1

    new_user_id = max(df['user_id']) + 1  # create a new user id
    new_user_ratings = [
        {'user_id': new_user_id, 'book_id': book_id, 'rating': rating}
        for book_id, rating in cold_start_ratings
    ]

    new_ratings_df = pd.DataFrame(new_user_ratings)
    df_with_new = pd.concat([df, new_ratings_df], ignore_index=True)

    reader = Reader(rating_scale=(1, 5))
    data_with_new = Dataset.load_from_df(df_with_new[['user_id', 'book_id', 'rating']], reader)

    trainset_with_new = data_with_new.build_full_trainset()
    algo = SVDpp()
    algo.fit(trainset_with_new)

    testset_new_user = trainset_with_new.build_anti_testset(fill=0)
    testset_new_user = [x for x in testset_new_user if x[0] == new_user_id]
    predictions_new_user = algo.test(testset_new_user)

    top_n_new_user = get_top_n(predictions_new_user, n=5)
    top_n_result = top_n_new_user[new_user_id]

    answer = "Based on the questionnaire, the books recommended to you are: \n"
    for book in top_n_result:
        answer += f"{books.iloc[book[0]]['title']} by {books.iloc[book[0]]['authors']}; \n"
    res = {
        "fulfillment_response": {
            "messages": [{"text": {"text": [answer]}}]
        }
    }
    return res


def get_top_n(predictions, n=10):
    # Map the predictions to each user
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Sort the predictions for each user and retrieve the k highest ones
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


if __name__ == '__main__':
    app.run()
