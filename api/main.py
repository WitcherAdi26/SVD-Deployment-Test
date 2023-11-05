from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Working 123"

import os
import pandas as pd
import numpy as np
from surprise.model_selection import cross_validate, train_test_split, KFold
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD, accuracy
from collections import defaultdict
from heapq import nlargest
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import pearsonr

dataset=pd.read_csv("/api/transaction_dataset.csv")
print(dataset.head())
reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(dataset[[ 'user_id','pid', 'rating']], reader)

kf = KFold(n_splits=10)
kf.split(data)
svd = SVD(n_factors=500,n_epochs=20, lr_all=0.005, reg_all=0.02)
result=cross_validate(svd, data, measures=['RMSE', 'MAE'])

trainset = data.build_full_trainset()
svd.fit(trainset)

user_items = defaultdict(list)
i=1
for user_id, pid, rating in dataset[['user_id', 'pid', 'rating']].values:
    #if(i%5000==0): print(i)
    i+=1
    user_items[user_id].append((pid, rating))


@app.route('/', methods=['GET'])
def returnchar():
    def get_top_n_recommendations(model, user_id, n=5):
        unrated_items = []
        for item in trainset.all_items():
            if item not in user_items[user_id]:
                unrated_items.append(item)
    
        predictions = []
        for item_id in unrated_items:
            predicted_rating = model.predict(user_id, item_id).est
            predictions.append((item_id, predicted_rating))
    
        top_n_recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
        top_n_recommendations= [str(product) for product,rating in top_n_recommendations]
        return ",".join(top_n_recommendations)
        #id1,id2,id3

    user_id = request.args.get('query')
    recommendations=get_top_n_recommendations(svd,user_id,n=5)
    print(type(recommendations))
    return jsonify(recommendations)




    #output eg "prod1,prod2,prod3"

    # d = {}
    # ch = request.args.get('query')
    # if ch and len(ch) == 1:  # Check if ch is a single character
    #     ans = str(ord(ch))
    #     d['output'] = ans
    # else:
    #     d['error'] = "Invalid input: Please provide a single character"
    # return jsonify(d)


if __name__ == "__main__":
    app.run()
