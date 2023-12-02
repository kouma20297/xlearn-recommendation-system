import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import xlearn as xl
from collections import defaultdict
from sklearn.model_selection import train_test_split

# データの読み込み前処理
m_cols = ["movie_id", "title", "genre"]
movies = pd.read_csv(
    "/app/学習用データ/ml-10M100K/movies.dat",
    names=m_cols,
    sep="::",
    encoding="latin-1",
    engine="python",
)

# genreをlist形式で保持する
movies["genre"] = movies.genre.apply(lambda x: x.split("|"))

# 最初の数行を表示
print(movies.head())

# 列の情報を表示
print(movies.info())


# ユーザが付与した映画のタグ情報の読み込み
t_cols = ["user_id", "movie_id", "tag", "timestamp"]
user_tagged_movies = pd.read_csv(
    "/app/学習用データ/ml-10M100K/tags.dat",
    names=t_cols,
    sep="::",
    engine="python",
)

# データファイルのtag列を小文字にする
user_tagged_movies["tag"] = user_tagged_movies["tag"].str.lower()


# tagを映画ごとにlist形式で保持する
movie_tags = user_tagged_movies.groupby("movie_id").agg({"tag": list})

# タグ情報を結合する
movies = movies.merge(movie_tags, on="movie_id", how="left")


# 評価値データの読み込み
r_cols = ["user_id", "movie_id", "rating", "timestamp"]
ratings = pd.read_csv(
    "/app/学習用データ/ml-10M100K/ratings.dat",
    names=r_cols,
    sep="::",
    engine="python",
)


# データ量が多いため、ユーザー数を1000に絞って、試していく
valid_user_ids = sorted(ratings.user_id.unique())[:1000]
ratings = ratings[ratings["user_id"].isin(valid_user_ids)]

# 映画のデータと評価のデータを結合する
movielens = ratings.merge(movies, on="movie_id")

print(f"unique_users={len(movielens.user_id.unique())}, unique_movies={len(movielens.movie_id.unique())}")


# データの分割（トレーニングデータ、バリデーションデータ、テストデータ）
movies_train, movies_temp = train_test_split(movielens, test_size=0.2, random_state=42)
movies_val, movies_test = train_test_split(movies_temp, test_size=0.5, random_state=42)

user_tagged_movies_train, user_tagged_movies_temp = train_test_split(
    user_tagged_movies, test_size=0.2, random_state=42
)
user_tagged_movies_val, user_tagged_movies_test = train_test_split(
    user_tagged_movies_temp, test_size=0.5, random_state=42
)

ratings_train, ratings_temp = train_test_split(ratings, test_size=0.2, random_state=42)
ratings_val, ratings_test = train_test_split(ratings_temp, test_size=0.5, random_state=42)

# 各データセットのサイズを確認
print("Training Data:")
print("Movies:", movies_train.shape)
print("Tags:", user_tagged_movies_train.shape)
print("Ratings:", ratings_train.shape)

print("\nValidation Data:")
print("Movies:", movies_val.shape)
print("Tags:", user_tagged_movies_val.shape)
print("Ratings:", ratings_val.shape)

print("\nTesting Data:")
print("Movies:", movies_test.shape)
print("Tags:", user_tagged_movies_test.shape)
print("Ratings:", ratings_test.shape)


# 因子数
factors = 10
# 評価数の閾値
minimum_num_rating = 200
# エポック数
n_epochs = 50
# 学習率
lr = 0.01
# 補助情報の利用
use_side_information = False

# 評価値がminimum_num_rating件以上ある映画に絞る
filtered_movies_train = movielens.groupby("movie_id").filter(lambda x: len(x) >= minimum_num_rating)

# ユーザーが評価した映画
user_evaluated_movies = filtered_movies_train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()


print(movies_train.head())


# 例: 欠損値をゼロで埋める
movies = movies.fillna(0)


# 行列のインデックスと映画/ユーザーを対応させる辞書を作成
unique_user_ids = sorted(filtered_movies_train.user_id.unique())
unique_movie_ids = sorted(filtered_movies_train.movie_id.unique())
user_id2index = dict(zip(unique_user_ids, range(len(unique_user_ids))))
movie_id2index = dict(zip(unique_movie_ids, range(len(unique_movie_ids))))

# FM用にデータの整形
# filtered_movies_trainをx.yに変換している xがユーザーと映画に関する特徴量で、yが対応する評価値が格納されている
train_data_for_fm = []
y = []
for i, row in filtered_movies_train.iterrows():
    x = {"user_id": str(row["user_id"]), "movie_id": str(row["movie_id"])}
    if use_side_information:
        # ここに必要なサイド情報を追加
        pass
    train_data_for_fm.append(x)
    y.append(row["rating"])

y = np.array(y)

vectorizer = DictVectorizer()
X = vectorizer.fit_transform(train_data_for_fm)  # fit_transformを追加


# データをファイルに保存
with open(
    r"C:\Users\nakas\OneDrive\デスクトップ\開発\startupboost\レコメンドシステム\xlearn\学習用データ\ml-10M100K\train_data_for_fm.txt", "w"
) as f:
    for i in range(X.shape[0]):
        line = " ".join([f"{j}:{X[i, j]}" for j in range(X.shape[1])])
        f.write(f"{y[i]} {line}\n")


# FMモデルの初期化
fm_model = xl.FMModel(task="reg", metric="rmse", lr=lr, opt="sgd", k=factors, epoch=n_epochs)


# モデルの学習
train_file = r"C:\Users\nakas\OneDrive\デスクトップ\開発\startupboost\レコメンドシステム\xlearn\学習用データ\ml-10M100K\train_data_for_fm.txt"
model_file = "/app/学習用データ/ml-10M100K/model.out"
fm_model.fit(
    train_file,
    model_file,
)

# テストデータを使ってモデルの性能評価
y_pred = fm_model.predict(X)
pred_matrix = y_pred.reshape(len(unique_user_ids), len(unique_movie_ids))


# 学習用に出てこないユーザーや映画の予測評価値は、平均評価値とする
average_score = movielens_train.rating.mean()
movie_rating_predict = movielens_test.copy()
pred_results = []
for i, row in movielens_test.iterrows():
    user_id = row["user_id"]
    if user_id not in user_id2index or row["movie_id"] not in movie_id2index:
        pred_results.append(average_score)
        continue
    user_index = user_id2index[row["user_id"]]
    movie_index = movie_id2index[row["movie_id"]]
    pred_score = pred_matrix[user_index, movie_index]
    pred_results.append(pred_score)
movie_rating_predict["rating_pred"] = pred_results


# 各ユーザーに対するレコメンドリストの作成

pred_user2items = defaultdict(list)

for user_id in unique_user_ids:
    user_index = user_id2index[user_id]
    movie_indexes = np.argsort(-pred_matrix[user_index, :])
    for movie_index in movie_indexes:
        movie_id = unique_movie_ids[movie_index]
        if movie_id not in user_evaluated_movies[user_id]:
            pred_user2items[user_id].append(movie_id)
        if len(pred_user2items[user_id]) == 10:
            break

pred_user2items

# user_id=2のユーザーが学習データで評価を付けた映画一覧
movielens_train[movielens_train.user_id == 2]


# user_id=2に対するおすすめ(318, 50, 527)
movies[movies.movie_id.isin([318, 50, 527])]
