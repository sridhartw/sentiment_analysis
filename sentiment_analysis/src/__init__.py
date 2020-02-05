from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from add_features import add_doc2vec, add_tf_idf
from get_data import get_data_as_dataframe
from cleanse_data import clean_text
from get_sentiments import get_sentiments
from sklearn.model_selection import train_test_split
import pandas as pd

from plot_roc_curve import show_roc_curve

raw_df = get_data_as_dataframe('../resource/sample_g_play_data.html')
raw_df["review_clean"] = raw_df["review"].apply(lambda x: clean_text(x))
# pd.set_option('display.max_colwidth', 0)
transformed_df= get_sentiments(raw_df)
transformed_df= pd.concat([transformed_df, add_doc2vec(transformed_df)], axis=1)
transformed_df= pd.concat([transformed_df, add_tf_idf(transformed_df)], axis=1)

label = "is_bad_review"
ignore_cols = [label, "review", "review_clean","star_rating"]
features = [c for c in transformed_df.columns if c not in ignore_cols]

# split the data into train and test

X_train, X_test, y_train, y_test = train_test_split(transformed_df[features], transformed_df[label], test_size = 0.20, random_state = 42)


# train a random forest classifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)


# # show feature importance
feature_importances_df = pd.DataFrame({"feature": features, "importance": classifier.feature_importances_}).sort_values("importance", ascending = False)
print(feature_importances_df.head(20))


show_roc_curve(classifier, X_test, y_test)






# with pd.option_context('display.max_rows', None, 'display.max_columns', None) :
#     print(transformed_df)

