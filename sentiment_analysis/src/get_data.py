from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

def get_data_as_dataframe(filename) :
    soup = BeautifulSoup(open(filename, encoding="utf-8"), "html.parser")
    review = soup.find_all("span", {"jsname": "fbQN7e"})
    rating = soup.find_all("div", {"class": "pf5lIe"})

    feedbacks=[]
    for i in range(0,len(review)):
        stars=len(rating[i].find_all("div", {"class": "vQHuPe bUWb7c"}))
        review_text = review[i].text
        feedbacks.append([stars,review_text])

    raw_df = pd.DataFrame(feedbacks, columns=["star_rating", "review"])
    raw_df["is_bad_review"] = np.where(raw_df['star_rating']>2,0,1)
    return raw_df






