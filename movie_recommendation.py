from flask import Flask, render_template,request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df=pd.read_csv('movie_metadata.csv')
df.drop_duplicates(subset="movie_title",keep=False,inplace=True)
df['ID']=np.arange(len(df))
important_features=['ID','movie_title','director_name','genres','actor_1_name','actor_2_name','actor_3_name']
df=df[important_features]
for x in important_features:
    df[x]=df[x].fillna(' ')
df['movie_title']=df['movie_title'].apply(lambda x:x.replace(u'\xa0',u''))
df['movie_title']=df['movie_title'].apply(lambda x:x.strip())
def combine(row):
        return row['director_name']+" "+row["genres"]+" "+row['actor_1_name']+" "+row['actor_2_name']+" "+row['actor_3_name']
df["combined"]=df.apply(combine,axis=1)
cv=CountVectorizer()
count=cv.fit_transform(df["combined"])
cosine_simi=cosine_similarity(count)
def get_tittle(Id):
    return df[df.ID==Id]["movie_title"].values[0]
def get_id(tittle):
    return df[df.movie_title==tittle]['ID'].values[0]
def recommend(user_liking):
    movie_index=get_id(user_liking)
    similar_movies=list(enumerate(cosine_simi[movie_index]))
    sorted_similar_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)
    sorted_similar_movies
    i=0
    j=0
    l=[1]*10
    for movie in sorted_similar_movies:    
        x=get_tittle(movie[0])
        if i==0:
            i=i+1
        else:    
            l[j]=x
            j=j+1
            i=i+1
        if i>10:
            break
    return l              

app = Flask(__name__)

@app.route('/')
def home():
   return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='GET':
        user_liking=request.args.get('movie')
    else:
        user_liking=request.form['movie']
    output=recommend(user_liking)  
    return render_template('after.html',output=output)
if __name__ == '__main__':
   app.run(debug=True)