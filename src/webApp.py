import streamlit
import streamlit as st
import numpy as np
import model
from PIL import Image

st.write(
    '''
    # Movies Recommending Website.
    '''
)



st.sidebar.header("User Input Parameters")

def user_input_features():
    new_user_id = 0.0
    new_rating_ave = 0.0
    new_action = st.sidebar.slider('How much do you like actoin in movies ?',0.0,5.0,0.5)
    new_adventure = st.sidebar.slider('How much do you like adventure in movies?',0.0,5.0,0.5)
    new_animation = st.sidebar.slider('How much do you like animation in movies?',0.0,5.0,0.5)
    new_childrens = st.sidebar.slider('How much do you like children movies?',0.0,5.0,0.5)
    new_comedy = st.sidebar.slider('How much do you like comedy in movies?',0.0,5.0,0.5)
    new_crime = st.sidebar.slider('How much do you like crime in movies?',0.0,5.0,0.5)
    new_documentary = st.sidebar.slider('How much do you like documantaries?',0.0,5.0,0.5)
    new_drama = st.sidebar.slider('How much do you like drama in movies?',0.0,5.0,0.5)
    new_fantasy = st.sidebar.slider('How much do you like fantasy in movies?',0.0,5.0,0.5)
    new_horror = st.sidebar.slider('How much do you like horror in movies?',0.0,5.0,0.5)
    new_mystery = st.sidebar.slider('How much do you like mystery in movies?',0.0,5.0,0.5)
    new_romance = st.sidebar.slider('How much do you like romance movies?',0.0,5.0,0.5)
    new_scifi = st.sidebar.slider('How much do you like sci-fi movies?',0.0,5.0,0.5)
    new_thriller = st.sidebar.slider('How much do you like thriller movies?',0.0,5.0,0.5)
    new_rating_count =3
    features=user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave,
                      new_action, new_adventure, new_animation, new_childrens,
                      new_comedy, new_crime, new_documentary,
                      new_drama, new_fantasy, new_horror, new_mystery,
                      new_romance, new_scifi, new_thriller]])
    return features

data=user_input_features()


model=model.MovieRecomModel()
st.write(
    '''
    ### Recommended movies:
    '''
)
if st.button('Recommend'):
    res=model.predict(data)
    st.write(res)
