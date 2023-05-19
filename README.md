# MovieRecommender
This project is a movie recommender system that utilizes a content-based filtering algorithm with a neural network. The system recommends movies to users based on their preferences and similarities between movies. It uses the MovieLens dataset for training and evaluation.

## Data Processing
The first step in the project was importing and processing the MovieLens dataset. The dataset contains movie ratings provided by users, along with movie metadata such as genre, release year, and other information. The data was preprocessed and transformed to make it suitable for training the neural network model.

# Neural Network Model
A suitable neural network model was built for the movie recommender system. The model takes into account various features of movies, such as genre, release year, and user ratings. It uses these features to learn patterns and make predictions about which movies are likely to be preferred by users. The model was trained using the movies-users dataset, with appropriate loss and optimization functions.

# Deployment
For the deployment of the movie recommender system, the Streamlit framework was used to build a web application. Streamlit provides an easy-to-use interface for creating interactive web applications with Python. The application allows users to input their preferences and receive personalized movie recommendations based on the trained neural network model.

The web application was deployed on the Streamlit Sharing web platform, which allows for easy sharing and hosting of Streamlit applications. Users can access the movie recommender system by visiting the deployed web application on Streamlit Sharing.

# Usage
To use the movie recommender system, follow these steps:

Visit the [deployed web application URL](https://abdelhakiem-movierecommender-srcwebapp-ec7krz.streamlit.app/).
Input your preferences for movie genres, and any other relevant criteria.
Click the "Recommend" button to receive personalized movie recommendations.
The system will display a list of recommended movies based on your preferences and the trained neural network model.
