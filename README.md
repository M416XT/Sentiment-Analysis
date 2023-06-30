# Sentiment Analysis
Link to the Data Set: https://www.kaggle.com/datasets/kazanova/sentiment140

The code demonstrates a sentiment analysis project using Twitter data. The project involves training a machine learning model to predict the sentiment (positive, negative, or neutral) of tweets. The code utilizes various libraries and frameworks such as pandas, scikit-learn, NLTK, Gensim, and Keras.

The dataset used in this project contains tweets along with their corresponding target labels. The target labels represent the polarity of the tweet, with 0 indicating negative sentiment, 2 indicating neutral sentiment, and 4 indicating positive sentiment. The dataset is preprocessed by cleaning the text, removing special characters, links, and user mentions. Stop words are also removed, and stemming is performed to reduce words to their base form.

The dataset is then split into training and test sets. Word2Vec, a popular word embedding technique, is applied to generate word vectors representing the semantic meaning of words. The Word2Vec model is trained on the training data, and the resulting word vectors are used to create an embedding layer for the neural network model.

The text is tokenized using the Tokenizer class from Keras, and the sequences are padded to ensure uniform length. The target labels are encoded using a label encoder. An embedding layer is added to the sequential model, followed by a dropout layer to prevent overfitting. The LSTM (Long Short-Term Memory) layer is employed to capture sequential dependencies in the text data. Finally, a dense layer with a sigmoid activation function is added to output the predicted sentiment.

The model is compiled with binary cross-entropy loss and the Adam optimizer. Callbacks are implemented to reduce the learning rate on plateaus and stop training early if the validation accuracy does not improve. The model is then trained on the training data and evaluated on the test data.

The accuracy and loss metrics are plotted using Matplotlib. Predictions can be made on new text inputs using the trained model, and a confusion matrix is generated to visualize the performance of the model. The classification report provides detailed precision, recall, and F1-score for each class, and the overall accuracy score is calculated.

Finally, the trained model, Word2Vec model, tokenizer, and label encoder are saved for future use. The above code provides a comprehensive pipeline for sentiment analysis on Twitter data, which can be extended and modified for different text classification tasks.
