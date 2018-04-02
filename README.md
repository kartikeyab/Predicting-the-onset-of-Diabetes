# Predicting the onset of Diabetes
Using a Neural Network to predict diabetes in women  using diagnostic features . I used Keras for the building the Neural Network. Added
Dropout layers to avoid overfitting.
The kaggle dataset that i used is - https://www.kaggle.com/uciml/pima-indians-diabetes-database

The neural network that I have built is 5 layered . The performance on the test set was around 80.1% initially when i did not use dropout layers . After adding the dropout layers, accuracy increased to around 86% . Probably because it was overfitting and not perfoming that well on the test set when dropout was not used. 

Will update this with a Sci-Kit version soon !
