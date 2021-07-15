# Data Science Portfolio

# [Project 1: Case study of the survival rate in Titanic historical accident](https://github.com/lilosa88/Titanic)

- The sinking of the Titanic is one of the most infamous shipwrecks in history. While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others. This project is part of [kaggle's competitions](https://www.kaggle.com/c/titanic/overview) its main goal involves building a predictive model that answers the following question: “what sorts of people were more likely to survive?”. The work was carried out using passenger data (i.e. name, age, gender, socio-economic class, etc).

- From mere data exploration we found that women and kids had more chances to survive. Furthermore, the class ticket played an important role: 1st class passengers had more chances to survive. The age and the place of embarking was decisive with respect to mens' suvival. People travelling with 1 or 3 people were more likely to survive than if they had either zero or more than three family members.

- Different Classification Machine Learning models and Deep Learning models were used in oder to predict the survival of passengers aboard of the Titanic. Among the ML classification we made use of Random Forest, Logistic Regression and KNN.

- The best accuracy (0.86) was obtained from a Deep Learning Model. The use of Dropout layers was used in order to avoid overfitting.  


# [Project 2: Case study of the Diabetes rate](https://github.com/lilosa88/Diabetes)

- The dataset was obtained from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.). Data was collected using direct questionnaires from the patients of Sylhet Diabetes Hospital in Sylhet, Bangladesh and approved by a doctor. This dataset contains the signs and symptpoms data of newly diabetic or soon to be diabetic patients. The objective of this project was to build a predictive model that can  predict whether a patient is prone to be diabetic or not. 

- From mere data exploration we found that all features but "itching" and "delayed healing" have an influence on the diabetes rate. More specifically, the ones with major impact are:
-Gender: Women are more prone to get diabetes.
-Polyuria: If the patient presents an excessive or an abnormally large production or passage of urine then he/she is more prone to get diabetes.
-Polydipsia: If the patient presents an excessive thirst or excess drinking, then he/she is more prone to get diabetes.
-Polyphagia:  If the patient presents an abnormally strong sensation of hunger or desire to eat often leading to or accompanied by over-eating, then she/he is more prone to get diabetes.

- The best accuracy (0.9711) was obtained with KNN with k=3.

# [Project 3: Case study to predict final prices of houses in Iowa](https://github.com/lilosa88/PricingHouse)

- This project was carried out in the context of a [kaggle's competitions](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). In order to buy a house there are many different parameters that influence price negotiations. Therefore, the idea was to create a model that predicts the sales prices given a dataset with 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa. 

- The features that had a huge amount of missing values (approx. more than 2000 missing values), were treated accordingly. We found that indeed such values were not missing as they could be related to another particular feature (i.e. For PoolQuality the missing values means that house did not have any pool). The real missing values were not so numerous. Thus, an imputation technique (i.e. mean value) was used in order to retrieve sensible data. Similarly, categorical variables were filled using the mode value. 

- Concerning the feature engineering, new columns were created in order to better understand the whole dataset. The redundant or useless information was dropped. Moreover, categorical variables such as strings were transformed into continuous variables, making use of dummy variables; the corresponding normalization of the whole dataset was carried out using MinMaxScaler. 

- Two machine Learning model were tested: Ridge Regression and Lasso. The hyperparameter 'alpha' has the following values: 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000. The best alpha value for the Ridge Regression was found to be equal to 10, whilst we used 100 for Lasso. Finally the best accuaracy obtained was given by Lasso: The train Lasso Accuracy was 0.9229 and the test Lasso Accuracy was 0.8902.


# [Project 4: Case study in order to predict the total sunspots number](https://github.com/lilosa88/Sunspots)
- This project was carried out in the context of one of [kaggle's competitions](https://www.kaggle.com/robervalt/sunspots). I had the opportunity to work on it as a part of the curriculum of one of my specialization courses called [DeepLearning.AI TensorFlow Developer Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/L6R6AFWVXHZT) which is given by DeepLearning.AI. This specialization is made of four courses: 
1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning 
2. Convolutional Neural Networks in TensorFlow 
3. Natural Language Processing in TensorFlow 
4. Sequences, Time Series and Prediction

Specifically, this project is part of the fourth course of this specialization. 

- Sunspots are temporary phenomena on the Sun's photosphere that appear as spots darker than the surrounding areas. They are regions of reduced surface temperature caused by concentrations of magnetic field fluxes that inhibit convection. Sunspots usually appear in pairs of opposite magnetic polarity. Their number varies according to the approximately 11-year solar cycle. We have a dataset that contains the monthly mean total sunspot number, from 1749/01/01 to 2017/08/31. The idea is to create a Deep Learning model that is capable of predicting the total sunspot number in the future. 

- Loking at how the monthly mean total sunspot number changes with time, we can observe a bit of seaonality. However, it is not very regular given that some peaks are much higher than others. We also have a bit of noise, but there is no general trend.

-  We split our series into a training and validation datatests. We select split_time= 3000. We set all the constants for our neural network model. Window_size = 20, batch_size = 32 and shuffle_buffer_size = 1000.

- For the Neural Netwrok model: We use Convolutional layer with the activation function as a "relu". We use two Long Short Term Memory layers. We use three Dense layers with the activation function as a "relu". We use lr=1e-5 and epoch=100. The loss function used was Huber. The metric was mae.
  

# [Project 5: Digit Recognizer](https://github.com/lilosa88/DigitRecognizion)

- This project is one of [kaggle's competitions](https://www.kaggle.com/c/digit-recognizer). As previously, I carried out this work as a part of a specialization course called [DeepLearning.AI TensorFlow Developer Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/L6R6AFWVXHZT) which is given by DeepLearning.AI. This specialization is made of 4 courses: 
1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning 
2. Convolutional Neural Networks in TensorFlow 
3. Natural Language Processing in TensorFlow 
4. Sequences, Time Series and Prediction

Specifically, this project is part of the second course of this specialization. 

- MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerged, MNIST remains a reliable resource for researchers and learners alike.

- The objective of this study was to correctly identify digits from a dataset of tens of thousands of handwritten images.

-  For the feature engineering we:
  
    - Defined X and Y from the df_train dataset.
    - We normalized, dividing by 255 (maximum value that you can find in one row of the df_train dataset).
    - Reshaping, following X = X.values.reshape(-1, 28,28,1)
    - Label encoding for the y label
    - Split into train and test

- We compared the performance of two following two neural networks: Simple Model (Accuracy 0.97238) and Model with double convolutions and pooling (Accuracy 0.9864). In both case the activation functions used were 'relu' and 'softmax', the lr = 0.001 and as loss function we use categorical_crossentropy.
  
# [Project 6: Fashion-MNIST](https://github.com/lilosa88/Fashion-MNIST-)

- This project belongs to one of [kaggle's competitions](https://www.kaggle.com/zalando-research/fashionmnist). I carried out this work as part of a specialization course called [DeepLearning.AI TensorFlow Developer Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/L6R6AFWVXHZT), which is given by DeepLearning.AI. This specialization is made of 4 courses: 
1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning 
2. Convolutional Neural Networks in TensorFlow 
3. Natural Language Processing in TensorFlow 
4. Sequences, Time Series and Prediction

  Specifically, this project is part of the first course of this specialization. 

- MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerged, MNIST remains a reliable resource for researchers and learners alike.

- Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

- The objective of this study is to correctly identify the different Zalando's articles from the dataset.

-  For the feature engineering we did the following:
  
    - Since the Fashion MNIST data is available directly in the tf.keras datasets API, we used load_data to get two sets of two lists, which represent the training and testing values of the corresponding imagesm which contain the clothing items and their labels.
 
    - We normalized by the column length (i.e. 255).

    - We reshaped the images (only for the second model), using training_images.reshape(60000, 28, 28, 1) and test_images.reshape(10000, 28, 28, 1)

- We compared the performance using the following neural networks: Simple Neural Network (Accuracy 0.9299) and Neural Network with convolutions and pooling (Accuracy 0.9953). 

# [Project 7: Horse or Human](https://github.com/lilosa88/Horse-or-Human)

- This project belongs to [kaggle's competitions](https://www.kaggle.com/sanikamal/horses-or-humans-dataset) and I carried out as a part of a specialization called [DeepLearning.AI TensorFlow Developer Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/L6R6AFWVXHZT) which is given by DeepLearning.AI. This specialization is conformed by 4 courses: 
1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning 
2. Convolutional Neural Networks in TensorFlow 
3. Natural Language Processing in TensorFlow 
4. Sequences, Time Series and Prediction

  Specifically this project is part of the first course in this specialization. 

- Horses or Humans is a dataset of 300×300 images, created by Laurence Moroney, that is licensed CC-By-2.0 for anybody to use in learning or testing computer vision algorithms.

- The objective of this study is to correctly identify if the image is a horse or a human.

-  For the feature engineering we:
  
    - We define each directory using os library.

    - Use of data generators. It read the pictures in our source folders, convert them to float32 tensors, and feed them (with their labels) to our network. We          have one generator for the training images and one for the validation images. The two generators yield batches of images of size 300x300 and their labels          (binary). 

    - Data that goes into neural networks should usually be normalized in some way to make it more amenable to processing by the network. In our case, we will          preprocess our images by normalizing the pixel values to be in the [0, 1] range (originally all values are in the [0, 255] range). In Keras this can be done        via the keras.preprocessing.image.ImageDataGenerator class using the rescale parameter. This ImageDataGenerator class allows you to instantiate generators of      augmented image batches (and their labels) via .flow(data, labels) or .flow_from_directory(directory). These generators can then be used with the Keras model      methods that accept data generators as inputs: fit, evaluate_generator, and predict_generator.

- For the Neural Network with convolutions and pooling (Accuracy 0.9555). 

# [Project 8: Dogs vs Cats](https://github.com/lilosa88/Dogs-vs-Cats)

- This project belongs to [kaggle's competitions](https://www.kaggle.com/c/dogs-vs-cats/overview) and I carried out as a part of a specialization called [DeepLearning.AI TensorFlow Developer Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/L6R6AFWVXHZT) which is given by DeepLearning.AI. This specialization is conformed by 4 courses: 
1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning 
2. Convolutional Neural Networks in TensorFlow 
3. Natural Language Processing in TensorFlow 
4. Sequences, Time Series and Prediction

  Specifically this project is part of the second course in this specialization. 

- Dogs vs Cats is a dataset of 1500 images for each dogs and cats. Each image have a different sizes and is unlabel.

- The objective of this study is to correctly identify if the image is a dog or a cat.

-  For the feature engineering we:
  
   - We define each directory using os library.

   - Use of data generators. It read the pictures in our source folders, convert them to float32 tensors, and feed them (with their labels) to our network. We have      one generator for the training images and one for the validation images. The two generators yield batches of images of size 150x150 and their labels (binary). 

   - Data that goes into neural networks should usually be normalized in some way to make it more amenable to processing by the network. In our case, we will            preprocess our images by normalizing the pixel values to be in the [0, 1] range (originally all values are in the [0, 255] range). In Keras this can be done        via the keras.preprocessing.image.ImageDataGenerator class using the rescale parameter. This ImageDataGenerator class allows you to instantiate generators of      augmented image batches (and their labels) via .flow(data, labels) or .flow_from_directory(directory). These generators can then be used with the Keras model      methods that accept data generators as inputs: fit, evaluate_generator, and predict_generator.

   - Imagen Augmentation (only in the second and third model) which is a very simple, but powerful tool to help to avoid overfitting. To put it simply, if you are      training a model to spot cats, and your model has never seen what a cat looks like when lying down, it might not recognize that in future. Augmentation simply      amends your images on-the-fly while training using transforms like rotation, among others.

- We applied four different models:
  - Neural Network with Convolution and Pooling (Accuracy 0.9875). We found overfitting.  
  - Neural Network with Convolution and Pooling making use of Imagen Augmentation (Accuracy 0.7970)  
  - Neural Network with Convolution, Pooling and Dropout making use of Image Augmentation (Accuracy 0.8105)  
  - Pre-trained Neural Network (using InceptionV3) with Convolution, Pooling and Dropout (Accuracy 0.9610)  

# [Project 9: Rock, Paper and Scissors](https://github.com/lilosa88/Rock-Paper-and-Scissors)

- This project belongs to [kaggle's competitions](https://www.kaggle.com/c/rock-paper-scissors/overview) and I carried out as a part of a specialization called [DeepLearning.AI TensorFlow Developer Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/L6R6AFWVXHZT) which is given by DeepLearning.AI. This specialization is conformed by 4 courses: 
1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning 
2. Convolutional Neural Networks in TensorFlow 
3. Natural Language Processing in TensorFlow 
4. Sequences, Time Series and Prediction

  Specifically this project is part of the second course in this specialization. 

- Rock, Paper and Scissors is a dataset containing 2,892 images of diverse hands in Rock/Paper/Scissors poses.

- The objective of this study is to correctly identify if the image.

-  For the feature engineering we:
  
   - We define each directory using os library.

   - Use of data generators. It read the pictures in our source folders, convert them to float32 tensors, and feed them (with their labels) to our network. We have      one generator for the training images and one for the validation images. The two generators yield batches of images of size 150x150 and their labels (binary). 

   - Data that goes into neural networks should usually be normalized in some way to make it more amenable to processing by the network. In our case, we will            preprocess our images by normalizing the pixel values to be in the [0, 1] range (originally all values are in the [0, 255] range). In Keras this can be done        via the keras.preprocessing.image.ImageDataGenerator class using the rescale parameter. This ImageDataGenerator class allows you to instantiate generators of      augmented image batches (and their labels) via .flow(data, labels) or .flow_from_directory(directory). These generators can then be used with the Keras model      methods that accept data generators as inputs: fit, evaluate_generator, and predict_generator.

   - Imagen Augmentation (only in the second and third model) which is a very simple, but powerful tool to help to avoid overfitting. To put it simply, if you are      training a model to spot cats, and your model has never seen what a cat looks like when lying down, it might not recognize that in future. Augmentation simply      amends your images on-the-fly while training using transforms like rotation, among others.
 
  - Neural Network with Convolution, Pooling and Dropout making use of Image Augmentation (Accuracy 0.9806)  

# [Project 10: Spam Detection](https://github.com/lilosa88/Spam-Detection-)

- Dataset is obtained from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection). The SMS Spam Collection is a public set of SMS labeled messages that have been collected for mobile phone spam research. The objective is to build a predictive model that predicts if a message is spam or not. 

- For the preprocessing:

   - We clean the data by removing punctuations, stopwords and applying lowercase. Thus we use PorterStemmer, stemming is the process of reducing words to their        word stem.
   - We convert our sentences into vectors using Bag of words model.
   - We applying encoding into the column label.
   - Train and test split. 

- Machine Learning Model: Naive Bayes Model
 
  Train Random Forest's Accuracy:  0.9887
 
  Test Random Forest's Accuracy:  0.9838 
  
# [Project 11: Sarcasm-detection](https://github.com/lilosa88/Sarcasm-detection)

- This project belongs to [kaggle's competitions](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home) and I carried out as a part of a specialization called [DeepLearning.AI TensorFlow Developer Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/L6R6AFWVXHZT) which is given by DeepLearning.AI. This specialization is conformed by 4 courses: 
1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning 
2. Convolutional Neural Networks in TensorFlow 
3. Natural Language Processing in TensorFlow 
4. Sequences, Time Series and Prediction

  Specifically this project is part of the third course in this specialization. 
  
- We will make use of a public data-sets published by [Rishabh Misra](https://rishabhmisra.github.io/publications/) with details on [Kaggle] (https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home). News Headlines dataset for Sarcasm Detection is collected from two news website. [TheOnion](https://www.theonion.com/) aims at producing sarcastic versions of current events and we collected all the headlines from News in Brief and News in Photos categories (which are sarcastic). We collect real (and non-sarcastic) news headlines from [HuffPost](https://www.huffingtonpost.com/). 

- For the preprocessing:

  - Test and train split: This dataset has about 27,000 records. So, we train on 20,000 and validate on the rest. 

  - For both train and test data we apply the following steps:

      - We apply tokenizer with vocab_size = 1000 and oov_tok = OOV.
      - We apply the fit_on_texts method. 
      - We apply the word_index method. 
      - We turn the sentences into lists of values based on these tokens.To do so, we apply the method texts_to_sequences.
      - We apply the method pad_sequences that use padding. 
      - We convert the two train and test sets into arrays

- Neural Network with one Embedding layer, one GlobalAveragePooling1D layer and two Dense layers (Accuracy 0.8800).

# [Project 12: Guessing-following-words](https://github.com/lilosa88/Guessing-following-words)

- This project was carried out as a part of a specialization called [DeepLearning.AI TensorFlow Developer Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/L6R6AFWVXHZT) which is given by DeepLearning.AI. This specialization is conformed by 4 courses: 
1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning 
2. Convolutional Neural Networks in TensorFlow 
3. Natural Language Processing in TensorFlow 
4. Sequences, Time Series and Prediction

  Specifically this project is part of the third course in this specialization. 
  
- We taken a traditional Irish song and the purpouse is to create a model that allow us to guees the following word in a sentence.

- Preprocessing: We convert the text into a list of broken sentences. To do so we make use of tokenizer and padding.
 
 - Neural Network with one Embedding layer, one Bidiractional LSTM layer and one Dense layers (Accuracy 0.9536).

# [Project 13: Creating Poetry](https://github.com/lilosa88/Poetry)

- This project was carried out as a part of a specialization called [DeepLearning.AI TensorFlow Developer Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/L6R6AFWVXHZT) which is given by DeepLearning.AI. This specialization is conformed by 4 courses: 
1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning 
2. Convolutional Neural Networks in TensorFlow 
3. Natural Language Processing in TensorFlow 
4. Sequences, Time Series and Prediction

  Specifically this project is part of the third course in this specialization. 
  
- We create a file with a combinations of songs. The purpouse is to create a model that allow us to complete a sentence.

- Preprocessing: We convert the text into a list of broken sentences. To do so we make use of tokenizer and padding.
 
 - Neural Network with one Embedding layer, one Bidiractional LSTM layer and one Dense layers (Accuracy 0.7726).

# [Project 14: Fake-News-Classifier](https://github.com/lilosa88/Fake-News-Classifier)

- TThis project belongs to [kaggle's competitions](https://www.kaggle.com/c/fake-news/data). The objective is to sevelop a machine learning or Deep Learning program to identify when an article might be fake news.


- Preprocessing: We remove all the missing values. Followed by cleaning the data by removing punctuations, stopwords and applying lowercase. Thus we use PorterStemmer, stemming is the process of reducing words to their word stem.

- First model: Bag of words and Machine Learning model 
    - Train MultinomialNB Algorithm's Accuracy: 0.924
    - Test MultinomialNB Algorithm's Accuracy: 0.901

- Second model: One hot representation and Neural Network with LSTM
    - Train Neural Network's Accuracy: 0.996
    - Test Neural Network's Accuracy: 0.906

- Third model: One hot representation and Neural Network with a Bidirectional LSTM and a Dropout layer
    - Train Neural Network's Accuracy: 0.996
    - Test Neural Network's Accuracy: 0.907

- Fourth model: TF-idf Vectorizer and Machine Learning model 
    - Train MultinomialNB Algorithm's Accuracy: 0.918
    - Test MultinomialNB Algorithm's Accuracy: 0.880

# [Project 15: IMDB](https://github.com/lilosa88/IMDB)

- This project was carried out as a part of a specialization called [DeepLearning.AI TensorFlow Developer Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/L6R6AFWVXHZT) which is given by DeepLearning.AI. This specialization is conformed by 4 courses: 
1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning 
2. Convolutional Neural Networks in TensorFlow 
3. Natural Language Processing in TensorFlow 
4. Sequences, Time Series and Prediction

  Specifically this project is part of the third course in this specialization. 
  
- From TensorFlow Data Services (TFTS) library we used the IMDB reviews dataset. This dataset was authored by [Andrew Mass et al at Stanford](http://ai.stanford.edu/~amaas/data/sentiment/). 


- Preprocessing: 
  - Test and train split: This dataset has about 50,000 records. So, we train on 25,000 and validate on the rest.
  - For both train and test data we apply the following steps:

      - We apply tokenizer with vocab_size = 1000 and oov_tok = OOV.
      - We apply the fit_on_texts method. 
      - We apply the word_index method. 
      - We turn the sentences into lists of values based on these tokens.To do so, we apply the method texts_to_sequences.
      - We apply the method pad_sequences that use padding. 
      - We convert the two train and test sets into arrays

- First model: Neural Network with One Embedding layer, one Flatten layer and two Dense layers (Accuracy 1.0) 
 
- Second model: Neural Network with One Embedding layer, one Global Average Pooling 1D layer and two Dense layers (Accuracy 0.9509) 
  
- Third model: Neural Network with One Embedding layer, one Bidirectional layer with GRU(32) and two Dense layers: This adds a layer of neurons (Accuracy 0.9832) 
 
- Fourth model: Neural Netwrok with One Embedding layer, one Bidirectional layer with LSTM(32) and two Dense layers (Accuracy 1.0)
  
- Fifth model: Neural Network with One Embedding layer, one Conv1D layer, one GlobalAveragePooling1D and Two Dense layers (Accuracy 1.0)

With a pre-tokenize dataset:
  
- Sixth model: Neural Network with One Embedding layer, one Global Average Pooling 1D layer and two Dense layers (Accuracy 0.9411)

- Seventh model: Neural Network with One Embedding layer, one Bidirectional layer and two Dense layers (Accuracy 0.8928)
  
- Eigth model:Neural Netwrok with One Embedding layer, two Bidirectional layer and two Dense layers (Accuracy 0.9774)
