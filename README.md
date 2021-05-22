# Data Science Portfolio

# [Project 1: Case study of the survival rate in Titanic](https://github.com/lilosa88/Titanic)

- The sinking of the Titanic is one of the most infamous shipwrecks in history. While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others. This project that is part of [kaggle's competitions](https://www.kaggle.com/c/titanic/overview) has as objective to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

- From data exploration we found that women and kids had more chances to survive. As well, the class ticket played a role, being the 1st class the onces with more chances to survive. The age and the place of embarking was decisive for the suvival for mens. People that travel with 1 or 3 people than 0 or more than 3, had more chances to survive.

- Different Classification Machine Learning model and Deep Learning model were used in oder to predict the survival of passengers aboard Titanic. Among the classification ML we used Random Forest, Logistic Regression and KNN.

- The best accuracy (0.86) was obtained from Deep Learning Model. The use of Dropout layers was used in order to avoid overfitting.  


# [Project 2: Case study of the Diabetes rate](https://github.com/lilosa88/Diabetes)

- Dataset is obtained from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.). This has been collected using direct questionnaires from the patients of Sylhet Diabetes Hospital in Sylhet, Bangladesh and approved by a doctor. This dataset contains the sign and symptpom data of newly diabetic or would be diabetic patient. The objective is to build a predictive model that predicts if a pacient is prone to be diabetic or not using the dataset. 

- From data exploration we found that except for the features itching and delayed healing, the rest of the features has an influence on the diabetes rate. Specificately the once that has more influence are:
-Gender: Women are more prone to get diabetes.
-Polyuria: If the pacient present an excessive or an abnormally large production or passage of urine then is more prone to get diabetes.
-Polydipsia: If the pacient present an excessive thirst or excess drinking is more prone to get diabetes.
-Polyphagia:  If the pacient present an feels an abnormally strong sensation of hunger or desire to eat often leading to or accompanied by overeating is more prone to get diabetes.

- The best accuracy (0.9711) was obtained with KNN with k=3.

# [Project 3: Case study to predict final prices of houses in Iowa](https://github.com/lilosa88/PricingHouse)

- This project belongs to [kaggle's competitions](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). In order to buy a house there are many different parameters that influences price negotiations. Therefore, the idea is to create a model that predicts the sales prices given a dataset with 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa. 

- The features that had a huge amount of missing values (approx. more than 2000 missing values), were treated accordingly. We found that indded this values were not missing just they were related to a missing of that particular feature (i.e. For PoolQuality the missing values means that that house did not have any pool). For the rest of the features with missing values, as these features had maximum 4 missing values out of 2919 we fill the data with the corresponding media value for the cases where the feature is float64. For the categorical variables we fill it with the mode. 

- For the feature engineering new columns were created in order to better understanding of the data. The redundant or useless information was dropped. As well the categorical variable as strings they were transformed into continuos variables making use of dummy variables and the respective normalization of the whole dataset was carried out using MinMaxScaler. 

- Two machine Learning model were tested: Ridge Regression and Lasso. The hyperparameter 'alpha' has the following values: 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000. The best for Ridge Regression alpha was 10 and for Lasso was 100. At the end the best accuaracy obtained was given by Lasso: The train Lasso Accuracy was 0.9229 and the test Lasso Accuracy was 0.8902.


# [Project 4: Case study in order to predict the total sunspots number](https://github.com/lilosa88/Sunspots)
- This project belongs to [kaggle's competitions](https://www.kaggle.com/robervalt/sunspots) and I carried out as a part of a specialization called [DeepLearning.AI TensorFlow Developer Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/L6R6AFWVXHZT) which is given by DeepLearning.AI. This specialization is conformed by 4 courses: 
1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning 
2. Convolutional Neural Networks in TensorFlow 
3. Natural Language Processing in TensorFlow 
4. Sequences, Time Series and Prediction

Specifically this project is part of the second course in this specialization. 

- Sunspots are temporary phenomena on the Sun's photosphere that appear as spots darker than the surrounding areas. They are regions of reduced surface temperature caused by concentrations of magnetic field flux that inhibit convection. Sunspots usually appear in pairs of opposite magnetic polarity. Their number varies according to the approximately 11-year solar cycle. We have a dataset that contains the monthly mean total sunspot number, from 1749/01/01 to 2017/08/31. The idea is to create a Deep Learning model that is capable of predicting the total sunspot number in the future. 

- Loking at how the monthly mean total sunspot number changes with the time we observe a bit of seaonality. However it is not very regular with some peaks much higher than others. We also have a bit of noise but there is not general trend.

-  We split our series into a training and validation datatests. We select split_time= 3000. We set all the constants for our neural network model. Window_size = 20, batch_size = 32 and shuffle_buffer_size = 1000.

- For the Neural Netwrok model: We use Convolutional layer with the activation function as a "relu". We use two Long Short Term Memory layers. We use three Dense layers with the activation function as a "relu". We use lr=1e-5 and epoch=100. The loss function used was Huber. The metric was mae.
  

# [Project 5: Digit Recognizer](https://github.com/lilosa88/DigitRecognizion)

- This project belongs to [kaggle's competitions](https://www.kaggle.com/c/digit-recognizer) and I carried out as a part of a specialization called [DeepLearning.AI TensorFlow Developer Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/L6R6AFWVXHZT) which is given by DeepLearning.AI. This specialization is conformed by 4 courses: 
1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning 
2. Convolutional Neural Networks in TensorFlow 
3. Natural Language Processing in TensorFlow 
4. Sequences, Time Series and Prediction

Specifically this project is part of the second course in this specialization. 

- MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

- The objective of this study is to correctly identify digits from a dataset of tens of thousands of handwritten images.

-  For the feature engineering we:
  
    - Defined X and Y from the df_train dataset
    - So we normalize dividing by 255 (maximum value that you can find in one row of the df_train dataset).
    - Resahping, following X = X.values.reshape(-1, 28,28,1)
    - Label encoding for the y label
    - Split into train and test

- We compare the performance of two following two neural networks: Simple Model (Accuracy 0.97238) and Model with double convolutions and pooling (Accuracy 0.9864). In both case the activation functions used were 'relu' and 'softmax', the lr = 0.001 and as loss function we use categorical_crossentropy.
  
  # [Project 6: Fashion-MNIST](https://github.com/lilosa88/Fashion-MNIST-)

- This project belongs to [kaggle's competitions](https://www.kaggle.com/zalando-research/fashionmnist) and I carried out as a part of a specialization called [DeepLearning.AI TensorFlow Developer Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/L6R6AFWVXHZT) which is given by DeepLearning.AI. This specialization is conformed by 4 courses: 
1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning 
2. Convolutional Neural Networks in TensorFlow 
3. Natural Language Processing in TensorFlow 
4. Sequences, Time Series and Prediction

  Specifically this project is part of the first course in this specialization. 

- MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

- Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

- The objective of this study is to correctly identify the different Zalando's articles from the dataset.

-  For the feature engineering we:
  
    - The Fashion MNIST data is available directly in the tf.keras datasets API. Using load_data we get two sets of two lists, these will be the training and testing values for the graphics that contain the clothing items and their labels.
 
- The values in the number are between 0 and 255. Since we will train a neural network, we need that all values are between 0 and 1. Therefore, we normalize dividing by 255.

- We reshape the images (only for the second model), following training_images.reshape(60000, 28, 28, 1) and test_images.reshape(10000, 28, 28, 1)

- We compare the performance of the two following two neural networks: Simple Neural Network (Accuracy 0.9299) and Neural Network with convolutions and pooling (Accuracy 0.9953). 

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

# [Project 10: Rock, Paper and Scissors](https://github.com/lilosa88/Rock-Paper-and-Scissors)

# Objective

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

# [Project 11: Spam Detection](https://github.com/lilosa88/Spam-Detection-)
# Objective

Dataset is obtained from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection). The SMS Spam Collection is a public set of SMS labeled messages that have been collected for mobile phone spam research. The objective is to build a predictive model that predicts if a message is spam or not. 

# Preprocessing

- We clean the data by removing punctuations, stopwords and applying lowercase. Thus we use PorterStemmer, stemming is the process of reducing words to their word stem.
- We convert our sentences into vectors using Bag of words model.
- We applying encoding into the column label.
- Train and test split. 

# Machine Learning Model

- Naive Bayes Model
 
 Train Random Forest's Accuracy:  0.9887
 
 Test Random Forest's Accuracy:  0.9838 
