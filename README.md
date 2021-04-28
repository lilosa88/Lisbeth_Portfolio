# Data Science Portafolio

# [Project 1: Case study of the survival rate in Titanic:](https://github.com/lilosa88/Titanic)

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

# [Project 3: Case study to predict final prices of houses in Iowa:](https://github.com/lilosa88/PricingHouse)

- This project belongs to [kaggle's competitions](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). In order to buy a house there are many different parameters that influences price negotiations. Therefore, the idea is to create a model that predicts the sales prices given a dataset with 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa. 

# [Project 4: Prediction of Sunspots:](https://github.com/lilosa88/Sunspots)
- Sunspots are temporary phenomena on the Sun's photosphere that appear as spots darker than the surrounding areas. Create a model to predict the number of suspots.

# [Project 5:Digit Recognition:](https://github.com/lilosa88/DigitRecognizion)
- MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

- Create a model that correctly identify digits from a dataset of tens of thousands of handwritten images. 
