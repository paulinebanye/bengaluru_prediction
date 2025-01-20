# **Predicting Property Prices in Bengaluru**
This project focuses on building a predictive model to predict house prices in Bengaluru, India by preparing the dataset, performing exploratory data analysis, and applying advanced machine learning techniques. Through data cleaning, outlier removal, and feature engineering, the dataset was optimized for accurate predictions. Various models were tested, and hyperparameter tuning was used to select the best-performing one. The final model is ready for deployment to provide reliable price estimates, assisting buyers, sellers, and real estate professionals in making informed decisions. 

<div style="text-align: center"><img src="imgs/villas-project.jpg" alt="house" width="80%" height="50%"></div>

---

> ## Table of contents
- [Overview](#overview)
- [Technologies](#technologies)
- [Setting up the project](#setting-up-the-project)
- [Project Workflow](#project-workflow)
  - [Data Preparation](#data-preparation)
  - [Feature Engineering](#feature-engineering)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Outlier Detection and Removal](#outlier-detection-and-removal)
  - [Building the prediction model](#building-the-prediction-model)
  - [Validation and Model Comparison](#validation-and-model-comparison)
  - [Identifying the Best Model](#identifying-the-best-model)
  - [Combining Training and Validation Data](#combining-training-and-validation-data)
  - [Test Set Evaluation](#test-set-evaluation)
  - [Storing the Model](#storing-the-model)
  - [Link to the Model on Kaggle](#link-to-the-model-on-kaggle)
- [Future Improvements](#future-improvements)
- [Status](#status)
- [Contributing to the project](#contributing-to-the-project)


#

> ## **Overview**
To develop a reliable predictive model for property prices by cleaning and analyzing data, extracting relevant features, and implementing machine learning algorithms.  

**Dataset**\
This model leverages a dataset containing key features such as Area Type, Availability, Location, Price, Size, Society, Total Sqft, Bath, and Balcony to make accurate predictions.  
- **area_type**: Property location type (e.g., Super built-up Area).  
- **availability**: Indicates when the property is ready (e.g., Ready To Move).  
- **location**: The area of the property (e.g., Electronic City Phase II).  
- **size**: Number of rooms (e.g., 2 BHK).  
- **society**: Housing society name.  
- **total_sqft**: Total property area in square feet.  
- **bath**: Number of bathrooms.  
- **balcony**: Number of balconies.  
- **price**: Property price in lakhs (₹).  

> ## Technologies

<p align="justify">
*Note: This project was setup and developed on a system running Windows 10. The stacks used for the project include:
</p>

| <b><u>Tools</u></b> | <b><u>Usage</u></b>   |
| :------------------ | :-------------------- |
| **`Python 3.11`**   | Programming language. |
| **`Anaconda 23.7.4`** | Anaconda Navigator for managing packages and environments. |
| **`Jupyter Lab`**    | Integrated development environment for Python. |

#

> ## Setting Up The Project

<p align="justify">
The first step requires the download and installation of Python 3.11 and Anaconda.
</p>

<p align="justify">
After the installation of the Python program, setup the project environment with the Anaconda environment. This helps to create an isolated Python environment containing all the packages necessary for the project.
</p>

```python
# create a conda environment
(base) conda create -n pred python=3.11

# activate the conda environment
(base) conda activate pred

# install jupyter lab
(base) conda install -n pred -c conda-forge jupyterlab

# start jupyter lab
(pred) jupyter lab
```

Once the conda environment is active, the next step is the installation of all the dependencies needed for this project.

\*Note:

- If a "pip command not found error" is encountered, download get-pip.py and run `phython get-pip.py` to install it.


A few of the dependencies are listed below.

| <b><u>Modules</u></b>     | <b><u>Usage</u></b>           |
| :------------------------ | :---------------------------- |
| **`pandas`** | Data manipulation and analysis. |
| **`seaborn`** | Data visualization library. |
| **`sckit learn`** | Evaluating model performance. |
| **`rapidfuzz`** | Fuzzy string matching. |

An exhaustive list can be found in the requirements.txt file included in this project. The modules can be 'batch installed' using the `pip install -r requirements.txt` command.


> ## **Project Workflow**
- ### **Data Preparation**:
  The dataset was carefully reviewed and cleaned. Missing values were addressed either by removing irrelevant columns or imputing data. Columns like total_sqft and availability, were standardized to ensure uniformity, features such as price per square foot were calculated to enhance analysis.

  ```python
  # Fill missing data
  for col in df.columns[df.isnull().any()]:
      if df[col].dtype == "object":
          df[col].fillna(df[col].mode()[0], inplace=True)
      else:
          df[col].fillna(df[col].median(), inplace=True)
  ```

- ### **Feature Engineering**
  Location names were cleaned to correct errors and identify common names, and mixed  data, like ranges in total_sqft, was resolved for consistency.

- ### **Exploratory Data Analysis**
  The dataset was analyzed to identify trends and patterns, and visualizations were created to provide insights into the data. This analysis helped in understanding the relationships between features.
  - Top 10 locations by price per square feet.
  - Total properties per area type.
  - Correlation between features
  - Average property size per area type.
  - Distribution of area types by location
  - Availability dates and house prices per square feet over time
  - Amentities distribution across locations

- ### **Outlier Detection and Removal**
  Outliers were identified and removed to prevent them from skewing the model's predictions. This was achieved by analyzing the distribution of features and removing data points that deviated significantly from the norm.

- ### **Building the prediction model**
  The dataset was split into training, validation, and test sets to train and evaluate the model. Various machine learning models were tested, and hyperparameter tuning was used to optimize their performance.

  The training process focused on fitting each model to the training data and optimizing their hyperparameters using GridSearchCV. This method also accounted for the encoding and scaling of the categorical and numerical features, ensuring the models were trained on standardized data.

  ```python
  # # select columns based on the datatype (numeric & categorical)
  numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
  categorical_features = X_train.select_dtypes(include=['object']).columns

  # specify preprocessing step for the columns
  preprocessor = ColumnTransformer(
      transformers=[
          ('num', StandardScaler(), numeric_features),
          ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
      ])
  ```

- ### **Validation and Model Comparison**
  Once trained, the models were evaluated on the validation set using the mean squared error (MSE) metric. This allowed comparison of their predictive accuracy and identification of the most effective model.

- ### **Identifying the Best Model**
  The model with the lowest validation MSE was selected as the best-performing model. Its configuration and performance were analyzed to confirm its suitability for the test set.

- ### **Combining Training and Validation Data**
  To provide the final model with as much data as possible, the training and validation datasets were combined, and the model was retrained on this merged dataset.

- ### **Test Set Evaluation**
  The final evaluation was conducted using the test set. This provided an unbiased estimate of the model's performance on unseen data, ensuring the model's generalizability to real-world applications.

- ### **Storing the Model**
  To ensure reproducibility and future usability, the final trained model was serialized and stored using Python's `joblib` library.

  ```python
  # Save the trained model to a file
  import joblib
  joblib.dump(final_model, 'bengalaru_model.joblib')
  print("Model stored successfully!")
  ```
- ### **Link to the Model on Kaggle**
  [Pauline's Bengaluru House Prediction](https://www.kaggle.com/models/beepauline/paulines-bengaluru-house-prediction)

> ## Future Improvements
To enhance the accessibility and impact of this model, several deployment strategies could be explored in the future:
- Expanding the dataset beyond Bengaluru.
- Incorporating geolocation data to provide more accurate location-based predictions.
- Developing a user-friendly interface for easy access to the model's predictions.
- Implementing a web-based application to allow users to interact with the model directly.


> ## Status
This project is a work in progress and is currently under development.


> ## Contributing to the project

If you find something worth contributing, such as datasets (from diverse locations), please fork the repo, make a pull request and add valid and well-reasoned explanations about your changes or comments.

Before adding a pull request, please note:

- It should be inviting and clear.
- Any additions should be relevant.
- It should be easy to contribute to.

This repository is not meant to contain everything. Only good quality verified information.

All **`suggestions`** are welcome!

> ###### Readme created by Pauline Banye (2025)