# Breast-Cancer-Causal-Inference
![causal_inference](https://user-images.githubusercontent.com/79056802/184502427-3547b825-abb5-4a9b-919b-7db1a0c7d7ca.png)

## Overview
> Breast-cancer Diagnostic The second greatest cause of cancer death in women, after lung cancer, is breast cancer, which is the most prevalent invasive cancer in females. Since 1989, significant progress has been made in the detection and treatment of breast cancer. More than 3.1 million Americans have survived breast cancer, according to the American Cancer Society (ACS). About 1 in 38 women will develop breast cancer in their lifetime (2.6 percent ). Early detection of the disease and precise diagnosis both increase the likelihood of long-term survival for a person with breast cancer.The prognosis, or anticipated long-term behavior of the disease, heavily influences the choice of appropriate therapy immediately following surgery.

> This project aims at inferring the causation of breast cancer based on the characteristics breast mass. With this information, disease indicators can be predicted and early treatment can be started for a given patient. Early treatment gives the patient a higher chance of survival and cure from breast cancer. This project will build causal graphs from the dataset and use this graphs for analysis. From the causal graphs, we will select features that have direct influence to cancer diagnosis. We will then use this selected features to train a Linear Regression model. We will compare the accuracy of this model with a model that is developed by using all the features in the dataset.

## Installation
```
git clone https://github.com/wakura-mbuya/Breast-Cancer-Causal-Inference.git
cd Breast-Cancer-Causal-Inference
pip install -r requirements.txt
```

## Data
The data used for this project was obtained from [Kaggle](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data). It was collected from diagnosis of breast cancer through processes such as mammography, FNA with visual interpretations and surgical biopsy. Features in the data are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.The data can be found [here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

#### Categorical variables in the data

      * Diagnosis(Malignant / benign)

#### Continuous variables

      * the circumference (mean of distances from the center to points on the perimeter)
      * the concavity (severity of concave portions of the contour)
      * points that are concave (number of concave portions of the contour)
      * fractal dimension of symmetry (“coastline approximation” — 1)
      * the texture (standard deviation of gray-scale values)
      * Perimeter\s area
      * suppleness (local variation in radius lengths)
      * compactness (area2 / perimeter2 — 1.0)

## Notebooks

> All the analysis and examples of implementation can be found here in the form of .ipynb file. The project used 2 jupyter notebooks: <i>eda.ipynb</i> and <i>causal

## Scripts

> All the modules for the analysis are found here. We defined two scripts that modularizes the code we used in the notebook. The causality.py script includes functions used for causal analysis. utilities.py script include functions for performing comman tasks such as loading data and visualization.

## Tests

> All the unit and integration tests are found here

## Author

👤 **Wakura Mbuya**

- GitHub: [Wakura Mbuya](https://github.com/wakura-mbuya)
- LinkedIn: [Wakura Mbuya](https://www.linkedin.com/in/ken-wakura-b72234218)

## Show your support

Give a ⭐ if you like this project!
