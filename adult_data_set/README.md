# Adult Dataset

This dataset is taken from https://archive.ics.uci.edu/ml/datasets/Adult

Also taken from Kaggle https://www.kaggle.com/wenruliu/adult-income-dataset/notebooks

| **abstract**: Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset. | ![img](https://archive.ics.uci.edu/ml/assets/MLimages/Large2.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |

| **Data Set Characteristics:**  | Multivariate         | **Number of Instances:**  | 48842 | **Area:**               | Social     |
| ------------------------------ | -------------------- | ------------------------- | ----- | ----------------------- | ---------- |
| **Attribute Characteristics:** | Categorical, Integer | **Number of Attributes:** | 14    | **Date Donated**        | 1996-05-01 |
| **Associated Tasks:**          | Classification       | **Missing Values?**       | Yes   | **Number of Web Hits:** | 1906418    |



**Source:**

Donor:

Ronny Kohavi and Barry Becker
Data Mining and Visualization
Silicon Graphics.
e-mail: ronnyk '@' live.com for questions.



**Data Set Information:**

Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

Prediction task is to determine whether a person makes over 50K a year.



**Attribute Information:**

Listing of attributes:

\>50K, <=50K.

age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

**Relevant Papers:**

Ron Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid", Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, 1996
[[Web Link\]](http://robotics.stanford.edu/~ronnyk/nbtree.pdf)

## Training

### Tensorflow NN ( with out  Standard Scalar )

```
Epoch: 0        Loss: 3.7956421
Epoch: 500      Loss: 3.7956421
Epoch: 1000     Loss: 3.7956421
Epoch: 1500     Loss: 3.7956421
Epoch: 2000     Loss: 3.7956421
Epoch: 2500     Loss: 3.7956421
Epoch: 3000     Loss: 3.7956421
Epoch: 3500     Loss: 3.7956421
Epoch: 4000     Loss: 3.7956421
Epoch: 4500     Loss: 3.7956421
Epoch: 5000     Loss: 3.7956421
Epoch: 5500     Loss: 3.7956421
Epoch: 6000     Loss: 3.7956421
Epoch: 6500     Loss: 3.7956421
Epoch: 7000     Loss: 3.7956421
Epoch: 7500     Loss: 3.7956421
Epoch: 8000     Loss: 3.7956421
Epoch: 8500     Loss: 3.7956421
Epoch: 9000     Loss: 3.7956421
Epoch: 9500     Loss: 3.7956421
Epoch: 10000    Loss: 3.7956421
Epoch: 10500    Loss: 3.7956421
Epoch: 11000    Loss: 3.7956421
Epoch: 11500    Loss: 3.7956421
Epoch: 12000    Loss: 3.7956421
Epoch: 12500    Loss: 3.7956421
Epoch: 13000    Loss: 3.7956421
Epoch: 13500    Loss: 3.7956421
Epoch: 14000    Loss: 3.7956421
Epoch: 14500    Loss: 3.7956421
Epoch: 15000    Loss: 3.7956421
Epoch: 15500    Loss: 3.7956421
Epoch: 16000    Loss: 3.7956421
Epoch: 16500    Loss: 3.7956421
Epoch: 17000    Loss: 3.7956421
Epoch: 17500    Loss: 3.7956421
Epoch: 18000    Loss: 3.7956421
Epoch: 18500    Loss: 3.7956421
Epoch: 19000    Loss: 3.7956421
Epoch: 19500    Loss: 3.7956421
Epoch: 20000    Loss: 3.7956421
Precision Score = 1.0 , Recall Score = 0.0031201248049922 , Accuracy = 0.7645107794361525
```

### tensorflow NN ( with Standard Scalar )

```
Epoch: 1000     Loss: 0.38121223
Epoch: 1500     Loss: 0.38116452
Epoch: 2000     Loss: 0.3811807
Epoch: 2500     Loss: 0.38116992
Epoch: 3000     Loss: 0.38127676
Epoch: 3500     Loss: 0.38117567
Epoch: 4000     Loss: 0.38118824
Epoch: 4500     Loss: 0.38116214
Epoch: 5000     Loss: 0.381172
Epoch: 5500     Loss: 0.38115495
Epoch: 6000     Loss: 0.38115293
Epoch: 6500     Loss: 0.38109973
Epoch: 7000     Loss: 0.38117087
Epoch: 7500     Loss: 0.38123262
Epoch: 8000     Loss: 0.38127348
Epoch: 8500     Loss: 0.38125506
Epoch: 9000     Loss: 0.38124475
Epoch: 9500     Loss: 0.38121778
Epoch: 10000    Loss: 0.38132796
Epoch: 10500    Loss: 0.38119754
Epoch: 11000    Loss: 0.38120332
Epoch: 11500    Loss: 0.38118866
Epoch: 12000    Loss: 0.38121364
Epoch: 12500    Loss: 0.38121584
Epoch: 13000    Loss: 0.38108385
Epoch: 13500    Loss: 0.38121992
Epoch: 14000    Loss: 0.3811831
Epoch: 14500    Loss: 0.3811838
Epoch: 15000    Loss: 0.3811719
Epoch: 15500    Loss: 0.3811667
Epoch: 16000    Loss: 0.3811848
Epoch: 16500    Loss: 0.38120025
Epoch: 17000    Loss: 0.38121003
Epoch: 17500    Loss: 0.38122192
Epoch: 18000    Loss: 0.38122833
Epoch: 18500    Loss: 0.3812673
Epoch: 19000    Loss: 0.38125345
Epoch: 19500    Loss: 0.38123137
Epoch: 20000    Loss: 0.38120726
Precision Score = 0.7028779894608836 , Recall Score = 0.45085803432137284 , Accuracy = 0.8252564338799828
time took  0:29:04.437904
```



### not using Standard Scalar

              precision    recall  f1-score   support
    
           0       0.80      0.99      0.89     12435
           1       0.91      0.19      0.31      3846
    
    accuracy                           0.80     16281
    
    macro avg       0.86      0.59      0.60     16281
    weighted avg       0.83      0.80      0.75     16281
    
    
    Accuracy Report
    
    0.8039432467293164
  

### using Standard Scalar

              precision    recall  f1-score   support
    
           0       0.87      0.94      0.91     12435
           1       0.75      0.56      0.64      3846
    
    accuracy                           0.85     16281
    
    macro avg       0.81      0.75      0.77     16281
    weighted avg       0.84      0.85      0.84     16281
    
    Accuracy Report
    
    0.8508691112339537


