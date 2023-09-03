**Multi-class Classification in Perceptron Using One-vs-Rest Approach and L2 Regularisation**

We simulate the classification using Single Layer Perceptron enhanced with regularisation coefficient

We extend the perceptron\_train() function to update the weight vectors using the formula:

$$w^{k+1}=w^k+target\*input-2\*\lambda\*w^k$$

Where $\lambda$ is the regularization coefficient ={0.01, 0.1, 1.0, 10.0, 100.0}. After obtaining the three sets of weight vectors (and biases), we use the three weight vectors for multi-class one-vs-rest classification. These are the results of the experiment:

| |**Multi-class with L2 Regularisation Train Accuracy**|||||
| :- | :-: | :- | :- | :- | :- |
| |**regularisation coefficient (?)**|||||
|**class**|**0.01**|**0.1**|**1**|**10**|**100**|
|**class-1**|100%|100%|0%|0%|0%|
|**class-2**|0%|0%|0%|0%|0%|
|**class-3**|100%|100%|100%|100%|100%|

*Table 1: **Train** Accuracies among three classes on One-vs-Rest approach with L2 Regularisation*

| |**Multi-class with L2 Regularisation Test Accuracy**|||||
| :- | :-: | :- | :- | :- | :- |
| |**regularisation coefficient ($\lambda$)**|||||
|**class**|**0.01**|**0.1**|**1**|**10**|**100**|
|**class-1**|100%|100%|0%|0%|0%|
|**class-2**|0%|0%|0%|0%|0%|
|**class-3**|100%|100%|100%|100%|100%|

*Table 2: **Test**  Accuracies among three classes on One-vs-Rest approach with L2 Regularisation*

As shown in the two tables above, the regularization coefficient for **0.01** and **0.1** show that they have the same accuracy as the *multi-class classification without L2 regularisation*. Both class-1 and class-3 have the same **100%** accuracy on train and test dataset. However, the other three regularization coefficient (**1, 10, 100**) show *only 100% accuracy on class-3* in both train and test dataset. On the other hand, these coefficients produce 0% accuracy on either train and test dataset on class-1 and class-2. It shows that the two coefficient 0.01 and 0.1, give the best performance compared to other coefficients. 
