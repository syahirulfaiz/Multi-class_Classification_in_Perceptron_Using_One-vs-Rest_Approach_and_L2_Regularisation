**Multi-class Classification in Perceptron Using One-vs-Rest Approach and L2 Regularisation**

We simulate the classification using Single Layer Perceptron enhanced with regularisation coefficient

We extend the perceptron\_train() function to update the weight vectors using the formula:

$$w^{k+1}=w^k+target\*input-2\*\lambda\*w^k$$

Where $\lambda$ is the regularization coefficient ={0.01, 0.1, 1.0, 10.0, 100.0}. After obtaining the three sets of weight vectors (and biases), we use the three weight vectors for multi-class one-vs-rest classification. These are the results of the experiment:

| |**Multi-class with L2 Regularisation Train Accuracy**|||||
| :- | :-: | :- | :- | :- | :- |
| |**regularisation coefficient ($\lambda$)**|||||
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


## HOW TO ##

1. Extract the zip and locate the 'program' directory/folder.
2. Open the terminal (Unix) or command prompt (Windows)
3. Change directory using 'cd' to the 'program' folder.
4. Type 'py perceptron.py' or 'python perceptron.py' in the command line.
5. When program is running, we can choose one of 5 (five) choices to produce the result required by the assignment:

a. choice (1) to obtain answer for assignment number 4 and 5
b. choice (2) and (3) to obtain answer number 4
c. choice (4) to obtain answer number 6
d. choice (5) to obtain answer number 7

8. For example, when we choose (1), we will see 'perceptron_train array', this is the predicted array result of the training process; 
'accuracy train' is the train classification accuracy for each classifier;
'perceptron_test array' is the Predicted array result of the testing process; 
'accuracy test' is the test classification accuracy for each classifier;
and 'weight vector' is the updated weight array (for answer number 5)

9. Same example applies for the choice (2) or (3)

10. When we choose (4) or (5), we will see 'accuracy train class-x' or 'accuracy test class-x' where this denote the train or test accuracy for each class x={1,2,3}.

11. When we choose (5), we need to specify the regularisation coefficient={0.01, 0.1, 1, 10, 100}, and we will have the answer for number 7.


