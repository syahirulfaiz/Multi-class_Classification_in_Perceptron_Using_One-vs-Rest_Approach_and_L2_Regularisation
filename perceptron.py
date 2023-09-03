import numpy
import re

COLUMN = 5

BIAS = 0
WEIGHT = numpy.zeros(COLUMN-1)

BIAS_1 = 0
BIAS_2 = 0
BIAS_3 = 0
WEIGHT_1 = numpy.zeros(COLUMN-1)
WEIGHT_2 = numpy.zeros(COLUMN-1)
WEIGHT_3 = numpy.zeros(COLUMN-1)

CHOICE = 0
REGULARISATION = 0

################################ FUNCTION DEFINITION ######################################

def load_data(kelas, file_name):
    
    global CHOICE    
    
    features = []
    target = []
    original_target =[]
    
    with open(file_name) as F:
        for line in F:
            p = line.strip().split(",")
            x = numpy.zeros(COLUMN-1)
            y = numpy.zeros(1)

            y = int(re.search('\d',p[COLUMN-1]).group(0))
            
            if CHOICE=='1':   
                if (y==1) | (y==2):
                    if (y==2) : y = -1
                else :
                    continue
            
            elif CHOICE=='2':
                if (y==2) | (y==3):
                    if (y==2) : y = 1
                    elif (y==3) : y = -1 
                else :
                    continue
            
            elif CHOICE=='3':
                if (y==1) | (y==3):
                    if (y==3) : y = -1 
                else :
                    continue
            
            elif (CHOICE =='4') | (CHOICE =='5'):
                if kelas =='1':
                    if (y==2) | (y==3):
                        y = -1 
                    else :
                        y = 1
                elif kelas =='2':    
                    if (y==1) | (y==3):
                        y = -1 
                    else :
                        y = 1
                elif kelas =='3':
                    if (y==1) | (y==2):
                        y = -1 
                    else :
                        y = 1

            target.append(y)
            original_target.append(p[COLUMN-1])
            for i in range(COLUMN-1):
                x[i] = float(p[i])  
            features.append(x)

    return features,target,original_target

def evaluation(desired_target, predicted_target):
    correct = 0
    for ith_row in range(len(desired_target)):
        if desired_target[ith_row]==predicted_target[ith_row]:
            correct = correct + 1        
    return correct/len(desired_target)*100


def evaluation_one_vs_rest(desired_target, predicted_target):
    correct_1 = 0
    sum_1 = 0
    correct_2 = 0
    sum_2 = 0
    correct_3 = 0
    sum_3 = 0
    
    for ith_row in range(len(desired_target)):
        if desired_target[ith_row]=='class-1':
            if desired_target[ith_row]==predicted_target[ith_row]:
                correct_1 = correct_1 + 1
            sum_1 = sum_1 + 1
        elif desired_target[ith_row]=='class-2':
            if desired_target[ith_row]==predicted_target[ith_row]:
                correct_2 = correct_2 + 1
            sum_2 = sum_2 + 1
        elif desired_target[ith_row]=='class-3':
            if desired_target[ith_row]==predicted_target[ith_row]:
                correct_3 = correct_3 + 1
            sum_3 = sum_3 + 1

    accuracy_1 = correct_1/sum_1*100
    accuracy_2 = correct_2/sum_1*100
    accuracy_3 = correct_3/sum_1*100
    
    return accuracy_1, accuracy_2, accuracy_3


def perceptron_train(kelas, features, target, max_iteration):
    
    global BIAS
    global BIAS_1
    global BIAS_2
    global BIAS_3
    
    for epoch in range(max_iteration):
        predicted_target = []
        
        for ith_row in range(len(features)):
            if kelas == '1':
                activation = numpy.dot(WEIGHT_1,features[ith_row]) + BIAS_1
            elif kelas == '2':
                activation = numpy.dot(WEIGHT_2,features[ith_row]) + BIAS_2
            elif kelas == '3':
                activation = numpy.dot(WEIGHT_3,features[ith_row]) + BIAS_3
            else:
                activation = numpy.dot(WEIGHT,features[ith_row]) + BIAS
                               
            y_activation = target[ith_row] * activation
            ##if ya<=0 (misclassified), then update weight and bias
            if y_activation <=0 :
                if kelas == '1':
                    for jth_column in range(COLUMN-1):
                        if (CHOICE == '5'):                            
                            WEIGHT_1[jth_column] = WEIGHT_1[jth_column] + target[ith_row] * features[ith_row][jth_column] - 2 * float(REGULARISATION) * WEIGHT_1[jth_column]
                        else :        
                            WEIGHT_1[jth_column] = WEIGHT_1[jth_column] + target[ith_row] * features[ith_row][jth_column] 
                    BIAS_1 = BIAS_1 + target[ith_row]
                elif kelas == '2':
                    for jth_column in range(COLUMN-1):
                        if (CHOICE == '5'):
                            WEIGHT_2[jth_column] = WEIGHT_2[jth_column] + target[ith_row] * features[ith_row][jth_column] - 2 * float(REGULARISATION) * WEIGHT_2[jth_column]
                        else :
                            WEIGHT_2[jth_column] = WEIGHT_2[jth_column] + target[ith_row] * features[ith_row][jth_column] 
                    BIAS_2 = BIAS_2 + target[ith_row]
                elif kelas == '3':
                    for jth_column in range(COLUMN-1):
                        if (CHOICE == '5'): 
                            WEIGHT_3[jth_column] = WEIGHT_3[jth_column] + target[ith_row] * features[ith_row][jth_column] - 2 * float(REGULARISATION) * WEIGHT_3[jth_column]
                        else :
                            WEIGHT_3[jth_column] = WEIGHT_3[jth_column] + target[ith_row] * features[ith_row][jth_column] 
                    BIAS_3 = BIAS_3 + target[ith_row]
                else:
                    for jth_column in range(COLUMN-1):
                        WEIGHT[jth_column] = WEIGHT[jth_column] + target[ith_row] * features[ith_row][jth_column] 
                    BIAS = BIAS + target[ith_row]
                
            predicted_target.append(numpy.sign(activation))
                 
    return predicted_target


def perceptron_test(features,desired_target):
    global BIAS

    correct_test = 0
    predicted_target = []
    
    for ith_row in range(len(features)):
        activation = numpy.dot(WEIGHT,features[ith_row]) + BIAS
        predicted_target.append(numpy.sign(activation))

    return predicted_target   


def perceptron_test_one_vs_all(features):
    
    predicted_target = []
    
    for ith_row in range(len(features)):

        activation_1 = (numpy.dot(WEIGHT_1,features[ith_row]) + BIAS_1)
        activation_2 = (numpy.dot(WEIGHT_2,features[ith_row]) + BIAS_2)
        activation_3 = (numpy.dot(WEIGHT_3,features[ith_row]) + BIAS_3)

        if (activation_1 > activation_2) & (activation_1 > activation_3):
            label = 'class-1'
        elif (activation_2 > activation_1) & (activation_2 > activation_3):
            label = 'class-2'
        elif (activation_3 > activation_1) & (activation_3 > activation_2):
            label = 'class-3'
            
        predicted_target.append(label)
        
    return predicted_target



def flow():
    global CHOICE
    CHOICE=input('Please choose classification options (1-5) :\n\n(1).class-1 vs class-2\n(2).class-2 vs class-3\n(3).class-1 vs class-3\n(4).1 vs rest\n(5).1 vs rest with L2 \n(0).exit()\n\nYour choice:')
    if (CHOICE != '1') & (CHOICE != '2') & (CHOICE != '3') & (CHOICE != '4') & (CHOICE != '5') & (CHOICE != '0') :
        flow()

################################ MAIN PROGRAM ######################################

flow()
if (CHOICE == '1') | (CHOICE == '2') | (CHOICE == '3') :
    F,C,Original_Class=load_data(0,'train.data')
    array_train = perceptron_train(0,F,C,20)
    print('\nperceptron_train array =',array_train)

    accuracy = evaluation(C, array_train)
    print('\naccuracy train =',accuracy,'%')

    F,C,Original_Class=load_data(0,'test.data')
    array_test = perceptron_test(F,C)
    print('\nperceptron_test array =',array_test)

    accuracy = evaluation(C, array_test)
    print('\naccuracy test =',accuracy,'%')

    if (CHOICE == '1') : print('\nweight vector: ',WEIGHT)
    
elif (CHOICE == '4') :
    F,C,Original_Class=load_data('1','train.data')
    array_train_1 = perceptron_train('1',F,C,20)
    F,C,Original_Class=load_data('2','train.data')
    array_train_2 = perceptron_train('2',F,C,20)
    F,C,Original_Class=load_data('3','train.data')
    array_train_3 = perceptron_train('3',F,C,20)

    array_train = perceptron_test_one_vs_all(F)
    print('\nperceptron_train array =',array_train)

    accuracy_1, accuracy_2, accuracy_3 = evaluation_one_vs_rest(Original_Class,array_train)
    print('\naccuracy train class-1 =',accuracy_1,'%')
    print('\naccuracy train class-2 =',accuracy_2,'%')
    print('\naccuracy train class-3 =',accuracy_3,'%')

    F,C,Original_Class=load_data(0,'test.data')
    array_test = perceptron_test_one_vs_all(F)
    print('\nperceptron_test array =',array_test)

    accuracy_1, accuracy_2, accuracy_3 = evaluation_one_vs_rest(Original_Class,array_test)
    print('\naccuracy test class-1 =',accuracy_1,'%')
    print('\naccuracy test class-2 =',accuracy_2,'%')
    print('\naccuracy test class-3 =',accuracy_3,'%')

elif (CHOICE == '5') :
    REGULARISATION = input('\nPlease input your REGULARISATION COEFFICIENT: ')

    print('the L2 regularisation coefficient=',REGULARISATION)
    
    F,C,Original_Class=load_data('1','train.data')
    array_train_1 = perceptron_train('1',F,C,20)
    F,C,Original_Class=load_data('2','train.data')
    array_train_2 = perceptron_train('2',F,C,20)
    F,C,Original_Class=load_data('3','train.data')
    array_train_3 = perceptron_train('3',F,C,20)

    array_train = perceptron_test_one_vs_all(F)
    print('\nperceptron_train array =',array_train)

    accuracy_1, accuracy_2, accuracy_3 = evaluation_one_vs_rest(Original_Class,array_train)
    print('\naccuracy train class-1 =',accuracy_1,'%')
    print('\naccuracy train class-2 =',accuracy_2,'%')
    print('\naccuracy train class-3 =',accuracy_3,'%')

    F,C,Original_Class=load_data(0,'test.data')
    array_test = perceptron_test_one_vs_all(F)
    print('\nperceptron_test array =',array_test)

    accuracy_1, accuracy_2, accuracy_3 = evaluation_one_vs_rest(Original_Class,array_test)
    print('\naccuracy test class-1 =',accuracy_1,'%')
    print('\naccuracy test class-2 =',accuracy_2,'%')
    print('\naccuracy test class-3 =',accuracy_3,'%')

    print('\nweight vector classifier 1 (class-1 vs rest)=',WEIGHT_1)
    print('weight vector classifier 2 (class-2 vs rest)=',WEIGHT_2)
    print('weight vector classifier 3 (class-3 vs rest)=',WEIGHT_3)
elif (CHOICE == '0') :
    exit()


