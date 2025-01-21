from NaiveBayes import NaiveBayes
#NaiveBayes class is general class for text classification 
#initialized by NaiveBayes([Category1 , Category2 , ... ,CategoryN])
#after defining categories N Category class will be created
#each can be accessed by NaiveBayes["CategoryI"]

#Training 
#NaiveBays.fit(Text , CategoryI) Will redirect the Text for CategoryI 

#CategoryI will breakdown the text into terms and add the frequency of each term
#to dict CategoryI.terms and will update data needed such as
#TotalDocCount , TotalTermCount

#Testing 
#NaiveBays.test(Text ,use_log ,  smoothing )
#will take Text and call all Category.Score(Text)
#which will calculate score for each , NaiveBayes will find max score
#max score of all categories will be the right choice for NaiveBays



#ham_count can be accessed by model["0"].TotalDocCount
#spam_count can be accessed by model["1"].TotalDocCount
#ham_fd can be accessed by model["0"].terms
#spam_fd can be accessed by model["1"].terms

def nb_train(x , y) -> NaiveBayes:
    """
    Trains a NaiveBayes model on the given data.

    Parameters:
    x (list): A list of feature vectors or documents for training.
    y (list): A list of labels ('0' or '1') corresponding to each document in x.

    Returns:
    NaiveBayes: A trained NaiveBayes model with classes '0' and '1'.
    """
    
    model = NaiveBayes(["0" , "1"])
    model.fit_all(x , y)
    return model 


def nb_test(docs , model : NaiveBayes , use_log = False , smoothing = False ) -> list[str]:
    """
    Tests a trained NaiveBayes model on a list of documents and returns the predicted labels.

    Parameters:
    docs (list): A list of feature vectors or documents to be classified.
    model (NaiveBayes): A trained NaiveBayes model.
    use_log (bool): Whether to use logarithmic probabilities for classification (default: False).
    smoothing (bool): Whether to apply smoothing to the probabilities (default: False).

    Returns:
    list[str]: A list of predicted labels ('0' or '1') for each document in docs.
    """
    Tested_List = []
    for doc in docs:
        Tested_List.append(model.test(doc , use_log , smoothing))
    return Tested_List


def f_score(y_true, y_pred):
    T_p = T_n = F_p = F_n = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]: 
            if y_true[i] == '1':  
                T_p += 1
            else:  
                T_n += 1
        else:
            if y_true[i] == '1' and y_pred[i] != '1':  
                F_n += 1
            elif y_true[i] != '1' and y_pred[i] == '1': 
                F_p += 1
    
    precision = T_p / (T_p + F_p) if (T_p + F_p) > 0 else 0
    recall = T_p / (T_p + F_n) if (T_p + F_n) > 0 else 0
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f_score

#---------------------------------------------------------Project   
#importing load_data  



# from DataBase import load_data


#split the data into pair of Text - category for train
x_train, y_train = load_data("project2/SPAM_training_set")
#Feeding Data into model
model = nb_train(x_train, y_train)
#split the data into pair of Text - category for train
x_test, y_test = load_data("project2/SPAM_test_set")
#applying model on test and returning list of predicted 

y_pred1 = nb_test(x_test, model, use_log = True, smoothing = True)
y_pred2 = nb_test(x_test, model, use_log = True, smoothing = False)
y_pred3 = nb_test(x_test, model, use_log = False, smoothing = True)
y_pred4 = nb_test(x_test, model, use_log = False, smoothing = False)

print(f"smoothing = True , log = True")
print(f_score(y_test,y_pred1))
print(f"smoothing = True , log = False")
print(f_score(y_test,y_pred2))
print(f"smoothing = False , log = True")
print(f_score(y_test,y_pred3))
print(f"smoothing = False , log = False")
print(f_score(y_test,y_pred4))






