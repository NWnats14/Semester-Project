from data_eval import *
from data_learn import *
from data_table import *
from data_util import *
from random import sample

def imbalanced_knn(table, instance, k, numerical_columns, nominal_columns, label_column):
    majority_instances = [row for row in table if row[label_column] == 0]
    minority_instances = [row for row in table if row[label_column] == 1]

    oversampled_minority = minority_instances + sample(minority_instances, len(majority_instances) - len(minority_instances))
    balanced_instances = majority_instances + oversampled_minority
    balanced_table = DataTable(table.columns())
    for inst in balanced_instances:
        balanced_table.append(inst.values())
    k_nearest = knn(balanced_table, instance, k, numerical_columns, nominal_columns)
    return k_nearest
    
    
def imbalanced_knn_eval(train, test, vote_fun, k, label_col, numeric_cols, nominal_cols=[]):
    """Returns a confusion matrix resulting from running knn over the
    given test set. 

    Args:
        train: The training set.
        test: The test set.
        vote_fun: The function to use for selecting labels.
        k: The k nearest neighbors for knn.
        label_col: The column to use for predictions. 
        numeric_cols: The columns compared using Euclidean distance.
        nominal_cols: The nominal columns for knn (match or no match).

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the given voting function returns multiple labels, the
        first such label is selected.

    """
    total_label = DataTable.combine(train, test, train.columns(),True)
    label_list = list(set(row[label_col] for row in total_label))
    label_list.sort()
    confusion_matrix = DataTable(['actual']+label_list)
    for i in label_list:
        confusion_matrix.append([i]+[0 for x in range(len(label_list))])
    for inst in test:
        neighbors = imbalanced_knn(train, inst, k, numeric_cols, nominal_cols,label_col     )
        instances =[]
        scores = []
        for key in neighbors.keys():
            instances += neighbors[key]    
            scores += [1-key for x in range(len(neighbors[key]))]
        predicted_label = vote_fun(instances,scores,label_col)[0]
        actual_label = inst[label_col]
        predicted_index=label_list.index(predicted_label)
        actual_index=label_list.index(actual_label)
        if predicted_label == actual_label:
            confusion_matrix.update(predicted_index, actual_label, confusion_matrix[predicted_index][actual_label]+1)
        else:
            confusion_matrix.update(actual_index, predicted_label, confusion_matrix[actual_index][predicted_label]+1)
    return confusion_matrix
