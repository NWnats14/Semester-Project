"""Machine learning algorithm evaluation functions. 

NAME: Nick Wunderle
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import *
from data_util import *
from data_learn import *
from math import ceil
from random import randint



#----------------------------------------------------------------------
# HW-8
#----------------------------------------------------------------------

def bootstrap(table): 
    """Creates a training and testing set using the bootstrap method.

    Args: 
        table: The table to create the train and test sets from.

    Returns: The pair (training_set, testing_set)

    """
    train_index = [randint(0, table.row_count() - 1) for i in range(table.row_count())]
    train_set = table.rows(train_index)
    
    test_index = list(set(range(table.row_count())) - set(train_index))
    test_set = table.rows(test_index)
    
    return train_set, test_set



def stratified_holdout(table, label_col, test_set_size):
    """Partitions the table into a training and test set using the holdout
    method such that the test set has a similar distribution as the
    table.

    Args:
        table: The table to partition.
        label_col: The column with the class labels. 
        test_set_size: The number of rows to include in the test set.

    Returns: The pair (training_set, test_set)

    """
    labels, counts = frequencies(table, label_col)
    label_dist = []
    for count in counts:
        label_dist.append(count/table.row_count())
    
    parts = partition(table, [label_col])
    train_table = DataTable(table.columns())
    test_table = DataTable(table.columns())
    
    index = 0
    for part in parts:
        train_part, test_part = holdout(part, int(label_dist[index]*test_set_size))
        train_table = union_all([train_table, train_part])
        test_table = union_all([test_table, test_part])
        index+=1
    return train_table, test_table
    
    
def tdidt_eval_with_tree(dt_root, test, label_col):
    """Evaluates the given test set using tdidt over the training
    set, returning a corresponding confusion matrix.

    Args:
       td_root: The decision tree to use.
       test: The testing data set.
       label_col: The column being predicted.
       columns: The categorical columns

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the naive bayes returns multiple labels for a given test
        instance, the first such label is selected.

    """
    #TODO
    return None



def random_forest(table, remainder, F, M, N, label_col, columns):
    """Returns a random forest build from the given table. 
    
    Args:
        table: The original table for cleaning up the decision tree.
        remainder: The table to use to build the random forest.
        F: The subset of columns to use for each classifier.
        M: The number of unique accuracy values to return.
        N: The total number of decision trees to build initially.
        label_col: The column with the class labels.
        columns: The categorical columns used for building the forest.

    Returns: A list of (at most) M pairs (tree, accuracy) consisting
        of the "best" decision trees and their corresponding accuracy
        values. The exact number of trees (pairs) returned depends on
        the other parameters (F, M, N, and so on).

    """
    train_set, test_set = stratified_holdout(table, label_col, (1/3)*table.row_count())
    boot_samples = bootstrap(remainder)
    initial_trees = []
    for sample in boot_samples:
        initial_trees.append(tdidt_F(sample, label_col, F, columns))
    for tree in initial_trees:
        print(tdidt_eval_with_tree(tree, test_set, label_col))
    



def random_forest_eval(table, train, test, F, M, N, label_col, columns):
    """Builds a random forest and evaluate's it given a training and
    testing set.

    Args: 
        table: The initial table.
        train: The training set from the initial table.
        test: The testing set from the initial table.
        F: Number of features (columns) to select.
        M: Number of trees to include in random forest.
        N: Number of trees to initially generate.
        label_col: The column with class labels. 
        columns: The categorical columns to use for classification.

    Returns: A confusion matrix containing the results. 

    Notes: Assumes weighted voting (based on each tree's accuracy) is
        used to select predicted label for each test row.

    """
    #TODO
    pass


#----------------------------------------------------------------------
# HW-7
#----------------------------------------------------------------------


def tdidt_eval(train, test, label_col, columns):
    """Evaluates the given test set using tdidt over the training
    set, returning a corresponding confusion matrix.

    Args:
       train: The training data set.
       test: The testing data set.
       label_col: The column being predicted.
       columns: The categorical columns

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the naive bayes returns multiple labels for a given test
        instance, the first such label is selected.

    """
    #TODO
    pass


def tdidt_stratified(table, k_folds, label_col, columns):
    """Evaluates tdidt prediction approach over the table using stratified
    k-fold cross validation, returning a single confusion matrix of
    the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        columns: The categorical columns for tdidt. 

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    #TODO
    pass


#----------------------------------------------------------------------
# HW-6
#----------------------------------------------------------------------

def stratify(table, label_column, k):
    """Returns a list of k stratified folds as data tables from the given
    data table based on the label column.

    Args:
        table: The data table to partition.
        label_column: The column to use for the label. 
        k: The number of folds to return. 

    Note: Does not randomly select instances for the folds, and
        instead produces folds in order of the instances given in the
        table.

    """
    labels = distinct_values(table, label_column)
    partitions = partition(table, [label_column])
    folds = [DataTable(table.columns()) for i in range(k)]
    for label in labels:
        part_table = partitions[labels.index(label)]
        for i, row in enumerate(part_table):
            fold_index = i % k
            folds[fold_index].append(row.values())
    return folds


def union_all(tables):
    """Returns a table containing all instances in the given list of data
    tables.

    Args:
        tables: A list of data tables. 

    Notes: Returns a new data table that contains all the instances of
       the first table in tables, followed by all the instances in the
       second table in tables, and so on. The tables must have the
       exact same columns, and the list of tables must be non-empty.

    """
    if tables == []:
        raise ValueError('no tables given')
    number_cols = set()
    col_order = []
    for table in tables:
        number_cols.add(table.column_count())
        col_order.append(table.columns())
    for col in range(len(col_order)):
        if col_order[col] != col_order[col-1]:
            raise ValueError('mismatched column names')
    if len(number_cols) != 1:
        raise ValueError('tables do not have the same number of columns')
    new_table = DataTable(tables[0].columns())
    for table in tables:
        if table.row_count() > 0:
            for row in table:
                new_table.append(row.values())
    return new_table


def naive_bayes_eval(train, test, label_col, continuous_cols, categorical_cols=[]):
    """Evaluates the given test set using naive bayes over the training
    set, returning a corresponding confusion matrix.

    Args:
       train: The training data set.
       test: The testing data set.
       label_col: The column being predicted.
       continuous_cols: The continuous columns (estimated via PDF)
       categorical_cols: The categorical columns

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the naive bayes returns multiple labels for a given test
        instance, the first such label is selected.

    """
    total_label = DataTable.combine(train, test, train.columns(),True)
    label_list = list(set(row[label_col] for row in total_label))
    label_list.sort()
    confusion_matrix = DataTable(['actual']+label_list)
    for i in label_list:
        confusion_matrix.append([i]+[0 for x in range(len(label_list))])
    for inst in test:
        predicted_label = naive_bayes(train, inst, label_col, continuous_cols, categorical_cols)[0][0]
        actual_label = inst[label_col]
        predicted_index=label_list.index(predicted_label)
        actual_index=label_list.index(actual_label)
        if predicted_label == actual_label:
            confusion_matrix.update(predicted_index, actual_label, confusion_matrix[predicted_index][actual_label]+1)
        else:
            confusion_matrix.update(actual_index, predicted_label, confusion_matrix[actual_index][predicted_label]+1)
    return confusion_matrix


def naive_bayes_stratified(table, k_folds, label_col, cont_cols, cat_cols=[]):
    """Evaluates naive bayes over the table using stratified k-fold cross
    validation, returning a single confusion matrix of the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        cont_cols: The continuous columns for naive bayes. 
        cat_cols: The categorical columns for naive bayes. 

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    label_list = list(set(row[label_col] for row in table))
    label_list.sort()
    confusion_matrix = DataTable(['actual']+label_list)
    for i in label_list:
        confusion_matrix.append([i]+[0 for x in range(len(label_list))])
    folds = stratify(table, label_col, k_folds)
    fold_index = 0
    for fold in folds:
        temp = folds.pop(fold_index)
        new_table = union_all(folds)
        for inst in fold:
            predicted_label = naive_bayes(new_table, inst, label_col, cont_cols, cat_cols)[0][0]
            actual_label = inst[label_col]
            predicted_index=label_list.index(predicted_label)
            actual_index=label_list.index(actual_label)
            if predicted_label == actual_label:
                confusion_matrix.update(predicted_index, actual_label, confusion_matrix[predicted_index][actual_label]+1)
            else:
                confusion_matrix.update(actual_index, predicted_label, confusion_matrix[actual_index][predicted_label]+1)
        folds.insert(fold_index, temp)
        fold_index += 1
    return confusion_matrix
        
        
def knn_stratified(table, k_folds, label_col, vote_fun, k, num_cols, nom_cols=[]):
    """Evaluates knn over the table using stratified k-fold cross
    validation, returning a single confusion matrix of the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        vote_fun: The voting function to use with knn.
        num_cols: The numeric columns for knn.
        nom_cols: The nominal columns for knn.

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    label_list = list(set(row[label_col] for row in table))
    label_list.sort()
    confusion_matrix = DataTable(['actual']+label_list)
    for i in label_list:
        confusion_matrix.append([i]+[0 for x in range(len(label_list))])
    folds = stratify(table, label_col, k_folds)
    fold_index = 0
    for fold in folds:
        temp = folds.pop(fold_index)
        new_table = union_all(folds)
        for inst in fold:
            neighbors = knn(new_table, inst, k, num_cols, nom_cols)
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
        folds.insert(fold_index, temp)
        fold_index += 1
    return confusion_matrix


#----------------------------------------------------------------------
# HW-5
#----------------------------------------------------------------------

def holdout(table, test_set_size):
    """Partitions the table into a training and test set using the holdout method. 

    Args:
        table: The table to partition.
        test_set_size: The number of rows to include in the test set.

    Returns: The pair (training_set, test_set)

    """
    train_table = table.copy()
    test_table = DataTable(table.columns())
    for i in range(test_set_size):
        rand_row = randint(0, train_table.row_count()-1)
        test_table.append(train_table[rand_row].values())
        del train_table[rand_row]
    return train_table, test_table


def knn_eval(train, test, vote_fun, k, label_col, numeric_cols, nominal_cols=[]):
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
    label_list=distinct_values(test,label_col)
    label_list.sort()
    confusion_matrix = DataTable(['actual']+label_list)
    for i in label_list:
        confusion_matrix.append([i]+[0 for x in range(len(label_list))])
    for inst in test:
        neighbors = knn(train, inst, k, numeric_cols, nominal_cols)
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


def accuracy(confusion_matrix, label):
    """Returns the accuracy for the given label from the confusion matrix.
    
    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the accuracy of.

    """
    true_p, true_n, false_p, false_n = get_counts(confusion_matrix, label)       
    return (true_p+true_n)/(true_p+true_n+false_n+false_p)


def precision(confusion_matrix, label):
    """Returns the precision for the given label from the confusion
    matrix.

    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the precision of.

    """
    true_p, true_n, false_p, false_n = get_counts(confusion_matrix, label)
    return true_p/(true_p+false_p)
    

def recall(confusion_matrix, label): 
    """Returns the recall for the given label from the confusion matrix.

    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the recall of.

    """
    true_p, true_n, false_p, false_n = get_counts(confusion_matrix, label)
    return true_p/(true_p+false_n)
    

def get_counts(confusion_matrix, label):
    true_p = 0
    true_n = 0
    false_p = 0
    false_n = 0
    
    for row in confusion_matrix:
        if row['actual'] == label:
            true_p = row[label]
            for col in confusion_matrix.columns():
                if col != label and col != 'actual':
                    false_n += row[col]
        else:
            for col in confusion_matrix.columns():
                if col == label:
                    false_p += row[col]
                elif col != 'actual' and col != label:
                    true_n += row[col] 
    return true_p, true_n, false_p, false_n


