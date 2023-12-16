"""Machine learning algorithm implementations.

NAME: Nick Wunderle
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import *
from data_util import *
from decision_tree import *

from random import randint
import math


#----------------------------------------------------------------------
# HW-8
#----------------------------------------------------------------------


def random_subset(F, columns):
    """Returns F unique column names from the given list of columns. The
    column names are selected randomly from the given names.

    Args: 
        F: The number of columns to return.
        columns: The columns to select F column names from.

    Notes: If F is greater or equal to the number of names in columns,
       then the columns list is just returned.

    """
    if F >= len(columns):
        return columns
        
    new_subset = []
    for i in range(F):
        random_col = columns[randint(0, len(columns)-1)]
        if random_col not in new_subset:
            new_subset.append(random_col)
        else:
            i-=1
    return new_subset



def tdidt_F(table, label_col, F, columns): 
    """Returns an initial decision tree for the table using information
    gain, selecting a random subset of size F of the columns for
    attribute selection. If fewer than F columns remain, all columns
    are used in attribute selection.

    Args:
        table: The table to build the decision tree from. 
        label_col: The column containing class labels. 
        F: The number of columns to randomly subselect
        columns: The categorical columns. 

    Notes: The resulting tree can contain multiple leaf nodes for a
        particular attribute value, may have attributes without all
        possible attribute values, and in general is not pruned.

    """
    if table.row_count() == 0:
        return None
    if same_class(table, label_col):
        return build_leaves(table, label_col)
    if not columns:
        return build_leaves(table, label_col)
    entropy_values = calc_e_new(table, label_col, random_subset(F, columns))
    best_e = min(entropy_values.keys())
    best_attribute = entropy_values[best_e][0]
    partitions = partition(table, best_attribute)
    if len(partitions)==1:
        return build_leaves(partitions[0], label_col)
    node = AttributeNode(name=best_attribute, values={})
    for subset in partitions:
        child_tree = tdidt(subset, label_col, columns)
        node.values[subset[0][best_attribute]] = child_tree
    return node



def closest_centroid(centroids, row, columns):
    """Given k centroids and a row, finds the centroid that the row is
    closest to.

    Args:
        centroids: The list of rows serving as cluster centroids.
        row: The row to find closest centroid to.
        columns: The numerical columns to calculate distance from. 
    
    Returns: The index of the centroid the row is closest to. 

    Notes: Uses Euclidean distance (without the sqrt) and assumes
        there is at least one centroid.

    """
    distances = {}
    for index, inst in enumerate(centroids):
        temp_dist = 0
        for col in columns:
            temp_dist += (row[col] - inst[col])**2
        distances.setdefault(temp_dist,[]).append(index)
    distances = sorted(distances.items())
    return distances[0][1][0]
    
        
def select_k_random_centroids(table, k):
    """Returns a list of k random rows from the table to serve as initial
    centroids.

    Args: 
        table: The table to select rows from.
        k: The number of rows to select values from.
    
    Returns: k unique rows. 

    Notes: k must be less than or equal to the number of rows in the table. 

    """
    if k > table.row_count():
        raise ValueError('not enough instances')
    random_centroids = []
    if k == 0:
        return random_centroids
    index = 0
    while index < k:
        random_row = table[randint(0, table.row_count() -1)]
        if random_row not in random_centroids:
            random_centroids.append(random_row)
            index+=1
    return random_centroids
    
    
def k_means(table, centroids, columns): 
    """Returns k clusters from the table using the initial centroids for
    the given numerical columns.

    Args:
        table: The data table to build the clusters from.
        centroids: Initial centroids to use, where k is length of centroids.
        columns: The numerical columns for calculating distances.

    Returns: A list of k clusters, where each cluster is represented
        as a data table.

    Notes: Assumes length of given centroids is number of clusters k to find.

    """
    k = len(centroids)
    
    new_cents = [DataRow(table.columns(), [-1 for x in range(len(table.columns()))]) for y in range(k)]
    #print(new_cents)
    #print(centroids)
    clusters = [DataTable(table.columns()) for i in range(k)]
    for i, inst in enumerate(centroids):
        clusters[i].append(inst.values())
    for row in table:
        if row not in centroids:
            closest = closest_centroid(centroids, row, columns)
            clusters[closest].append(row.values())
    for i, cluster in enumerate(clusters):
        for col in cluster.columns():
            new_cents[i][col] = mean(cluster, col)
    #print(new_cents)
    return clusters



def tss(clusters, columns):
    """Return the total sum of squares (tss) for each cluster using the
    given numerical columns.

    Args:
        clusters: A list of data tables serving as the clusters
        columns: The list of numerical columns for determining distances.
    
    Returns: A list of tss scores for each cluster. 

    """
    if clusters == []:
        return []
    total_sums = []
    centroids = [DataRow(clusters[0].columns(), [0 for col in clusters[0].columns()]) for i in range(len(clusters))]
    print(centroids)
    for i, cluster in enumerate(clusters):
        for col in columns:
            centroids[i][col] = mean(cluster, col)
    for i, cluster in enumerate(clusters):
        dist = 0
        for row in cluster:
            for col in columns:
                dist += (row[col] - centroids[i][col])**2
        total_sums.append(dist)
    return total_sums
        
            
#----------------------------------------------------------------------
# HW-7
#----------------------------------------------------------------------


def same_class(table, label_col):
    """Returns true if all of the instances in the table have the same
    labels and false otherwise.

    Args: 
        table: The table with instances to check. 
        label_col: The column with class labels.

    """
    col_values = column_values(table, label_col)
    if len(set(col_values)) == 1:
        return True
    return False


def build_leaves(table, label_col):
    """Builds a list of leaves out of the current table instances.
    
    Args: 
        table: The table to build the leaves out of.
        label_col: The column to use as class labels

    Returns: A list of LeafNode objects, one per label value in the
        table.

    """
    leaves = []
    total_count = table.row_count()
    labels, counts = frequencies(table, label_col)
    for label in labels:
        leaves.append(LeafNode(label, counts[labels.index(label)], total_count))
    return leaves


def calc_e_new(table, label_col, columns):
    """Returns entropy values for the given table, label column, and
    feature columns (assumed to be categorical). 

    Args:
        table: The table to compute entropy from
        label_col: The label column.
        columns: The categorical columns to use to compute entropy from.

    Returns: A dictionary, e.g., {e1:['a1', 'a2', ...], ...}, where
        each key is an entropy value and each corresponding key-value
        is a list of attributes having the corresponding entropy value. 

    Notes: This function assumes all columns are categorical.

    """ 
    entropy_values = {}    
    
    if table.row_count() == 0:
        return {0 : [col for col in columns]}
    for col in columns:
        part_e = []
        table_part = partition(table, col)
        total_e = 0
        for sub_table in table_part:
            entropy = 0
            part_labels = distinct_values(sub_table, label_col)
            label_counts = {label:0 for label in part_labels}
            for label in part_labels:
                for row in sub_table:
                    if row[label_col] == label:
                        label_counts[label] +=1
            label_prob = {x:label_counts[x]/sub_table.row_count() for x in part_labels}
            for key in label_prob:
                entropy -= label_prob[key] * math.log(label_prob[key], 2)
            part_e.append(entropy)
        for i in range(len(part_e)):
            total_e += part_e[i]*(table_part[i].row_count()/table.row_count())
        entropy_values.setdefault(total_e, []).append(col)      
    return entropy_values


def tdidt(table, label_col, columns): 
    """Returns an initial decision tree for the table using information
    gain.

    Args:
        table: The table to build the decision tree from. 
        label_col: The column containing class labels. 
        columns: The categorical columns. 

    Notes: The resulting tree can contain multiple leaf nodes for a
        particular attribute value, may have attributes without all
        possible attribute values, and in general is not pruned.

    """
    if table.row_count() == 0:
        return None
    if same_class(table, label_col):
        return build_leaves(table, label_col)
    if not columns:
        return build_leaves(table, label_col)
    entropy_values = calc_e_new(table, label_col, columns)
    best_e = min(entropy_values.keys())
    best_attribute = entropy_values[best_e][0]
    partitions = partition(table, best_attribute)
    if len(partitions)==1:
        return build_leaves(partitions[0], label_col)
    node = AttributeNode(name=best_attribute, values={})
    for subset in partitions:
        child_tree = tdidt(subset, label_col, columns)
        node.values[subset[0][best_attribute]] = child_tree
    return node


def summarize_instances(dt_root):
    """Return a summary by class label of the leaf nodes within the given
    decision tree.

    Args: 
        dt_root: The subtree root whose (descendant) leaf nodes are summarized. 

    Returns: A dictionary {label1: count, label2: count, ...} of class
        labels and their corresponding number of instances under the
        given subtree root.

    """
    summary = {}
    def traverse_tree(node):
        if isinstance(node, LeafNode):
            label = node.label
            count = node.count
            summary[label] = summary.get(label, 0) + count
        elif isinstance(node, list):
            for leaf_node in node:
                traverse_tree(leaf_node)
        elif isinstance(node, AttributeNode):
            for value, child_node in node.values.items():
                traverse_tree(child_node)
                
    traverse_tree(dt_root)
    return summary

def resolve_leaf_nodes(dt_root):
    """Modifies the given decision tree by combining attribute values with
    multiple leaf nodes into attribute values with a single leaf node
    (selecting the label with the highest overall count).

    Args:
        dt_root: The root of the decision tree to modify.

    Notes: If an attribute value contains two or more leaf nodes with
        the same count, the first leaf node is used.

    """
    #TODO
    print(dt_root)
    if isinstance(dt_root, LeafNode):
        return dt_root
    elif isinstance(dt_root, list):
        return [LeafNode(l.label, l.count, l.total) for l in dt_root]
        
    new_dt_root = AttributeNode(dt_root.name, {})
    for val, child in dt_root.values.items():
        new_dt_root.values[val] = resolve_leaf_nodes(child)
    if isinstance(new_dt_root, AttributeNode):
        for val, child in new_dt_root.values.items():
            label_counts = {}
            label_total = 0
            if isinstance(child, AttributeNode):
                for new_child in child.values.values():
                    for leaf in new_child:
                        if leaf.label not in label_counts:
                            label_counts[leaf.label] = leaf.count
                            label_total = leaf.total
                        else:
                            label_counts[leaf.label].count += leaf.count
            if isinstance(child, list):
                for leaf in child:
                    if leaf.label not in label_counts:
                        label_counts[leaf.label] = leaf.count
                        label_total = leaf.total
                    else:
                        label_counts[leaf.label].count += leaf.count
            label_counts = sorted(label_counts.items(), key=lambda x:x[1], reverse=True)
            new_dt_root.values[val] = [LeafNode(label_counts[0][0], label_counts[0][1], label_total)]
    return new_dt_root
        
    


def resolve_attribute_values(dt_root, table):
    """Return a modified decision tree by replacing attribute nodes
    having missing attribute values with the corresponding summarized
    descendent leaf nodes.
    
    Args:
        dt_root: The root of the decision tree to modify.
        table: The data table the tree was built from. 

    Notes: The table is only used to obtain the set of unique values
        for attributes represented in the decision tree.

    """
    #TODO
    pass


def tdidt_predict(dt_root, instance): 
    """Returns the class for the given instance given the decision tree. 

    Args:
        dt_root: The root node of the decision tree. 
        instance: The instance to classify. 

    Returns: A pair consisting of the predicted label and the
       corresponding percent value of the leaf node.

    """
    #TODO
    pass
    
#----------------------------------------------------------------------
# HW-6
#----------------------------------------------------------------------

def naive_bayes(table, instance, label_col, continuous_cols, categorical_cols=[]):
    """Returns the labels with the highest probabibility for the instance
    given table of instances using the naive bayes algorithm.

    Args:
       table: A data table of instances to use for estimating most probably labels.
       instance: The instance to classify.
       continuous_cols: The continuous columns to use in the estimation.
       categorical_cols: The categorical columns to use in the estimation. 

    Returns: A pair (labels, prob) consisting of a list of the labels
        with the highest probability and the corresponding highest
        probability.

    """
    label_values = set(column_values(table, label_col))
    label_counts = {label:0 for label in label_values}
    for label in label_values:
        for row in table:
            if row[label_col] == label:
                label_counts[label] +=1
    label_prob = {label:0 for label in label_values}
    for label in label_values:
        label_prob[label] = label_counts[label]/table.row_count()
    cond_prob = {label:0 for label in label_values}
    for label in label_values:
        temp_prob = 1
        for col in categorical_cols:
            match_count = 0
            for row in table:
                if row[label_col] == label:
                    if row[col] == instance[col]:
                        match_count +=1
            temp_prob = temp_prob * (match_count/label_counts[label])
        cond_prob[label] = temp_prob
    continuous_prob = {label:0 for label in label_values}
    table_part = partition(table, [label_col])
    if continuous_cols != []:
        for table_iter in table_part:
            label = distinct_values(table_iter, label_col)[0]
            temp_prob = 1
            for col in continuous_cols:
                means = mean(table_iter, col)
                sdev = std_dev(table_iter, col)
                temp_prob *= gaussian_density(instance[col], means, sdev)
            continuous_prob[label] = temp_prob
            
    total_prob = {label:0 for label in label_values}
    for label in label_values:
        if continuous_cols != []:
            total_prob[label] = label_prob[label]*cond_prob[label]*continuous_prob[label]
        else:
            total_prob[label] = label_prob[label]*cond_prob[label]
    high_prob = -1
    list_pred = []
    for key in total_prob.keys():
        if total_prob[key] > high_prob:
            high_prob = total_prob[key]
            list_pred = [key]         
        elif total_prob[key] == high_prob:
            list_pred.append(key)
        pred = (list_pred, high_prob)
    return pred
    

def gaussian_density(x, mean, sdev):
    """Return the probability of an x value given the mean and standard
    deviation assuming a normal distribution.

    Args:
        x: The value to estimate the probability of.
        mean: The mean of the distribution.
        sdev: The standard deviation of the distribution.

    """
    if sdev != 0:
        coef = 1/(math.sqrt(2*math.pi)*sdev)
        exponent = -1*((x-mean)**2/(2*sdev**2))
    return (coef*math.e**exponent)


#----------------------------------------------------------------------
# HW-5
#----------------------------------------------------------------------

def knn(table, instance, k, numerical_columns, nominal_columns=[]):
    """Returns the k closest distance values and corresponding table
    instances forming the nearest neighbors of the given instance. 

    Args:
        table: The data table whose instances form the nearest neighbors.
        instance: The instance to find neighbors for in the table.
        k: The number of closest distances to return.
        numerical_columns: The numerical columns to use for comparison.
        nominal_columns: The nominal columns to use for comparison (if any).

    Returns: A dictionary with k key-value pairs, where the keys are
        the distances and the values are the corresponding rows.

    Notes: 
        The numerical and nominal columns must be disjoint. 
        The numerical and nominal columns must be valid for the table.
        The resulting score is a combined distance without the final
        square root applied.

    """
    
    for col in numerical_columns:
        if col in nominal_columns:
            raise ValueError('column cannot be numerical and nominal')
        if col not in table.columns():
            raise ValueError('not valid column name')
    for col in nominal_columns:
        if col not in table.columns():
            raise ValueError('not valid column name')
    
    distances = {}
    for row in table:
        temp_dist = 0
        for col in numerical_columns:
            temp_dist += (instance[col] - row[col])**2
        for col in nominal_columns:
            if instance[col] != row[col]:
                temp_dist += 1
        distances.setdefault(temp_dist,[]).append(row)
        
    sorted_dist = sorted(distances.items())
    k_nearest = {}
    count = 0
    
    for distance, neighbor in sorted_dist:
        k_nearest[distance] = neighbor
        count += 1
        if count >= k:
            break
    
    return k_nearest
        


def majority_vote(instances, scores, labeled_column):
    """Returns the labels in the given instances that occur the most.

    Args:
        instances: A list of instance rows.
        labeled_column: The column holding the class labels.

    Returns: A list of the labels that occur the most in the given
    instances.

    """
    check_set = set()
    for row in instances:
        check_set.add(row[labeled_column])
    label_counts = {}
    high = 0
    for i in check_set:
        count = 0
        for row in instances:
            if row[labeled_column] == i:
                count += 1
        label_counts.setdefault(count, []).append(i)
        if count > high:
            high = count
    sorted_count = dict(sorted(label_counts.items()))
    return sorted_count[high]


def weighted_vote(instances, scores, labeled_column):
    """Returns the labels in the given instances with the largest total
    sum of corresponding scores.

    Args:
        instances: The list of instance rows.
        scores: The corresponding scores for each instance.
        labeled_column: The column with class labels.

    """
    weighted_dict = {}
    for index, instance in enumerate(instances):
        label = instance[labeled_column]
        score = scores[index]
        if label in weighted_dict:
            weighted_dict[label] += score
        else:
            weighted_dict[label] = score     
    high_label = []
    high_score = 0
    sorted_weight = dict(sorted(weighted_dict.items(),key = lambda x:x[1],reverse=True))
    for label, score in sorted_weight.items():
        if score >= high_score:
            high_score = score
            high_label.append(label)   
    return high_label

