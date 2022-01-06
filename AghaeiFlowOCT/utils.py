import numpy as np
from sklearn.tree import _tree

def cart_to_rules(cart):
    """ Convert a fitted DecisionTreeClassifier to rules. """
    branch_rules = {}
    classification_rules = {}
    
    n_features_ = cart.n_features_
    classes_ = cart.classes_
    tree_ = cart.tree_
    
    def recurse(cart_node, rules_node):
        """ Recursively obtain branch/classification rules. """
        if tree_.feature[cart_node] != _tree.TREE_UNDEFINED:
            # Split at node in CART is of the form x[j] <= b; retrieve j and b
            j = tree_.feature[cart_node]
            b = tree_.threshold[cart_node]
            a = np.zeros(n_features_)
            a[j] = 1
            branch_rules[rules_node] = (a, b)
            recurse(tree_.children_left[cart_node], 2*rules_node)
            recurse(tree_.children_right[cart_node], 2*rules_node+1)
        else:
            value = tree_.value[cart_node]
            class_index = np.argmax(value)
            classification_rules[rules_node] = classes_[class_index]
    
    recurse(0, 1)
    
    return branch_rules, classification_rules

def extend_rules_to_full_tree(depth, branch_rules, classification_rules, n_features):
    """ A lot of the code assumes the rules are for a full binary tree of a given depth. """
    new_branch_rules = {}
    new_classification_rules = {}
    
    leaf_nodes = list(range(2**depth, 2**(depth+1)))
    def recurse(root, label):
        """ Recursively fill out an empty subtree. """
        if root in leaf_nodes:
            new_classification_rules[root] = label
        else:
            new_branch_rules[root] = (np.zeros(n_features), 1)
            recurse(2*root, label)
            recurse(2*root+1, label)
    
    for t in branch_rules:
        new_branch_rules[t] = branch_rules[t]
    
    for t in classification_rules:
        if t in leaf_nodes:
            new_classification_rules[t] = classification_rules[t]
        else:
            recurse(t, classification_rules[t])
    
    return new_branch_rules, new_classification_rules

def predict_with_rules(x, branch_rules, classification_rules):
    """ Classify a single observation using rules. """
    # Assume that if branch_rules is None, then classification_rules[1] returns some class label
    if branch_rules is None:
        return classification_rules[1]
    # Traverse the tree until we reach a leaf
    t = 1
    while t in branch_rules:
        a, b = branch_rules[t]
        t = 2*t if np.dot(a, x) <= b else 2*t+1
    return classification_rules[t]
