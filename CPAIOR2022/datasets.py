import math
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

class CandidateThresholdBinarizer(TransformerMixin, BaseEstimator):
    """ Binarize continuous data using candidate thresholds.
    
    For each feature, sort observations by values of that feature, then find
    pairs of consecutive observations that have different class labels and
    different feature values, and define a candidate threshold as the average of
    these two observations’ feature values.
    
    Attributes
    ----------
    candidate_thresholds_ : dict mapping features to list of thresholds
    """
    
    def fit(self, X, y):
        """ Finds all candidate split thresholds for each feature.
        
        Parameters
        ----------
        X : pandas DataFrame with observations, X.columns used as feature names
        y : pandas Series with labels
        
        Returns
        -------
        self
        """
        X_y = X.join(y)
        self.candidate_thresholds_ = {}
        for j in X.columns:
            thresholds = []
            sorted_X_y = X_y.sort_values([j, y.name]) # Sort by feature value, then by label
            prev_feature_val, prev_label = sorted_X_y.iloc[0][j], sorted_X_y.iloc[0][y.name]
            for idx,row in sorted_X_y.iterrows():
                curr_feature_val, curr_label = row[j], row[y.name]
                if (curr_label != prev_label and
                        not math.isclose(curr_feature_val, prev_feature_val)):
                    thresh = (prev_feature_val + curr_feature_val)/2
                    thresholds.append(thresh)
                prev_feature_val, prev_label = curr_feature_val, curr_label
            self.candidate_thresholds_[j] = thresholds
        return self
    
    def transform(self, X):
        """ Binarize numerical features using candidate thresholds.
        
        Parameters
        ----------
        X : pandas DataFrame with observations, X.columns used as feature names
        
        Returns
        -------
        Xb : pandas DataFrame that is the result of binarizing X
        """
        check_is_fitted(self)
        Xb = pd.DataFrame()
        for j in X.columns:
            for threshold in self.candidate_thresholds_[j]:
                binary_test_name = "X[{}] <= {}".format(j, threshold)
                Xb[binary_test_name] = (X[j] <= threshold)
        return Xb

def preprocess_dataset(X_train, X_test, y_train=None, numerical_features=None, categorical_features=None, binarization=None):
    """ Preprocess a dataset.
    
    Numerical features are scaled to the [0,1] interval by default, but can also
    be binarized, either by considering all candidate thresholds for a
    univariate split, or by binning. Categorical features are one-hot encoded.
    
    Parameters
    ----------
    X_train
    X_test
    y_train : pandas Series of training labels, only needed for binarization
        with candidate thresholds
    numerical_features : list of numerical features
    categorical_features : list of categorical features
    binarization : {'all-candidates', 'binning'}, default=None
        Binarization technique for numerical features.
        all-candidates
            Use all candidate thresholds.
        binning
            Perform binning using scikit-learn's KBinsDiscretizer.
        None
            No binarization is performed, features scaled to the [0,1] interval.
    
    Returns
    -------
    X_train_new : pandas DataFrame that is the result of binarizing X
    """
    if numerical_features is None:
        numerical_features = []
    if categorical_features is None:
        categorical_features = []
    
    numerical_transformer = MinMaxScaler()
    if binarization == 'all-candidates':
        numerical_transformer = CandidateThresholdBinarizer()
    elif binarization == 'binning':
        numerical_transformer = KBinsDiscretizer(encode='onehot-dense')
    #categorical_transformer = OneHotEncoder(drop='if_binary', sparse=False, handle_unknown='ignore') # Should work in scikit-learn 1.0
    categorical_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')
    ct = ColumnTransformer([("num", numerical_transformer, numerical_features),
                            ("cat", categorical_transformer, categorical_features)])
    X_train_new = ct.fit_transform(X_train, y_train)
    X_test_new = ct.transform(X_test)
    
    return X_train_new, X_test_new

################################################################################
# Functions for loading datasets from files into pandas DataFrames and Series
################################################################################

def load_acute_inflammations(decision_number):
    """ Load the Acute Inflammations dataset.
    
    Contains a mix of numerical and categorical attributes. Decided to not use
    this dataset in the paper for this reason.
    
    https://archive.ics.uci.edu/ml/datasets/Acute+Inflammations
    """
    if decision_number not in [1, 2]:
        raise ValueError("problem_number must be 1 or 2")
    df = pd.read_csv("datasets/diagnosis.data",
            names=["a1","a2","a3","a4","a5","a6","d1","d2"], decimal=',',
            encoding='utf-16', delim_whitespace=True)
    y = df["d{}".format(decision_number)]
    X = df.drop(columns=["d1","d2"])
    return X, y

def load_acute_inflammations_1():
    """ Load the Acute Inflammations dataset with decision 1 as the label.
    
    Contains a mix of numerical and categorical attributes.
    """
    return load_acute_inflammations(1)

def load_acute_inflammations_2():
    """ Load the Acute Inflammations dataset with decision 1 as the label.
    
    Contains a mix of numerical and categorical attributes.
    """
    return load_acute_inflammations(2)

def load_balance_scale():
    """ Load the Balance Scale dataset.
    
    Contains only categorical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/Balance+Scale
    """
    names = ["class","left weight","left distance","right weight","right distance"]
    df = pd.read_csv("datasets/balance-scale.data", names=names)
    y = df["class"]
    X = df.drop(columns="class")
    return X, y

def load_banknote_authentication():
    """ Load the Banknote Authentication dataset.
    
    Contains only numerical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/banknote+authentication
    """
    names = ["variance","skewness","curtosis","entropy","class"]
    df = pd.read_csv("datasets/data_banknote_authentication.txt", names=names)
    y = df["class"]
    X = df.drop(columns="class")
    return X, y

def load_blood_transfusion():
    """ Load the Blood Transfusion dataset.
    
    Contains only numerical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
    """
    names = ["recency","frequency","monetary","time","class"]
    df = pd.read_csv("datasets/transfusion.data", header=0, names=names)
    y = df["class"]
    X = df.drop(columns="class")
    return X, y

def load_car_evaluation():
    """ Load the Car Evaluation dataset.
    
    Contains only categorical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
    """
    names = ["buying","maint","doors","persons","lug_boot","safety","class"]
    df = pd.read_csv("datasets/car.data", names=names)
    y = df["class"]
    X = df.drop(columns="class")
    return X, y

def load_chess():
    """ Load the Chess (King-Rook vs. King-Pawn) dataset.
    
    Contains only categorical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King-Pawn%29
    """
    names = list("a{}".format(j+1) for j in range(36)) + ["class"]
    df = pd.read_csv("datasets/kr-vs-kp.data", names=names)
    y = df["class"]
    X = df.drop(columns="class")
    return X, y

def load_climate_model_crashes():
    """ Load the Climate Model Crashes dataset.
    
    Contains only numerical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/climate+model+simulation+crashes
    """
    df = pd.read_csv("datasets/pop_failures.dat", delim_whitespace=True)
    y = df["outcome"]
    X = df.drop(columns=["Study","Run","outcome"])
    return X, y

def load_congressional_voting_records():
    """ Load the Congressional Voting Records dataset.
    
    Contains only categorical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
    """
    names = list("a{}".format(j+1) for j in range(17))
    df = pd.read_csv("datasets/house-votes-84.data", names=names)
    y = df["a1"]
    X = df.drop(columns="a1")
    return X, y

def load_glass_identification():
    """ Load the Glass Identification dataset.
    
    Contains only numerical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/glass+identification
    """
    df = pd.read_csv("datasets/glass.data",
            names=["Id number","RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","Type of glass"],
            index_col="Id number")
    y = df["Type of glass"]
    X = df.drop(columns="Type of glass")
    return X, y

def load_hayes_roth():
    """ Load the Hayes-Roth dataset.
    
    Contains only categorical attributes. Dataset already partitions examples
    into train and test sets.
    
    https://archive.ics.uci.edu/ml/datasets/Hayes-Roth
    """
    df = pd.read_csv("datasets/hayes-roth.data",
            names=["name","hobby","age","educational level","marital status","class"],
            index_col="name")
    y_train = df["class"]
    X_train = df.drop(columns="class")
    df = pd.read_csv("datasets/hayes-roth.test",
            names=["hobby","age","educational level","marital status","class"])
    y_test = df["class"]
    X_test = df.drop(columns="class")
    return X_train, X_test, y_train, y_test

def load_image_segmentation():
    """ Load the Image Segmentation dataset.
    
    Contains only numerical attributes. Dataset already partitions examples
    into train and test sets.
    
    http://archive.ics.uci.edu/ml/datasets/image+segmentation
    """
    names = ["class"] + list("a{}".format(j+1) for j in range(19))
    df = pd.read_csv("datasets/segmentation.data", skiprows=5, names=names)
    y_train = df["class"]
    X_train = df.drop(columns="class")
    df = pd.read_csv("datasets/segmentation.test", skiprows=5, names=names)
    y_test = df["class"]
    X_test = df.drop(columns="class")
    return X_train, X_test, y_train, y_test

def load_ionosphere():
    """ Load the Ionosphere dataset.
    
    Contains only numerical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/ionosphere
    """
    names = list("a{}".format(j+1) for j in range(34)) + ["class"]
    df = pd.read_csv("datasets/ionosphere.data", names=names)
    y = df["class"]
    X = df.drop(columns="class")
    return X, y

def load_monks_problems(problem_number):
    """ Load the MONK's Problems dataset.
    
    Contains only categorical attributes. Test set is a "full" set of examples
    and the training set is simply a subset of the test set.
    
    https://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems
    """
    if problem_number not in {1, 2, 3}:
        raise ValueError("problem_number must be 1, 2, or 3")
    for t in ['train', 'test']:
        filename = "datasets/monks-{}.{}".format(problem_number, t)
        df = pd.read_csv(filename,
                names=["class","a1","a2","a3","a4","a5","a6","Id"],
                index_col="Id",
                delim_whitespace=True)
        y = df["class"]
        X = df.drop(columns="class")
        if t == 'train':
            X_train, y_train = X, y
        else:
            X_test, y_test = X, y
    return X_train, X_test, y_train, y_test

def load_monks_problems_1():
    """ Load the MONK's problem 1 dataset.
    
    Contains only categorical attributes. Test set is a "full" set of examples
    and the training set is simply a subset of the test set.
    """
    return load_monks_problems(1)

def load_monks_problems_2():
    """ Load the MONK's problem 2 dataset.
    
    Contains only categorical attributes. Test set is a "full" set of examples
    and the training set is simply a subset of the test set.
    """
    return load_monks_problems(2)

def load_monks_problems_3():
    """ Load the MONK's problem 3 dataset.
    
    Contains only categorical attributes. Test set is a "full" set of examples
    and the training set is simply a subset of the test set.
    """
    return load_monks_problems(3)

def load_parkinsons():
    """ Load the Parkinsons dataset.
    
    Contains only numerical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/parkinsons
    """
    df = pd.read_csv("datasets/parkinsons.data", index_col="name")
    y = df["status"]
    X = df.drop(columns="status")
    return X, y

def load_soybean_small():
    """ Load the Soybean (Small) dataset.
    
    Contains only categorical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/Soybean+%28Small%29
    """
    names = list("a{}".format(j+1) for j in range(35)) + ["class"]
    df = pd.read_csv("datasets/soybean-small.data", names=names)
    y = df["class"]
    X = df.drop(columns="class")
    return X, y

def load_tictactoe_endgame():
    """ Load the Tic-Tac-Toe Endgame dataset.
    
    Contains only categorical attributes.
    
    https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame
    """
    names = list("a{}".format(j+1) for j in range(9)) + ["class"]
    df = pd.read_csv("datasets/tic-tac-toe.data", names=names)
    y = df["class"]
    X = df.drop(columns="class")
    return X, y
