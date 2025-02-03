# Python Module: Geometric Algorithms for Convex Combinations and Coresets

This Python module provides implementations of geometric algorithms for determining convex combinations, finding sets of farthest points, and computing coresets as described in the paper "More output-sensitive geometric algorithms" by Clarkson. The module is designed to be used in computational geometry, machine learning, and data analysis tasks where efficient geometric computations are required.

---

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
   - [isConvexCombination](#isconvexcombination)
   - [farthestPointsSetUsingMinMax](#farthestpointssetusingminmax)
   - [clarksonCoreset](#clarksoncoreset)
   - [computeClarksonCoreset](#computeclarksoncoreset)
3. [Acknowledgments](#acknowledgments)

---

## Installation

To use this module, ensure you have Python 3.10.2 installed.

To download the required modules, use miniconda [here](https://docs.conda.io/en/latest/miniconda.html).

Make sure you have gurobi 11.0.3 installed.

---

## Usage

### isConvexCombination(X, ind_E, s)
Description:
Determines whether a point s is a convex combination of the points in the set E. If s is a convex combination of E, the function returns None. Otherwise, it returns a witness vector that certifies that s is not a convex combination of E.

Parameters:
```
X (numpy.ndarray): A 2D array representing the dataset, where each row is a point.

ind_E (list): A list of indices corresponding to the points in the set E.

s (numpy.ndarray): A 1D array representing the point to be checked.
```
Returns:
```
None if s is a convex combination of E.

A witness vector (numpy.ndarray) if s is not a convex combination of E.
```

---

### farthestPointsSetUsingMinMax(X)
Description:
Finds the set of points that are farthest apart using a simple min-max approach along each dimension of X.

Parameters:
```
X (numpy.ndarray): A 2D array representing the dataset, where each row is a point.
```
Returns:
```
A list of indices corresponding to the farthest points in X.
```

---

### clarksonCoreset(X, ind_E, ind_S, dataset_name)
Description:
Computes the Clarkson coreset for a given dataset X, using the indices of the set E and the subset S. This function implements the "clarkson-cs" method described in the paper "More output-sensitive geometric algorithms".

Parameters:
```
X (numpy.ndarray): A 2D array representing the dataset, where each row is a point.

ind_E (list): A list of indices corresponding to the points in the set E.

ind_S (list): A list of indices corresponding to the points in the subset S.

dataset_name (str): A name or identifier for the dataset (used for logging or debugging).
```
Returns:
```
A list of indices representing the Clarkson coreset.
```

---

### computeClarksonCoreset(X, dataset_name)
Description:
Computes the Clarkson coreset for a given dataset X using the clarksonCoreset function. This is a higher-level function that automates the process of selecting the sets E and S.

Parameters:
```
X (numpy.ndarray): A 2D array representing the dataset, where each row is a point.

dataset_name (str): A name or identifier for the dataset (used for logging or debugging).
```
Returns:
```
A list of indices representing the Clarkson coreset.
```

---

## Acknowledgments

This module is inspired by the paper "More output-sensitive geometric algorithms" by Clarkson.

For any questions or support, please contact the maintainer at amet97vikram@gmail.com
