from time import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from tqdm import tqdm

data_path = "/data/local/AA/data/"
coresets_path = "/data/local/AA/results/coresets/"

EPSILON = 1e-6

def triangleAlgorithm(X, ind_E, s, epsilon=EPSILON):
    S = X[ind_E].copy()
    p = X[s].copy()

    # Step 0: Initialization
    # Find v with minimum distance to p
    idx = 0
    for vi in range(1,len(S)):
        if np.linalg.norm(p - S[vi]) < np.linalg.norm(p - S[idx]):
            idx = vi
    p_prime = S[idx].copy()
    v = S[idx].copy()

    while True:
        # Step 1: Stopping Criteria and Pivot Selection
        # Check if current point is ε-approximate solution
        if np.linalg.norm(p - p_prime) < epsilon * np.linalg.norm(p - v):
            return p_prime

        # Find pivot point
        pivot_index = None
        for vj in range(len(S)):
            if np.linalg.norm(p_prime - S[vj]) >= np.linalg.norm(p - S[vj]):
                pivot_index = vj
                break

        # If no pivot exists, return p_prime as witness
        if pivot_index is None:
            return p_prime

        vj = S[pivot_index].copy()
        v = S[pivot_index].copy()

        # Compute alpha
        numerator = np.dot((p - p_prime).T, vj - p_prime)
        denominator = np.square(np.linalg.norm(vj - p_prime))

        alpha = numerator / denominator

        # Update p_prime
        p_prime = (1 - alpha) * p_prime + alpha * vj

def isConvexCombinationTA(X, ind_E, s):
    try:
        epsilon = EPSILON

        # Run Triangle Algorithm
        result = triangleAlgorithm(X, ind_E, s)

        # Check proximity conditions
        R = max(np.linalg.norm(X[s] - X[vi]) for vi in ind_E)

        # Check relative error condition
        if np.linalg.norm(result - X[s]) / R < epsilon:
            return None

        # Check witness condition
        for vi in ind_E:
            if np.linalg.norm(result - X[vi]) < np.linalg.norm(X[s] - X[vi]):
                return result

        return result

    except Exception as e:
        print(f"Error in convex combination check: {e}")
        return np.array([])

# Determines whether a point s is a convex combination
# of the points in the set E.
# Returns None if s is a convex combination of E,
# otherwise returns a witness vector that certifies
# that s is not a convex combination of E.
def isConvexCombinationCK(X, ind_E, s):
    E = X[ind_E].copy()
    P = X[s].copy()

    # initialize the dimensions of the data
    k = E.shape[0]
    d = E.shape[1]

    # initialize the model and parameters
    model = gp.Model("ConvexCombination")
    model.setParam("OutputFlag", 0)

    # adding model variables -- the lambda coefficients of the convex combination equation should be between 0 and 1
    lambdas = model.addMVar((k,), lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="lambdas")

    # adding model constraints -- the sum of the lambda coefficients should be 1
    model.addConstr(lambdas.sum() == 1, name="sum_of_lambdas")

    # adding model constraints -- the convex combination equation => E.T x lambdas = P
    model.addMConstr(E.T, lambdas, '=', P, "convex_combination_equation")

    model.setParam("InfUnbdInfo", 1)

    # optimize the model
    # model.write("_gurobi_lp/convex_combination.lp")
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return None
    else:
        # model.computeIIS()
        # model.write("_gurobi_lp/convex_combination.ilp")

        # Computing the Farkas dual, when model is infeasible.
        # The Farkas dual is a certificate of infeasibility.
        # It is a vector that satisfies the following conditions:
        # y.T * A * x <= y.T * b
        # when the original problem : A * x = b is infeasible.
        # Here y will be our witness vector.
        dual = []
        for i in range(d):
            constr = model.getConstrByName("convex_combination_equation[{}]".format(i))
            assert constr is not None
            dual.append(constr.getAttr(GRB.Attr.FarkasDual))
        return np.array(dual)

def isConvexCombination(X, ind_E, s, method="CK"):
    if method == "CK":
        return isConvexCombinationCK(X, ind_E, s)
    elif method == "TA":
        return isConvexCombinationTA(X, ind_E, s)
    else:
        raise ValueError("Invalid method name")

# finds set of points that are farthest apart
# using simple min max along each dimension of X
def farthestPointsSetUsingMinMax(X):
    n = X.shape[0]
    d = X.shape[1]

    ind_E = set()

    for i in range(d):
        p1 = X[:,i].argmin()
        p2 = X[:,i].argmax()
        ind_E.add(p1)
        ind_E.add(p2)

    return list(ind_E)

# proposed coreset
# "clarkson-cs" in the paper "More output-sensitive geometric algorithms"
def clarksonCoreset(X, ind_E, ind_S, dataset_name, method):
    t_start = time()
    try:
        pbar = tqdm(total=len(ind_S), desc="clarkson-cs computation:")
        while len(ind_S) > 0:
            if len(ind_S) % 1000 == 0:
                pbar.write(
                    "Current Size of coreset: {}".format(len(ind_E)))
            s = ind_S.pop(0)
            witness_vector = isConvexCombination(X, ind_E, s, method)
            if witness_vector is not None:
                max_dot_product = np.dot(-1*witness_vector, X[s])
                p_prime = None
                for p in ind_S:
                    dot_product = np.dot(-1*witness_vector, X[p])
                    if dot_product > max_dot_product:
                        max_dot_product = dot_product
                        p_prime = p
                if p_prime is not None:
                    ind_E.append(p_prime)
                    ind_S.append(s)
                    ind_S.remove(p_prime)
                else:
                    ind_E.append(s)
            pbar.update(1)
        pbar.close()
    except Exception as e:
        print(e)
    X_C = X[ind_E].copy()
    t_end = time()
    if dataset_name is not None:
        np.savez(
            coresets_path + dataset_name + "_clarkson_coreset.npz",
            X=X_C,
            cs_time=t_end - t_start
        )
    return X_C

def computeClarksonCoresetWrapper(X, dataset_name, method):
    X_C = None

    # Assert that X is a numpy array
    assert isinstance(X, np.ndarray), "X must be a numpy array"

    # initialize two extreme points via farthestPointsSetUsingMinMax function
    # maintain the initialized indices as set E. Note: len(E) < len(X)
    # maintain the indices not belonging to E as set S. Note: len(S) = len(X) - len(E)
    # any index not belonging to E is a candidate for the next coreset
    ind_E = farthestPointsSetUsingMinMax(X)
    ind_S = np.setdiff1d(np.arange(len(X)), np.array(ind_E)).tolist()

    # obtain initial coreset using Clarkson's algorithm
    X_C = clarksonCoreset(X, ind_E, ind_S, dataset_name, method)

    return X_C

def computeClarksonCoreset(X, dataset_name=None, method="CK"):
    if dataset_name is None:
        return computeClarksonCoresetWrapper(X, dataset_name, method)
    try:
        data = np.load(coresets_path + dataset_name + "_clarkson_coreset.npz")
        return data["X"]
    except FileNotFoundError:
        return computeClarksonCoresetWrapper(X, dataset_name, method)

if __name__ == "__main__":
    pass
