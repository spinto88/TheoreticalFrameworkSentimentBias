import numpy as np 
from scipy.optimize import differential_evolution, Bounds
from typing import List, Dict
from src.schemas import AnalysisOutput, OutletScore, SubjectScore

def build_tensor(data):
    """
    data: lista de dicts con estructura:
        {
            "outlet": str,
            "subject": str,
            "mention_type": "negative" | "neutral" | "positive",
            "amount_of_mentions": int
        }

    returns:
        matrix: list[list[list[int]]]  # m x k x 3
        outlets: list[str]
        subjects: list[str]
    """

    # 1. extraer conjuntos únicos
    outlets = sorted({d.outlet for d in data})
    subjects = sorted({d.subject for d in data})

    # 2. index mapping
    outlet_to_idx = {o: i for i, o in enumerate(outlets)}
    subject_to_idx = {s: j for j, s in enumerate(subjects)}

    # 3. inicializar matriz m x k x 3
    m, k = len(outlets), len(subjects)
    matrix = [[[0, 0, 0] for _ in range(k)] for _ in range(m)]

    # 4. mapping de tipo → índice
    type_to_idx = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

    # 5. llenar la matriz
    for d in data:
        i = outlet_to_idx[d.outlet]
        j = subject_to_idx[d.subject]
        t = type_to_idx[d.mention_type]

        matrix[i][j][t] += d.amount_of_mentions

    return np.array(matrix), outlets, subjects

def build_output(outlets, subjects, z, a, b):
    """
    outlets: list[str] size m
    subjects: list[str] size k
    z: list[float] size m
    a: list[float] size k
    b: list[float] size k
    """

    # outlets
    outlet_scores = [
        OutletScore(outlet=outlets[i], z=float(z[i]))
        for i in range(len(outlets))
    ]

    # subjects
    subject_scores = [
        SubjectScore(subject=subjects[j], a=float(a[j]), b=float(b[j]))
        for j in range(len(subjects))
    ]

    return AnalysisOutput(
        outlets=outlet_scores,
        subjects=subject_scores
    )

def log_likelihood(x, mentions_matrix):

    """ Function to maximize """

    n_rows = mentions_matrix.shape[0]
    n_cols = mentions_matrix.shape[1]

    z, a, b = x[:n_rows], x[n_rows:(n_rows + n_cols)], x[(n_rows + n_cols):]
    
    logl = 0.00
    for i in range(n_rows):
        for j in range(n_cols):
            z_ij = z[i] * a[j] + b[j]
            N_ij = np.sum(mentions_matrix[i][j])
            logl += (mentions_matrix[i][j][2] - mentions_matrix[i][j][0]) * z_ij - N_ij * (np.log(np.exp(z_ij) + 1 + np.exp(-z_ij)))        
            logl -= 0.5*(z[i]**2 + a[j]**2 + b[j]**2)
    
    return logl
    
def negative_log_likelihood(x, mentions_matrix):
    return -log_likelihood(x, mentions_matrix)
 
def run_analysis(data): 

    mentions_matrix, outlets, subjects = build_tensor(data)

    n_media_outlets = len(outlets)
    n_subjects = len(subjects)

    # Bounds
    lb = [-5] * (n_media_outlets + 2 * n_subjects)
    ub = [5] * (n_media_outlets + 2 * n_subjects)
    bounds = Bounds(lb, ub)

    solution = differential_evolution(negative_log_likelihood, args=(mentions_matrix,), bounds=bounds, maxiter=1000, popsize=25)

    z_predicted, a_predicted, b_predicted = solution.x[:n_media_outlets], solution.x[n_media_outlets:(n_media_outlets + n_subjects)], solution.x[(n_media_outlets + n_subjects):]

    return build_output(outlets, subjects, z_predicted, a_predicted, b_predicted)

if __name__ == "__main__":

    import json 

    input_data = json.load(open("../input.json","r"))

    print(run_analysis(input_data["data"]))




