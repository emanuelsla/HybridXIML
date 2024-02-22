import numpy as np


def compute_gower_distance(x_1, x_2, metric_cols=[], character_cols=[]):
    squared_diff = 0
    for col in metric_cols:
        squared_diff += (x_1[col] - x_2[col]) ** 2
    euclidean = np.sqrt(squared_diff)
    w_euclidean = euclidean * len(metric_cols) / (len(metric_cols) + len(character_cols))

    if len(character_cols) > 0:
        equal = 0
        for col in character_cols:
            if x_1[col] == x_2[col]:
                equal += 1
        dice = 1 - (equal / len(character_cols))
        w_dice = dice * len(character_cols) / (len(metric_cols) + len(character_cols))
    else:
        w_dice = 0

    gower = w_euclidean + w_dice

    return gower


def compute_distance_matrix_between_dfs(df1, df2, metric_cols, character_cols, distance_func=compute_gower_distance):
    dist_matrix = [list]*len(df1)
    for i in range(0, len(df1)):
        dist_vector = [0] * len(df2)
        for j in range(0, len(df2)):
            dist = distance_func(df1.iloc[i], df2.iloc[j], metric_cols, character_cols)
            dist_vector[j] = dist
        dist_matrix[i] = dist_vector
    dist_matrix = np.asarray(dist_matrix)
    return dist_matrix


def compute_distance_matrix(df, metric_cols, character_cols, distance_func=compute_gower_distance):
    dist_matrix = [list] * len(df)
    for i in range(0, len(df)):
        dist_vector = [0] * len(df)
        for j in range(len(df)):
            dist = distance_func(df.iloc[i], df.iloc[j], metric_cols, character_cols)
            dist_vector[j] = dist
        dist_matrix[i] = dist_vector
    dist_matrix = np.asarray(dist_matrix)
    return dist_matrix