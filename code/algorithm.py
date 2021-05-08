import numpy as np
import scipy.spatial.distance


def findpeak(data, idx, r):
    data_point = data[idx, :].reshape(1, -1)
    threshold = 0.01
    shift = np.amax(data) - np.amin(data)
    while shift > threshold:
        distances = scipy.spatial.distance.cdist(data_point, data, metric='euclidean')
        neighbors = np.where(distances <= r)[-1]
        data_points_neighbors = data[neighbors, :]
        mean = np.mean(data_points_neighbors, axis=0).reshape(1, 3)
        shift = scipy.spatial.distance.cdist(data_point, mean, metric='euclidean')
        data_point = mean
    else:
        peak = mean
    return peak


def meanshift(data, r):
    labels = np.zeros((np.size(data, 0), 1), dtype=int)
    peaks = np.empty((0, len(data[0])))
    for i in range(0, len(data)):
        peak_potential = findpeak(data, i, r)
        distances_peak = scipy.spatial.distance.cdist(peak_potential, peaks, metric='euclidean')
        neighbors_peak = np.where(distances_peak <= r / 2)[-1]
        if len(neighbors_peak) == 0:
            labels[i] = np.amax(labels) + 1
            peaks = np.append(peaks, peak_potential, axis=0)
        elif len(neighbors_peak) == 1:
            labels[i] = labels[neighbors_peak]
        else:
            neighbors_peak_shor = np.where(np.amin(distances_peak))
            labels[i] = labels[neighbors_peak_shor]
    return labels, peaks


def meanshift_opt(data, r, c):
    labels = np.zeros((np.size(data, 0), 1), dtype=int)
    peaks = np.empty((0, len(data[0])))
    for i in range(0, len(data)):
        if labels[i] != 0:
            continue
        peak_potential, cpts = findpeak_opt(data, i, r, c)
        distances_peak = scipy.spatial.distance.cdist(peak_potential, peaks, metric='euclidean')
        neighbors_peak = np.where(distances_peak <= r / 2)[-1]
        if len(neighbors_peak) == 0:
            labels[i] = np.amax(labels) + 1
            labels[cpts] = labels[i]
            peaks = np.append(peaks, peak_potential, axis=0)

            distances_peak_points = scipy.spatial.distance.cdist(peak_potential, data, metric='euclidean')
            neighbors_peak = np.where(distances_peak_points <= r)[-1]
            labels[neighbors_peak] = labels[i]

        elif len(neighbors_peak) == 1:
            labels[i] = labels[neighbors_peak]
            labels[cpts] = labels[i]
        else:
            neighbors_peak_shor = np.where(np.amin(distances_peak))
            labels[i] = labels[neighbors_peak_shor]
            labels[cpts] = labels[i]
    return labels, peaks


def findpeak_opt(data, idx, r, c):
    data_point = data[idx, :].reshape(1, -1)
    cpts = np.zeros((np.size(data, 0), 1), dtype=int)
    threshold = 0.01
    shift = np.amax(data) - np.amin(data)
    while shift > threshold:
        distances = scipy.spatial.distance.cdist(data_point, data, metric='euclidean')
        neighbors = np.where(distances <= r)[-1]
        data_points_neighbors = data[neighbors, :]
        mean = np.mean(data_points_neighbors, axis=0).reshape(1, 3)
        shift = scipy.spatial.distance.cdist(data_point, mean, metric='euclidean')
        data_point = mean

        neighbors_close = np.where(distances <= r/c)[-1]
        cpts[neighbors_close] = 1
    else:
        peak = mean
    return peak, cpts