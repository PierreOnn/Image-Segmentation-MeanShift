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
    peak_labels = []
    for i in range(0, len(data)):
        peak_potential = findpeak(data, i, r)
        distances_peak = scipy.spatial.distance.cdist(peak_potential, peaks, metric='euclidean')
        neighbors_peak = np.where(distances_peak <= r / 2)[-1]
        if neighbors_peak.size == 0:
            labels[i] = np.amax(labels) + 1
            peak_labels = np.append(peak_labels, labels[i], axis=0)
            peaks = np.append(peaks, peak_potential, axis=0)
        elif neighbors_peak.size == 1:
            labels[i] = peak_labels[neighbors_peak]
        else:
            labels[i] = peak_labels[np.random.choice(neighbors_peak)]
    return labels, peaks


def meanshift_opt(data, r, c):
    labels = np.zeros((np.size(data, 0), 1), dtype=int)
    peaks = np.empty((0, len(data[0])))
    peak_labels = []
    for i in range(0, len(data)):
        if labels[i] != 0:
            continue
        peak_potential, cpts = findpeak_opt(data, i, r, c)
        distances_peak = scipy.spatial.distance.cdist(peak_potential, peaks, metric='euclidean')
        neighbors_peak = np.where(distances_peak <= r / 2)[-1]
        if neighbors_peak.size == 0:
            labels[i] = np.amax(labels) + 1
            peak_labels = np.append(peak_labels, labels[i], axis=0)
            peaks = np.append(peaks, peak_potential, axis=0)

            distances_peak_points = scipy.spatial.distance.cdist(peak_potential, data, metric='euclidean')
            peak_basin = (distances_peak_points <= r)
            speedup = (peak_basin + cpts.transpose())
            speedup_close = np.nonzero(speedup)[-1]
            labels[speedup_close] = labels[i]
        elif neighbors_peak.size == 1:
            labels[i] = peak_labels[neighbors_peak]
            speedup = cpts.transpose()
            speedup_close = np.nonzero(speedup)[-1]
            labels[speedup_close] = labels[i]
        else:
            labels[i] = peak_labels[np.random.choice(neighbors_peak)]
            speedup = cpts.transpose()
            speedup_close = np.nonzero(speedup)[-1]
            labels[speedup_close] = labels[i]
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
        mean = np.mean(data_points_neighbors, axis=0).reshape(1, -1)
        shift = scipy.spatial.distance.cdist(data_point, mean, metric='euclidean')
        data_point = mean

        neighbors_close = np.where(distances <= r / c)[-1]
        cpts[neighbors_close] = 1
    else:
        peak = mean
    return peak, cpts
