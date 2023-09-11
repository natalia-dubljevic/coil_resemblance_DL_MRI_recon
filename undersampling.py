import numpy as np
from sympy import divisors
import math

def centered_circle(image_shape,radius):
    """
    Description: creates a boolean centered circle image with a pre-defined radius
    Args:
        param image_shape: shape of the desired image
        param radius: radius of the desired circle
    Returns: Indices of values inside the circle
    """

    center_x = image_shape[0] // 2
    center_y = image_shape[1] // 2
    
    X,Y = np.indices(image_shape)
    circle_indices = np.argwhere(((X-center_x)**2+(Y-center_y)**2) < radius**2)

    return circle_indices

def centered_square(image_shape, acs_size):
    # acs size is a tuple of rows, cols
    center_x = image_shape[0] // 2
    center_y = image_shape[1] // 2

    mask = np.zeros(image_shape)
    half_side_x, half_side_y = acs_size[0] // 2, acs_size[1] // 2
    startx, starty = center_x - half_side_x, center_y - half_side_y
    mask[startx:startx + acs_size[0], starty:starty + acs_size[1]] = 1
    # mask_indices = np.argwhere(mask==1)

    return mask


def gaussian2d_circle(R_factor, radius, pattern_shape=(218, 170), cov=None):
    """
    Description: creates a 2D gaussian sampling pattern of a 2D image. Autocalibration
    circle at center
    Args:
        R factor: acceleration factor in the desired direction
        radius: radius of circle at center of Gaussian
        pattern_shape: shape of the desired sampling pattern.
        param cov: covariance matrix of the distribution
    Returns: sampling pattern image. It is a boolean image
    """

    N = pattern_shape[0] * pattern_shape[1]  # Image length
    circle_indices  = centered_circle(pattern_shape, radius)

    required_points = int(N / R_factor) - circle_indices.shape[0]  # recall it's size # inds, 2

    center = np.array([1.0 * pattern_shape[0] / 2 - 0.5, \
                       1.0 * pattern_shape[1] / 2 - 0.5])

    if cov is None:
        cov = np.array([[(1.0 * pattern_shape[0] / 4) ** 2, 0], \
                        [0, (1.0 * pattern_shape[1] / 4) ** 2]])

    samples = np.array([0])

    m = 1  # Multiplier. We have to increase this value
    # until the number of points (disregarding repeated points)
    # is equal to factor

    while (samples.shape[0] < required_points):

        samples = np.random.multivariate_normal(center, cov, m * required_points)
        samples = np.rint(samples).astype(int)  # locations rounded to nearest integer
        indexesx = np.logical_and(samples[:, 0] >= 0, samples[:, 0] < pattern_shape[0])
        indexesy = np.logical_and(samples[:, 1] >= 0, samples[:, 1] < pattern_shape[1])
        indexes = np.logical_and(indexesx, indexesy)
        samples = samples[indexes]

        # remove the circle indices from the running-- they will be included no matter what
        samples = np.array([x for x in samples.tolist() if x not in circle_indices.astype(int).tolist()])

        samples = np.unique(samples[:, 0] + 1j * samples[:, 1])  # I guess this gives you the unique coordinates
        samples = np.column_stack((samples.real, samples.imag)).astype(int)
        if samples.shape[0] < required_points:  # recall shape is just # points, 2 (for 2 coords)
            m *= 2
            continue
    # we probably overshot, so let's randomly remove some
    indexes = np.arange(samples.shape[0], dtype=int)  # indices o
    np.random.shuffle(indexes)
    samples = samples[indexes][:required_points]

    samples = np.append(circle_indices, samples, axis=0)
    under_pattern = np.zeros(pattern_shape)
    under_pattern[samples[:, 0], samples[:, 1]] = 1
    return under_pattern

def uniform_grid(R_factor, acs_size=(24, 24), pattern_shape=(218, 170)):
    N = pattern_shape[0] * pattern_shape[1]  # Image length
    square_mask = centered_square(pattern_shape, acs_size)
    required_points = N / R_factor - square_mask.sum()
    available_points = N - square_mask.sum()
    new_R = round(available_points / required_points)
    if R_factor==12:
        new_R = 15  # looks nicer
    print(f'R: {R_factor}, new_R: {new_R}')
    new_R_divisors = divisors(new_R)
    middle_index = math.ceil(len(new_R_divisors) / 2) - 1
    row_factor = max(new_R_divisors[middle_index], int(new_R / new_R_divisors[middle_index]))  # when given the option, remove more column points since the columns are taller
    col_factor = int(new_R / row_factor)

    square_mask[::row_factor, ::col_factor] = 1
    true_R = N / np.sum(square_mask)

    return square_mask, true_R

def uniform_grid_basic(R_factor, acs_size=(24, 24), pattern_shape=(218, 170)):
    # use this for R=6, since otherwise it does some weird things (but not for 218 x 180)
    N = pattern_shape[0] * pattern_shape[1]  # Image length
    square_mask = centered_square(pattern_shape, acs_size)
    R_divisors = divisors(R_factor)
    middle_index = math.ceil(len(R_divisors) / 2) - 1
    row_factor = max(R_divisors[middle_index], int(R_factor / R_divisors[middle_index]))  # when given the option, remove more column points since the columns are taller
    col_factor = int(R / row_factor)

    square_mask[::row_factor, ::col_factor] = 1
    true_R = N / np.sum(square_mask)

    return square_mask, true_R

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    R = 10
    #mask, true_R = uniform_grid(R, pattern_shape=(218, 170))
    for i in range(50):
        print(i)
        mask = gaussian2d_circle(R, 12, pattern_shape=(218, 180))
        np.save(f'gauss_undersampling_masks/218_180/gauss_mask_R={R}_v{i}.npy', mask)
        #plt.imshow(mask, origin='lower')
        #plt.savefig('plots/undersampling_test.png')
    #print(true_R)
    #np.save(f'nu_undersampling_masks/218_174/uniform_mask_R={R}.npy', mask)
    #plt.imshow(mask, origin='lower')
    #plt.savefig('plots/undersampling_test.png')