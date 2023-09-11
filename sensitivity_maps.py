import numpy as np
import matplotlib.pyplot as plt
import skimage
from coil_centers import coil_centers

def ellipse_grid_2d(x, y, x0=84.5, y0=108.5, a=85, b=109, smooth_sigma=3) -> np.ndarray:
    # recall, in equation of ellipse is this is equal to 1
    ellipse = (x - x0)**2 / a**2 + (y - y0)**2 / b**2
    ellipse_grid = np.zeros(ellipse.shape)  # switched since, recall, x direction is the numpy y coordinate
    ellipse_grid[ellipse < 1] = 1
    ellipse_grid = skimage.filters.gaussian(ellipse_grid, sigma=smooth_sigma)
    return ellipse_grid

def gaussian_2d(x, y, x0, y0, sigma_x=90*np.sqrt(2), sigma_y=90*np.sqrt(2)) -> np.ndarray:
    # if sigma_x = sigma_y, we have a circle gaussian. Otherwise, elliptical
    values =  np.exp(-(((x-x0) / sigma_x) ** 2 + ((y-y0) / sigma_y) ** 2) / 2)
    return values

def cone_2d(x, y, x0, y0, slope=1 / 276 * 2 * np.pi) -> np.ndarray:
    # 276 since this is the length across the diagonal of the image
    # Need to fine tune this and make it more generalized, but the idea is that within our map, the
    # phase will range from 0 to 2pi from far corner to far corner

    radii = np.sqrt((x-x0)**2 + (y-y0)**2)
    return radii * slope 


def create_map(center_x, center_y, x_dim=170, y_dim=218, sigma_x=70, sigma_y=70):
    x = np.arange(x_dim)
    y = np.arange(y_dim)      
    xv, yv = np.meshgrid(x, y)      
               
    ellipse_grid = ellipse_grid_2d(xv, yv)

    # MAKE SENSITIVITY MAPS
    abs_map = gaussian_2d(xv, yv, center_x, center_y, sigma_x=sigma_x, sigma_y=sigma_y) #* ellipse_grid
    phase_map = cone_2d(xv, yv, center_x, center_y) #* ellipse_grid
    sens_map = abs_map * np.exp(1j * phase_map)
    return sens_map, phase_map

    
def create_mod_rect_map(x_cord, y_cord, x_len, y_len, x_phase, y_phase, x_dim=170, y_dim=218, smooth_sigma=5):
    # x_cord and y_cord are bottom left of rectangle
    # x_phase and y_phase are for the center of the phase cone
    x = np.arange(x_dim) 
    y = np.arange(y_dim)
    xv, yv = np.meshgrid(x, y)

    abs_map = np.zeros((y_dim, x_dim))
    abs_map[y_cord:y_cord + y_len, x_cord:x_cord + x_len ] = 1
    phase_map = cone_2d(xv, yv, x_phase, y_phase)
    sens_map = abs_map * np.exp(1j * phase_map)

    return sens_map, phase_map

def check_overlap(maps: list) -> np.ndarray:
    out = np.diff(np.vstack(maps).reshape(len(maps), -1), n=len(maps) - 1, axis=0) == 0
    out = out.reshape(maps[0].shape)
    return out.astype(int)


def check_box_overlap(maps: list) -> np.ndarray:
    smaps = np.round(np.abs(np.stack(map_list, axis=0))).astype(int)
    smaps_equal = np.where(np.sum(smaps, axis=0) == 4, 1, 0)
    return smaps_equal

if __name__ == "__main__":
    nx = 174  # pixels in x direction (so NOT number of rows)
    ny = 218
    shape = str(ny) + '_' + str(nx)

    x = np.arange(nx) 
    y = np.arange(ny)
    xv, yv = np.meshgrid(x, y)

    # import matplotlib as mpl
    # cmap0 = mpl.cm.gray
    # map, phase_map = create_map(10, 15, sigma_y=30)

    # abs = np.abs(map)
    # plt.imshow(abs, cmap=cmap0)
    # plt.axis('off')
    # fig = plt.gcf()
    # norm2 = mpl.colors.Normalize(vmin=0, vmax=1)
    # cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=cmap0), ticks=[0, 1])
    # cbar.ax.set_yticklabels(['0', '1'], fontsize='20')
    # plt.savefig('final_report_plots/smap_sim_mod_abs.png', dpi=300, bbox_inches='tight')
    # plt.close()
    # 
    # cmap = mpl.cm.magma
    # phase = phase_map
    # plt.imshow(phase, cmap=cmap)
    # plt.axis('off')
    # norm1 = mpl.colors.Normalize(vmin=0, vmax=3.1415)
    # cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm1, cmap=cmap), ticks=[0, 3.1415/2, 3.1415])
    # cbar.ax.set_yticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$'], fontsize='20')
    # plt.savefig('final_report_plots/smap_sim_mod_phase.png', dpi=300, bbox_inches='tight')
    # plt.close()

    style = 'box'
    map_config = 100

    if style == 'box':
        bboxes, phases = coil_centers(map_config, shape=shape)
    else:
        centers, sigmas = coil_centers(map_config, shape=shape)

    plt.figure(dpi=300)
    abs_map = np.zeros((ny, nx))
    map_list = []
    phase_list = []

    for i in range(4):
        if style == 'box':
            b, p = bboxes[i], phases[i]
            sens_map, phase_map = create_mod_rect_map(*b, *p, x_dim=nx, y_dim=ny)  # returns phase map as well so it's nice and unwrapped for plotting
        else:
            c, sx, sy = centers[i], sigmas[i][0], sigmas[i][1]
            sens_map, phase_map = create_map(*c, sigma_x=sx, sigma_y=sy, x_dim=nx, y_dim=ny)

        map_list.append(sens_map)
        phase_list.append(phase_map)
        plt.subplot(2,4,i+1)
        plt.imshow(np.abs(sens_map), origin='lower')
        plt.axis('off')
        plt.subplot(2,4,i+5)
        plt.imshow(phase_map, origin='lower')
        plt.axis('off')
        abs_map += np.abs(sens_map)
    plt.savefig(f'plots/smaps/sens_map{map_config}_test.png', bbox_inches='tight')
    plt.clf()

    overlap = check_box_overlap(map_list)
    print(f'Overlap percent: {np.sum(overlap)/(nx*ny):.2f}')
    plt.imshow(overlap, vmin=0, vmax=1)
    plt.savefig(f'plots/smaps/sens_map{map_config}_overlap.png', bbox_inches='tight')
    plt.clf()

    sens_4_channel = np.stack(map_list, axis=0)
    phase_4_channel  = np.stack(phase_list, axis=0)
    print(sens_4_channel.shape)
    np.save(f'sensitivity_maps/{shape}/square_new/smap_{map_config}.npy', sens_4_channel)
    plt.imshow(abs_map, origin='lower')
    plt.colorbar()
    plt.savefig(f'plots/smaps/abs_map{map_config}_test.png')  # shows overlap of absolute value maps