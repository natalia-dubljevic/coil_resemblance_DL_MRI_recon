import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import matplotlib as mpl
import pandas as pd
from scipy.stats import bootstrap
import seaborn as sns

smap1 = np.load(r'sensitivity_maps/218_170/square_new/smap_102.npy')
smap1 = np.load(r'sensitivity_maps/218_170/circle_ring/circle_R_8.npy')
smap1 = np.moveaxis(smap1, -1, 0)
smap1_phase = np.angle(smap1)  # just makes phase visuals look a bit nicer, sometimes there's a bit of a wraparound effect
#smap1_phase *= np.abs(smap1)

img = np.load('/home/pasala/Data/12-channel/train/target_slice_images/e14089s3_P53248_s80_nlinv_img.npy')
mask = np.load('undersampling_masks/218_170/uniform_mask_R=6.npy')

cmap0 = cm.get_cmap('magma', 2)

to_plot = 'violin'

if to_plot == 'maps':
    fig1 = plt.figure(constrained_layout=False)
    gs1 = fig1.add_gridspec(nrows=2, ncols=4, left=0.05, right=0.91, wspace=0.1, hspace=0.01)

    for i in range(8):
        col = i // 2
        ax = fig1.add_subplot(gs1[0, col])
        ax.imshow(np.abs(smap1[col, :, :]), cmap=cmap0)
        ax.axis('off')
        ax.set_title(f'Coil {col+1}')
        if col == 0:
            ax.text(-25, 170, 'Magnitude', rotation='vertical', ha='center', fontsize='x-large')

        ax = fig1.add_subplot(gs1[1, col])
        ax.axis('off')
        a = ax.imshow(smap1_phase[col, :, :], cmap='magma')
        if col == 0:
            ax.text(-25, 135, 'Phase', rotation='vertical', ha='center', fontsize='x-large')
        if col == 3:
            cmap = cm.magma
            norm1 = mpl.colors.Normalize(vmin=0, vmax=3.1415)
            cax1 = fig1.add_axes([0.92, 0.133, 0.03, 0.34])
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm1, cmap=cmap), cax=cax1, ticks=[0, 3.1415/2, 3.1415])
            cbar.ax.set_yticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$'])

            norm2 = mpl.colors.Normalize(vmin=0, vmax=1)
            cax2 = fig1.add_axes([0.92, 0.52, 0.03, 0.34])
            plt.colorbar(cm.ScalarMappable(norm=norm2, cmap=cmap0), cax=cax2, ticks=[0, 1])

    plt.savefig('final_report_plots/smaps_2.png', bbox_inches='tight', dpi=300)

if to_plot == 'processing':
    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=3, wspace=0.16, hspace=0.17)
    gs.tight_layout(fig, w_pad=0.0)

    # nlinv image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.abs(img[0, :, :]), cmap='gray')
    ax1.axis('off')
    ax1.set_title(f'NLINV')

    # sens map
    ax2 = fig.add_subplot(gs[0, 1])
    abs = np.abs(smap1[0, :, :])
    ax2.imshow(abs, cmap=cmap0)
    ax2.axis('off')
    ax2.set_title('Magnitude')

    phase = smap1_phase[0, :, :]
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(phase, cmap='magma')
    ax3.axis('off')
    ax3.set_title('Phase')

    # us map
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(mask, cmap='binary_r')
    ax4.axis('off')
    ax4.set_title('Mask (R=6)')

    # recon
    input_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img * smap1[0, :, :], axes=(-1, -2))), axes=(-1, -2)) * mask
    input_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input_kspace, axes=(-1, -2))), axes=(-1, -2))
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(np.abs(input_img[0, :, :]), cmap='gray')
    ax5.axis('off')
    ax5.set_title(f'Input Data')

    # target
    target = img[0, :, :] * smap1[0, :, :]
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(np.abs(target), cmap='gray')
    ax6.axis('off')
    ax6.set_title(f'Target')

    # colorbars
    cmap = cm.magma
    norm1 = mpl.colors.Normalize(vmin=0, vmax=3.1415)
    cax1 = fig.add_axes([0.895, 0.525, 0.02, 0.355])
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm1, cmap=cmap), cax=cax1, ticks=[0, 3.1415/2, 3.1415])
    cbar.ax.set_yticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$'])

    norm2 = mpl.colors.Normalize(vmin=0, vmax=1)
    cax2 = fig.add_axes([0.625, 0.525, 0.02, 0.355])
    plt.colorbar(cm.ScalarMappable(norm=norm2, cmap=cmap0), cax=cax2, ticks=[0, 1])


    plt.savefig('final_report_plots/processing.png', bbox_inches='tight', dpi=300)

if to_plot == 'processing_v2':
    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=3, wspace=0.16, hspace=0.17)
    gs.tight_layout(fig, w_pad=0.0)

    # nlinv image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.abs(img[0, :, :]), cmap='gray')
    ax1.axis('off')
    ax1.set_title(f'NLINV')

    # sens map
    ax2 = fig.add_subplot(gs[0, 1])
    abs = np.abs(smap1[0, :, :])
    ax2.imshow(abs, cmap=cmap0)
    ax2.axis('off')
    ax2.set_title('Magnitude')

    phase = smap1_phase[0, :, :]
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(phase, cmap='magma')
    ax3.axis('off')
    ax3.set_title('Phase')

    # coil image
    ax4 = fig.add_subplot(gs[1, 0])
    target = img[0, :, :] * smap1[0, :, :]
    ax4.imshow(np.abs(target), cmap='gray')
    ax4.axis('off')
    ax4.set_title(f'Coil Image')

    # us mask
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(mask, cmap='binary_r')
    ax5.axis('off')
    ax5.set_title('Mask (R=6)')

    # input data
    input_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img * smap1[0, :, :], axes=(-1, -2))), axes=(-1, -2)) * mask
    input_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input_kspace, axes=(-1, -2))), axes=(-1, -2))
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(np.abs(input_img[0, :, :]), cmap='gray')
    ax6.axis('off')
    ax6.set_title(f'Input Data')

    # colorbars
    cmap = cm.magma
    norm1 = mpl.colors.Normalize(vmin=0, vmax=3.1415)
    cax1 = fig.add_axes([0.895, 0.525, 0.02, 0.355])
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm1, cmap=cmap), cax=cax1, ticks=[0, 3.1415/2, 3.1415])
    cbar.ax.set_yticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$'])

    norm2 = mpl.colors.Normalize(vmin=0, vmax=1)
    cax2 = fig.add_axes([0.625, 0.525, 0.02, 0.355])
    plt.colorbar(cm.ScalarMappable(norm=norm2, cmap=cmap0), cax=cax2, ticks=[0, 1])


    plt.savefig('final_report_plots/processing_v2.png', bbox_inches='tight', dpi=300)
    
if to_plot == 'input_output_data':
    input_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img * smap1, axes=(-1, -2))), axes=(-1, -2)) * mask
    input_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input_kspace, axes=(-1, -2))), axes=(-1, -2))

    for i in range(input_img.shape[0]):
        re_slice, im_slice = np.real(input_img[i, :, :]), np.imag(input_img[i, :, :])
        plt.imshow(re_slice, cmap='gray')
        plt.axis('off')
        plt.savefig(f'final_report_plots/re_slice{i}.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.imshow(im_slice, cmap='gray')
        plt.axis('off')
        plt.savefig(f'final_report_plots/im_slice{i}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    plt.imshow(np.real(np.squeeze(img)), cmap='gray')
    plt.axis('off')
    plt.savefig(f'final_report_plots/target_re_slice.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.imshow(np.imag(np.squeeze(img)), cmap='gray')
    plt.axis('off')
    plt.savefig(f'final_report_plots/target_im_slice.png', dpi=300, bbox_inches='tight')
    plt.close()


if to_plot == 'jitter_v2':
    base_c = '#B997E1'
    model_c = '#EF9755'
    base_med_c = '#380D6B'
    model_med_c = '#94450B'

    plt.style.use('seaborn')
    Rs = [4, 8]  # can be modifed for more Rs, just make sure to fix adding the new axes and figsize to 14 (for 4 plots). also the legend intitiation
    smaps_versions = ['smap_9', 'smap10', 'smap101', 'smap102', 'smap103', 'smap104', 'smap11']

    fig_n = plt.figure(figsize=(10, 12))  
    fig_s = plt.figure(figsize=(10, 12))
    folder_path = '/home/pasala/ENSF619_Final_Project/results'
    smaps_versions = ['smap_9', 'smap_10', 'smap_101', 'smap_102', 'smap_103', 'smap_104', 'smap_11']


    for i, R in enumerate(Rs):
        base_s_meds = []
        model_s_meds = []
        base_p_meds = []
        model_p_meds = []

        for j, smap in enumerate(smaps_versions):
            base_results = pd.read_csv(f'{folder_path}/trad_results_R_{R}_square_new.csv')
            model_results = pd.read_csv(f'{folder_path}/results_smap_branch_v0_R_{R}_square_new.csv')
            results = pd.merge(base_results, model_results, how='left')

            base_s = results.loc[results['smap_style'] == smap, 'ssim_trad'].to_numpy()
            model_s = results.loc[results['smap_style'] == smap, 'ssim_model'].to_numpy()
            base_p = results.loc[results['smap_style'] == smap, 'psnr_trad'].to_numpy()
            model_p = results.loc[results['smap_style'] == smap, 'psnr_model'].to_numpy()

            # this is where we're going to put the jitter lines. But we need to add jitter ourselves unless we want to use seaborn
            base_idx, model_idx = np.full(len(base_s), j, dtype=np.float64), np.full(len(model_s), j + 0.4,  dtype=np.float64)
            jitter = np.random.uniform(low=-0.15, high=0.15, size=len(base_s))
            base_idx += jitter
            model_idx += jitter

            # deal with SSIM
            if j == 0:
                ax_s = fig_s.add_subplot(2, 1, i+1)
                ax_p = fig_n.add_subplot(2, 1, i+1)

                # fix things like ticks
                ax_s.set_xticks([0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2])
                ax_s.set_xticklabels(['Map 1', 'Map 2', 'Map 3', 'Map 4', 'Map 5', 'Map 6', 'Map 7'], fontsize='18')
                ax_s.tick_params(labelsize='18')
                ax_s.set_ylim(0, 1.03)

                ax_p.set_xticks([0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2])
                ax_p.set_xticklabels(['Map 1', 'Map 2', 'Map 3', 'Map 4', 'Map 5', 'Map 6', 'Map 7'], fontsize='18')
                ax_p.set_ylim(0, 55)
                ax_p.tick_params(labelsize='18')

                # and titles
                ax_s.set_title(f'R={R}', fontsize='20')
                ax_p.set_title(f'R={R}', fontsize='20')
                ax_s.set_ylabel('SSIM', fontsize='18')
                ax_p.set_ylabel('pSNR', fontsize='18')

            

            ax_s.scatter(base_idx, base_s, alpha=0.05, c=base_c)
            base_s_median = np.median(base_s)
            ax_s.scatter(model_idx, model_s, alpha=0.05, c=model_c)
            model_s_median = np.median(model_s)
            base_s_meds.append(base_s_median)
            model_s_meds.append(model_s_median)
            
            #ax_s.hlines(base_s_median, j - 0.15, j + 0.15, colors=base_med_c)
            #ax_s.hlines(model_s_median, j + 0.25, j + 0.55, colors=model_med_c)

            # deal with psnr
            ax_p.scatter(base_idx, base_p, alpha=0.05, c=base_c)
            base_p_median = np.median(base_p)
            ax_p.scatter(model_idx, model_p, alpha=0.05, c=model_c)
            model_p_median = np.median(model_p)
            base_p_meds.append(base_p_median)
            model_p_meds.append(model_p_median)

            if R == 8 and j == 6:  # so last one
                legend_elements_s = [mpl.lines.Line2D([0], [0], marker='o', markersize=10, color=base_c, lw=0, label='SENSE SSIM'),
                                     mpl.lines.Line2D([0], [0], marker='o', markersize=10, color=model_c, lw=0, label='Model SSIM'),
                                     mpl.lines.Line2D([0], [0], marker='o', markersize=9,color=base_med_c, lw=3, label='SENSE median SSIM'),
                                     mpl.lines.Line2D([0], [0], marker='o', markersize=9,color=model_med_c, lw=3, label='Model median SSIM')]
                legend_elements_p = [mpl.lines.Line2D([0], [0], marker='o', markersize=10, color=base_c, lw=0, label='SENSE pSNR'),
                                     mpl.lines.Line2D([0], [0], marker='o', markersize=10, color=model_c, lw=0, label='Model pSNR'),
                                     mpl.lines.Line2D([0], [0], marker='o', markersize=9,color=base_med_c, lw=3, label='SENSE median pSNR'),
                                     mpl.lines.Line2D([0], [0], marker='o', markersize=9,color=model_med_c, lw=3, label='Model median pSNR')]
                ax_s.legend(handles=legend_elements_s, fontsize='15')
                ax_p.legend(handles=legend_elements_p, fontsize='15')


            #ax_p.hlines(base_p_median, j - 0.15, j + 0.15, colors=base_med_c)
            #ax_p.hlines(model_p_median, j + 0.25, j + 0.55, colors=model_med_c)

        base = np.array([0, 1, 2, 3, 4, 5, 6])
        model = base + 0.4
        ax_s.plot(base, base_s_meds, 'o-', c=base_med_c)
        ax_s.plot(model, model_s_meds, 'o-', c=model_med_c)

        ax_p.plot(base, base_p_meds, 'o-', c=base_med_c)
        ax_p.plot(model, model_p_meds, 'o-', c=model_med_c)

        # do legend stuff
        legend_elements = [mpl.lines.Line2D([0], [0], color=base_med_c, lw=4, label='SENSE median SSIM'),
                           mpl.lines.Line2D([0], [0], color=model_med_c, lw=4, label='Model median SSIM'),
                           mpl.lines.Line2D([0], [0], marker='o', color=base_c, label='SENSE SSIM'),
                           mpl.lines.Line2D([0], [0], marker='o', color=model_c, label='Model SSIM')]

        #plt.style.use('default')
        #lax_s = fig_s.add_axes([0, -0.3, 0.5, 0.3])
        #lax_s.axis('off')
        #lax_s.legend(handles=legend_elements)

    fig_s.savefig('final_report_plots/ssim_jitter_v2_smap.png', dpi=300, bbox_inches='tight')
    fig_n.savefig('final_report_plots/psnr_jitter_v2_smap.png', dpi=300, bbox_inches='tight')

if to_plot == 'jitter_v3':
    base_c = '#B997E1'
    model_c = '#EF9755'
    base_med_c = '#380D6B'
    model_med_c = '#94450B'

    plt.style.use('seaborn')
    Rs = [6, 8]  # can be modifed for more Rs, just make sure to fix adding the new axes and figsize to 14 (for 4 plots). also the legend intitiation
    smaps_versions = ['smap_9', 'smap_10', 'smap_101', 'smap_102', 'smap_103', 'smap_104', 'smap_11']
    smaps_versions = ['circle_R_8', 'circle_R_9', 'circle_R_10', 'circle_R_11', 'circle_R_12']

    fig_n = plt.figure(figsize=(10, 12))  
    fig_s = plt.figure(figsize=(10, 12))
    folder_path = '/home/pasala/ENSF619_Final_Project/results/realistic_results'

    for i, R in enumerate(Rs):
        base_s_meds = []
        base_s_ses = []
        model_s_meds = []
        model_s_ses = []
        base_p_meds = []
        base_p_ses = []
        model_p_meds = []
        model_p_ses = []

        for j, smap in enumerate(smaps_versions):
            base_results = pd.read_csv(f'{folder_path}/trad_results_R_{R}_{smap}.csv')
            if smap == 'circle_R_8':
                model_results = pd.read_csv(f'{folder_path}/model_results_smap_branch_v{smap}_2_R_{R}_{smap}.csv')
            else:
                model_results = pd.read_csv(f'{folder_path}/model_results_smap_branch_v{smap}_1_R_{R}_{smap}.csv')
            results = pd.merge(base_results, model_results, how='left')

            base_s = results.loc[results['smap_style'] == smap, 'ssim_trad'].to_numpy()
            model_s = results.loc[results['smap_style'] == smap, 'ssim_model'].to_numpy()
            base_p = results.loc[results['smap_style'] == smap, 'psnr_trad'].to_numpy()
            model_p = results.loc[results['smap_style'] == smap, 'psnr_model'].to_numpy()

            # this is where we're going to put the jitter lines. But we need to add jitter ourselves unless we want to use seaborn
            base_idx, model_idx = np.full(len(base_s), j, dtype=np.float64), np.full(len(model_s), j + 0.4,  dtype=np.float64)
            jitter = np.random.uniform(low=-0.15, high=0.15, size=len(base_s))
            base_idx += jitter
            model_idx += jitter

            # deal with SSIM
            if j == 0:
                ax_s = fig_s.add_subplot(2, 1, i+1)
                ax_p = fig_n.add_subplot(2, 1, i+1)

                # fix things like ticks
                #ax_s.set_xticks([0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2])
                #ax_s.set_xticklabels(['Map 1', 'Map 2', 'Map 3', 'Map 4', 'Map 5', 'Map 6', 'Map 7'], fontsize='18')
                ax_s.set_xticks([0.2, 1.2, 2.2, 3.2, 4.2])
                ax_s.set_xticklabels(['Map 1', 'Map 2', 'Map 3', 'Map 4', 'Map 5'], fontsize='18')
                ax_s.tick_params(labelsize='18')
                ax_s.set_ylim(0, 1.03)

                #ax_p.set_xticks([0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2])
                #ax_p.set_xticklabels(['Map 1', 'Map 2', 'Map 3', 'Map 4', 'Map 5', 'Map 6', 'Map 7'], fontsize='18')
                ax_p.set_xticks([0.2, 1.2, 2.2, 3.2, 4.2])
                ax_p.set_xticklabels(['Map 1', 'Map 2', 'Map 3', 'Map 4', 'Map 5'], fontsize='18')
                ax_p.set_ylim(0, 55)
                ax_p.tick_params(labelsize='18')

                # and titles
                ax_s.set_title(f'R={R}', fontsize='20')
                ax_p.set_title(f'R={R}', fontsize='20')
                ax_s.set_ylabel('SSIM', fontsize='18')
                ax_p.set_ylabel('pSNR', fontsize='18')

            

            ax_s.scatter(base_idx, base_s, alpha=0.03, c=base_c)
            base_res = bootstrap((base_s,), np.median, n_resamples=1000)
            base_s_median = np.median(base_s)

            ax_s.scatter(model_idx, model_s, alpha=0.03, c=model_c)
            model_s_median = np.median(model_s)
            model_res = bootstrap((model_s,), np.median, n_resamples=1000)

            base_s_meds.append(base_s_median)
            base_s_ses.append(base_res.standard_error)
            model_s_meds.append(model_s_median)
            model_s_ses.append(model_res.standard_error)
            
            #ax_s.hlines(base_s_median, j - 0.15, j + 0.15, colors=base_med_c)
            #ax_s.hlines(model_s_median, j + 0.25, j + 0.55, colors=model_med_c)

            # deal with psnr
            ax_p.scatter(base_idx, base_p, alpha=0.03, c=base_c)
            base_res = bootstrap((base_p,), np.median, n_resamples=500)
            base_p_median = np.median(base_p)

            ax_p.scatter(model_idx, model_p, alpha=0.03, c=model_c)
            model_p_median = np.median(model_p)
            model_res = bootstrap((model_p,), np.median, n_resamples=500)

            base_p_meds.append(base_p_median)
            model_p_meds.append(model_p_median)
            base_p_ses.append(base_res.standard_error)
            model_p_ses.append(model_res.standard_error)

            if R == 10 and j == 4:  # so last one
                legend_elements_s = [mpl.lines.Line2D([0], [0], marker='o', markersize=10, color=base_c, lw=0, label='SENSE SSIM'),
                                     mpl.lines.Line2D([0], [0], marker='o', markersize=10, color=model_c, lw=0, label='Model SSIM'),
                                     mpl.lines.Line2D([0], [0], marker='o', markersize=9,color=base_med_c, lw=3, label='SENSE median SSIM'),
                                     mpl.lines.Line2D([0], [0], marker='o', markersize=9,color=model_med_c, lw=3, label='Model median SSIM')]
                legend_elements_p = [mpl.lines.Line2D([0], [0], marker='o', markersize=10, color=base_c, lw=0, label='SENSE pSNR'),
                                     mpl.lines.Line2D([0], [0], marker='o', markersize=10, color=model_c, lw=0, label='Model pSNR'),
                                     mpl.lines.Line2D([0], [0], marker='o', markersize=9,color=base_med_c, lw=3, label='SENSE median pSNR'),
                                     mpl.lines.Line2D([0], [0], marker='o', markersize=9,color=model_med_c, lw=3, label='Model median pSNR')]
                ax_s.legend(handles=legend_elements_s, fontsize='15')
                ax_p.legend(handles=legend_elements_p, fontsize='15')


            #ax_p.hlines(base_p_median, j - 0.15, j + 0.15, colors=base_med_c)
            #ax_p.hlines(model_p_median, j + 0.25, j + 0.55, colors=model_med_c)

        #base = np.array([0, 1, 2, 3, 4, 5, 6])
        base = np.array([0, 1, 2, 3, 4])
        model = base + 0.4
        #ax_s.plot(base, base_s_meds, 'o-', c=base_med_c)
        #ax_s.plot(model, model_s_meds, 'o-', c=model_med_c)
#
        #ax_p.plot(base, base_p_meds, 'o-', c=base_med_c)
        #ax_p.plot(model, model_p_meds, 'o-', c=model_med_c)

        ax_s.errorbar(base, base_s_meds, yerr=base_s_ses, fmt='o-', mfc=base_med_c, c=base_med_c)
        ax_s.errorbar(model, model_s_meds, yerr=model_s_ses, fmt='o-', mfc=model_med_c, c=model_med_c)

        ax_p.errorbar(base, base_p_meds, yerr=base_p_ses, fmt='o-', mfc=base_med_c, c=base_med_c)
        ax_p.errorbar(model, model_p_meds, yerr=model_p_ses, fmt='o-', mfc=model_med_c, c=model_med_c)

        print(base_s_meds)
        print(model_s_meds)
        print(base_p_meds)
        print(model_p_meds)

        # do legend stuff
        legend_elements = [mpl.lines.Line2D([0], [0], color=base_med_c, lw=4, label='SENSE median SSIM'),
                           mpl.lines.Line2D([0], [0], color=model_med_c, lw=4, label='Model median SSIM'),
                           mpl.lines.Line2D([0], [0], marker='o', color=base_c, label='SENSE SSIM'),
                           mpl.lines.Line2D([0], [0], marker='o', color=model_c, label='Model SSIM')]

        #plt.style.use('default')
        #lax_s = fig_s.add_axes([0, -0.3, 0.5, 0.3])
        #lax_s.axis('off')
        #lax_s.legend(handles=legend_elements)

    fig_s.savefig('plots/ssim_jitter_v3_realistic.png', dpi=300, bbox_inches='tight')
    fig_n.savefig('plots/psnr_jitter_v3_realistic.png', dpi=300, bbox_inches='tight')

if to_plot == 'sim_smaps':
    cmap = cm.magma
    abs = np.abs(smap1[3, :, :])
    plt.imshow(abs, cmap='gray')
    plt.axis('off')

    fig = plt.gcf()
    #norm2 = mpl.colors.Normalize(vmin=0, vmax=1)
    #cbar = plt.colorbar(cm.ScalarMappable(norm=norm2, cmap=cmap0), ticks=[0, 1])
    #cbar.ax.set_yticklabels(['0', '1'], fontsize='20')
    plt.colorbar(cmap='gray')

    plt.savefig('plots/processing_examples/smap_sim_abs.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    phase = smap1_phase[3, :, :]
    plt.imshow(phase, cmap=cmap)
    plt.axis('off')
    norm1 = mpl.colors.Normalize(vmin=0, vmax=3.1415)
    #cbar = plt.colorbar(cm.ScalarMappable(norm=norm1, cmap=cmap), ticks=[0, 3.1415/2, 3.1415])
    #cbar = plt.colorbar(ticks=[0, 3.1415/2, 3.1415])
    #cbar = plt.colorbar(ticks=[0, -3.1415/2, -3.1415])
    cbar = plt.colorbar(ticks=[0, -3.1415/4, -3.1415/2])
    #cbar.ax.set_yticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$'], fontsize='20')
    cbar.ax.set_yticklabels(['0', r'-$\frac{\pi}{4}$',r'-$\frac{\pi}{2}$'], fontsize='20')

    plt.savefig('plots/processing_examples/smap_sim_phase.png', dpi=300, bbox_inches='tight')

if to_plot == 'gfactor_comp':
    trad_R_6 = np.load('/mnt/c/Users/natal/Documents/Scripts/smap_testing/g_factor_maps/SENSE_circle_R_8_nomask_R=8.npy')
    trad_R_10 = np.load('/mnt/c/Users/natal/Documents/Scripts/smap_testing/g_factor_maps/SENSE_circle_R_8_nomask_R=10.npy')
    model_R_6 = [np.load('results/g_factor_maps/model_smap_branch_vcircle_R_8_2_results_R_8_ex1.npy'),
                 np.load('results/g_factor_maps/model_smap_branch_vcircle_R_8_2_results_R_8_ex2.npy'),
                 np.load('results/g_factor_maps/model_smap_branch_vcircle_R_8_2_results_R_8_ex3.npy')]
    model_R_10 = [np.load('results/g_factor_maps/model_smap_branch_vcircle_R_8_2_results_R_10_ex1_purenoise.npy'),
                  np.load('results/g_factor_maps/model_smap_branch_vcircle_R_8_2_results_R_10_ex2_purenoise.npy'),
                  np.load('results/g_factor_maps/model_smap_branch_vcircle_R_8_2_results_R_10_ex3_purenoise.npy')]
    trad_R_6 /= np.sqrt(8)
    all_6 = np.concatenate((trad_R_6, model_R_6[0], model_R_6[1], model_R_6[2]), axis=None)
    all_10 = np.concatenate((trad_R_6, model_R_6[0], model_R_6[1], model_R_6[2]), axis=None)

    min_6, max_6 = np.nanmin(all_6), np.nanmax(all_6)
    min_10, max_10 = np.nanmin(all_10), np.nanmax(all_10)

    cmap = cm.viridis
    fig, axes = plt.subplots(nrows=1, ncols=4)
    axes[0].imshow(trad_R_6, cmap=cmap, vmin=min_6, vmax=max_6)
    axes[0].axis('off')
    axes[0].set_title('SENSE')
    axes[1].imshow(model_R_6[0], cmap=cmap, vmin=min_6, vmax=max_6)
    axes[1].axis('off')
    axes[2].imshow(model_R_6[1], cmap=cmap, vmin=min_6, vmax=max_6)
    axes[2].axis('off')
    axes[2].set_title('Deep Learning Model')
    axes[3].imshow(model_R_6[2], cmap=cmap, vmin=min_6, vmax=max_6)
    axes[3].axis('off')

    norm = mpl.colors.Normalize(vmin=min_6, vmax=max_6)
    cax = fig.add_axes([0.92, 0.35, 0.03, 0.29])
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='G-factor')

    fig.savefig('plots/analysis/gfactor_R_8_nomask.png', dpi=300, bbox_inches='tight')

    #fig, axes = plt.subplots(nrows=1, ncols=4)
    #axes[0].imshow(trad_R_10, cmap='RdBu_r', vmin=min_10, vmax=max_10)
    #axes[0].axis('off')
    #axes[1].imshow(model_R_10[0], cmap='RdBu_r', vmin=min_10, vmax=max_10)
    #axes[1].axis('off')
    #axes[2].imshow(model_R_10[1], cmap='RdBu_r', vmin=min_10, vmax=max_10)
    #axes[2].axis('off')
    #axes[3].imshow(model_R_10[2], cmap='RdBu_r', vmin=min_10, vmax=max_10)
    #axes[3].axis('off')
#
    #norm = mpl.colors.Normalize(vmin=min_10, vmax=max_10)
    #cax = fig.add_axes([0.92, 0.35, 0.03, 0.29])
    #cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    #fig.savefig('plots/analysis/gfactor_R_10_purenoise.png', dpi=300, bbox_inches='tight')

if to_plot == 'gfactor_comp_models':
    model_rad_8_R_6 = [np.load(f'results/g_factor_maps/model_smap_branch_vcircle_R_8_2_results_R_6_ex{i}_purenoise.npy') for i in [1, 2, 3]]
    model_rad_9_R_6 = [np.load(f'results/g_factor_maps/model_smap_branch_vcircle_R_9_1_results_R_6_ex{i}_purenoise.npy') for i in [1, 2, 3]]
    model_rad_10_R_6 = [np.load(f'results/g_factor_maps/model_smap_branch_vcircle_R_10_1_results_R_6_ex{i}_purenoise.npy') for i in [1, 2, 3]]
    model_rad_11_R_6 = [np.load(f'results/g_factor_maps/model_smap_branch_vcircle_R_11_1_results_R_6_ex{i}_purenoise.npy') for i in [1, 2, 3]]
    model_rad_12_R_6 = [np.load(f'results/g_factor_maps/model_smap_branch_vcircle_R_12_1_results_R_6_ex{i}_purenoise.npy') for i in [1, 2, 3]]

    model_rad_8_R_10 = [np.load(f'results/g_factor_maps/model_smap_branch_vcircle_R_8_2_results_R_10_ex{i}_purenoise.npy') for i in [1, 2, 3]]
    model_rad_9_R_10 = [np.load(f'results/g_factor_maps/model_smap_branch_vcircle_R_9_1_results_R_10_ex{i}_purenoise.npy') for i in [1, 2, 3]]
    model_rad_10_R_10 = [np.load(f'results/g_factor_maps/model_smap_branch_vcircle_R_10_1_results_R_10_ex{i}_purenoise.npy') for i in [1, 2, 3]]
    model_rad_11_R_10 = [np.load(f'results/g_factor_maps/model_smap_branch_vcircle_R_11_1_results_R_10_ex{i}_purenoise.npy') for i in [1, 2, 3]]
    model_rad_12_R_10 = [np.load(f'results/g_factor_maps/model_smap_branch_vcircle_R_12_1_results_R_10_ex{i}_purenoise.npy') for i in [1, 2, 3]]

    all_6 = np.concatenate((model_rad_8_R_6[0], model_rad_8_R_6[1], model_rad_8_R_6[2],
                            model_rad_9_R_6[0], model_rad_9_R_6[1], model_rad_9_R_6[2],
                            model_rad_10_R_6[0], model_rad_10_R_6[1], model_rad_10_R_6[2],
                            model_rad_11_R_6[0], model_rad_11_R_6[1], model_rad_11_R_6[2],
                            model_rad_12_R_6[0], model_rad_12_R_6[1], model_rad_12_R_6[2]), axis=None)
    all_10 = np.concatenate((model_rad_8_R_10[0], model_rad_8_R_10[1], model_rad_8_R_10[2],
                             model_rad_9_R_10[0], model_rad_9_R_10[1], model_rad_9_R_10[2],
                             model_rad_10_R_10[0], model_rad_10_R_10[1], model_rad_10_R_10[2],
                             model_rad_11_R_10[0], model_rad_11_R_10[1], model_rad_11_R_10[2],
                             model_rad_12_R_10[0], model_rad_12_R_10[1], model_rad_12_R_10[2]), axis=None)
    
    min_6, max_6 = np.nanmin(all_6), np.nanmax(all_6)
    min_10, max_10 = np.nanmin(all_10), np.nanmax(all_10)

    fig = plt.figure()
    for i in range(3):
        ax1 = fig.add_subplot(3, 5, (i * 5) + 1)
        ax2 = fig.add_subplot(3, 5, (i * 5) + 2)
        ax3 = fig.add_subplot(3, 5, (i * 5) + 3)
        ax4 = fig.add_subplot(3, 5, (i * 5) + 4)
        ax5 = fig.add_subplot(3, 5, (i * 5) + 5)
            
        ax1.imshow(model_rad_8_R_6[i], cmap='RdBu_r', vmin=min_6, vmax=max_6)
        ax2.imshow(model_rad_9_R_6[i], cmap='RdBu_r', vmin=min_6, vmax=max_6)
        ax3.imshow(model_rad_10_R_6[i], cmap='RdBu_r', vmin=min_6, vmax=max_6)
        ax4.imshow(model_rad_11_R_6[i], cmap='RdBu_r', vmin=min_6, vmax=max_6)
        ax5.imshow(model_rad_12_R_6[i], cmap='RdBu_r', vmin=min_6, vmax=max_6)

        axes = [ax1, ax2, ax3, ax4, ax5]
        for ax in axes:
            ax.axis('off')

    fig.savefig('plots/analysis/gfactor_comp_model_R=6.png', dpi=300, bbox_inches='tight')

    fig = plt.figure()
    for i in range(3):
        ax1 = fig.add_subplot(3, 5, (i * 5) + 1)
        ax2 = fig.add_subplot(3, 5, (i * 5) + 2)
        ax3 = fig.add_subplot(3, 5, (i * 5) + 3)
        ax4 = fig.add_subplot(3, 5, (i * 5) + 4)
        ax5 = fig.add_subplot(3, 5, (i * 5) + 5)

        ax1.imshow(model_rad_8_R_10[i], cmap='RdBu_r', vmin=min_10, vmax=max_10)
        ax2.imshow(model_rad_9_R_10[i], cmap='RdBu_r', vmin=min_10, vmax=max_10)
        ax3.imshow(model_rad_10_R_10[i], cmap='RdBu_r', vmin=min_10, vmax=max_10)
        ax4.imshow(model_rad_11_R_10[i], cmap='RdBu_r', vmin=min_10, vmax=max_10)
        ax5.imshow(model_rad_12_R_10[i], cmap='RdBu_r', vmin=min_10, vmax=max_10)

        axes = [ax1, ax2, ax3, ax4, ax5]
        for ax in axes:
            ax.axis('off')

    fig.savefig('plots/analysis/gfactor_comp_model_R=10.png', dpi=300, bbox_inches='tight')

if to_plot == 'violin':
    exp = 'realistic'

    base_c = '#B997E1'
    model_c = '#EF9755'
    base_med_c = '#380D6B'
    model_med_c = '#94450B'

    Rs = [8]  # can be modifed for more Rs, just make sure to fix adding the new axes and figsize to 14 (for 4 plots). also the legend intitiation

    if exp == 'realistic':
        smaps_versions = ['circle_R_8', 'circle_R_9', 'circle_R_10', 'circle_R_11', 'circle_R_12']
    else:
        smaps_versions = ['smap_20', 'smap_40', 'smap_60', 'smap_80', 'smap_100']

    fig_s = plt.figure(figsize=(10, 12))  
    fig_p = plt.figure(figsize=(10, 12))
    fig_ph = plt.figure(figsize=(10, 12))
    folder_path = f'/home/pasala/ENSF619_Final_Project/results/{exp}_results'

    for i, R in enumerate(Rs):
        base_s_meds = []
        model_s_meds = []
        base_p_meds = []
        model_p_meds = []
        base_ph_meds = []
        model_ph_meds = []

        base_s_stds = []
        model_s_stds = []
        base_p_stds = []
        model_p_stds = []

        for j, smap in enumerate(smaps_versions):
            base_results = pd.read_csv(f'{folder_path}/trad_results_accel_{R}_{smap}_noise_sim.csv')
            model_results = pd.read_csv(f'{folder_path}/model_results_smap_branch_v{smap}_noise_final_accel_{R}_{smap}.csv')


            model_results['Method'] = 'DL model'
            base_results['Method'] = 'CG-SENSE'
            #model_results.rename(columns={'ssim_model':'SSIM', 'psnr_model':'PSNR'}, inplace=True)
            #base_results.rename(columns={'ssim_trad':'SSIM', 'psnr_trad':'PSNR'}, inplace=True)
            model_results.rename(columns={'ssim_model':'SSIM', 'psnr_model':'PSNR', 'phase_model':'IW-Absolute phase disparity'}, inplace=True)
            base_results.rename(columns={'ssim_trad':'SSIM', 'psnr_trad':'PSNR', 'phase_trad':'IW-Absolute phase disparity'}, inplace=True)
            
            if j == 0:
                results = pd.concat([base_results, model_results], join='inner')
            else:
                results = pd.concat([results, base_results, model_results], join='inner')
            

            results['slice'] = results['img_id'].str.split('_', expand=True)[2].str[1:].astype(int)
            results = results.loc[(results['slice'] >= 77) & (results['slice'] < 178), :]

            base_s = results.loc[(results['smap_style'] == smap) & (results['Method'] == 'CG-SENSE'), 'SSIM'].to_numpy()
            model_s = results.loc[(results['smap_style'] == smap) & (results['Method'] == 'DL model'), 'SSIM'].to_numpy()
            base_p = results.loc[(results['smap_style'] == smap) & (results['Method'] == 'CG-SENSE'), 'PSNR'].to_numpy()
            model_p = results.loc[(results['smap_style'] == smap) & (results['Method'] == 'DL model'), 'PSNR'].to_numpy()
            base_ph = results.loc[(results['smap_style'] == smap) & (results['Method'] == 'CG-SENSE'), 'IW-Absolute phase disparity'].to_numpy()
            model_ph = results.loc[(results['smap_style'] == smap) & (results['Method'] == 'DL model'), 'IW-Absolute phase disparity'].to_numpy()

            model_s_median = np.mean(model_s)
            base_s_median = np.mean(base_s)
            model_p_median = np.mean(model_p)
            base_p_median = np.mean(base_p)
            model_ph_median = np.median(model_ph)
            base_ph_median = np.median(base_ph)
            
            model_s_std = np.std(model_s)
            base_s_std = np.std(base_s)
            model_p_std = np.std(model_p)
            base_p_std = np.std(base_p)
            base_s_stds.append(base_s_std)
            model_s_stds.append(model_s_std)
            base_p_stds.append(base_p_std)
            model_p_stds.append(model_p_std)


            base_s_meds.append(base_s_median)
            model_s_meds.append(model_s_median)
            base_p_meds.append(base_p_median)
            model_p_meds.append(model_p_median)
            base_ph_meds.append(base_ph_median)
            model_ph_meds.append(model_ph_median)

            # deal with SSIM
            if j == 0:
                ax_s = fig_s.add_subplot(2, 1, i+1)
                ax_p = fig_p.add_subplot(2, 1, i+1)
                ax_ph = fig_ph.add_subplot(2, 1, i+1)

                # and titles
                ax_s.set_title(f'R={R}', fontsize='20')
                ax_p.set_title(f'R={R}', fontsize='20')
                ax_ph.set_title(f'R={R}', fontsize='20')
                ax_s.set_ylabel('SSIM', fontsize='18')
                ax_p.set_ylabel('pSNR', fontsize='18')
                ax_ph.set_ylabel('Absolute phase disparity', fontsize='18')

        if exp == 'realistic':
            results['smap_style'] = results['smap_style'].map({'circle_R_8':'8', 
                                                                'circle_R_9':'9', 
                                                                'circle_R_10':'10', 
                                                                'circle_R_11':'11', 
                                                                'circle_R_12':'12'})
            results.rename(columns={'smap_style':'Coil radius (cm)'}, inplace=True)
            p = sns.violinplot(data=results, x='Coil radius (cm)', y='SSIM', hue='Method', split=True, ax=ax_s, inner='quartile')
            p.set_xlabel('Fraction of sensitivity overlap', fontsize=18)
            p = sns.violinplot(data=results, x='Coil radius (cm)', y='PSNR', hue='Method', split=True, ax=ax_p, inner='quartile')
            p.set_xlabel('Fraction of sensitivity overlap', fontsize=18)
            p = sns.violinplot(data=results, x='Coil radius (cm)', y='IW-Absolute phase disparity', hue='Method', split=True, ax=ax_ph, inner='quartile', cut=0)
            p.set_xlabel('Fraction of sensitivity overlap', fontsize=18)

        else:
            results['smap_style'] = results['smap_style'].map({'smap_20':'0.20', 
                                                               'smap_40':'0.40', 
                                                               'smap_60':'0.60', 
                                                               'smap_80':'0.80', 
                                                               'smap_100':'1.00'})
            results.rename(columns={'smap_style':'Fraction of sensitivity overlap'}, inplace=True)
            #sns.set(font_scale = 1)
            p = sns.violinplot(data=results, x='Fraction of sensitivity overlap', y='SSIM', hue='Method', split=True, ax=ax_s, inner='quartile')
            p.set_xlabel('Fraction of sensitivity overlap', fontsize=18)
            p = sns.violinplot(data=results, x='Fraction of sensitivity overlap', y='PSNR', hue='Method', split=True, ax=ax_p, inner='quartile')
            p.set_xlabel('Fraction of sensitivity overlap', fontsize=18)
            p = sns.violinplot(data=results, x='Fraction of sensitivity overlap', y='IW-Absolute phase disparity', hue='Method', split=True, ax=ax_ph, inner='quartile', cut=0)
            p.set_xlabel('Fraction of sensitivity overlap', fontsize=18)

        ax_s.axhline(results['SSIM'].mean(), alpha=0.3, c='C2')
        ax_p.axhline(results['PSNR'].mean(), alpha=0.3, c='C2')
        ax_ph.axhline(results['IW-Absolute phase disparity'].mean(), alpha=0.3, c='C2')

        #ax_s.set_ybound(0, 1)
        #ax_p.set_ybound(10, 55)
        #ax_ph.set_ybound(0, 0.25)

        print(base_s_meds)
        print(base_s_stds)
        print(model_s_meds)
        print(model_s_stds)
        print(base_p_meds)
        print(base_p_stds)
        print(model_p_meds)
        print(model_p_stds)
        print(base_ph_meds)
        print(model_ph_meds)


    fig_s.savefig(f'plots/ssim_violin_{exp}_final_noise_accel_{R}.png', dpi=400, bbox_inches='tight')
    fig_p.savefig(f'plots/psnr_violin_{exp}_final_noise_accel_{R}.png', dpi=400, bbox_inches='tight')
    fig_ph.savefig(f'plots/phase_metric_violin_{exp}_final_noise_accel_{R}.png', dpi=400, bbox_inches='tight')

    
