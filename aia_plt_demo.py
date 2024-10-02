import os
from glob import glob
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, SqrtStretch

import sunpy.map
from sunpy.physics.differential_rotation import differential_rotate
from sunpy.net import Fido
from sunpy.net import attrs as a

import slice_tools as st

wave = 193
path_fits = './test/'
do_download = False
do_preprocess = True
do_difference = True
do_slice = True
standard_time = '2022-10-02T20:20:00'
save_aligned_fits = False
#FOVs = [240,-20,940,690]
FOVs = [240,-300,940,690]
pres = [300,360]
inif = [0,0,0,0]

channel = f'sdoaia{wave}'
sdoaia = matplotlib.colormaps[channel]
aia_maps = []

def align_fits_to_time_map(fits_file_path, standard_time, save_aligned_fits=False):
    aia_map = sunpy.map.Map(fits_file_path)
    aia_map = aia_map / aia_map.exposure_time
    standard_time = Time(standard_time)
    derotated_map = differential_rotate(aia_map, time=standard_time)
    if save_aligned_fits:
        output_filename = fits_file_path.replace(".fits", "_aligned.fits")
        derotated_map.save(output_filename, overwrite=True)
        print(f"自转校正后的FITS文件已保存到: {output_filename}")  
    return derotated_map

def save_data_to_fits(data, wcs_info, save_path, file_prefix, header_info=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    hdu = fits.PrimaryHDU(data, header=wcs_info.to_header())
    if header_info is not None:
        for key, value in header_info.items():
            hdu.header[key] = value
    fits_file_name = f"{file_prefix}.fits"
    fits_file_path = os.path.join(save_path, fits_file_name)
    hdu.writeto(fits_file_path, overwrite=True)
    return fits_file_path

if do_download:
    #User settings!!!
    t0 = '2022-10-02T20:20:00'
    t1 = '2022-10-02T20:25:00'
    email = 'y.w.ni@smail.nju.edu.cn'
    dt = 60*u.second
    #Require the ${path_save} already exists!

    if os.path.exists(path_fits):
        print(f"The folder:{path_save} already exists!")
    else:
        print(f"Creating a new folder: {path_save}")
        os.mkdir(path_save)
    
    print('Initiate data download...')
    print('Commence data retrieval process->')
    search_tuple1 = (
                    a.Time(t0, t1),
                    a.jsoc.Series('aia.lev1_euv_12s'),
                    a.jsoc.Segment('image'),
                    a.Wavelength(wave*u.AA),
                    a.Sample(dt),
                    a.jsoc.Notify(email)
                    )

    res_img1 = Fido.search(*search_tuple1)
    print('Downloading data ->')
    downloaded_files_img = Fido.fetch(res_img1,path=path_fits)
    print('Data download complete.')

if do_preprocess:
    fits_files = sorted(glob(os.path.join(path_fits, "*.fits")))
    
    if not fits_files:
        print("未找到任何FITS文件。")
    else:
        aia_maps = []
        
        for fits_file in fits_files:
            print(f"正在处理文件: {fits_file}")
            aligned_map = align_fits_to_time_map(fits_file, standard_time, save_aligned_fits=save_aligned_fits)
            aia_maps.append(aligned_map)

if do_difference:
    if len(aia_maps) < 2:
        print("无法进行运行差分，地图数量不足。")
    else:
        m_seq = sunpy.map.Map(aia_maps, sequence=True)
        m_seq_base = sunpy.map.Map([m - m_seq[0].quantity for m in m_seq[1:]], sequence=True)
        m_seq_running = sunpy.map.Map(
            [m - prev_m.quantity for m, prev_m in zip(m_seq[1:], m_seq[:-1])],
            sequence=True
        )

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(projection=m_seq.maps[0])
        ani = m_seq.plot(axes=ax1, norm=matplotlib.colors.LogNorm(vmin=10, vmax=5e3))
        plt.show()

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(projection=m_seq_base.maps[0])
        ani = m_seq_base.plot(axes=ax2, title='Base Difference', norm=colors.Normalize(vmin=-10, vmax=10), cmap='Greys_r')
        plt.colorbar(extend='both', label=m_seq_base[0].unit.to_string())
        plt.show()

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(projection=m_seq_running.maps[0])
        ani = m_seq_running.plot(axes=ax3, title='Running Difference', norm=colors.Normalize(vmin=-10, vmax=10), cmap='Greys_r')
        plt.colorbar(extend='both', label=m_seq_running[0].unit.to_string())
        plt.show()

        base_diff_maps = list(m_seq_base.maps)
        running_diff_maps = list(m_seq_running.maps)

if do_slice:
    proc_map = base_diff_maps
    fits_files = sorted(glob(os.path.join(path_fits, "*.fits")))
    for idx, (map_obj, fits_file) in enumerate(zip(proc_map, fits_files)):
        hdudata = fits.open(fits_file)
        hdu = hdudata[1].header
        b0 = hdu['CRLT_OBS']
        rsun = hdu['R_SUN']
        date_str = hdu['T_OBS']
        date_part = date_str[:10].replace('-', '') 
        time_part = date_str[11:19].replace(':', '')
        fpkl = f"PLT_{date_part}T{time_part}.pkl"
        fname = f"PLT_{date_part}T{time_part}.fits"
        bottom_left = SkyCoord(FOVs[0] * u.arcsec, FOVs[1] * u.arcsec, frame=map_obj.coordinate_frame)
        top_right = SkyCoord(FOVs[2] * u.arcsec, FOVs[3] * u.arcsec, frame=map_obj.coordinate_frame)
        submap = map_obj.submap(bottom_left, top_right=top_right)
        redo_slice = 'N'
        while redo_slice.upper() != 'Y':
            print("是否需要重新生成数据切片 (Y/N)? 默认[Y]:", end=" ")
            user_input = input().strip().upper()
            if user_input == '':
                redo_slice = 'Y'
            else:
                redo_slice = user_input
            if redo_slice == 'Y':
                print("重新生成数据切片")
                if idx==0:
                    GridInfo, sps, init_info = st.ai_sph_slice(submap, colormap=sdoaia, vrange=[-10,10],
                                                    is_norm=True, init_wave=True, init_info=inif)
                    inif = init_info
                else:
                    GridInfo, sps, _ = st.ai_sph_slice(submap, colormap=sdoaia, vrange=[-10,10],
                                                    is_norm=True, init_wave=False, init_info=inif)
                with open(fpkl, 'wb') as f:
                    pickle.dump((GridInfo, sps, inif), f)
                values, fov_range, yt, zt = st.get_slice_data(submap, GridInfo, pres[0], pres[1], 
                                            is_norm=True,len2deg=True,isplot=True,
                                            colormap=sdoaia, vrange=[-10, 10], selected_points=sps,
                                            colorplt=sdoaia, prange=[-10, 10])
                redo_slice = 'N'
            else:

                wcs_info = WCS(naxis=2)
                wcs_info.wcs.crpix = [int(pres[1]/2), int(pres[0]/2)]
                wcs_info.wcs.crval = [(GridInfo[2]+GridInfo[3])/2, (GridInfo[0]+GridInfo[1])/2]
                wcs_info.wcs.cdelt = [(GridInfo[3]-GridInfo[2])/pres[1], (GridInfo[1]-GridInfo[0])/pres[0]]
                wcs_info.wcs.ctype = ["RA---TAN", "DEC--TAN"]
                header_info = {'BUNIT': 'DN',
                               'DATE-OBS': date_str}
                fits_file_path = save_data_to_fits(values, wcs_info, path_fits, fname, header_info=header_info)
                print(f"FITS文件已保存到: {fits_file_path}")
                break

        

   

    