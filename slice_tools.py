import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames

def pixel_to_helioprojective(selected_points, rsun, swap_submap):
    # 获取太阳半径
    
    # 存储转换后的结果
    converted_points = []
    
    # 遍历选定的点
    for x_in, y_in in selected_points:
        # 将像素坐标转换为wcs坐标
        wcs_coord = swap_submap.pixel_to_world(x_in*u.pix, y_in*u.pix)
        
        # 计算对应的 y 和 z 坐标
        y_out = wcs_coord.Tx / rsun
        z_out = wcs_coord.Ty / rsun
        
        # 只保留值，不保留单位
        y_out_value = y_out.value
        z_out_value = z_out.value
        
        # 存储转换后的坐标
        converted_points.append((y_out_value, z_out_value))
    
    return converted_points

def rot3d(raxis, angle, degree=False):
    if isinstance(angle, np.ndarray) and angle.size == 1:
        angle = angle.item()  # 将单个值的 NumPy 数组转换为标量值

    if degree:
        angle = np.radians(angle)

    if raxis == 1:
        rot3darr = np.array([[1, 0, 0],
                              [0, np.cos(angle), -np.sin(angle)],
                              [0, np.sin(angle), np.cos(angle)]])
    elif raxis == 2:
        rot3darr = np.array([[np.cos(angle), 0, np.sin(angle)],
                              [0, 1, 0],
                              [-np.sin(angle), 0, np.cos(angle)]])
    elif raxis == 3:
        rot3darr = np.array([[np.cos(angle), -np.sin(angle), 0],
                              [np.sin(angle), np.cos(angle), 0],
                              [0, 0, 1]])
    else:
        raise ValueError("Invalid rotation axis")

    return rot3darr

def arcsec_to_pixel(yt, zt, swap_submap):
    
    world_coords = SkyCoord(yt * u.arcsec, zt * u.arcsec, frame=swap_submap.coordinate_frame)
    
    # 使用world_to_pixel函数将世界坐标转换为像素坐标
    pixel_coords = swap_submap.world_to_pixel(world_coords)
    
    # 提取像素坐标的数值部分
    x_pixel_values = pixel_coords.x.value
    y_pixel_values = pixel_coords.y.value
    
    return x_pixel_values, y_pixel_values

def spherical_to_cartesian(theta, phi):

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def cartesian_to_spherical(cartesian):

    x, y, z = cartesian
    theta = np.arccos(z)  # 纬度
    phi = np.arctan2(y, x)  # 经度
    return theta, phi

def rotation_matrix(theta_A, phi_A):

    # 绕 z 轴旋转 -phi_A，使点 A 移动到 xz 平面
    Rz = np.array([
        [np.cos(-phi_A), -np.sin(-phi_A), 0],
        [np.sin(-phi_A), np.cos(-phi_A), 0],
        [0, 0, 1]
    ])
    
    # 绕 y 轴旋转 -theta_A，使点 A 移动到北极 (0, 0, 1)
    Ry = np.array([
        [np.cos(-theta_A), 0, np.sin(-theta_A)],
        [0, 1, 0],
        [-np.sin(-theta_A), 0, np.cos(-theta_A)]
    ])
    
    # 总的旋转矩阵
    return Ry @ Rz

def rotate_points_spherical(theta, phi, theta_A, phi_A):
    
    B_cartesian = spherical_to_cartesian(theta, phi)
    R = rotation_matrix(theta_A, phi_A)
    B_rotated_cartesian = R @ B_cartesian
    theta_rot, phi_rot = cartesian_to_spherical(B_rotated_cartesian)
    
    return theta_rot, phi_rot

def fit_sphere_polar(points, initial_center):

    # 将初始中心转化为球面坐标
    init_phi = np.deg2rad(initial_center[0][0])
    init_theta = np.deg2rad(90 - initial_center[0][1])

    # 定义目标函数，最小化这些点和某个假设北极的纬度差异
    def objective(center):
        # 使用旋转后的球面坐标计算纬度差异
        rotated_theta_vals = []
        for lon, lat in points:
            theta = np.deg2rad(90 - lat)
            phi = np.deg2rad(lon)
            theta_rot, phi_rot = rotate_points_spherical(theta, phi, center[0], center[1])
            rotated_theta_vals.append(theta_rot)

        return np.var(rotated_theta_vals)  # 目标是最小化纬度的方差

    # 初始极点猜测
    initial_guess = [init_theta, init_phi]

    # 使用优化方法找到最佳的旋转角度
    result = minimize(objective, initial_guess, method='L-BFGS-B', tol=1e-9)
    theta_opt, phi_opt = result.x

    # 最佳旋转角度对应的新极点的经纬度
    new_pole = [theta_opt, phi_opt]

    return new_pole  # 返回新的极点 (theta, phi)

def inverse_transform(chi_0, mu, theta_O, phi_O):

    x_prime = np.sin(chi_0) * np.cos(mu)
    y_prime = np.sin(chi_0) * np.sin(mu)
    z_prime = np.cos(chi_0)

    x1 = np.cos(theta_O) * x_prime + np.sin(theta_O) * z_prime
    z1 = -np.sin(theta_O) * x_prime + np.cos(theta_O) * z_prime
    y1 = y_prime

    S_x = np.cos(phi_O) * x1 - np.sin(phi_O) * y1
    S_y = np.sin(phi_O) * x1 + np.cos(phi_O) * y1
    S_z = z1

    theta_p = np.arccos(S_z)
    phi_p = np.arctan2(S_y, S_x)
    
    return theta_p, phi_p

def is_point_on_small_arc(theta_p, phi_p, theta_A, phi_A, theta_B, phi_B):
    cos_gamma_AB = np.sin(theta_A) * np.sin(theta_B) + np.cos(theta_A) * np.cos(theta_B) * np.cos(phi_A - phi_B)
    gamma_AB = np.arccos(np.clip(cos_gamma_AB, -1.0, 1.0))
    cos_gamma_AP = np.sin(theta_A) * np.sin(theta_p) + np.cos(theta_A) * np.cos(theta_p) * np.cos(phi_A - phi_p)
    gamma_AP = np.arccos(np.clip(cos_gamma_AP, -1.0, 1.0))
    cos_gamma_BP = np.sin(theta_B) * np.sin(theta_p) + np.cos(theta_B) * np.cos(theta_p) * np.cos(phi_B - phi_p)
    gamma_BP = np.arccos(np.clip(cos_gamma_BP, -1.0, 1.0))
    return gamma_AP <= gamma_AB and np.abs(gamma_AP + gamma_BP - gamma_AB) < 1e-6

def objective_function(mu, chi_0, theta_A, phi_A, theta_B, phi_B, theta_O, phi_O):

    theta_p, phi_p = inverse_transform(chi_0, mu, theta_O, phi_O)
    
    if is_point_on_small_arc(theta_p, phi_p, theta_A, phi_A, theta_B, phi_B):
        return 0
    else:
        cos_dist_A = np.sin(theta_A) * np.sin(theta_p) + np.cos(theta_A) * np.cos(theta_p) * np.cos(phi_A - phi_p)
        dist_A = np.arccos(np.clip(cos_dist_A, -1.0, 1.0))
        
        cos_dist_B = np.sin(theta_B) * np.sin(theta_p) + np.cos(theta_B) * np.cos(theta_p) * np.cos(phi_B - phi_p)
        dist_B = np.arccos(np.clip(cos_dist_B, -1.0, 1.0))
        
        return min(dist_A, dist_B)

def find_mu_for_chi0(chi_0, theta_A, phi_A, theta_B, phi_B, theta_O, phi_O):
    result = minimize(objective_function, 0, bounds=[(-np.pi, np.pi)], args=(chi_0, theta_A, phi_A, theta_B, phi_B, theta_O, phi_O))
    return result.x[0]

def get_data(x_index, y_index, map):
    data = map.data
    # 定义要插值的坐标
    x_coords = range(data.shape[1])
    y_coords = range(data.shape[0])
    
    # 创建插值函数
    interpolated_data = RegularGridInterpolator((y_coords, x_coords), data)
    
    # 对坐标进行插值
    interpolated_values = interpolated_data((y_index.flatten(), x_index.flatten()))
    
    # 将插值结果重新整形为与 x_index 和 y_index 相同的形状
    interpolated_values_reshaped = interpolated_values.reshape(x_index.shape)
    
    return interpolated_values_reshaped

def get_sph_slice(B0, rsun, swap_submap):

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(projection=swap_submap)
    image = swap_submap.plot(axes=ax, cmap='bwr', origin='lower', vmin = -4, vmax = 4)
    swap_submap.draw_limb(axes=ax)
    swap_submap.draw_grid(axes=ax)
    ax.title.set_visible(False)
    selected_center = []
    def select_center(event):
        if event.inaxes is None:
          return
    
        global x_c, y_c
        # 获取鼠标点击位置的坐标
        x_c = event.xdata
        y_c = event.ydata
        selected_center.append((x_c, y_c))

        # 在图像上绘制已选取的点
        plt.plot(x_c, y_c, 'b+', markersize=10)
        plt.draw()

        # 判断是否已经选取了足够的点
        if len(selected_center) >= 1:
          print(f'已选取{1}个点，停止选择...')
          plt.close()  # 关闭当前绘图窗口以继续执行代码
    fig.canvas.mpl_connect('button_press_event', select_center)
    plt.show()
    plt.close()

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(projection=swap_submap)
    image = swap_submap.plot(axes=ax, cmap='bwr', origin='lower', vmin = -4, vmax = 4)
    swap_submap.draw_limb(axes=ax)
    swap_submap.draw_grid(axes=ax)
    plt.plot(x_c, y_c, 'b+', markersize=10)
    ax.title.set_visible(False)
    selected_points = []
    def select_points(event):
        if event.inaxes is None:
            return
        
        # 获取鼠标点击位置的坐标
        x = event.xdata
        y = event.ydata

        # 将选取的点坐标添加到列表中
        selected_points.append((x, y))
    
        # 在图像上绘制已选取的点
        plt.plot(x, y, 'r+', markersize=10)   
        plt.draw()

        # 判断是否已经选取了足够的点
        if event.button == 3 and len(selected_points) > 3:
            num_sp = len(selected_points)
            print(f'已选取{num_sp}个点，停止选择...')
            plt.close()  # 关闭当前绘图窗口以继续执行代码
    fig.canvas.mpl_connect('button_press_event', select_points)
    plt.show()
    plt.close()

    R_nB0 = rot3d(2, -B0)

    converted_center = pixel_to_helioprojective(selected_center, rsun, swap_submap)
    ysc = np.array([ycs for ycs, _ in converted_center])
    zsc = np.array([zcs for _, zcs in converted_center])
    xsc = np.sqrt(1.0 - ysc ** 2 - zsc ** 2)

    f_sc = np.zeros((3,1))
    f_sc[0,:] = xsc
    f_sc[1,:] = ysc
    f_sc[2,:] = zsc
    f_c = np.dot(R_nB0, f_sc)
    xcc = f_c[0]
    ycc = f_c[1]
    zcc = f_c[2]
    # 计算经度和纬度
    phcc = np.arctan2(ycc, xcc)
    thcc = np.arccos(zcc)

    # 将弧度转换为度数
    loncc = np.degrees(phcc)
    latcc = 90 - np.degrees(thcc)
    print("中心经纬度：",loncc,latcc)

    converted_points = pixel_to_helioprojective(selected_points, rsun, swap_submap)
    yi = np.array([yis for yis, _ in converted_points])
    zi = np.array([zis for _, zis in converted_points])
    xi = np.sqrt(1.0 - yi ** 2 - zi ** 2)

    # 计算theta_i和phi_i
    thd = -thcc
    phd = -phcc
    num = xi.size
    th_i = np.zeros(num)
    ph_i = np.zeros(num)
    f_0 = np.zeros((3,num))
    f_0[0,:] = xi
    f_0[1,:] = yi
    f_0[2,:] = zi
    R_z = rot3d(3, phd)
    R_y = rot3d(2, thd)
    R_t0 = np.dot(R_z, R_nB0)
    R_t = np.dot(R_y, R_t0)
    f_1 = np.dot(R_t, f_0)
    for ii in range(num):
        th_i[ii] = np.arccos(f_1[2,ii])
        ph_i[ii] = np.arctan2(f_1[1,ii], f_1[0,ii])

    # 判断数组的第一个值并根据条件修改剩余值
    if ph_i[0] < 0:
        # 如果第一个值小于0，数组中所有大于0的值减去2π
        ph_i[1:][ph_i[1:] > 0] -= 2 * np.pi
    else:
        # 如果第一个值大于等于0，数组中所有小于0的值加上2π
        ph_i[1:][ph_i[1:] < 0] += 2 * np.pi

    thi_min = 0.0
    thi_max = np.max(th_i)
    phi_min = np.min(ph_i)
    phi_max = np.max(ph_i)

    thi_deg_min = thi_min * 180/np.pi
    thi_deg_max = thi_max * 180/np.pi
    phi_deg_min = phi_min * 180/np.pi
    phi_deg_max = phi_max * 180/np.pi

    GridInfo = np.zeros(8)
    # 交互式地获取phi_min，phi_max, thi_min和thi_max
    GridInfo[0] = np.deg2rad(float(input('给出弧面的脊宽度最大值[单位：deg](' + str(thi_deg_min) + '): ')))
    GridInfo[1] = np.deg2rad(float(input('给出弧面的脊宽度最大值[单位：deg](' + str(thi_deg_max) + '): ')))
    GridInfo[2] = np.deg2rad(float(input('给出弧面的角宽度最小值[单位：deg](' + str(phi_deg_min) + '): ')))
    GridInfo[3] = np.deg2rad(float(input('给出弧面的角宽度最大值[单位：deg](' + str(phi_deg_max) + '): ')))
    GridInfo[4] = thcc
    GridInfo[5] = phcc
    GridInfo[6] = rsun
    GridInfo[7] = B0

    return GridInfo, selected_points

def ai_sph_slice(swap_submap, colormap='bwr',vrange=[-4,4], 
                 is_norm=False, init_wave=False, init_info=[0,0,0,0]):


    fig = plt.figure(figsize=(6,6))  # 创建一个单独的子图
    ax = fig.add_subplot(projection=swap_submap)
    swap_submap.plot(axes=ax, cmap=colormap, origin='lower', vmin=vrange[0], vmax=vrange[1])
    swap_submap.draw_limb(axes=ax)
    swap_submap.draw_grid(axes=ax)
    ax.title.set_visible(False)
    selected_center = []
    selected_center_pixel = []
    def select_center(event):
        if event.inaxes is None:
            return

        if event.button == 1:
            x_c = event.xdata
            y_c = event.ydata
            hp_coords = swap_submap.pixel_to_world(x_c * u.pix, y_c * u.pix)
            hg_coords = hp_coords.transform_to(frames.HeliographicStonyhurst(obstime=swap_submap.date))
            world_coords_lon = hg_coords.lon.to(u.deg).value
            world_coords_lat = hg_coords.lat.to(u.deg).value
            selected_center.append((world_coords_lon, world_coords_lat))
            pixel_coords = swap_submap.world_to_pixel(hp_coords)
            selected_center_pixel.append((pixel_coords.x.value,pixel_coords.y.value))
            plt.plot(pixel_coords.x.value, pixel_coords.y.value, 'r+', markersize=10)
            plt.draw()

        if event.button == 3:
            print(f'已选取{1}个点，停止选择...')
            plt.close()  # 关闭当前绘图窗口以继续执行代码
            return
    fig.canvas.mpl_connect('button_press_event', select_center)
    plt.show()
    plt.close()

    fig = plt.figure(figsize=(6,6))  # 创建一个单独的子图
    ax = fig.add_subplot(projection=swap_submap)
    swap_submap.plot(axes=ax, cmap=colormap, origin='lower', vmin=vrange[0], vmax=vrange[1])
    swap_submap.draw_limb(axes=ax)
    swap_submap.draw_grid(axes=ax)
    ax.title.set_visible(False)
    selected_points = []
    selected_points_pixel = []
    def select_points(event):
        if event.inaxes is None:
            return

        if event.button == 1:
            x = event.xdata
            y = event.ydata
            hp_coords = swap_submap.pixel_to_world(x * u.pix, y * u.pix)
            hg_coords = hp_coords.transform_to(frames.HeliographicStonyhurst(obstime=swap_submap.date))
            world_coords_lon = hg_coords.lon.to(u.deg).value  # 经度 (度)
            world_coords_lat = hg_coords.lat.to(u.deg).value  # 纬度 (度)
            selected_points.append((world_coords_lon, world_coords_lat))
            pixel_coords = swap_submap.world_to_pixel(hp_coords)
            selected_points_pixel.append((pixel_coords.x.value,pixel_coords.y.value))
            plt.plot(pixel_coords.x.value, pixel_coords.y.value, 'r+', markersize=10)
            plt.draw()

        if event.button == 3 and len(selected_points) > 3:
            num_sp = len(selected_points)
            print(f'已选取{num_sp}个点，停止选择...')
            plt.close() 
    fig.canvas.mpl_connect('button_press_event', select_points)
    plt.show()
    plt.close()

    init_center = np.array([(x, y) for x, y in selected_center])
    given_point = np.array([(x, y) for x, y in selected_points])
    new_pole = fit_sphere_polar(given_point,init_center)

    thcc = new_pole[0]
    phcc = new_pole[1]
    point_map = []
    for lon, lat in given_point:
            theta = np.deg2rad(90 - lat)
            phi = np.deg2rad(lon)
            theta_rot, phi_rot = rotate_points_spherical(theta, phi, thcc, phcc)
            point_map.append([theta_rot, phi_rot])
    theta_rot_vals = np.array([theta_rot for theta_rot, phi_rot in point_map])
    phi_rot_vals = np.array([phi_rot for theta_rot, phi_rot in point_map])

    if init_wave:
        fig = plt.figure(figsize=(6,6))  # 创建一个单独的子图
        ax = fig.add_subplot(projection=swap_submap)
        swap_submap.plot(axes=ax, cmap=colormap, origin='lower', vmin=vrange[0], vmax=vrange[1])
        swap_submap.draw_limb(axes=ax)
        swap_submap.draw_grid(axes=ax)
        ax.title.set_visible(False)
        selected_inclined = []
        selected_inclined_pixel = []
        o_pixel = np.array([(x, y) for x, y in selected_center_pixel])
        f_pixel = np.array([(x, y) for x, y in selected_points_pixel])
        plt.plot(o_pixel[0][0], o_pixel[0][1], 'bs', markersize=10)
        plt.plot(np.array(f_pixel[:,0]), np.array(f_pixel[:,1]), 'r+', markersize=10)
        def select_inclined(event):
            if event.inaxes is None:
                return

            if event.button == 1:
                x_c = event.xdata
                y_c = event.ydata
                hp_coords = swap_submap.pixel_to_world(x_c * u.pix, y_c * u.pix)
                hg_coords = hp_coords.transform_to(frames.HeliographicStonyhurst(obstime=swap_submap.date))
                world_coords_lon = hg_coords.lon.to(u.deg).value
                world_coords_lat = hg_coords.lat.to(u.deg).value
                selected_inclined.append((world_coords_lon, world_coords_lat))
                pixel_coords = swap_submap.world_to_pixel(hp_coords)
                selected_inclined_pixel.append((pixel_coords.x.value,pixel_coords.y.value))
                plt.plot(pixel_coords.x.value, pixel_coords.y.value, 'kx', markersize=10)
                plt.draw()

            if event.button == 3:
                print(f'已选取{2}个点，停止选择...')
                plt.close()  # 关闭当前绘图窗口以继续执行代码
                return
        fig.canvas.mpl_connect('button_press_event', select_inclined)
        plt.show()
        plt.close()
        theta_si = np.array([x for x, y in selected_inclined])
        phi_si = np.array([y for x, y in selected_inclined])
        init_info = [theta_si[0], theta_si[1], phi_si[0], phi_si[1]]
  
    theta_A = init_info[0]
    theta_B = init_info[1]
    phi_A = init_info[2]
    phi_B = init_info[3]

    print(theta_rot_vals)
    print(phi_rot_vals)
    phi_rot_vals[phi_rot_vals < 0] += 2 * np.pi

    thi_min = np.min(theta_rot_vals)
    thi_max = np.max(theta_rot_vals)
    phi_min = np.min(phi_rot_vals)
    phi_max = np.max(phi_rot_vals)

    thi_deg_min = np.rad2deg(thi_min)
    thi_deg_max = np.rad2deg(thi_max)
    phi_deg_min = np.rad2deg(phi_min)
    phi_deg_max = np.rad2deg(phi_max)


    GridInfo = np.zeros(7)
    # 交互式地获取phi_min，phi_max, thi_min和thi_max
    GridInfo[0] = np.deg2rad(float(input('给出弧面的脊宽度最大值[单位：deg](' + str(thi_deg_min) + '->' + str(thi_deg_max) + '): ')))
    GridInfo[1] = np.deg2rad(float(input('给出弧面的脊宽度最大值[单位：deg](' + str(thi_deg_min) + '->' + str(thi_deg_max) + '): ')))
    GridInfo[2] = np.deg2rad(float(input('给出弧面的角宽度最小值[单位：deg](' + str(phi_deg_min) + '->' + str(phi_deg_max) + '): ')))
    GridInfo[3] = np.deg2rad(float(input('给出弧面的角宽度最大值[单位：deg](' + str(phi_deg_min) + '->' + str(phi_deg_max) + '): ')))
    GridInfo[4] = thcc
    GridInfo[5] = phcc

    if is_norm:
        chi_0 = (GridInfo[0] + GridInfo[1]) / 2
        mu = find_mu_for_chi0(chi_0, theta_A, phi_A, theta_B, phi_B, thcc, phcc)
        GridInfo[6] = mu

    return GridInfo, selected_points_pixel, init_info

def get_slice_data(swap_submap,GridInfo,nth,nph,
                   is_norm=False,len2deg=False,isplot=False,
                   colormap='bwr',vrange=[-4,4],selected_points=None,
                   colorplt='bwr',prange=[-2,2]
                   ):

    Rsun = swap_submap.meta.get('rsun_obs', None)
    observer = swap_submap.observer_coordinate
    # 生成新的网格<thg,phg>
    thg = np.zeros((nph, nth))
    phg = np.zeros((nph, nth))
    ith_values = np.linspace(GridInfo[0], GridInfo[1], nth)
    if is_norm:
        mu = GridInfo[6]
        oiph_values = np.linspace(-np.pi, np.pi, nph)
        iph_values = oiph_values + mu
    else:
        iph_values = np.linspace(GridInfo[2], GridInfo[3], nph)
    phg, thg = np.meshgrid(iph_values, ith_values, indexing='ij')
    lon_grid = np.zeros((nph, nth))
    lat_grid = np.zeros((nph, nth))
    for jj in range(nth):
            for ii in range(nph):
                theta_rot, phi_rot = inverse_transform(thg[ii,jj], phg[ii,jj], GridInfo[4], GridInfo[5])
                lon_grid[ii,jj] = np.rad2deg(phi_rot)
                lat_grid[ii,jj] = 90 - np.rad2deg(theta_rot)
    hg_coords = SkyCoord(lon_grid * u.deg, lat_grid * u.deg, 
                     frame=frames.HeliographicStonyhurst, 
                     obstime=swap_submap.date)
    hp_coords = hg_coords.transform_to(frames.Helioprojective(observer=observer))
    pixel_f = swap_submap.world_to_pixel(hp_coords)
    x_pixel = pixel_f.x.value
    y_pixel = pixel_f.y.value
    cdelt1 = swap_submap.scale[0].value
    cdelt2 = swap_submap.scale[1].value
    reference_pixel_x = swap_submap.reference_pixel.x.value
    reference_pixel_y = swap_submap.reference_pixel.y.value
    yt = (x_pixel - reference_pixel_x) * cdelt1
    zt = (y_pixel - reference_pixel_y) * cdelt2


    fov_range = np.zeros(4)
    if len2deg:
        fov_range[0] = GridInfo[0]
        fov_range[1] = GridInfo[1]
        fov_range[2] = -np.pi
        fov_range[3] = np.pi
    else:
        fov_range[0] = 0
        fov_range[1] = Rsun*(GridInfo[1]-GridInfo[0])
        fov_range[2] = 0
        fov_range[3] = Rsun*np.abs(np.sin((GridInfo[1]+GridInfo[0])/2.))*(GridInfo[3] - GridInfo[2])
    values = get_data(x_pixel, y_pixel, swap_submap)

    if isplot is True:
        fig = plt.figure(figsize=(12, 6))

        ax1 = fig.add_subplot(1, 2, 1, projection=swap_submap)
        swap_submap.plot(axes=ax1, cmap=colormap, origin='lower', vmin=vrange[0], vmax=vrange[1], aspect='auto')
        swap_submap.draw_limb(axes=ax1)
        swap_submap.draw_grid(axes=ax1)
        ax1.title.set_visible(False)
        ax1.set_title('Swap Submap')  # 设置标题
        # 判断 selected_points 是否存在并绘制在第一个子图中
        if selected_points is not None:
            for x_in, y_in in selected_points:
               ax1.plot(x_in, y_in, 'r+', markersize=10)
        new_yg_flat = x_pixel.ravel()
        new_zg_flat = y_pixel.ravel()
        ax1.plot(new_yg_flat, new_zg_flat, 'g+', markersize=0.05)
        ax1.set_frame_on(False)

        # 第二个子图绘制 values 数组
        ax2 = fig.add_subplot(1, 2, 2)  # 第二个子图
        im = ax2.imshow(values, cmap=colorplt, origin='lower', vmin=prange[0], vmax=prange[1], extent=fov_range, aspect='auto')  # 在 ax2 上绘制 imshow 图像
        ax2.set_xlabel('Spine Length (arcsec)')
        ax2.set_ylabel('Averaged Arc Length (arcsec)')
        ax2.set_title('Slice Image')
        # 添加颜色条
        fig.colorbar(im, ax=ax2)
        plt.show()

    return values, fov_range, yt, zt


