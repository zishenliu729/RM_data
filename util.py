import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import drjit as dr
import os
import random
def count_subfolders_with_os(folder_path):
    """使用os模块统计子文件夹数量"""
    try:
        # 检查路径是否存在且是一个目录
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"路径不存在: {folder_path}")
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"不是一个目录: {folder_path}")
        
        # 统计子文件夹数量
        count = 0
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                count += 1
        return count
    except Exception as e:
        print(f"错误: {e}")
        return -1

def cars_movement(scene,displacement_vec):
    j= 1
    formatted_num = f"{j:03d}"
    a = scene.get(f"mesh-car_x_p_{formatted_num}")
    b = scene.get(f"mesh-car_x_n_{formatted_num}")
    c = scene.get(f"mesh-car_y_p_{formatted_num}")
    d = scene.get(f"mesh-car_y_n_{formatted_num}")
    while a is not None or b is not None or c is not None or d is not None:

        # Compute and render a coverage map at 0.5m above the ground
            

        j += 1
        formatted_num = f"{j:03d}"
        a = scene.get(f"mesh-car_x_p_{formatted_num}")
        b = scene.get(f"mesh-car_x_n_{formatted_num}")
        c = scene.get(f"mesh-car_y_p_{formatted_num}")
        d = scene.get(f"mesh-car_y_n_{formatted_num}")

        if a != None:
            scene.get(f"mesh-car_x_p_{formatted_num}").position += displacement_vec["x_p"]
        if b != None:
            scene.get(f"mesh-car_x_n_{formatted_num}").position += displacement_vec["x_n"]
        if c != None:
            scene.get(f"mesh-car_y_p_{formatted_num}").position += displacement_vec["y_p"]
        if d != None:
            scene.get(f"mesh-car_y_n_{formatted_num}").position += displacement_vec["y_n"]

    return scene

def tx_movement(scene,move_vec):
    scene.get("tx").position += move_vec
    return scene

def cars_desity_change(scene,radio):
    total_car = []
    for sh in scene.objects:
        if "mesh-car" in sh:
            total_car.append(sh)
    num_car = len(total_car)
    if num_car == 0:
        return scene
    n = round(num_car*radio)
    n = max(0,min(n,num_car))
    remove_car = random.sample(total_car,n)
    scene.edit(remove=remove_car)
    return scene

def process_masks(rm_mask, building_mask, car_mask):
    """
    处理三张mask图像：使mask3在mask1或mask2为1的位置均为0
    
    参数:
        mask1, mask2, mask3: 输入的灰度图mask（numpy数组），元素值为0或1，
                           形状需完全相同 (height, width)
    返回:
        处理后的mask3（numpy数组）
    """
    # 检查三张图尺寸是否一致
    if not (rm_mask.shape == building_mask.shape == car_mask.shape):
        raise ValueError("三张mask图像的尺寸必须完全相同！")
    
    # 步骤1：找到mask1或mask2中为1的位置（逻辑或运算）
    # 注：若mask是0-255的灰度图，需先转换为0/1（如 mask1 == 255）
    # 这里假设输入mask已是0/1的二值图
    mask_obstacle = np.logical_or(car_mask, building_mask)  # 结果为布尔数组，True表示需要清零的位置
    
    # 步骤2：将mask3中上述位置设为0
    processed_mask3 = rm_mask.copy()  # 避免修改原数组
    processed_mask3[mask_obstacle] = 0  # 符合条件的位置强制设为0
    
    return processed_mask3



def process_masks_3channel(rm_mask, building_mask, car_mask,tx_mask):
    """
    处理三张mask图像：使mask3在mask1或mask2为1的位置均为0
    
    参数:
        mask1, mask2, mask3: 输入的灰度图mask（numpy数组），元素值为0或1，
                           形状需完全相同 (height, width)
    返回:
        处理后的mask3（numpy数组）
    """
    
    # 步骤1：找到mask1或mask2中为1的位置（逻辑或运算）
    # 注：若mask是0-255的灰度图，需先转换为0/1（如 mask1 == 255）
    # 这里假设输入mask已是0/1的二值图
    # mask_obstacle = np.logical_or(car_mask, building_mask)  # 结果为布尔数组，True表示需要清零的位置
    
    # 步骤2：将mask3中上述位置设为0
    processed_mask3 = rm_mask.copy()  # 避免修改原数组
    building_bool = (building_mask == 1)  # 形状 (h, w)，True表示建筑物位置
    car_bool = (car_mask == 1)            # 车辆位置
    tx_bool = (tx_mask == 1)  
    processed_mask3[building_bool] = [255, 128, 0]  # 符合条件的位置强制设为0
    processed_mask3[car_bool] = [0, 255, 255]
    processed_mask3[tx_bool] = [255,0,0]
     # 将mask3中上述位置设为0
    
    return processed_mask3

def gray_generate(radio_map,building_mask,car_mask,metric,file_name,db_scale: bool = True):
    
    rm_real = radio_map.path_gain.numpy().squeeze(axis=0)
    if metric=="rss" and db_scale:
        rm_values *= 1000
    valid = np.logical_and(rm_real > 0., np.isfinite(rm_real))
    opacity = valid.astype(np.float32)
    any_valid = np.any(valid)
    rm_real[valid] = 10. * np.log10(rm_real[valid])

    vmin = rm_real[valid].min() if any_valid else 0

    vmax = rm_real[valid].max() if any_valid else 0
    normalizer = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    color_map = matplotlib.colormaps.get_cmap('gray')
    texture = color_map(normalizer(rm_real))
    # Eliminate alpha channel
    texture = texture[..., :3]
    # Colors from the color map are gamma-compressed, go back to linear
    texture = np.power(texture, 2.2)
     # Pre-multiply alpha to avoid fringe
    texture *= opacity[..., None]
    
    texture_uint8 = (texture * 255).astype(np.uint8)
    texture_single = texture_uint8[..., 0] 
    
    texture_final = process_masks(texture_single, building_mask, car_mask)
    return plt.imsave(file_name, texture_final, cmap='gray')

def viridis_generate(radio_map,building_mask,car_mask,tx_mask,metric,file_name,db_scale: bool = True):
    
    rm_real = radio_map.path_gain.numpy().squeeze(axis=0)
    if metric=="rss" and db_scale:
        rm_values *= 1000
    valid = np.logical_and(rm_real > 0., np.isfinite(rm_real))
    opacity = valid.astype(np.float32)
    any_valid = np.any(valid)
    rm_real[valid] = 10. * np.log10(rm_real[valid])

    vmin = rm_real[valid].min() if any_valid else 0

    vmax = rm_real[valid].max() if any_valid else 0
    normalizer = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    color_map = matplotlib.colormaps.get_cmap('viridis')
    texture = color_map(normalizer(rm_real))
    # Eliminate alpha channel
    texture = texture[..., :3]
    # Colors from the color map are gamma-compressed, go back to linear
    texture = np.power(texture, 2.2)
     # Pre-multiply alpha to avoid fringe
    texture *= opacity[..., None]
    
    texture_uint8 = (texture * 255).astype(np.uint8)
    texture_single = texture_uint8[..., :3] 
    
    texture_final = process_masks_3channel(texture_single, building_mask, car_mask,tx_mask)
    return plt.imsave(file_name, texture_final)

def ground_3d_to_topview_pixel(scene, ground_vertices, image_width, image_height):
    """
    将地面3D顶点坐标（z≈0）转换为俯视图的像素坐标（基于场景整体边界）
    
    参数:
        scene: 场景对象（用于获取整体边界）
        ground_vertices: 形状为 (N, 3) 的numpy数组，存储地面顶点的3D坐标 (x, y, z)
        image_width: 俯视图宽度（像素）
        image_height: 俯视图高度（像素）
    
    返回:
        形状为 (N, 2) 的numpy数组，存储像素坐标 (u, v)
    """
    # 1. 提取x和y坐标（忽略z，地面z≈0）
    # 修正：提取前两列（x和y），形状为 (N, 2)
    # loc = ground_vertices[:, :2]  # 改为[:, :2]，获取(x, y)
    
    # 2. 获取场景整体边界（x和y方向）
    scene_bbox = scene.mi_scene.bbox()
    # 处理空场景的边界（避免±inf导致计算错误）
    scene_min = scene_bbox.min
    scene_min = dr.select(dr.isinf(scene_min), -1.0, scene_min)  # 用dr.Vector3f统一处理
    scene_max = scene_bbox.max
    scene_max = dr.select(dr.isinf(scene_max), 1.0, scene_max)
    
    # 提取x和y方向的边界（转换为numpy数组便于计算）
    # 注意：根据实际API调整属性访问方式（可能是.x/.y或[0]/[1]）
    x_min, y_min = scene_min.x, scene_min.y
    x_max, y_max = scene_max.x, scene_max.y
    
    # 处理边界相同的特殊情况（避免除零）
    if x_max == x_min:
        x_max = x_min + 1e-6
    if y_max == y_min:
        y_max = y_min + 1e-6
    
    # 3. 计算x和y方向的尺寸
    size_x = x_max - x_min
    size_y = y_max - y_min
    
    pixel_coords = {}  # 存储结果：{网格ID: 像素坐标数组(N,2)}
    for mesh_id, vertices in ground_vertices.items():
        # 提取当前网格顶点的x和y坐标（忽略z）
        # vertices形状为(N,3)，取前两列得到(N,2)的(x,y)
        loc = vertices[:, :2]
        
        # 归一化到[0,1]范围
        norm_x = (loc[:, 0] - x_min) / size_x
        norm_y = (loc[:, 1] - y_min) / size_y
        
        # 映射到像素坐标
        pixel_u = norm_x * (image_width - 1)
        pixel_v = norm_y * (image_height - 1)  # 若需翻转y轴，改为(1 - norm_y) * ...
        
        # 转换为整数像素坐标，存入结果字典
        pixel_coords[mesh_id] = np.column_stack([pixel_u, pixel_v]).astype(int)
    
    # 转换为整数像素坐标
    return pixel_coords

def ground_to_pixel(scene, ground_vertices, image_width, image_height):
    """
    将地面3D顶点坐标（z≈0）转换为俯视图的像素坐标（基于场景整体边界）
    
    参数:
        scene: 场景对象（用于获取整体边界）
        ground_vertices: 形状为 (N, 3) 的numpy数组，存储地面顶点的3D坐标 (x, y, z)
        image_width: 俯视图宽度（像素）
        image_height: 俯视图高度（像素）
    
    返回:
        形状为 (N, 2) 的numpy数组，存储像素坐标 (u, v)
    """
    # 1. 提取x和y坐标（忽略z，地面z≈0）
    # 修正：提取前两列（x和y），形状为 (N, 2)
    # loc = ground_vertices[:, :2]  # 改为[:, :2]，获取(x, y)
    
    # 2. 获取场景整体边界（x和y方向）
    scene_bbox = scene.mi_scene.bbox()
    # 处理空场景的边界（避免±inf导致计算错误）
    scene_min = scene_bbox.min
    scene_min = dr.select(dr.isinf(scene_min), -1.0, scene_min)  # 用dr.Vector3f统一处理
    scene_max = scene_bbox.max
    scene_max = dr.select(dr.isinf(scene_max), 1.0, scene_max)
    
    # 提取x和y方向的边界（转换为numpy数组便于计算）
    # 注意：根据实际API调整属性访问方式（可能是.x/.y或[0]/[1]）
    x_min, y_min = scene_min.x, scene_min.y
    x_max, y_max = scene_max.x, scene_max.y
    
    # 处理边界相同的特殊情况（避免除零）
    if x_max == x_min:
        x_max = x_min + 1e-6
    if y_max == y_min:
        y_max = y_min + 1e-6
    
    # 3. 计算x和y方向的尺寸
    size_x = x_max - x_min
    size_y = y_max - y_min
    
    
    loc = ground_vertices[:2]
        
        # 归一化到[0,1]范围
    norm_x = (loc[0] - x_min) / size_x
    norm_y = (loc[1] - y_min) / size_y
        
        # 映射到像素坐标
    pixel_u = norm_x * (image_width - 1)
    pixel_v = norm_y * (image_height - 1)  # 若需翻转y轴，改为(1 - norm_y) * ...
        
        # 转换为整数像素坐标，存入结果字典
    pixel_coords = np.column_stack([pixel_u, pixel_v]).astype(int)
    
    # 转换为整数像素坐标
    return pixel_coords

def world_to_pixel(world_points, sensor, image_width, image_height):
    """
    将3D世界坐标转换为2D像素坐标
    :param world_points: 形状为 (N, 3) 的numpy数组，存储3D顶点坐标 (x, y, z)
    :param sensor: Mitsuba传感器（相机）对象
    :param image_width: 渲染图片的宽度（像素）
    :param image_height: 渲染图片的高度（像素）
    :return: 形状为 (N, 2) 的numpy数组，存储像素坐标 (u, v)
    """
    # 1. 获取相机的视图变换矩阵（世界坐标 → 相机坐标）
    view_transform = sensor.world_transform
    # 2. 获取相机的投影矩阵（相机坐标 → 归一化设备坐标NDC）
    proj_transform = sensor.projection_transform()
    
    # 3. 转换3D点为齐次坐标（添加w=1）
    homogeneous_points = np.hstack([world_points, np.ones((len(world_points), 1))])  # 形状 (N, 4)
    
    # 4. 应用视图变换（世界坐标 → 相机坐标）
    camera_points = view_transform.transform_points(homogeneous_points)  # 结果仍是齐次坐标 (N, 4)
    
    # 5. 应用投影变换（相机坐标 → NDC坐标，范围[-1,1]）
    ndc_points = proj_transform.transform_points(camera_points)  # (N, 4)
    
    # 6. 透视除法（齐次坐标转非齐次）
    ndc_x = ndc_points[:, 0] / ndc_points[:, 3]
    ndc_y = ndc_points[:, 1] / ndc_points[:, 3]
    
    # 7. NDC坐标 → 像素坐标（[-1,1] → [0, width/height]）
    # 注意：图像坐标系中y轴向下，需要翻转
    pixel_u = (ndc_x + 1) * 0.5 * image_width
    pixel_v = (1 - (ndc_y + 1) * 0.5) * image_height  # 翻转y轴
    
    # 8. 转换为整数像素坐标（可选，根据需求保留浮点或取整）
    return np.column_stack([pixel_u, pixel_v]).astype(int)

def get_2d_vertices(scene, obj_type, with_all_vert):
    """
    获取场景中指定类型物体的地面顶点3D坐标
    
    参数:
        scene: 场景对象
        obj_type: 物体类型，可选值: "only_building"|"only_car"|"both"
        with_all_vert: 是否返回所有顶点坐标
    
    返回:
        根据参数组合返回不同的顶点数组组合
    """
    # 验证输入参数有效性
    valid_types = {"only_building", "only_car", "both"}
    if obj_type not in valid_types:
        raise ValueError(f"obj_type必须是{valid_types}中的一种，当前为{obj_type}")
    
    # 初始化存储容器
    all_vertices = {}                 # 所有顶点
    ground_vertices_buildings = {}     # 建筑物地面顶点
    ground_vertices_cars = {}          # 车辆地面顶点

    # 一次遍历完成所有顶点收集，减少重复计算
    for sh in scene.mi_scene.shapes():
        # 只处理网格类型
        if not sh.is_mesh():
            continue
        mesh_id = sh.id()
        # 获取并转换顶点坐标 (3D世界坐标)
        # 注意：根据实际API调整顶点缓冲区的获取方式
        vertices_buffer = sh.vertex_positions_buffer()
        vertices_np = np.array(vertices_buffer, dtype=np.float32).reshape(-1, 3)
        all_vertices[mesh_id] = vertices_np
        
        # 获取材质名称用于区分物体类型
        mat_name = sh.bsdf().radio_material.name
        
        # 只对目标类型物体进行地面顶点筛选
        process_building = (obj_type in ["only_building", "both"]) and (mat_name == "itu_brick")
        process_car = (obj_type in ["only_car", "both"]) and (mat_name == "itu_metal")
        
        if not (process_building or process_car):
            continue
        
        # 计算地面阈值（z轴为高度方向）
        # 这里使用当前网格的z最小值作为局部地面参考（更精准）
        local_min_z = np.min(vertices_np[:, 2])
        epsilon = 0.1
        z_min = local_min_z - epsilon
        z_max = local_min_z + epsilon
        
        # 筛选地面顶点
        is_ground = (vertices_np[:, 2] >= z_min) & (vertices_np[:, 2] <= z_max)
        ground_vertices = vertices_np[is_ground]
        
        # 根据物体类型分类存储
        if process_building:
            ground_vertices_buildings[mesh_id] = ground_vertices
        if process_car:
            ground_vertices_cars[mesh_id] = ground_vertices
    
    
    
    # 根据参数组合返回结果
    if with_all_vert:
        if obj_type == "only_building":
            return all_vertices, ground_vertices_buildings
        elif obj_type == "only_car":
            return all_vertices, ground_vertices_cars
        else:  # both
            return all_vertices, ground_vertices_buildings, ground_vertices_cars
    else:
        if obj_type == "only_building":
            return ground_vertices_buildings
        elif obj_type == "only_car":
            return ground_vertices_cars
        else:  # both
            return ground_vertices_buildings, ground_vertices_cars

def create_building_mask(building_vertices_dict, image_width, image_height):
    """
    创建建筑物地面区域的掩码图像（适配字典格式输入）
    
    参数:
        building_vertices_dict: 字典格式 {网格ID: 像素坐标数组(N,2)}，
                               每个数组存储一个建筑物的地面顶点像素坐标
        image_width: 图像宽度（像素）
        image_height: 图像高度（像素）
    
    返回:
        灰度图像数组（0为背景，1为建筑物内部），形状为 (image_height, image_width)
    """
    # 创建空白图像（初始全为0）
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    # 遍历字典中的每个建筑物顶点数组（忽略键，只处理值）
    for vertices in building_vertices_dict.values():
        # 过滤超出图像范围的顶点（避免填充错误）
        valid_vertices = vertices[
            (vertices[:, 0] >= 0) & (vertices[:, 0] < image_width) &
            (vertices[:, 1] >= 0) & (vertices[:, 1] < image_height)
        ]
        
        if len(valid_vertices) < 3:  # 至少需要3个点才能构成多边形
            continue
        
        # 转换为OpenCV需要的格式（int32类型的二维数组）
        pts = valid_vertices.astype(np.int32).reshape((-1, 1, 2))
        
        # 填充多边形内部（值设为1）
        cv2.fillPoly(mask, [pts], color=1)
        
        # 旋转后图像尺寸会变为 (原宽度, 原高度)
        rotated = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
        # 步骤2：沿height中心轴（垂直中轴线）镜像翻转（水平翻转）
        # 翻转后图像左右颠倒
        flipped = cv2.flip(rotated, flipCode=1)
    
    return flipped

def create_tx_mask(tx_vertices_dict, image_width, image_height):
    """
    创建建筑物地面区域的掩码图像（适配字典格式输入）
    
    参数:
        building_vertices_dict: 字典格式 {网格ID: 像素坐标数组(N,2)}，
                               每个数组存储一个建筑物的地面顶点像素坐标
        image_width: 图像宽度（像素）
        image_height: 图像高度（像素）
    
    返回:
        灰度图像数组（0为背景，1为建筑物内部），形状为 (image_height, image_width)
    """
    # 创建空白图像（初始全为0）
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    # 遍历字典中的每个建筑物顶点数组（忽略键，只处理值）
    for i in range(tx_vertices_dict.shape[0]):
        # 过滤超出图像范围的顶点（避免填充错误）
        valid_vertices = tx_vertices_dict[i][
            (tx_vertices_dict[i][0] >= 0) & (tx_vertices_dict[i][0] < image_width) &
            (tx_vertices_dict[i][1] >= 0) & (tx_vertices_dict[i][1] < image_height)
        ]
        
        # 转换为OpenCV需要的格式（int32类型的二维数组）
        pts = valid_vertices.astype(np.int32)
        
        # 填充多边形内部（值设为1）
        cv2.fillPoly(mask, [pts], color=1)
        
        # 旋转后图像尺寸会变为 (原宽度, 原高度)
        rotated = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
        # 步骤2：沿height中心轴（垂直中轴线）镜像翻转（水平翻转）
        # 翻转后图像左右颠倒
        flipped = cv2.flip(rotated, flipCode=1)
    
    return flipped