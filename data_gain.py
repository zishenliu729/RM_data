import os
# 限制仅 0 号 GPU 可见（根据需要修改为目标 GPU 编号，如 1、2 等）
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import cv2
# Import or install Sionna

try:
    import sionna.rt
except ImportError as e:
    import os
    os.system("pip install sionna-rt")
    import sionna.rt

import util

no_preview = False # Toggle to False to use the preview widget
                  # instead of rendering for scene visualization

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      RadioMapSolver, PathSolver,transform_mesh
from sionna.rt.utils import r_hat, subcarrier_frequencies

scene = load_scene("Hong Hum_256.xml",
                   merge_shapes=False)


# Configure a transmitter that is located at the front of "car_2"
scene.add(Transmitter("tx", position=[4,18,1.5], orientation=[np.pi,0,0]))
scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="tr38901", polarization="V")
scene.rx_array = scene.tx_array

# Create radio map solver
rm_solver = RadioMapSolver()


displacement_vec_x_p = [8,0,0]
displacement_vec_x_n = [-8,0,0]
displacement_vec_y_p = [0,8,0]
displacement_vec_y_n = [0,-8,0]
num_displacements = 2
image_width = 256
image_height = 256
cam =  Camera(position=[-14,35,450], look_at=[-14,35,0])
for i in range(num_displacements+1): 
    rm = rm_solver(scene=scene,
                    samples_per_tx=20**6,
                    refraction=True,
                    max_depth=10,
                    cell_size=[1,1])
    
    
    ground_vertices, ground_vertices_buildings_np, ground_vertices_cars_np = util.get_2d_vertices(scene=scene, obj_type="both", with_all_vert=True)
    tx_2d_vertice = np.array(scene.get("tx").position).squeeze(-1)[:2]
    building_pixel_coords = util.ground_3d_to_topview_pixel(scene = scene,
                                                        ground_vertices=ground_vertices_buildings_np, 
                                                        image_width=image_width, 
                                                        image_height=image_height)
    car_pixel_coords = util.ground_3d_to_topview_pixel(scene = scene,
                                                        ground_vertices=ground_vertices_cars_np, 
                                                        image_width=image_width, 
                                                        image_height=image_height)
    tx_pixel_coords = util.ground_to_pixel(scene = scene,
                                                        ground_vertices=tx_2d_vertice, 
                                                        image_width=image_width, 
                                                        image_height=image_height)
    building_mask = util.create_building_mask(building_pixel_coords, image_width, image_height)
    car_mask = util.create_building_mask(car_pixel_coords, image_width, image_height)
    tx_mask = util.create_tx_mask(tx_pixel_coords, image_width, image_height)
    cv2.imwrite("building_ground_mask.png", building_mask * 255)
    cv2.imwrite(f"cars_{i}.png", car_mask * 255)
    cv2.imwrite(f"tx.png", tx_mask * 255)
    util.gray_generate(rm,building_mask,car_mask,metric="path_gain",file_name=f"/home/super/Zishen/RM_data_sio/RM_data/path_gain_HH_{i}.png")
    util.viridis_generate(rm,building_mask,car_mask,tx_mask,metric="path_gain",file_name=f"/home/super/Zishen/RM_data_sio/RM_data/path_gain_HH_v_{i}.png")

    scene.render_to_file(camera=cam,  filename=f"HH_{i}.png", radio_map=rm,
                     resolution=(1024,1024),
                num_samples=1024,
                rm_vmin=-140)
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
            scene.get(f"mesh-car_x_p_{formatted_num}").position += displacement_vec_x_p
        if b != None:
            scene.get(f"mesh-car_x_n_{formatted_num}").position += displacement_vec_x_n
        if c != None:
            scene.get(f"mesh-car_y_p_{formatted_num}").position += displacement_vec_y_p
        if d != None:
            scene.get(f"mesh-car_y_n_{formatted_num}").position += displacement_vec_y_n


