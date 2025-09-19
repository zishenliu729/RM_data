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

num_scenes = util.count_subfolders_with_os(f"dataset/scenes")
displacement_vec = {
"x_p": [8,0,0],
"x_n": [-8,0,0],
"y_p":[0,8,0],
"y_n":[0,-8,0]}

num_displacements = 2
image_width = 256
image_height = 256
cam =  Camera(position=[-14,35,450], look_at=[-14,35,0])
for idx in range(1,num_scenes+1):
    path =f"dataset/scenes/scene_{idx}/scene_{idx}.xml"
    scene = load_scene(path,merge_shapes=False)
    scene.file_path = path
    # scene_nocar = load_scene(f"dataset/scenes/scene_{idx}/scene_{idx}.xml",merge_shapes=False)
    # #2. Render the scene geometry.
    # for sh in scene_nocar.objects:
    #     if "mesh-car" in sh:
    #         scene_nocar.edit(remove=sh)

            
  
    
    scene.add(Transmitter("tx", position=[4,18,1.5], orientation=[np.pi,0,0]))
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="tr38901", polarization="V")
    scene.rx_array = scene.tx_array

    # Create radio map solver
    rm_solver = RadioMapSolver()


    
    
    ground_vertices_buildings_np = util.get_2d_vertices(scene=scene, obj_type="only_building", with_all_vert=False)
    building_pixel_coords = util.ground_3d_to_topview_pixel(scene = scene,
                                                            ground_vertices=ground_vertices_buildings_np, 
                                                            image_width=image_width, 
                                                            image_height=image_height)
    building_mask = util.create_building_mask(building_pixel_coords, image_width, image_height)
    cv2.imwrite(f"dataset/buildings/building_ground_mask_{idx}.png", building_mask * 255)

    tx_2d_vertice = np.array(scene.get("tx").position).squeeze(-1)[:2]
    tx_pixel_coords = util.ground_to_pixel(scene = scene,
                                                            ground_vertices=tx_2d_vertice, 
                                                            image_width=image_width, 
                                                            image_height=image_height)
    tx_mask = util.create_tx_mask(tx_pixel_coords, image_width, image_height)
    cv2.imwrite(f"dataset/tx/tx_{idx}.png", tx_mask * 255)

    if not os.path.exists(f'dataset/cars/car_{idx}'):
        os.makedirs(f'dataset/cars/car_{idx}')
    if not os.path.exists(f'dataset/path_gain/path_gain_{idx}'):
        os.makedirs(f'dataset/path_gain/path_gain_{idx}')
    if not os.path.exists(f'dataset/visual/visual_{idx}'):
        os.makedirs(f'dataset/visual/visual_{idx}')
    for i in range(num_displacements+1): 
        rm = rm_solver(scene=scene,
                        samples_per_tx=20**6,
                        refraction=True,
                        max_depth=10,
                        cell_size=[1,1])
        
        
        ground_vertices_cars_np = util.get_2d_vertices(scene=scene, obj_type="only_car", with_all_vert=False)
        car_pixel_coords = util.ground_3d_to_topview_pixel(scene = scene,
                                                            ground_vertices=ground_vertices_cars_np, 
                                                            image_width=image_width, 
                                                            image_height=image_height)
        
        
        car_mask = util.create_building_mask(car_pixel_coords, image_width, image_height)
        cv2.imwrite(f"dataset/cars/car_{idx}/car_{idx}_{i}.png", car_mask * 255)
        
        # util.gray_generate(rm,building_mask,car_mask,metric="path_gain",file_name=f"dataset/path_gain/path_gain_{idx}/path_gain_{i}.png")
        util.viridis_generate(rm,building_mask,car_mask,tx_mask,metric="path_gain",file_name=f"dataset/path_gain/path_gain_{idx}/path_gain_v_{i}.png")
        scene.render_to_file(camera=cam, filename=f"dataset/visual/visual_{idx}/path_gain_visual_{i}.png", radio_map=rm,
                        resolution=(1024,1024),
                    num_samples=1024,
                    rm_vmin=-140)
        
        scene = util.cars_movement(scene, displacement_vec)
        # scene = util.cars_desity_change(scene, radio=0.3)
        
        


