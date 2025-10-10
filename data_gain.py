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
                      RadioMapSolver, PathSolver,transform_mesh, ITURadioMaterial
from sionna.rt.utils import r_hat, subcarrier_frequencies

num_scenes = util.count_subfolders_with_os(f"dataset/scenes")
# num_scenes = 1
displacement_vec = {
"x_p": [8,0,0],
"x_n": [-8,0,0],
"y_p":[0,8,0],
"y_n":[0,-8,0]}

num_displacements = 2
image_width = 256
image_height = 256
cam =  Camera(position=[0,0,35], look_at=[0,0,0])
for idx in range(1,num_scenes+1):
    path =f"dataset/scenes/scene_{idx}/scene_{idx}.xml"
    # idx =4
    # path =f"dataset/scenes/scene_indoor_MU/indoor.xml"
    scene = load_scene(path,merge_shapes=False)
    scene.file_path = path
    # scene_nocar = load_scene(f"dataset/scenes/scene_{idx}/scene_{idx}.xml",merge_shapes=False)
    # #2. Render the scene geometry.
    # for sh in scene_nocar.objects:
    #     if "mesh-car" in sh:
    #         scene_nocar.edit(remove=sh)

            
  
    
    scene.add(Transmitter("tx", position=[1.5,1.5,2.5], orientation=[np.pi,0,0],power_dbm=0))
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="tr38901", polarization="V")
    scene.rx_array = scene.tx_array

    # people = scene.add(cube_name="people",
    #              size=[1.0, 0.2, 1.75],  # 长5m，宽3m，厚0.2m
    #              position=[3.0, 2.0, 0.875]) 
 
    # human_material = ITURadioMaterial(
    #                 name="human_tissue",
    #                 permittivity=40.0,   # 介电常数
    #                 conductivity=0.5     # 电导率（S/m）
    #             )
    
    # people.material = human_material

    # Create radio map solver
    rm_solver = RadioMapSolver()

    # mapping from physical position to pixel position (building)
    ground_vertices_buildings_np = util.get_2d_vertices(scene=scene, obj_type="only_building", with_all_vert=False)
    building_pixel_coords = util.ground_3d_to_topview_pixel(scene = scene,
                                                            ground_vertices=ground_vertices_buildings_np, 
                                                            image_width=image_width, 
                                                            image_height=image_height)
    building_mask = util.create_building_mask(building_pixel_coords, image_width, image_height)
    cv2.imwrite(f"dataset/buildings/building_ground_mask_{idx}.png", building_mask * 255)

    # mapping from physical position to pixel position (tx)
    tx_2d_vertice = np.array(scene.get("tx").position).squeeze(-1)[:2]
    tx_pixel_coords = util.ground_to_pixel(scene = scene,
                                                            ground_vertices=tx_2d_vertice, 
                                                            image_width=image_width, 
                                                            image_height=image_height)
    tx_mask = util.create_tx_mask(tx_pixel_coords, image_width, image_height)
    cv2.imwrite(f"dataset/tx/tx_{idx}.png", tx_mask * 255)

    # create folders
    if not os.path.exists(f'dataset/cars/car_{idx}'):
        os.makedirs(f'dataset/cars/car_{idx}')
    if not os.path.exists(f'dataset/path_gain/path_gain_{idx}'):
        os.makedirs(f'dataset/path_gain/path_gain_{idx}')
    if not os.path.exists(f'dataset/visual/visual_{idx}'):
        os.makedirs(f'dataset/visual/visual_{idx}')

    # create radio map solver
    for i in range(num_displacements+1): 
        rm = rm_solver(scene=scene,
                        samples_per_tx=20**7,
                        refraction=True,
                        max_depth=10,
                        cell_size=[0.1,0.1])
        
        
        # ground_vertices_cars_np = util.get_2d_vertices(scene=scene, obj_type="only_car", with_all_vert=False)
        # car_pixel_coords = util.ground_3d_to_topview_pixel(scene = scene,
        #                                                     ground_vertices=ground_vertices_cars_np, 
        #                                                     image_width=image_width, 
        #                                                     image_height=image_height)
        
        
        # car_mask = util.create_building_mask(car_pixel_coords, image_width, image_height)
        # cv2.imwrite(f"dataset/cars/car_{idx}/car_{idx}_{i}.png", car_mask * 255)
        
        ## 2D radio map synthesis (BEV)
        # gray image
        util.gray_generate(rm,building_mask,metric="path_gain",file_name=f"dataset/path_gain/path_gain_{idx}/path_gain_{i}.png")
        ## viridis image
        # util.viridis_generate(rm,building_mask,tx_mask,metric="path_gain",file_name=f"dataset/path_gain/path_gain_{idx}/path_gain_v_{i}.png")
        
        # render visual image through sionnaRT tools
        scene.render_to_file(camera=cam, filename=f"dataset/visual/visual_{idx}/path_gain_visual_{i}.png", radio_map=rm,
                        resolution=(1024,1024),
                    num_samples=1024,
                    rm_vmin=-100)
        # scene = util.people_movement(scene, move_vec=[1,-1,0])
        # scene = util.people_random_movement(scene)
        # scene = util.cars_movement(scene, displacement_vec)
        # scene = util.cars_desity_change(scene, radio=0.5)
        # scene = util.tx_movement(scene, move_vec=[10,0,0])
        
        
        


