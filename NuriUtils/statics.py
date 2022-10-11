import numpy as np
import imageio
import os

# COCO
with open(os.path.join(os.path.dirname(__file__), "data/coco_category.txt"), "r") as f:
    lines = f.readlines()
DETECTION_CATEGORIES = [line.rstrip() for line in lines]
COCO_CATEGORIES = DETECTION_CATEGORIES
# 40 category of interests
with open(os.path.join(os.path.dirname(__file__), "data/matterport_category.txt"), "r") as f:
    lines = f.readlines()
CATEGORIES = {}
CATEGORIES['mp3d'] = [line.rstrip() for line in lines]
CATEGORIES['gibson'] = DETECTION_CATEGORIES
with open(os.path.join(os.path.dirname(__file__), "data/cv_colors.txt"), "r") as f:
    lines = f.readlines()
STANDARD_COLORS = [line.rstrip() for line in lines]

with open(os.path.join(os.path.dirname(__file__), "data/mp3d_trainset.txt"), "r") as f:
    lines = f.readlines()
MP3D_TRAIN_SCENE = [line.rstrip() for line in lines]

with open(os.path.join(os.path.dirname(__file__), "data/mp3d_valset.txt"), "r") as f:
    lines = f.readlines()
MP3D_VAL_SCENE = [line.rstrip() for line in lines]

with open(os.path.join(os.path.dirname(__file__), "data/gibson_trainset.txt"), "r") as f:
    lines = f.readlines()
GIBSON_TINY_TRAIN_SCENE = [line.rstrip() for line in lines]

with open(os.path.join(os.path.dirname(__file__), "data/gibson_testset.txt"), "r") as f:
    lines = f.readlines()
GIBSON_TINY_TEST_SCENE = [line.rstrip() for line in lines]

with open(os.path.join(os.path.dirname(__file__), "data/mp3d_catset.txt"), "r") as f:
    lines = f.readlines()
MP3D_CAT_TRAIN_SCENE = [line.rstrip() for line in lines]

new_objects_mpcat40_mapping = {'AlarmClockVintage': 'objects', 'BathSheetBlue': 'objects', 'BathTowelPink': 'objects', 'CDrack': 'shelving',
                               'CatObject': 'objects', 'CrackersBox': 'objects', 'CubeBox': 'objects', 'CutleryBlockSet': 'objects',
                               'DishTowel': 'objects', 'DogBowlBlue': 'objects', 'DogFoobler': 'objects', 'Elephant': 'objects', 'EyeCream': 'objects',
                               'FabricBasket': 'objects', 'FabricCube': 'objects', 'Fedora': 'objects', 'FileSorterPink': 'objects', 'GardenSwing': 'objects',
                               'GigabyteBox': 'objects', 'KidsPack': 'objects', 'KitchenTowel': 'objects', 'Notebook': 'objects',
                               'PepsiBox': 'objects', 'PitcherWhite': 'objects', 'PlantContainerQuadra': 'objects', 'PlantContainerUrn': 'objects',
                               'PlantPot': 'plant', 'PorcelainRamekin': 'objects', 'PowerPressureCooker': 'objects', 'ReversibleBookend': 'objects',
                               'RollLearnTurtle': 'objects', 'RoundPlanterRed': 'objects', 'Sandwich': 'objects', 'StackRings': 'objects', 'SteelMilkFrother': 'objects',
                               'StorageBinBlack': 'objects', 'TeaSet': 'objects', 'TeapotWhite': 'objects', 'TucanFrame': 'picture', 'TwistedPuzzle': 'objects',
                               'UpFlippinDog': 'objects', 'VeniceFrame': 'picture', 'VitaminCEster': 'objects', 'aquarium': 'objects', 'armchair2': 'chair',
                               'armchair3': 'chair', 'baguettePain': 'objects', 'banana': 'objects', 'bdOuverte': 'objects', 'blackTupelo': 'objects',
                               'blocks': 'objects', 'bookcase': 'shelving', 'bouilloire': 'objects', 'bucket': 'objects', 'cardboard-box': 'objects',
                               'chair_office': 'chair', 'chaiseOrange': 'chair', 'cheezit': 'objects', 'chefcan': 'objects', 'clothes-basket': 'furniture',
                               'comoCiliegio': 'cabinet', 'couchPoofyPillows': 'sofa', 'couchTable': 'table', 'credenza': 'cabinet', 'cup': 'objects',
                               'cup2': 'objects', 'desserte': 'shelving', 'fanPalm': 'objects', 'flowers': 'plant', 'frame3VerticalSummerPhotos': 'picture',
                               'frame3VerticalTowerPhotos': 'picture', 'frame_horizontal': 'picture', 'frame_square': 'picture', 'gardenUmbrella': 'furniture',
                               'hallTable': 'table', 'japanese_style_screen': 'furniture', 'lamp2': 'lighting', 'largeclamp': 'objects', 'leather_sofa': 'sofa',
                               'letters': 'objects', 'livres': 'objects', 'modern_armchair': 'chair', 'officeChair': 'chair', 'percolateur': 'objects',
                               'pizza': 'objects', 'plancheADecouper': 'objects', 'potplant2': 'plant', 'rocking_chair': 'chair', 'roundTable2': 'table', 'rug': 'objects',
                               'rug2': 'objects', 'secheCheveux': 'objects', 'shelves_storage_wood': 'furniture', 'singleChair': 'chair', 'small_bookshelves': 'shelving',
                               'smiley': 'objects', 'sneakers': 'objects', 'sofa4': 'sofa', 'sofa_white_beige': 'sofa', 'stair_ladder_loft': 'stairs', 'stairs_spiral': 'stairs',
                               'table': 'table', 'table_granmother_marble': 'table', 'throwPillow': 'cushion', 'toast': 'objects',
                               'toothpaste_brush': 'objects', 'wheelie_bin': 'objects', 'wine_bottle': 'objects'}

gibson_categories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                     'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                     'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                     'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

new_objects_gibson_mapping = {
    'Notebook': 'laptop', 'ReversibleBookend': 'book', 'toothpaste': 'surfboard', 'flowers': 'potted plant', 'cup': 'cup', 'officeChair': 'chair',
    'TwistedPuzzle': 'carrot', 'SteelMilkFrother': 'fork', 'bottle': 'bottle', 'pot': 'vase', 'VitaminCEster': 'surfboard', 'FabricCube': 'vase', 'PlantPot': 'vase',
    'desserte': 'bench', 'letters': 'carrot', 'bouilloire': 'surfboard', 'chair': 'chair', 'clock': 'clock', 'CubeBox': 'suitcase', 'banana':'banana',
    'plancheADecouper': 'surfboard', 'leather': 'couch', 'KitchenTowel': 'kite', 'credenza': 'bench', 'cardboard-box': 'suitcase', 'comoCiliegio': 'surfboard',
    'RoundPlanterRed': 'vase', 'blocks': 'carrot', 'PlantContainerQuadra': 'vase', 'pizza': 'pizza', 'livres': 'book', 'clothes-basket': 'surfboard', 'CutleryBlockSet': 'fork',
    'couchTable': 'dining table', 'Elephant': 'elephant', 'sneakers': 'surfboard', 'largeclamp': 'surfboard', 'CatObject': 'surfboard',
    'CDrack': 'dining table', 'cheezit': 'carrot', 'small': 'surfboard', 'bucket': 'vase', 'PorcelainRamekin': 'fork', 'bag': 'handbag', 'UpFlippinDog': 'surfboard',
    'Sandwich':'sandwich', 'smiley': 'carrot', 'modern': 'chair', 'CrackersBox': 'suitcase', 'roundTable2': 'dining table', 'EyeCream': 'surfboard', 'PowerPressureCooker': 'toaster',
    'bdOuverte': 'book', 'rug2': 'surfboard', 'wine': 'bottle', 'TeapotWhite': 'surfboard', 'KidsPack': 'backpack', 'pillow': 'boat', 'couchPoofyPillows': 'boat', 'mug': 'cup',
    'StackRings': 'carrot', 'laptop':'laptop', 'printer': 'surfboard', 'RollLearnTurtle': 'surfboard', 'cup2': 'cup', 'chefcan': 'surfboard', 'bowl': 'bowl',
    'Fedora': 'surfboard', 'DishTowel': 'kite', 'FileSorterPink': 'book', 'BathSheetBlue': 'kite', 'hallTable': 'dining table', 'FabricBasket': 'vase', 'DogFoobler': 'surfboard',
    'BathTowelPink': 'kite', 'TeaSet': 'fork', 'percolateur': 'surfboard', 'DogBowlBlue': 'bowl', 'PlantContainerUrn': 'vase',
    'shelves': 'dining table', 'potplant2': 'potted plant', 'PepsiBox': 'suitcase', 'secheCheveux': 'surfboard', 'lamp2': 'surfboard', 'rug': 'surfboard',
    'PitcherWhite': 'fork', 'toast': 'fork', 'throwPillow': 'boat', 'GigabyteBox': 'suitcase', 'AlarmClockVintage': 'clock', 'baguettePain': 'donut'
}

# 28 category of interests
# COI = ['chair', 'door', 'table', 'picture', 'cabinet', 'cushion', 'window', 'sofa', 'bed', 'curtain',
#        'chest_of_drawers', 'plant', 'sink', 'stairs', 'toilet', 'stool', 'towel', 'mirror', 'tv_monitor', 'shower',
#        'bathtub', 'counter', 'fireplace', 'shelving', 'blinds', 'gym_equipment', 'clothes', 'objects']
# COI = ['chair', 'door', 'table', 'picture', 'cabinet', 'cushion', 'window', 'sofa',
#        'bed', 'curtain', 'chest_of_drawers', 'plant', 'sink', 'stairs',  'toilet', 'stool', 'towel',
#        'mirror', 'tv_monitor', 'shower', 'column', 'bathtub', 'counter', 'fireplace', 'lighting', 'beam', 'railing',
#        'shelving', 'blinds', 'gym_equipment', 'seating', 'board', 'furniture', 'appliances', 'clothes', 'objects']
COI = ['chair', 'door', 'table', 'picture', 'cabinet', 'cushion', 'window', 'sofa',
       'bed', 'curtain', 'chest_of_drawers', 'plant', 'sink', 'stairs',  'toilet', 'stool', 'towel',
       'mirror', 'tv_monitor', 'shower', 'bathtub', 'counter', 'fireplace',
       'shelving', 'blinds', 'gym_equipment', 'seating', 'board', 'furniture', 'appliances', 'clothes', 'objects']
CATEGORIES_TO_IGNORE = ["stairs", "shelving", "mirror", "column"]
CATEGORIES_TO_IGNORE_IDX = np.concatenate([np.where([CATEGORIES_TO_IGNORE_ in CATEGORIES_ for CATEGORIES_ in CATEGORIES])[0] for CATEGORIES_TO_IGNORE_ in CATEGORIES_TO_IGNORE])
COI_INDEX = {}
COI_INDEX['mp3d'] = np.where([c in COI for c in CATEGORIES['mp3d']])[0]
COI = CATEGORIES['gibson']
COI_INDEX['gibson'] = np.where([c in COI for c in CATEGORIES['gibson']])[0]
DETECT_CATEGORY = ['mug', 'bottle', 'pot', 'bowl', 'chair', 'table', 'clock', 'bag', 'sofa', 'laptop', 'bed', 'microwave', 'cabinet', 'bookshelf', 'stove', 'printer', 'pillow']
ADD_OBJ_CATEGORY = ['AlarmClockVintage', 'BathSheetBlue', 'BathTowelPink', 'CDrack',
       'CatObject', 'CrackersBox', 'CubeBox', 'CutleryBlockSet',
       'DishTowel', 'DogBowlBlue', 'DogFoobler', 'Elephant', 'EyeCream',
       'FabricBasket', 'FabricCube', 'Fedora', 'FileSorterPink',
       'GigabyteBox', 'KidsPack', 'KitchenTowel', 'Notebook', 'PepsiBox',
       'PitcherWhite', 'PlantContainerQuadra', 'PlantContainerUrn',
       'PlantPot', 'PorcelainRamekin', 'PowerPressureCooker',
       'ReversibleBookend', 'RollLearnTurtle', 'RoundPlanterRed',
       'Sandwich', 'StackRings', 'SteelMilkFrother', 'StorageBinBlack',
       'TeaSet', 'TeapotWhite', 'TucanFrame', 'TwistedPuzzle',
       'UpFlippinDog', 'VeniceFrame', 'VitaminCEster', 'bag',
       'baguettePain', 'banana', 'bdOuverte', 'blocks', 'bottle',
       'bouilloire', 'bowl', 'bucket', 'cardboard-box', 'chair',
       'cheezit', 'chefcan', 'clock', 'clothes-basket', 'comoCiliegio',
       'couchPoofyPillows', 'couchTable', 'credenza', 'cup', 'cup2',
       'desserte', 'fanPalm', 'flowers', 'frame',
       'frame3VerticalSummerPhotos', 'frame3VerticalTowerPhotos',
       'gardenUmbrella', 'hallTable', 'japanese', 'lamp2', 'laptop',
       'largeclamp', 'leather', 'letters', 'livres', 'modern', 'mug',
       'officeChair', 'percolateur', 'pillow', 'pizza',
       'plancheADecouper', 'pot', 'potplant2', 'printer', 'rocking',
       'roundTable2', 'rug', 'rug2', 'secheCheveux', 'shelves',
       'singleChair', 'small', 'smiley', 'sneakers', 'throwPillow',
       'toast', 'toothpaste', 'wheelie', 'wine']


ADD_OBJ_MAPPING = {}
ADD_OBJ_MAPPING['mp3d'] = \
{
    'AlarmClockVintage': 'objects',
    'BathSheetBlue': 'objects',
    'BathTowelPink': 'objects',
    'CDrack': 'shelving',
    'CatObject': 'objects',
    'CrackersBox': 'objects',
    'CubeBox': 'objects',
    'CutleryBlockSet': 'objects',
    'DishTowel': 'objects',
    'DogBowlBlue': 'objects',
    'DogFoobler': 'objects',
    'Elephant': 'objects',
    'EyeCream': 'objects',
    'FabricBasket': 'objects',
    'FabricCube': 'objects',
    'Fedora': 'objects',
    'FileSorterPink': 'objects',
    'GigabyteBox': 'objects',
    'KidsPack': 'objects',
    'KitchenTowel': 'objects',
    'Notebook': 'objects',
    'PepsiBox': 'objects',
    'PitcherWhite': 'objects',
    'PlantContainerQuadra': 'objects',
    'PlantContainerUrn': 'objects',
    'PlantPot': 'plant',
    'PorcelainRamekin': 'objects',
    'PowerPressureCooker': 'objects',
    'ReversibleBookend': 'objects',
    'RollLearnTurtle': 'objects',
    'RoundPlanterRed': 'objects',
    'Sandwich': 'objects',
    'StackRings': 'objects',
    'SteelMilkFrother': 'objects',
    'StorageBinBlack': 'objects',
    'TeaSet': 'objects',
    'TeapotWhite': 'objects',
    'TucanFrame': 'picture',
    'TwistedPuzzle': 'objects',
    'UpFlippinDog': 'objects',
    'VeniceFrame': 'picture',
    'VitaminCEster': 'objects',
    'baguettePain': 'objects',
    'banana': 'objects',
    'bdOuverte': 'objects',
    'blocks': 'objects',
    'bouilloire': 'objects',
    'bucket': 'objects',
    'cardboard-box': 'objects',
    'cheezit': 'objects',
    'chefcan': 'objects',
    'clothes-basket': 'furniture',
    'comoCiliegio': 'cabinet',
    'couchPoofyPillows': 'sofa',
    'couchTable': 'table',
    'credenza': 'cabinet',
    'cup': 'objects',
    'cup2': 'objects',
    'desserte': 'shelving',
    'fanPalm': 'objects',
    'flowers': 'plant',
    'frame3VerticalSummerPhotos': 'picture',
    'frame3VerticalTowerPhotos': 'picture',
    'gardenUmbrella': 'furniture',
    'hallTable': 'table',
    'lamp2': 'lighting',
    'largeclamp': 'objects',
    'letters': 'objects',
    'livres': 'objects',
    'officeChair': 'chair',
    'percolateur': 'objects',
    'pizza': 'objects',
    'plancheADecouper': 'objects',
    'potplant2': 'plant',
    'roundTable2': 'table',
    'rug': 'objects',
    'rug2': 'objects',
    'secheCheveux': 'objects',
    'singleChair': 'chair',
    'smiley': 'objects',
    'sneakers': 'objects',
    'throwPillow': 'cushion',
    'toast': 'objects',
    'bag': 'objects',
    'bottle': 'objects',
    'bowl': 'objects',
    'clock': 'objects',
    'laptop': 'objects',
    'chair_office': 'chair',
    'frame_horizontal': 'picture',
    'frame_square': 'picture',
    'leather_sofa': 'sofa',
    'shelves_storage_wood': 'furniture',
    'small_bookshelves': 'shelving',
    'table_granmother_marble': 'table',
    'toothpaste_brush': 'objects',
    'wheelie_bin': 'objects',
    'wine_bottle': 'objects'
}

ALL_CATEGORIES = list(set(CATEGORIES['mp3d'] + CATEGORIES['gibson'] + DETECTION_CATEGORIES))
ALL_CATEGORIES = np.sort(np.unique(ALL_CATEGORIES))

task_cat2mpcat40_labels = [
    'chair',
    'table',
    'picture',
    'cabinet',
    'cushion',
    'sofa',
    'bed',
    'chest_of_drawers',
    'plant',
    'sink',
    'toilet',
    'stool',
    'towel',
    'tv_monitor',
    'shower',
    'bathtub',
    'counter',
    'fireplace',
    'gym_equipment',
    'seating',
    'clothes',
]
OBJECT_TARGET_CATEGORY = {}
OBJECT_TARGET_CATEGORY['gibson'] = ['chair', 'couch', 'potted plant', 'bed', 'toilet', 'tv']
OBJECT_TARGET_CATEGORY['mp3d'] = task_cat2mpcat40_labels
# coco_categories = {
#     "chair": 0,
#     "couch": 1,
#     "potted plant": 2,
#     "bed": 3,
#     "toilet": 4,
#     "tv": 5,
#     "dining-table": 6,
#     "oven": 7,
#     "sink": 8,
#     "refrigerator": 9,
#     "book": 10,
#     "clock": 11,
#     "vase": 12,
#     "cup": 13,
#     "bottle": 14
# }

coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # potted plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # dining-table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
}


# COI_TO_CATEGORY = {
#     np.where(['chair' == cat for cat in CATEGORIES])[0][0]: np.where(['chair' == cat for cat in DETECT_CATEGORY])[0][0],
#     np.where(['table' == cat for cat in CATEGORIES])[0][0]: np.where(['table' == cat for cat in DETECT_CATEGORY])[0][0],
#     np.where(['cabinet' == cat for cat in CATEGORIES])[0][0]: np.where(['cabinet' == cat for cat in DETECT_CATEGORY])[0][0],
#     np.where(['cushion' == cat for cat in CATEGORIES])[0][0]: np.where(['pillow' == cat for cat in DETECT_CATEGORY])[0][0],
#     np.where(['sofa' == cat for cat in CATEGORIES])[0][0]: np.where(['sofa' == cat for cat in DETECT_CATEGORY])[0][0],
#     np.where(['bed' == cat for cat in CATEGORIES])[0][0]: np.where(['bed' == cat for cat in DETECT_CATEGORY])[0][0],
#     np.where(['chest_of_drawers' == cat for cat in CATEGORIES])[0][0]: np.where(['cabinet' == cat for cat in DETECT_CATEGORY])[0][0],
#     np.where(['seating' == cat for cat in CATEGORIES])[0][0]: np.where(['sofa' == cat for cat in DETECT_CATEGORY])[0][0],
#     np.where(['furniture' == cat for cat in CATEGORIES])[0][0]: np.where(['cabinet' == cat for cat in DETECT_CATEGORY])[0][0]
# }
# for i in range(len(CATEGORIES), len(CATEGORIES) + len(DETECT_CATEGORY)):
#     COI_TO_CATEGORY[i] = i-len(CATEGORIES) # = {i: i - 41 for i in range(41, 41 + 17)}


AGENT_SPRITE = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "assets",
        "maps_topdown_agent_sprite",
        "100x100.png",
    )
)
AGENT_SPRITE = np.ascontiguousarray(np.flipud(AGENT_SPRITE))

OBJECT_YELLOW = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "assets",
        "maps_topdown_object",
        "star_yellow.png",
    )
)
OBJECT_YELLOW = np.ascontiguousarray(np.flipud(OBJECT_YELLOW))

OBJECT_YELLOW_DIM = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "assets",
        "maps_topdown_object",
        "star_dim.png",
    )
)
OBJECT_YELLOW_DIM = np.ascontiguousarray(np.flipud(OBJECT_YELLOW_DIM))

OBJECT_BLUE = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "assets",
        "maps_topdown_object",
        "star_blue.png",
    )
)
OBJECT_BLUE = np.ascontiguousarray(np.flipud(OBJECT_BLUE))

OBJECT_GRAY = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "assets",
        "maps_topdown_object",
        "star_gray.png",
    )
)
OBJECT_GRAY = np.ascontiguousarray(np.flipud(OBJECT_GRAY))

OBJECT_GREEN = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "assets",
        "maps_topdown_object",
        "star_green.png",
    )
)
OBJECT_GREEN = np.ascontiguousarray(np.flipud(OBJECT_GREEN))

OBJECT_PINK = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "assets",
        "maps_topdown_object",
        "star_pink.png",
    )
)

OBJECT_PINK = np.ascontiguousarray(np.flipud(OBJECT_PINK))

OBJECT_RED = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "assets",
        "maps_topdown_object",
        "star_red.png",
    )
)
OBJECT_RED = np.ascontiguousarray(np.flipud(OBJECT_RED))

OBJECT_START_FLAG = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "assets",
        "maps_topdown_flag",
        "flag1_blue.png",
    )
)
OBJECT_START_FLAG = np.ascontiguousarray(OBJECT_START_FLAG)

OBJECT_GOAL_FLAG = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "assets",
        "maps_topdown_flag",
        "flag1_red.png",
    )
)
OBJECT_GOAL_FLAG = np.ascontiguousarray(OBJECT_GOAL_FLAG)

AI2THOR_COI = ['AlarmClock', 'AluminumFoil', 'Apple', 'ArmChair', 'BaseballBat', 'BasketBall', 'Bathtub', 'BathtubBasin', 'Bed', 'Blinds', 'Book', 'Boots', 'Bottle', 'Bowl', 'Box', 'Bread', 'ButterKnife', 'Cabinet', 'Candle', 'CD', 'CellPhone', 'Chair', 'Cloth', 'CoffeeMachine', 'CoffeeTable', 'CounterTop', 'CreditCard', 'Cup', 'Curtains', 'Desk', 'DeskLamp', 'Desktop', 'DiningTable', 'DishSponge', 'DogBed', 'Drawer', 'Dresser', 'Dumbbell', 'Egg', 'Faucet', 'FloorLamp', 'Footstool', 'Fork', 'Fridge', 'GarbageBag', 'GarbageCan', 'HandTowel', 'HandTowelHolder', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamper', 'Lettuce', 'LightSwitch', 'Microwave', 'Mirror', 'Mug', 'Newspaper', 'Ottoman', 'Painting', 'Pan', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Poster', 'Pot', 'Potato', 'RemoteControl', 'RoomDecor', 'Safe', 'SaltShaker', 'ScrubBrush', 'Shelf', 'ShelvingUnit', 'ShowerCurtain', 'ShowerDoor', 'ShowerGlass', 'ShowerHead', 'SideTable', 'Sink', 'SinkBasin', 'SoapBar', 'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'Stool', 'StoveBurner', 'StoveKnob', 'TableTopDecor', 'TargetCircle', 'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'Toaster', 'Toilet', 'ToiletPaper', 'ToiletPaperHanger', 'Tomato', 'Towel', 'TowelHolder', 'TVStand', 'VacuumCleaner', 'Vase', 'Watch', 'WateringCan', 'Window', 'WineBottle']