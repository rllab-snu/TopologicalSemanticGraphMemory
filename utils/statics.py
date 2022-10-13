import numpy as np
import imageio
import os

# COCO
with open(os.path.join(os.path.dirname(__file__), "data/coco_category.txt"), "r") as f:
    lines = f.readlines()
DETECTION_CATEGORIES = [line.rstrip() for line in lines]

with open(os.path.join(os.path.dirname(__file__), "data/gibson_category.txt"), "r") as f:
    lines = f.readlines()
GIBSON_CATEGORIES = [line.rstrip() for line in lines]

CATEGORIES = {}
CATEGORIES['gibson'] = GIBSON_CATEGORIES
with open(os.path.join(os.path.dirname(__file__), "data/cv_colors.txt"), "r") as f:
    lines = f.readlines()
STANDARD_COLORS = [line.rstrip() for line in lines]

with open(os.path.join(os.path.dirname(__file__), "data/gibson_trainset.txt"), "r") as f:
    lines = f.readlines()
GIBSON_TINY_TRAIN_SCENE = [line.rstrip() for line in lines]

with open(os.path.join(os.path.dirname(__file__), "data/gibson_testset.txt"), "r") as f:
    lines = f.readlines()
GIBSON_TINY_TEST_SCENE = [line.rstrip() for line in lines]


with open(os.path.join(os.path.dirname(__file__), "data/mp3d_category_selected.txt"), "r") as f:
    lines = f.readlines()
COI_mp3d = [line.rstrip() for line in lines]

COI_INDEX = {}
COI = CATEGORIES['gibson']
COI_INDEX['gibson'] = np.where([c in COI for c in CATEGORIES['gibson']])[0]

ALL_CATEGORIES = list(set(CATEGORIES['gibson'] + DETECTION_CATEGORIES))
ALL_CATEGORIES = np.sort(np.unique(ALL_CATEGORIES))


AGENT_SPRITE = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "assets",
        "maps_topdown_agent_sprite",
        "100x100.png",
    )
)
AGENT_SPRITE = np.ascontiguousarray(np.flipud(AGENT_SPRITE))

OBJECT_YELLOW = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "assets",
        "maps_topdown_object",
        "star_yellow.png",
    )
)
OBJECT_YELLOW = np.ascontiguousarray(np.flipud(OBJECT_YELLOW))

OBJECT_YELLOW_DIM = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "assets",
        "maps_topdown_object",
        "star_dim.png",
    )
)
OBJECT_YELLOW_DIM = np.ascontiguousarray(np.flipud(OBJECT_YELLOW_DIM))

OBJECT_BLUE = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "assets",
        "maps_topdown_object",
        "star_blue.png",
    )
)
OBJECT_BLUE = np.ascontiguousarray(np.flipud(OBJECT_BLUE))

OBJECT_GRAY = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "assets",
        "maps_topdown_object",
        "star_gray.png",
    )
)
OBJECT_GRAY = np.ascontiguousarray(np.flipud(OBJECT_GRAY))

OBJECT_GREEN = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "assets",
        "maps_topdown_object",
        "star_green.png",
    )
)
OBJECT_GREEN = np.ascontiguousarray(np.flipud(OBJECT_GREEN))

OBJECT_PINK = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "assets",
        "maps_topdown_object",
        "star_pink.png",
    )
)

OBJECT_PINK = np.ascontiguousarray(np.flipud(OBJECT_PINK))

OBJECT_RED = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "assets",
        "maps_topdown_object",
        "star_red.png",
    )
)
OBJECT_RED = np.ascontiguousarray(np.flipud(OBJECT_RED))

OBJECT_START_FLAG = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "assets",
        "maps_topdown_flag",
        "flag1_blue.png",
    )
)
OBJECT_START_FLAG = np.ascontiguousarray(OBJECT_START_FLAG)

OBJECT_GOAL_FLAG = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "assets",
        "maps_topdown_flag",
        "flag1_red.png",
    )
)
OBJECT_GOAL_FLAG = np.ascontiguousarray(OBJECT_GOAL_FLAG)
