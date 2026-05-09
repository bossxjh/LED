from .libero import parse_libero
from .bcz import parse_bcz
from .taco_play import parse_taco
from .toto import parse_toto
from .jaco_play import parse_jaco
from .roboturk import parse_roboturk
from .cable_routing import parse_route
from .nyu_opening_door import parse_nyu_opening_door
from .franka_play import parse_franka_play
from .fractal import parse_fractal


DATASET_PARSERS = {
    "libero": parse_libero,#官方的libero数据集格式：.h5文件或者.hdf5文件
    "libero-goal": parse_libero,
    "libero-object": parse_libero,
    "libero-spatial": parse_libero,
    "libero-10": parse_libero,
    "libero-90": parse_libero,
    "bcz":parse_bcz,#bc-z数据集格式：
    "taco_play": parse_taco, #taco-play数据集格式：pickle文件
    "toto":parse_toto, #toto数据集格式：pickle文件
    "jaco":parse_jaco, #jaco-play数据集格式：pickle文件
    "roboturk":parse_roboturk, #roboturk数据集格式：pickle文件
    "cable_routing":parse_route, #cable_routing数据集格式：mp4视频文件
    "nyu_opening_door":parse_nyu_opening_door, #nyu_opening_door数据集格式：mp4视频文件
    "franka_play":parse_franka_play, #franka_play数据集格式：pickle文件
    "fractal":parse_fractal, #fractal数据集格式：tfrecord文件
}
