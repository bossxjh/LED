from .libero import parse_meta_libero
from .libero_rlds_v2 import parse_meta_libero_rlds
from .local import parse_meta_local_demo_dataset

DATASET_PARSERS_META = {
    "libero": parse_meta_libero,#官方的libero数据集格式：.h5文件或者.hdf5文件
    "libero-goal": parse_meta_libero,
    "libero-object": parse_meta_libero,
    "libero-spatial": parse_meta_libero,
    "libero-10": parse_meta_libero,
    "libero-90": parse_meta_libero,
    "rlds-libero": parse_meta_libero_rlds,#官方的libero数据集格式：.h5文件或者.hdf5文件
    "rlds-libero-goal": parse_meta_libero_rlds,
    "rlds-libero-object": parse_meta_libero_rlds,
    "rlds-libero-spatial": parse_meta_libero_rlds,
    "rlds-libero-10": parse_meta_libero_rlds,
    "rlds-libero-90": parse_meta_libero_rlds,
    "local": parse_meta_local_demo_dataset,
}
