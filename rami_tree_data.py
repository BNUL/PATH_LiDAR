"""
RAMI场景树参数数据
包含18种树的结构参数和叶面积数据
"""
import numpy as np

TREE_HEIGHTS = np.array([
    15.37, 19.35, 22.58, 25.76, 27.09, 19.86, 25.49, 27.99,
    30.51, 13.72, 10.90, 25.27, 30.49, 5.91, 11.27, 14.41,
    18.34, 20.70
])

CROWN_BASES = np.array([
    7.57, 10.14, 12.60, 14.46, 16.95, 12.93, 16.43, 15.23,
    18.36, 6.60, 1.82, 18.37, 16.52, 1.85, 5.93, 7.69,
    7.82, 7.56
])

CROWN_RADII = np.array([
    2.23, 1.92, 2.31, 2.78, 3.60, 1.95, 2.19, 3.11,
    3.56, 2.25, 1.60, 2.65, 3.15, 2.55, 1.93, 2.38,
    2.47, 2.65
])


TREE_LEAF_AREAS_DEFAULT = np.array([
    13.6225, 12.9206, 29.3120, 69.679, 120.983, 10.2282,
    32.4535, 75.3714, 93.9876, 20.288, 9.9413, 37.7483,
    143.369, 8.8554, 18.073, 27.528, 58.319, 92.0224
])


FOOTPRINT_CENTERS = {
    'RAMI_lu': (22.5, 82.5),
    'RAMI_up': (52.5, 82.5),
    'RAMI_ru': (82.5, 82.5),
    'RAMI_Left': (22.5, 52.5),
    'RAMI_mid': (52.5, 52.5),
    'RAMI_right': (82.5, 52.5),
    'RAMI_ld': (22.5, 22.5),
    'RAMI_down': (52.5, 22.5),
    'RAMI_rd': (82.5, 22.5)
}


FS_RAMI = np.array([
    0.3061627, 0.33171726, 0.3706362, 0.25579977, 0.20590448,
    0.26159445, 0.18429132, 0.19378647, 0.26492894
])

RAMI_FOLDER_INDEX = {
    'RAMI_lu': 0,
    'RAMI_up': 1,
    'RAMI_ru': 2,
    'RAMI_Left': 3,
    'RAMI_mid': 4,
    'RAMI_right': 5,
    'RAMI_ld': 6,
    'RAMI_down': 7,
    'RAMI_rd': 8
}

TREE_COUNTS = {
    'RAMI_lu': np.array([3, 1, 0, 1, 0, 7, 9, 3, 1, 1, 1, 16, 1, 0, 1, 7, 2, 0]),
    'RAMI_up': np.array([0, 4, 8, 2, 0, 5, 4, 4, 0, 0, 1, 4, 0, 0, 1, 0, 3, 0]),
    'RAMI_ru': np.array([0, 7, 17, 4, 0, 2, 9, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0]),
    'RAMI_Left': np.array([0, 1, 0, 0, 0, 5, 14, 5, 1, 1, 3, 4, 1, 1, 3, 3, 5, 7]),
    'RAMI_mid': np.array([1, 1, 0, 0, 0, 5, 17, 6, 0, 0, 0, 4, 2, 0, 1, 7, 5, 3]),
    'RAMI_right': np.array([1, 3, 12, 6, 3, 2, 2, 2, 0, 3, 0, 0, 0, 2, 6, 1, 3, 0]),
    'RAMI_ld': np.array([1, 0, 0, 0, 0, 5, 24, 7, 0, 2, 2, 5, 2, 0, 1, 3, 7, 0]),
    'RAMI_down': np.array([1, 1, 4, 0, 2, 11, 17, 7, 0, 0, 1, 0, 3, 2, 2, 3, 0, 0]),
    'RAMI_rd': np.array([0, 1, 4, 5, 0, 6, 15, 3, 1, 1, 1, 0, 0, 5, 3, 1, 1, 0]),
}


def get_tree_leaf_areas(rami_folder='RAMI_mid'):
    """
    获取指定RAMI场景的叶面积数据
    
    Parameters:
    -----------
    rami_folder : str
        RAMI场景文件夹名称
    
    Returns:
    --------
    np.ndarray
        叶面积数组
    """

  
    return TREE_LEAF_AREAS_DEFAULT.copy()


def get_crown_lengths():
    """计算树冠长度"""
    return TREE_HEIGHTS - CROWN_BASES


def get_fs0(rami_folder='RAMI_mid'):
    """
    获取指定RAMI场景的Fs0值（空隙率）
    
    Parameters:
    -----------
    rami_folder : str
        RAMI场景文件夹名称
    
    Returns:
    --------
    float
        Fs0值
    """
    idx = RAMI_FOLDER_INDEX.get(rami_folder, 4)  # 默认使用mid的索引
    return float(FS_RAMI[idx])


def get_tree_counts(rami_folder='RAMI_mid'):
    """
    获取指定RAMI场景footprint内的树数量分布
    
    Parameters:
    -----------
    rami_folder : str
        RAMI场景文件夹名称
    
    Returns:
    --------
    np.ndarray
        18种树类型的数量数组
    """
    return TREE_COUNTS[rami_folder].copy()
