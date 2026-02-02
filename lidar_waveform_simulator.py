"""
LiDAR Waveform Simulator - Python Version
===========================================

LiDAR波形模拟工具 - 基于PATH模型的天底观测激光雷达波形模拟

作者: 转换自MATLAB版本
日期: 2026-01-29
版本: v1.0 (重构版 - 核心类和函数已移至lidar_simulator_core.py)
"""

import numpy as np
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, List
import warnings

# 导入核心类和辅助函数
from lidar_simulator_core import (
    CanopyParameters,
    SimulationConfig,
    LiDARWaveformSimulator,
    load_pulse_waveform,
    calculate_tree_distribution_params
)

warnings.filterwarnings("ignore")


# ============================================================================
# 示例使用
# ============================================================================

def example_cylinder():
    """示例1: 圆柱形树冠"""
    print("\n" + "="*60)
    print("示例 1: 圆柱形树冠 (Cylinder Crown)")
    print("="*60)
    
    # 配置树冠参数
    canopy = CanopyParameters(
        crown_shape='cylinder',
        crown_density=0.02,  # trees/m² - 影响空隙率计算
        crown_radius=3.0,
        crown_height_range=(11.0, 17.0),  # 树冠基部11m，顶部17m
        leaf_area_density=0.6,
        leaf_angle_distribution=6,  # 6=spherical
        leaf_reflectance=0.5,
        ground_reflectance=0.2
    )
    
    # 配置模拟参数（Fs0将自动计算）
    config = SimulationConfig(
        use_pulse_shape=True,
        deltaz= 4,
        use_matlab_cylinder_allocation = False
    )
    
    # 创建模拟器并运行
    simulator = LiDARWaveformSimulator(canopy, config)
    height, waveform = simulator.simulate(plot=True)
    
    # 保存结果
    simulator.save_waveform('waveform_cylinder.txt', height, waveform)
    
    return height, waveform


def example_sphere():
    """示例2: 球形树冠"""
    print("\n" + "="*60)
    print("示例 2: 球形树冠 (Spherical Crown)")
    print("="*60)
    
    canopy = CanopyParameters(
        crown_shape='sphere',
        crown_density=0.02,  # trees/m²
        crown_radius=3.0,
        crown_height_range=(10.0, 16.0),
        leaf_area_density=0.8,
        leaf_angle_distribution=6,  # spherical
        leaf_reflectance=0.50,
        ground_reflectance=0.18
    )
    
    config = SimulationConfig(
        midlocat=0.5,  # 球形对称
        deltaz= 1,
        use_pulse_shape=True
    )
    
    simulator = LiDARWaveformSimulator(canopy, config)
    height, waveform = simulator.simulate(plot=True)
    simulator.save_waveform('waveform_sphere.txt', height, waveform)
    
    return height, waveform


def example_cone():
    """示例3: 锥形树冠"""
    print("\n" + "="*60)
    print("示例 3: 锥形树冠 (Conical Crown)")
    print("="*60)
    
    canopy = CanopyParameters(
        crown_shape='cone',
        crown_density=0.015,  # trees/m²
        crown_radius=8.0,
        crown_height_range=(5.0, 17.0),
        leaf_area_density=0.5,
        leaf_angle_distribution=6,  # spherical
        leaf_reflectance=0.3,
        ground_reflectance=0.2
    )
    
    config = SimulationConfig(
        deltaz= 1,
        use_pulse_shape=False
    )
    
    simulator = LiDARWaveformSimulator(canopy, config)
    height, waveform = simulator.simulate(plot=True)
    simulator.save_waveform('waveform_cone.txt', height, waveform)
    
    return height, waveform


def example_with_height_distribution():
    """示例4: 使用真实树高分布（基于叶面积加权）"""
    print("\n" + "="*60)
    print("示例 4: 使用真实树高分布（基于叶面积加权）")
    print("="*60)
    # 真实树结构数据
    tree_base_heights = np.array([8, 9, 8, 7, 6, 8, 6, 6, 8, 9, 8, 7, 6, 8, 6, 6])  # 树基部高度
    tree_radius = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])  # 树冠半径
    tree_lengths = np.array([5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8])  # 树冠长度
    tree_heights = tree_base_heights + tree_lengths  # 树顶高度
    LAD = 0.5  # 叶面积密度（假设值）
    # 计算树分布参数（封装复杂逻辑，cu默认为0.15）
    params = calculate_tree_distribution_params(
        tree_heights, tree_radius, tree_lengths,
        LAD=LAD, verbose=False
    )
    
    # 创建配置（footprint半径12.5m，面积490.87m²）
    canopy = CanopyParameters(
        crown_shape='cylinder',
        crown_density= params['tree_average_density'],
        crown_radius=tree_radius.tolist(),
        crown_height_range=(np.min(tree_base_heights), np.max(tree_heights)),
        leaf_area_density= LAD,
        leaf_angle_distribution=6,
        leaf_reflectance=0.568,
        distribution_type='non-overlapping',
        ground_reflectance=0.186544784
    )
    
    config = SimulationConfig(
        tree_heights=tree_base_heights.tolist(),
        tree_radius=tree_radius.tolist(),
        tree_length=tree_lengths.tolist(),
        use_pulse_shape=False,
        tree_distribution_params=params 
    )
    
    simulator = LiDARWaveformSimulator(canopy, config)
    height, waveform = simulator.simulate(plot=True)
    
    # 加载DART参考波形用于对比
    dart_waveform = None
    try:
        dart_data = np.loadtxt('RAMI_mid/LIDAR_DART_wave.txt', skiprows=6)
        dart_waveform = dart_data[:, 1]
    except Exception as e:
        print(f"警告: 无法加载DART参考波形: {e}")
    
    return height, waveform, dart_waveform


def example_with_pulse():
    """示例5: 考虑发射脉冲形状（使用高斯脉冲）"""
    print("\n" + "="*60)
    print("示例 5: 考虑发射脉冲形状")
    print("="*60)
    
    canopy = CanopyParameters(
        crown_shape='cylinder',
        crown_density=0.015,  # trees/m²
        crown_radius=3.0,
        crown_height_range=(5.0, 17.0),
        leaf_area_density=0.5,
        leaf_angle_distribution=6,  # spherical
        leaf_reflectance=0.56,
        ground_reflectance=0.18
    )
    
    config = SimulationConfig(
        use_pulse_shape=True  # 使用脉冲卷积
    )
    
    simulator = LiDARWaveformSimulator(canopy, config)
    height, waveform = simulator.simulate(plot=True)
    
    return height, waveform


def example_with_dart_pulse():
    """示例6: 使用DART发射脉冲"""
    print("\n" + "="*60)
    print("示例 6: 使用DART发射脉冲波形")
    print("="*60)
    
    # 尝试加载DART脉冲
    pulse = load_pulse_waveform('pulse.txt')
    
    canopy = CanopyParameters(
        crown_shape='cylinder',
        crown_density=0.015,  # trees/m²
        crown_radius=3.0,
        crown_height_range=(5.0, 17.0),
        leaf_area_density=0.5,
        leaf_angle_distribution=6,  # spherical
        leaf_reflectance=0.3,
        ground_reflectance=0.2
    )
    
    config = SimulationConfig(
        layer_height=0.15,
        use_pulse_shape=True if pulse is not None else False,
        pulse_waveform=pulse
    )
    
    simulator = LiDARWaveformSimulator(canopy, config)
    height, waveform = simulator.simulate(plot=True)
    
    return height, waveform


def example_rami_scene(rami_folder='RAMI_up', plot=True):
    """
    示例7: RAMI场景模拟
    
    RAMI (RAdiation transfer Model Intercomparison) 场景是标准的森林辐射传输验证场景
    
    Parameters:
    -----------
    rami_folder : str
        RAMI数据文件夹名称，可选：
        'RAMI_lu', 'RAMI_up', 'RAMI_ru',
        'RAMI_Left', 'RAMI_mid', 'RAMI_right',
        'RAMI_ld', 'RAMI_down', 'RAMI_rd'
        对应不同的footprint位置
    plot : bool
        是否绘图
        
    Returns:
    --------
    height : np.ndarray
        高度数组
    waveform : np.ndarray
        模拟波形
    dart_waveform : np.ndarray
        DART参考波形（用于对比）
    """
    import scipy.io
    import os
    from rami_tree_data import (TREE_HEIGHTS, CROWN_BASES, CROWN_RADII, 
                                 get_tree_leaf_areas, get_crown_lengths, 
                                 FOOTPRINT_CENTERS, get_fs0, get_tree_counts)
    
    print("\n" + "="*60)
    print(f"示例 7: RAMI场景模拟 ({rami_folder})")
    print("="*60)
    
    # 从配置文件加载RAMI场景树参数
    tree_heights = TREE_HEIGHTS
    crown_radii = CROWN_RADII
    tree_leaf_areas = get_tree_leaf_areas(rami_folder)
    crown_lengths = get_crown_lengths()

    
    # 获取预计算的场景数据
    center_x, center_y = FOOTPRINT_CENTERS[rami_folder]
    manual_Fs0 = get_fs0(rami_folder)
    tree_counts = get_tree_counts(rami_folder)
    
    print(f"\nFootprint位置: ({center_x}, {center_y})")
    print(f"Footprint内树木总数: {int(np.sum(tree_counts))}")
    print(f"树类型分布: {tree_counts.astype(int)}")
    
    # 使用封装的函数计算分布参数
    params = calculate_tree_distribution_params(
        tree_heights, crown_radii, crown_lengths,
        tree_counts=tree_counts,
        tree_leaf_areas=tree_leaf_areas,
        verbose=True
    )
       
    # RAMI场景使用固定的叶面积密度（物理参数，不是反推值）
    LAD_rami = 0.60  # RAMI场景的真实LAD
    print(f"使用固定LAD (u): {LAD_rami:.2f}")
    
    # 配置树冠参数
    canopy = CanopyParameters(
        crown_shape='cylinder',
        crown_height_range=(np.min(tree_heights), np.max(tree_heights)),
        leaf_area_density=LAD_rami,  # 使用RAMI场景固定LAD
        leaf_angle_distribution=6,  # spherical
        leaf_reflectance=0.568606,
        leaf_transmittance=0.0,
        ground_reflectance=0.186544784
    )
    
    # 读取DART发射脉冲
    pulse_file = os.path.join(rami_folder, 'pulse.txt')
    if os.path.exists(pulse_file):
        pulse = load_pulse_waveform(pulse_file)
    else:
        pulse = load_pulse_waveform('pulse.txt')
    
    # 配置模拟参数（直接使用18种树类型 + 加权数组）
    # 注意：_calculate_height_distribution需要的是 tree_counts * tree_leaf_areas
    weighted_tree_counts = tree_counts * tree_leaf_areas
    config = SimulationConfig(
        tree_heights=tree_heights,
        tree_radius=crown_radii,
        tree_length=crown_lengths,
        tree_weights=weighted_tree_counts,
        use_pulse_shape=True if pulse is not None else False,
        pulse_waveform=pulse,
        manual_Fs0=manual_Fs0,
        rami_mode=True,
        tree_distribution_params=params  # 自动解包zfre, layers_vb, layers_v, freConv
    )
    
    # 运行模拟
    simulator = LiDARWaveformSimulator(canopy, config)
    height, waveform = simulator.simulate(plot=False)
    
    # 读取DART参考波形
    dart_file = os.path.join(rami_folder, 'LIDAR_CONVOLVED_wave.txt')
    dart_waveform = None
    dart_height = None
    
    if os.path.exists(dart_file):
        try:
            dart_data = np.loadtxt(dart_file)
            if dart_data.ndim == 2 and dart_data.shape[1] >= 2:
                # DART格式：第1列是时间(ns)，第2列是返回信号
                # MATLAB: heightDart = RAMDartC(:,1).*-0.15
                # 时间转换为高度：光速c = 0.3 m/ns，往返距离 = c * t / 2 = 0.15 * t
                # 负号是因为时间增加对应高度降低（从上往下）
                dart_time = dart_data[:, 0]
                dart_height = dart_time * (-0.15)  # 时间(ns) × (-0.15) = 高度(m)
                dart_waveform = dart_data[:, 1]  # 返回信号
                
                # MATLAB代码中的过滤：heightDart >= -3.15 且 returnPhoto > 0
                valid_mask = (dart_height >= -3.15) & (dart_waveform > 0)
                
                # 找到第一个有效信号的位置
                if np.any(valid_mask):
                    first_valid = np.where(valid_mask)[0][0]
                    dart_height = dart_height[first_valid:]
                    dart_waveform = dart_waveform[first_valid:]
                    
                    # 再次应用高度过滤
                    valid_height = dart_height >= -3.15
                    dart_height = dart_height[valid_height]
                    dart_waveform = dart_waveform[valid_height]
                
                # 归一化
                if np.max(dart_waveform) > 0:
                    dart_waveform = dart_waveform / np.max(dart_waveform)
                
                print(f"\\n已加载DART参考波形: {len(dart_waveform)}个点")
                print(f"DART高度范围: {np.min(dart_height):.2f} - {np.max(dart_height):.2f} m")
            else:
                print(f"警告: DART文件格式不正确，列数={dart_data.shape[1] if dart_data.ndim == 2 else 1}")
        except Exception as e:
            print(f"警告: 无法读取DART波形文件: {e}")
    
    # 绘图对比
    if plot and dart_waveform is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        
        # 归一化Python波形
        waveform_norm = waveform / np.max(waveform) if np.max(waveform) > 0 else waveform
        
        # 左图：Python模拟
        axes[0].plot(waveform_norm, height, 'b-', linewidth=2, label='PATH Python')
        axes[0].set_xlabel('Normalized Return Signal', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Height (m)', fontsize=12, fontweight='bold')
        axes[0].set_title('PATH Simulation', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # 中图：DART参考
        axes[1].plot(dart_waveform, dart_height, 'r-', linewidth=2, label='DART Reference')
        axes[1].set_xlabel('Normalized Return Signal', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Height (m)', fontsize=12, fontweight='bold')
        axes[1].set_title('DART Reference', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # 右图：叠加对比
        axes[2].plot(waveform_norm, height, 'b-', linewidth=2, label='PATH Python', alpha=0.7)
        axes[2].plot(dart_waveform, dart_height, 'r--', linewidth=2, label='DART Reference', alpha=0.7)
        axes[2].set_xlabel('Normalized Return Signal', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Height (m)', fontsize=12, fontweight='bold')
        axes[2].set_title('Comparison', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.suptitle(f'RAMI Scene: {rami_folder}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    elif plot:
        simulator.plot_waveform(height, waveform)
    
    return height, waveform, dart_waveform


if __name__ == "__main__":
    """
    运行示例
    """
    print("\n" + "="*70)
    print("LiDAR波形模拟器 - Python版本")
    print("="*70)
    print("\n可用示例:")
    print("1. example_cylinder()          - 圆柱形树冠")
    print("2. example_sphere()            - 球形树冠")
    print("3. example_cone()              - 锥形树冠")
    print("4. example_with_height_distribution() - 使用真实树高分布")
    print("5. example_with_pulse()        - 考虑发射脉冲（高斯）")
    print("6. example_with_dart_pulse()   - 使用DART发射脉冲")
    print("7. example_rami_scene()        - RAMI标准场景")
    print("\n" + "="*70 + "\n")
    # example_cylinder()
    # 运行默认示例
    # example_with_height_distribution()
    # example_sphere()
    # example_cone()
    try:
        # 测试多个RAMI场景
        test_scenes = ['RAMI_lu', 'RAMI_up', 'RAMI_rd']
        print("\n" + "="*70)
        print("测试多个RAMI场景")
        print("="*70)
        
        for scene in test_scenes:
            print(f"\n{'='*60}")
            print(f"场景: {scene}")
            print(f"{'='*60}")
            height, waveform, dart_waveform = example_rami_scene(rami_folder=scene, plot=True)
            if waveform is not None and dart_waveform is not None:
                min_len = min(len(waveform), len(dart_waveform))
                waveform_norm = waveform[:min_len] / np.max(waveform[:min_len])
                dart_norm = dart_waveform[:min_len] / np.max(dart_waveform[:min_len])
                correlation = np.corrcoef(waveform_norm, dart_norm)[0, 1]
                print(f"\n结果:")
                print(f"  波形长度: {len(waveform)}")
                print(f"  最大回波: {np.max(waveform):.6f}")
                print(f"  相关性: {correlation:.4f}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
