"""
CHM-based LiDAR Waveform Simulator
===================================

从冠层高度模型(CHM)栅格数据模拟LiDAR波形

作者: weihua
日期: 2026-02-03
"""

import numpy as np
import matplotlib.pyplot as plt
from lidar_simulator_core import CanopyParameters, CHMWaveformSimulator


def simulate_from_chm(chm_file: str,
                     layer_height: float = 0.15,
                     height_threshold: float = 1.0,
                     branch_height_ratio: float = 1/3,
                     manual_Fs0: float = None,
                     manual_tree_top_height: float = None,
                     max_height_percentile: float = 80.0,
                     leaf_area_density: float = 0.3,
                     leaf_angle_distribution: int = 6,
                     leaf_reflectance: float = 0.35,
                     ground_reflectance: float = 0.35,
                     use_pulse_shape: bool = True,
                     plot: bool = True,
                     save_file: str = None):
    """
    从CHM文件模拟LiDAR波形
    
    Parameters:
    -----------
    chm_file : str
        CHM栅格文件路径(.tif格式)
    layer_height : float, optional
        层高 (m)，默认0.15m
    height_threshold : float, optional
        高度阈值 (m)，CHM < threshold 判定为间隙，默认1.0m
    branch_height_ratio : float, optional
        枝下高占平均树高的比例，默认1/3
    manual_Fs0 : float, optional
        手动设置空隙率 (0-1)，如果为None则根据CHM自动计算
    manual_tree_top_height : float, optional
        手动设置当前树顶高度 (m)，如果为None则使用max(CHM)
        用于CHM是早期数据但树已长高的情况（例如：GEDI波形显示25m但CHM最大值只有20m）
        注意：Fs分布不变，layers_v不变，但layers_vb会增加
    leaf_area_density : float, optional
        叶面积密度 (m²/m³)，默认0.5
    leaf_angle_distribution : int, optional
        叶倾角分布类型 (1-6)，默认6 (spherical)
        1: planophile, 2: erectophile, 3: plagiophile
        4: extremophile, 5: uniform, 6: spherical
    leaf_reflectance : float, optional
        叶片反射率 (0-1)，默认0.568606
    ground_reflectance : float, optional
        地面反射率 (0-1)，默认0.18
    use_pulse_shape : bool, optional
        是否使用脉冲卷积，默认True (使用GEDI高斯脉冲)
    plot : bool, optional
        是否绘制波形，默认True
    save_file : str, optional
        保存波形的文件名，如果为None则不保存
    
    Returns:
    --------
    height : np.ndarray
        高度数组 (m)
    waveform : np.ndarray
        模拟的波形数据
    """
    print("\n" + "="*70)
    print("CHM-based LiDAR Waveform Simulation")
    print("="*70)
    print(f"\nInput Parameters:")
    print(f"  CHM file: {chm_file}")
    print(f"  Layer height (cu): {layer_height} m")
    print(f"  Height threshold: {height_threshold} m")
    print(f"  Branch height ratio: {branch_height_ratio:.1%}")
    print(f"  Manual Fs0: {manual_Fs0 if manual_Fs0 is not None else 'Auto'}")
    print(f"  Manual tree top height: {manual_tree_top_height if manual_tree_top_height is not None else 'Auto (use max(CHM))'}")
    print(f"  Leaf area density (LAD): {leaf_area_density} m²/m³")
    print(f"  Leaf angle distribution: {leaf_angle_distribution}")
    print(f"  Leaf reflectance: {leaf_reflectance}")
    print(f"  Ground reflectance: {ground_reflectance}")
    print(f"  Use pulse shape: {use_pulse_shape}")
    print("="*70)
    
    # 创建树冠参数
    canopy_params = CanopyParameters(
        crown_shape='cylinder',  # CHM不需要指定形状
        leaf_area_density=leaf_area_density,
        leaf_angle_distribution=leaf_angle_distribution,
        leaf_reflectance=leaf_reflectance,
        ground_reflectance=ground_reflectance
    )
    
    # 创建CHM模拟器
    simulator = CHMWaveformSimulator(
        chm_file=chm_file,
        canopy_params=canopy_params,
        layer_height=layer_height,
        height_threshold=height_threshold,
        branch_height_ratio=branch_height_ratio,
        manual_Fs0=manual_Fs0,
        manual_tree_top_height=manual_tree_top_height,
        use_pulse_shape=use_pulse_shape
    )
    
    # 运行模拟
    height, waveform = simulator.simulate(plot=plot)
    
    # 保存结果
    if save_file is not None:
        simulator.save_waveform(save_file, height, waveform)
    
    return height, waveform


if __name__ == "__main__":
    # 基本使用示例
    print("\n" + "="*70)
    print("CHM波形模拟示例")
    print("="*70)
    
    chm_file = "6_chm.tif"
    height, waveform = simulate_from_chm(
        chm_file=chm_file,
        height_threshold=1.0,
        leaf_area_density=0.3,
        plot=True,
        save_file="chm_waveform_output.txt"
    )
    
    # # 示例2: 自定义参数
    # print("\n" + "="*70)
    # print("示例2: 自定义参数（更高阈值和手动Fs0）")
    # print("="*70)
    
    # height, waveform = simulate_from_chm(
    #     chm_file=chm_file,
    #     height_threshold=2.0,
    #     manual_Fs0=0.3,
    #     leaf_area_density=0.8,
    #     branch_height_ratio=0.25,
    #     plot=True
    # )
    
    # # 示例3: 比较不同高度阈值
    # print("\n" + "="*70)
    # print("示例3: 比较不同高度阈值的影响")
    # print("="*70)
    
    # compare_different_thresholds(chm_file, thresholds=[0.5, 1.0, 2.0])
    
    # # 示例4: 比较不同手动Fs0
    # print("\n" + "="*70)
    # print("示例4: 比较不同手动Fs0的影响")
    # print("="*70)
    
    # compare_manual_Fs0(chm_file, Fs0_values=[0.1, 0.3, 0.5])
