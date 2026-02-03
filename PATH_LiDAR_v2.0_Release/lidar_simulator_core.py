"""
LiDAR Simulator Core Components
================================

作者: weihua
日期: 2026-01-29
版本: v2.0
"""

import numpy as np
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
import warnings

warnings.filterwarnings("ignore")


@dataclass
class CanopyParameters:
    crown_shape: str = 'cylinder'
    crown_density: float = 0.01
    crown_radius: Union[float, List[float]] = 3.0
    crown_height_range: Tuple[float, float] = (5.0, 17.0)
    leaf_area_density: float = 0.5
    leaf_angle_distribution: int = 6  # 6: spherical (default)
    distribution_type: str = 'non-overlapping'  # 'poisson' (允许重叠) 或 'non-overlapping' (不重叠)
    leaf_reflectance: float = 0.568606
    leaf_transmittance: float = 0.4  # MATLAB assumes reflection only
    ground_reflectance: float = 0.18
    slope: float = 0.0
    aspect: float = 0.0
    
    def __post_init__(self):
        """验证参数有效性"""
        valid_shapes = ['cylinder', 'sphere', 'cone']
        if self.crown_shape.lower() not in valid_shapes:
            raise ValueError(f"crown_shape must be one of {valid_shapes}")
        
        if not 0 <= self.leaf_reflectance <= 1:
            raise ValueError("leaf_reflectance must be between 0 and 1")
        
        if not 0 <= self.leaf_transmittance <= 1:
            raise ValueError("leaf_transmittance must be between 0 and 1")
            
        if not 0 <= self.ground_reflectance <= 1:
            raise ValueError("ground_reflectance must be between 0 and 1")


@dataclass
class SimulationConfig:
    """
    模拟配置参数类
    
    Parameters:
    -----------
        
    midlocat : float
        树冠对称性参数
        0.5: 球形（上下对称）
        1.0: 圆柱形或锥形（不对称）
        
    use_pulse_shape : bool
        是否考虑发射脉冲形状
        True: 需要输入发射波形或进行高斯拟合
        False: 假设发射脉冲为瞬时脉冲（无宽度）
        
    pulse_waveform : np.ndarray, optional
        发射波形数据，如果use_pulse_shape=True且不提供，将使用高斯脉冲
        
    tree_heights : np.ndarray, optional
        真实树基部高度数组 (m)，当height_distribution=True时使用
        
    tree_radius : np.ndarray, optional
        每棵树的树冠半径数组 (m)，与tree_heights对应
        
    tree_length : np.ndarray, optional
        每棵树的树冠长度数组 (m)，与tree_heights对应
        实际树高 = tree_heights + tree_length
    
    manual_Fs0 : float, optional
        手动设置空隙率 Fs0 (0-1)，如果设置则不自动计算
        None表示自动计算
    
    use_matlab_cylinder_allocation : bool, optional
        是否使用MATLAB特殊的圆柱形Fs分配方式（/5和*4/5）
        True: 使用MATLAB特定分配（为匹配特定案例）
        False: 使用通用圆柱形分配（默认）
    
    use_mixed_crown_allocation : bool, optional
        是否使用混合树冠尺寸的通用分配算法
        True: 使用通用算法处理不同尺寸的圆柱形树冠
        False: 使用简单的单一路径长度分配（默认）
    
    footprint_radius : float, optional
        激光雷达footprint半径 (m)，用于计算Fs0
        默认12.5m（GEDI）
        footprint面积 = π × footprint_radius²
    
    rami_mode : bool, optional
        是否为RAMI场景模式，默认False
        RAMI模式下会使用特殊的Fs分配方式
    
    zfre_distribution : np.ndarray, optional
        RAMI场景的树冠长度分布频率（zfre）
        用于Fs(2, 2:end) = zfre * (1 - Fs0)
    
    tree_weights : np.ndarray, optional
        树木权重数组，与tree_heights长度相同
        用于freq_conv计算时的加权（如RAMI场景的叶面积加权）
    
    rami_layers_vb : int, optional
        RAMI模式下的layers_vb（总层数）
        MATLAB: layers_vb = ceil(max(treeHeight)/cu)
    
    rami_layers_v : int, optional
        RAMI模式下的layers_v（树冠层数）
        MATLAB: layers_v = ceil(CrownLengthmax(i)/cu)，即加权平均树冠长度
    
    tree_distribution_params : dict, optional
        树高分布参数字典（来自calculate_tree_distribution_params）
        如果提供，将自动解包以下参数：
        - zfre_distribution: zfre数组
        - rami_layers_vb: layers_vb值
        - rami_layers_v: layers_v值
        - manual_freq_conv: freConv数组
        - tree_weights: numOftreesWeibyla数组
        注意：此参数会覆盖单独设置的上述参数
    """
    layer_height: float = 0.15
    footprint_radius: float = 12.5
    midlocat: float = 1.0
    use_pulse_shape: bool = False
    pulse_waveform: Optional[np.ndarray] = None
    tree_heights: Optional[np.ndarray] = None
    tree_radius: Optional[np.ndarray] = None
    tree_length: Optional[np.ndarray] = None
    use_matlab_cylinder_allocation: bool = False
    use_mixed_crown_allocation: bool = False
    rami_mode: bool = False
    zfre_distribution: Optional[np.ndarray] = None
    tree_weights: Optional[np.ndarray] = None
    rami_layers_vb: Optional[int] = None
    rami_layers_v: Optional[int] = None
    manual_Fs0: Optional[float] = None
    manual_Fs: Optional[np.ndarray] = None
    manual_layers_v: Optional[int] = None
    manual_layers_vb: Optional[int] = None
    manual_freq_conv: Optional[np.ndarray] = None
    tree_distribution_params: Optional[dict] = None
    deltaz: Optional[float] = None  # 树冠随机分布的高度范围 (m)，用于生成均匀分布的freq_conv
    
    def __post_init__(self):
        """自动解包tree_distribution_params（如果提供）"""
        if self.tree_distribution_params is not None:
            params = self.tree_distribution_params
            # 自动设置相关参数（使用object.__setattr__因为dataclass是frozen-like）
            if 'zfre' in params and self.zfre_distribution is None:
                object.__setattr__(self, 'zfre_distribution', params['zfre'])
            if 'layers_vb' in params and self.rami_layers_vb is None:
                object.__setattr__(self, 'rami_layers_vb', params['layers_vb'])
            if 'layers_v' in params and self.rami_layers_v is None:
                object.__setattr__(self, 'rami_layers_v', params['layers_v'])
            if 'freConv' in params and self.manual_freq_conv is None:
                object.__setattr__(self, 'manual_freq_conv', params['freConv'])
            if 'numOftreesWeibyla' in params and self.tree_weights is None:
                object.__setattr__(self, 'tree_weights', params['numOftreesWeibyla'])


# ============================================================================
# 辅助函数
# ============================================================================

def load_pulse_waveform(filename: str) -> np.ndarray:
    """
    从文件加载发射脉冲波形
    
    Parameters:
    -----------
    filename : str
        脉冲波形文件路径
        
    Returns:
    --------
    pulse : np.ndarray
        脉冲波形数据
    """
    try:
        data = np.loadtxt(filename)
        if data.ndim == 2:
            pulse = data[:, -1]  # 假设最后一列是脉冲强度
        else:
            pulse = data
        # 归一化
        pulse = pulse / np.sum(pulse)
        return pulse
    except Exception as e:
        print(f"Error loading pulse waveform: {e}")
        return None


def calculate_tree_distribution_params(tree_heights, tree_radius, tree_length, 
                                      tree_counts=None, LAD=0.5, cu=0.15, verbose=True,
                                      tree_leaf_areas=None, footprint_radius=12.5):
    """
    计算基于叶面积加权的树高分布参数
    
    Parameters:
    -----------
    tree_heights : np.ndarray
        树顶高度数组 (m)
    tree_radius : np.ndarray
        树冠半径数组 (m)
    tree_length : np.ndarray
        树冠长度数组 (m)
    tree_counts : np.ndarray, optional
        每种树的数量，默认全为1
    LAD : float, optional
        叶面积密度 (m²/m³)，默认0.5
    cu : float, optional
        层高 (m)，默认0.15
    verbose : bool, optional
        是否输出详细信息，默认True
    tree_leaf_areas : np.ndarray, optional
        预定义的叶面积数组，如果提供则不使用LAD计算
    footprint_radius : float, optional
        激光雷达footprint半径 (m)，默认12.5m（GEDI）
    
    Returns:
    --------
    dict : 包含以下键值:
        - numOftreesWeibyla: 叶面积加权的树数量
        - zfre: 树冠长度频率分布
        - freConv: 加权树顶高度分布
        - layers_vb: 基于最大树高的层数
        - layers_v: 基于平均树冠长度的层数
        - sim_tree_tops_weighted: 加权树顶高度数组
        - meanLength: 加权平均树冠长度
        - true_LAI: 理论LAI
        - tree_leaf_areas: 每棵树的叶面积数组
    """
    if tree_counts is None:
        tree_counts = np.ones(len(tree_heights))
    
    # 计算footprint面积
    footprint_area = np.pi * footprint_radius ** 2
    
    # 计算或使用预定义的叶面积
    if tree_leaf_areas is None:
        tree_volumes = np.pi * tree_radius**2 * tree_length
        tree_leaf_areas = LAD * tree_volumes
        numOftreesWeibyla = tree_counts *tree_radius**2
    else:
        numOftreesWeibyla = tree_counts * tree_leaf_areas
    
    # 计算加权平均树冠长度
    meanLength = np.sum(tree_length * tree_counts) / np.sum(tree_counts)
    # 计算zfre分布（树冠长度的频率分布）
    # MATLAB: zL = meanLength:-cu:0 → [meanLength, meanLength-cu, ..., cu, 0]
    # MATLAB填充逻辑：
    #   zfre(end) = 最长的树 (>= zL(1))
    #   zfre(end-ii+1) = 树长在 zL(ii) ~ zL(ii+1) 之间
    #   zfre(1) = 0 (未赋值)
    # 结果：zfre = [0, 短树, 中树, ..., 长树]
    
    zL = np.arange(meanLength, -0.5*cu, -cu)  # 从meanLength到0附近
    zfre = np.zeros(len(zL))
    
    if verbose:
        print(f"zL范围: {zL[0]:.3f} → {zL[-1]:.3f}, 长度: {len(zL)}")
    
    # 模拟MATLAB的填充逻辑
    for ii in range(len(zL)):
        if ii == 0:
            # zfre(end) = 最长的树
            zfre[-1] = np.sum(tree_counts[tree_length >= zL[ii]])
        elif ii < len(zL):
            # zfre(end-ii+1) = 中间的树
            if ii + 1 < len(zL):
                mask = (tree_length <= zL[ii]) & (tree_length > zL[ii+1])
                zfre[-(ii+1)] = np.sum(tree_counts[mask])
    # zfre[0]保持为0（对应MATLAB的zfre(1)=0）
    
    # 归一化：MATLAB中zfre(1)=0不参与归一化，所以sum(zfre)实际是sum(zfre(2:end))
    # Python中需要确保归一化的分母是sum(zfre[1:])
    zfre_sum = np.sum(zfre[1:]) if len(zfre) > 1 else np.sum(zfre)
    if zfre_sum > 0:
        zfre[1:] = zfre[1:] / zfre_sum  # 只归一化非零部分
    # zfre[0]保持为0
    
    if verbose:
        print(f"zfre归一化后总和: {np.sum(zfre):.6f}")
        print(f"zfre非零值: {np.sum(zfre > 0)} 个")
        # 打印前10个和后10个非零值
        nonzero_indices = np.where(zfre > 0)[0]
        if len(nonzero_indices) > 0:
            print(f"zfre前5个非零值索引和数值:")
            for idx in nonzero_indices[:5]:
                print(f"  zfre[{idx}] = {zfre[idx]:.6f}")
            print(f"zfre后5个非零值索引和数值:")
            for idx in nonzero_indices[-5:]:
                print(f"  zfre[{idx}] = {zfre[idx]:.6f}")
    
    # 计算freConv（加权树顶高度分布）
    zConv = np.arange(np.ceil(np.max(tree_heights)), np.floor(np.min(tree_heights)) - cu, -cu)
    freConv = np.zeros(len(zConv))
    for ii in range(len(zConv)):
        if ii == len(zConv) - 1:
            freConv[ii] = np.sum(numOftreesWeibyla[tree_heights <= zConv[ii]])
        else:
            mask = (tree_heights <= zConv[ii]) & (tree_heights > zConv[ii+1])
            freConv[ii] = np.sum(numOftreesWeibyla[mask])
    freConv = freConv / np.sum(freConv) if np.sum(freConv) > 0 else freConv
    
    # 计算layers
    layers_vb = int(np.ceil(np.max(tree_heights) / cu))
    layers_v = int(np.ceil(meanLength / cu))
    
    # 创建加权树顶高度数组
    sim_tree_tops_weighted = []
    for i in range(len(tree_heights)):
        repeat_count = int(np.round(numOftreesWeibyla[i]))
        sim_tree_tops_weighted.extend([tree_heights[i]] * repeat_count)
    sim_tree_tops_weighted = np.array(sim_tree_tops_weighted)
    
    # 计算理论LAI
    total_leaf_area = np.sum(tree_leaf_areas * tree_counts)
    true_LAI = total_leaf_area / footprint_area
    treedensity = len(tree_heights) / footprint_area
    
    if verbose:
        print(f"Footprint半径: {footprint_radius:.2f} m, 面积: {footprint_area:.2f} m²")
        print(f"叶面积范围: {np.min(tree_leaf_areas):.2f} - {np.max(tree_leaf_areas):.2f} m^2")
        print(f"加权平均树冠长度: {meanLength:.3f} m")
        print(f"树冠长度分布bins数: {len(zfre)}")
        print(f"zfre总和: {np.sum(zfre):.6f}")
        print(f"freConv长度: {len(freConv)}, 总和: {np.sum(freConv):.6f}")
        print(f"layers_vb (基于max(treeHeight)): {layers_vb}")
        print(f"layers_v (基于meanLength): {layers_v}")
        print(f"实际树木数: {int(np.sum(tree_counts))}")
        print(f"加权树木数 (用于freConv): {int(np.sum(numOftreesWeibyla))}")
        print(f"理论LAI: {true_LAI:.3f}")
    
    return {
        'numOftreesWeibyla': numOftreesWeibyla,
        'zfre': zfre,
        'freConv': freConv,
        'layers_vb': layers_vb,
        'layers_v': layers_v,
        'sim_tree_tops_weighted': sim_tree_tops_weighted,
        'meanLength': meanLength,
        'true_LAI': true_LAI,
        'tree_leaf_areas': tree_leaf_areas,
        'tree_average_density': treedensity,
        'footprint_area': footprint_area
    }


class LiDARWaveformSimulator:
    """
    LiDAR波形模拟器主类
    """
    
    @staticmethod
    def _area_scatter_phase_function(rho_l: float, tau_l: float, 
                                     theta_0: float, phi_0: float, 
                                     theta: float, phi: float, 
                                     leaf_class: int) -> float:
        # Spherical分布的特殊处理（解析解）
        if leaf_class == 6:
            sza_r = np.deg2rad(theta_0 + 180)
            vza_r = np.deg2rad(theta)
            diff_phi = np.deg2rad(phi - phi_0)
            cos_beta = (np.cos(sza_r) * np.cos(vza_r) + 
                       np.sin(sza_r) * np.sin(vza_r) * np.cos(diff_phi))
            beta = np.arccos(cos_beta)
            omega = rho_l + tau_l
            func = (np.sin(beta) - beta * cos_beta) / np.pi
            Gamma_val = omega * func / 3.0 + (tau_l * cos_beta) / 3.0
            return np.abs(Gamma_val)
        
        # 其他分布使用数值积分
        if leaf_class == 5:  # uniform
            a, b = 0, 0
        elif leaf_class == 4:  # extremophile
            a, b = 1, 4
        elif leaf_class == 3:  # plagiophile
            a, b = -1, 4
        elif leaf_class == 2:  # erectophile
            a, b = -1, 2
        elif leaf_class == 1:  # planophile
            a, b = 1, 2
        else:
            a, b = 0, 0
        
        # 调用APF函数
        Gamma_val = LiDARWaveformSimulator._get_APF(
            a, b, rho_l, tau_l, theta_0, phi_0, theta, phi
        ) * np.pi
        return Gamma_val
    
    @staticmethod
    def _get_APF(a: float, b: float, rol: float, taul: float,
                 cteta: float, phi: float, cteta1: float, phi1: float) -> float:
        """
        计算双向反射/透射相关的面积投影因子 (APF)
        
        参数:
            a, b: 叶片角度分布参数
            rol: 反射率
            taul: 透射率
            cteta, phi: 入射方向的天顶角 & 方位角 (度)
            cteta1, phi1: 观测方向的天顶角 & 方位角 (度)
        
        返回:
            APF值
        """
        # 预处理角度（后向散射，观测方向与入射方向相反）
        cteta1 = cteta1 + 180
        
        ccteta = np.cos(np.deg2rad(cteta))
        ccteta1 = np.cos(np.deg2rad(cteta1))
        steta = np.sin(np.deg2rad(cteta))
        steta1 = np.sin(np.deg2rad(cteta1))
        phi_rad = np.deg2rad(phi)
        phi1_rad = np.deg2rad(phi1)
        
        kpi3 = 1.0 / (np.pi ** 3)
        
        # 积分网格
        n = 30
        m = 4 * n
        h_theta = 0.5 * np.pi / n
        h_fi = 2.0 * np.pi / m
        
        theta_i = 0.5 * h_theta
        fi_1 = 0.5 * h_fi
        
        integral = 0.0
        
        for i in range(n):
            fi_j = fi_1
            xx = 0.0
            c_i = np.cos(theta_i)
            s_i = np.sin(theta_i)
            
            for j in range(m):
                # 两个方向与叶片法线的夹角余弦
                yy = ccteta * c_i + steta * s_i * np.cos(phi_rad - fi_j)
                zz = ccteta1 * c_i + steta1 * s_i * np.cos(phi1_rad - fi_j)
                zz *= yy
                
                # 根据正负号选择反射或透射
                if zz <= 0:
                    xx += rol * abs(zz)
                else:
                    xx += taul * zz
                
                fi_j += h_fi
            
            xx *= h_fi
            yy = 1.0 + a * np.cos(b * theta_i)
            integral += yy * xx
            theta_i += h_theta
        
        integral *= h_theta
        return kpi3 * integral
    
    def __init__(self, canopy_params: CanopyParameters, sim_config: SimulationConfig):
        """
        初始化模拟器
        
        Parameters:
        -----------
        canopy_params : CanopyParameters
            树冠参数
        sim_config : SimulationConfig
            模拟配置
        """
        self.canopy = canopy_params
        self.config = sim_config
        
        # 计算G函数（使用完整方法）
        self.G = self._calculate_G()
        
        # 计算后向散射系数
        self.w_leaf = self._calculate_leaf_backscatter()
        self.w_ground = self.canopy.ground_reflectance
        
        # 计算空隙率
        # 如果手动设置了Fs0，使用手动值
        if self.config.manual_Fs0 is not None:
            self.Fs0 = self.config.manual_Fs0
        # 如果用户提供了真实树结构数据（基部高度、半径、长度），基于实际几何计算Fs0
        elif (self.config.tree_heights is not None and
            self.config.tree_radius is not None):
            # 基于真实树结构计算间隙率
            self.Fs0 = self._calculate_gap_probability_from_real_trees()
        else:
            # 基于树冠密度统计模型计算间隙率
            self.Fs0 = self._calculate_gap_probability()
        
        # 初始化路径长度分布
        self.Fs = None
        self.layers_v = None
        self.layers_vb = None
    
    def _calculate_gap_probability(self) -> float:
        """
        根据树冠参数计算空隙概率 (gap probability)
        
        支持两种分布模型:
        1. 'poisson' (泊松分布，允许树冠重叠):
           Fs0 = exp(-crown_coverage)
           适用于树木随机独立分布的森林
        
        2. 'non-overlapping' (不重叠分布):
           Fs0 = max(0, 1 - crown_coverage)
           适用于树冠相互排斥、不重叠的森林
        
        其中 crown_coverage = crown_density × π × crown_radius²
        
        Returns:
        --------
        Fs0 : float
            空隙概率 (0-1)
        """
        density = self.canopy.crown_density  # trees/m²
        
        # 获取树冠半径
        if isinstance(self.canopy.crown_radius, (list, np.ndarray)):
            radius = np.mean(self.canopy.crown_radius)
        else:
            radius = self.canopy.crown_radius
        
        # 计算树冠覆盖度
        crown_coverage = density * np.pi * radius ** 2
        
        # 根据分布类型计算空隙概率
        dist_type = self.canopy.distribution_type
        if dist_type == 'poisson':
            # 泊松分布：允许树冠重叠
            Fs0 = np.exp(-crown_coverage)
            model_name = "泊松分布（允许重叠）"
        elif dist_type == 'non-overlapping':
            # 不重叠分布：树冠相互排斥
            Fs0 = max(0.0, 1.0 - crown_coverage)
            model_name = "不重叠分布（树冠排斥）"
        else:
            raise ValueError(f"未知的分布类型: {dist_type}. "
                           f"请使用 'poisson' 或 'non-overlapping'")
        
        # 限制在合理范围内
        Fs0 = np.clip(Fs0, 0.01, 0.99)
        
        return Fs0
    
    def _calculate_gap_probability_from_real_trees(self) -> float:
        """
        基于真实树结构数据和footprint面积计算空隙概率
        
        当用户提供了tree_heights（基部高度）、tree_radius（树冠半径）和
        tree_length（树冠长度）时，基于实际树冠投影面积和footprint计算间隙率
        
        Returns:
        --------
        Fs0 : float
            空隙概率 (0-1)
        """
        tree_radii = self.config.tree_radius
        
        # 计算每棵树的树冠投影面积
        crown_areas = np.pi * np.array(tree_radii) ** 2
        total_crown_area = np.sum(crown_areas)
        
        # 计算激光雷达footprint面积
        footprint_area = np.pi * self.config.footprint_radius ** 2
        
        # 计算树冠覆盖率（相对于footprint面积）
        crown_coverage = total_crown_area / footprint_area if footprint_area > 0 else 0
        
        # 使用泊松分布计算间隙率
        dist_type = self.canopy.distribution_type
        if dist_type == 'poisson':
            Fs0 = np.exp(-crown_coverage)
            model_name = "泊松分布（基于真实树结构）"
        elif dist_type == 'non-overlapping':
            Fs0 = max(0.0, 1.0 - crown_coverage)
            model_name = "非重叠分布（基于真实树结构）"
        else:
            Fs0 = np.exp(-crown_coverage)
            model_name = "泊松分布（基于真实树结构）"
        
        # 限制在合理范围内
        Fs0 = np.clip(Fs0, 0.01, 0.99)
        
        return Fs0
    
    @staticmethod
    def _get_gFun(iorien: int, theta_L: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        计算叶片角度分布函数 g_L(theta_L)
        
        参数:
            iorien: 叶片法向分布类型 (1~6)
                1 - planophile     (a=1,  b=2)
                2 - erectophile    (a=-1, b=-2)
                3 - plagiophile    (a=-1, b=4)
                4 - extremophile   (a=1,  b=4)
                5 - uniform        (a=0,  b=任意)
                6 - spherical      g(θ) = sin(θ)
            theta_L: 叶倾角 (弧度)，可以是标量或 numpy 数组
        返回:
            g_L: 对应的叶片角度分布函数值
        """
        if iorien == 6:  # spherical
            g_L = np.sin(theta_L)
        else:
            # 根据类型设定 a, b 参数
            if iorien == 1:    # planophile
                a, b = 1, 2
            elif iorien == 2:  # erectophile
                a, b = -1, -2
            elif iorien == 3:  # plagiophile
                a, b = -1, 4
            elif iorien == 4:  # extremophile
                a, b = 1, 4
            elif iorien == 5:  # uniform
                a, b = 0, 0
            else:
                raise ValueError("iorien 必须是 1~6 之间的整数")
            
            g_L = (2 / np.pi) * (1 + a * np.cos(b * theta_L))
        
        return g_L
    
    @staticmethod
    def _get_G_function(iorien: int, theta: float, fi: float = 0.0) -> float:
        """
        计算投影函数 G(Omega) = (1/(2*pi)) ∫∫ g_L(θ_L) |cos(Ω, Ω_L)| dθ_L dφ_L
        
        参数:
            iorien: 叶片法向分布类型 (1~6)
            theta: 观测方向的天顶角 (度)
            fi: 观测方向的方位角 (度)
        返回:
            G_Fun: 投影函数 G(Omega)
        """
        # 转换为弧度
        theta_rad = np.deg2rad(theta)
        fi_rad = np.deg2rad(fi)
        
        # 积分网格设置
        n = 30
        m = 4 * n
        h_theta = 0.5 * np.pi / n
        h_fi = 2 * np.pi / m
        
        theta_i = 0.5 * h_theta
        fi_1 = 0.5 * h_fi
        
        G_Fun = 0.0
        
        for i in range(n):
            fi_j = fi_1
            c_i = np.cos(theta_i)
            s_i = np.sin(theta_i)
            xx = 0.0
            
            for j in range(m):
                # |cos(Ω · Ω_L)| = |cosθ cosθ_L + sinθ sinθ_L cos(φ - φ_L)|
                yy = np.cos(theta_rad) * c_i + np.sin(theta_rad) * s_i * np.cos(fi_rad - fi_j)
                xx += np.abs(yy)
                fi_j += h_fi
            
            xx *= h_fi
            g_val = LiDARWaveformSimulator._get_gFun(iorien, theta_i)
            G_Fun += g_val * xx
            theta_i += h_theta
        
        G_Fun *= h_theta
        G_Fun /= (2 * np.pi)
        
        return G_Fun
    
    def _calculate_G(self) -> float:
        """
        计算G函数（投影函数）
        对于天底观测 (theta=0)
        """
        leaf_class = self.canopy.leaf_angle_distribution
        
        # 天底角度
        theta = 0.0
        phi = 0.0
        
        # leaf_angle_distribution直接映射到iorien (1-6)
        # 1: planophile, 2: erectophile, 3: plagiophile
        # 4: extremophile, 5: uniform, 6: spherical
        iorien_map = {
            1: 1,  # planophile
            2: 2,  # erectophile
            3: 3,  # plagiophile
            4: 4,  # extremophile
            5: 5,  # uniform
            6: 6   # spherical
        }
        iorien = iorien_map.get(leaf_class, 6)  # 默认spherical
        
        # 使用完整的G函数计算
        G = self._get_G_function(iorien, theta, phi)
        
        return G
    
    def _calculate_leaf_backscatter(self) -> float:
        """
        计算叶片后向散射系数
        使用area_scatter_phase_function计算
        """
        rho_l = self.canopy.leaf_reflectance
        tau_l = 1 - rho_l - 0.05  # 透射率
        theta_0 = 0  # 从上往下入射
        phi_0 = 0
        theta = 0  # 垂直向上观测
        phi = 0
        
        # 直接使用用户输入的叶倾角分布类型 (1-6)
        # 1: planophile, 2: erectophile, 3: plagiophile
        # 4: extremophile, 5: uniform, 6: spherical
        leaf_class = int(self.canopy.leaf_angle_distribution)
        
        # 计算后向散射相函数
        Gamma = self._area_scatter_phase_function(
             rho_l, tau_l, theta_0, phi_0, theta, phi, leaf_class
        )
        
        # 后向散射系数 = 叶面积密度 × 层高 × 后向散射相函数
        u = self.canopy.leaf_area_density
        cu = self.config.layer_height
        w = u * cu * Gamma
    
        return w
    
    def _initialize_path_length_distribution(self, crown_length: float) -> np.ndarray:
        """
        初始化路径长度分布 (PLD)
        
        Parameters:
        -----------
        crown_length : float
            树冠长度
            
        Returns:
        --------
        Fs : np.ndarray
            路径长度分布，shape (2, n_layers+1)
            Fs[0, :] - 路径长度
            Fs[1, :] - 概率密度
        """
        cu = self.config.layer_height
        midLocat = self.config.midlocat
        
        layers_v = int(np.ceil(crown_length / cu))
        n_layers = int(np.ceil(layers_v * midLocat))
        step = cu / midLocat
        
        Fs = np.zeros((2, n_layers + 1))
        Fs[0, :] = np.arange(0, n_layers + 1) * step
        
        return Fs, layers_v
    
    def _calculate_shape_distribution(self, Fs: np.ndarray, 
                                     crown_length: float,
                                     crown_radius: float) -> np.ndarray:
        """
        根据树冠形状计算路径长度概率分布
        
        Parameters:
        -----------
        Fs : np.ndarray
            初始化的路径长度分布
        crown_length : float
            树冠长度
        crown_radius : float
            树冠半径
            
        Returns:
        --------
        Fs : np.ndarray
            更新后的路径长度分布
        """
        cu = self.config.layer_height
        midLocat = self.config.midlocat
        Fs0 = self.Fs0  # 使用自动计算的空隙率
        
        shape = self.canopy.crown_shape.lower()
        
        # RAMI模式：使用zfre分布（叶面积加权的冠层长度分布）
        if self.config.rami_mode and self.config.zfre_distribution is not None:
            Fs[1, 0] = Fs0  # Fs(2,1) = FsRAMI(i)
            zfre = self.config.zfre_distribution
            # MATLAB: Fs(2,2:end) = zfre * (1-FsRAMI(i))
            # zfre[0]对应最短路径(0附近)，zfre[-1]对应最长路径(meanLength)
            n_bins = min(len(zfre), Fs.shape[1] - 1)
            Fs[1, 1:n_bins+1] = zfre[:n_bins] * (1 - Fs0)
            print(f"[RAMI Mode] Applied zfre distribution: Fs0={Fs0:.4f}, zfre sum={(1-Fs0):.4f}")
            return Fs
        
        if shape == 'cylinder':
            # 圆柱形Fs分配
            if self.config.use_mixed_crown_allocation:
                # 混合树冠尺寸：使用通用算法处理不同尺寸的圆柱形树冠
                Fs = self._allocate_cylinder_distribution_general(Fs, crown_length)
            elif self.config.use_matlab_cylinder_allocation:
                # MATLAB特殊分配方式（仅用于匹配特定案例）
                # MATLAB waveform_lidar.m 320-321行:
                #   Fs(2,end) = (1-Fs0)/5;
                #   Fs(2,find((Fs(1,:) < 5.15) .* (Fs(1,:)>5))) = (1-Fs0)/5*4;
                #
                # 物理原理：
                # 通用圆柱形只有两个路径长度：0m（空隙Fs0）和最大长度（1-Fs0）
                # 但此特殊场景有两种尺寸的圆柱：
                #   - 小圆柱：半径r=1m，长度5m
                #   - 大圆柱：半径R=2m，长度8m
                # 路径长度分布：0m（空隙）、5m（小圆柱）、8m（大圆柱）
                #
                # 概率分配基于投影面积和密度：
                #   P_i = λ_i × π × r_i² × (1-Fs0) / Σ(λ_j × π × r_j²)
                # 当两种圆柱数量相同时（λ₁=λ₂），简化为面积比：
                #   P_small = r² / (r² + R²) = 1 / (1 + 4) = 1/5
                #   P_large = R² / (r² + R²) = 4 / (1 + 4) = 4/5
                #
                # 1/5的概率在最大路径长度8m（大圆柱应该是4/5，这里1/5可能是笔误或特定原因）
                Fs[1, -1] = (1 - Fs0) / 5
                # 4/5的概率在路径长度5m附近（小圆柱的最大路径长度）
                # mask用于定位5-5.15m区间的索引
                mask = (Fs[0, :] > 5.0) & (Fs[0, :] < 5.15)
                matching_indices = np.where(mask)[0]
                if len(matching_indices) > 0:
                    Fs[1, matching_indices] = (1 - Fs0) * 4 / 5 / len(matching_indices)
                else:
                    closest_idx = np.argmin(np.abs(Fs[0, :] - 5.1))
                    Fs[1, closest_idx] = (1 - Fs0) * 4 / 5
            else:
                # 通用圆柱形分配（默认）
                # 圆柱形树冠的路径长度固定，所有概率集中在最大路径长度
                Fs[1, -1] = 1 - Fs0
            
        elif shape == 'sphere':
            # 球形：根据椭球几何计算
            # MATLAB: Fs(2,2:end) = cu.*Fs(1,2:end)./(bratio*bratio*bottomRadius*bottomRadius*2).*(1-Fs0)./midLocat;
            b_ratio = 1.0  # 球形的长短轴比
            for i in range(1, Fs.shape[1]):
                Fs[1, i] = (cu * Fs[0, i] / (b_ratio * b_ratio * 
                           crown_radius * crown_radius * 2) * 
                           (1 - Fs0) / midLocat)
                           
        elif shape == 'cone':
            # 锥形：线性递减
            # MATLAB: Fs(2,2:end) = 2*cu.*(CrownLengthmax- Fs(1,2:end))./CrownLengthmax./CrownLengthmax.*(1-Fs0)/midLocat;
            for i in range(1, Fs.shape[1]):
                Fs[1, i] = (2 * cu * (crown_length - Fs[0, i]) / 
                           crown_length / crown_length * 
                           (1 - Fs0) / midLocat)
        
        # 最后设置零路径长度概率（空隙概率）
        # MATLAB: Fs(2,1) = Fs0;
        Fs[1, 0] = Fs0
        
        # 注意：MATLAB中没有归一化！直接返回
        return Fs
    
    def _allocate_cylinder_distribution_general(self, Fs: np.ndarray, crown_length: float) -> np.ndarray:
        """
        通用圆柱形Fs分配算法（自动识别树类型并计算概率）
        
        物理原理：
        1. 按(radius, length)分组，统计每种树的数量
        2. 计算每种树的密度 λ_i = count_i / (π × R_footprint²)
        3. 计算投影面积权重 w_i = λ_i × π × r_i² / Σ(λ_j × π × r_j²)
        4. 在对应路径长度处分配 Fs[1, idx] = (1 - Fs0) × w_i
        
        当所有树数量相同时，权重简化为面积比 w_i = r_i² / Σ(r_j²)
        
        Parameters:
        -----------
        Fs : np.ndarray
            初始化的路径长度分布
        crown_length : float
            标准波形的树冠长度（最大长度）
            
        Returns:
        --------
        Fs : np.ndarray
            分配后的路径长度分布
        """
        tree_radii = np.array(self.config.tree_radius)
        tree_lengths = np.array(self.config.tree_length)
        footprint_area = np.pi * self.config.footprint_radius ** 2
        Fs0 = self.Fs0
        
        # 按(radius, length)分组统计
        tree_types = {}  # key: (radius, length), value: {'count': n, 'indices': []}
        for i in range(len(tree_radii)):
            key = (tree_radii[i], tree_lengths[i])
            if key not in tree_types:
                tree_types[key] = {'count': 0, 'indices': []}
            tree_types[key]['count'] += 1
            tree_types[key]['indices'].append(i)
        
        # 计算每种树的密度和投影面积贡献
        total_weighted_area = 0.0
        for (radius, length), info in tree_types.items():
            density = info['count'] / footprint_area
            crown_area = np.pi * radius ** 2
            weighted_area = density * crown_area
            info['density'] = density
            info['weighted_area'] = weighted_area
            total_weighted_area += weighted_area
        
        # 计算每种树的概率权重
        for key in tree_types:
            tree_types[key]['weight'] = tree_types[key]['weighted_area'] / total_weighted_area
        
        # 初始化Fs[1, :]
        Fs[1, :] = 0
        Fs[1, 0] = Fs0
        
        print(f"\n通用圆柱形Fs分配（自动识别{len(tree_types)}种树类型）:")
        
        # 按长度从大到小排序
        sorted_types = sorted(tree_types.items(), key=lambda x: x[0][1], reverse=True)
        
        # 为每种树分配概率
        for (radius, length), info in sorted_types:
            count = info['count']
            density = info['density']
            weight = info['weight']
            
            print(f"  树类型: 半径={radius:.1f}m, 长度={length:.1f}m")
            print(f"    数量={count}, 密度={density:.6f} trees/m², 权重={weight:.4f}")
            
            # 找到对应长度的索引位置
            # 使用小范围匹配：length ± 0.15m
            tolerance = self.config.layer_height  # cu = 0.15m
            mask = np.abs(Fs[0, :] - length) <= tolerance
            matching_indices = np.where(mask)[0]
            
            if len(matching_indices) > 0:
                # 如果找到多个匹配索引，均分概率
                Fs[1, matching_indices] = (1 - Fs0) * weight / len(matching_indices)
                print(f"    分配到 Fs[1, {matching_indices}] (路径长度≈{length:.2f}m): "
                      f"{(1-Fs0)*weight:.4f}")
            else:
                # 如果没找到，分配到最接近的索引
                closest_idx = np.argmin(np.abs(Fs[0, :] - length))
                Fs[1, closest_idx] = (1 - Fs0) * weight
                print(f"    分配到 Fs[1, {closest_idx}] (路径长度={Fs[0, closest_idx]:.2f}m, "
                      f"最接近{length:.2f}m): {(1-Fs0)*weight:.4f}")
        
        return Fs
    
    def _calculate_height_distribution(self, tree_heights: np.ndarray,
                                       num_trees: np.ndarray) -> np.ndarray:
        """
        计算树高分布的卷积核
        
        Parameters:
        -----------
        tree_heights : np.ndarray
            树高数组
        num_trees : np.ndarray
            每个树高的树木数量
            
        Returns:
        --------
        freq_conv : np.ndarray
            高度分布频率
        """
        cu = self.config.layer_height
        
        z_conv = np.arange(np.ceil(np.max(tree_heights)), 
                          np.floor(np.min(tree_heights)) - cu, -cu)
        freq_conv = np.zeros(len(z_conv))
        
        for ii in range(len(z_conv)):
            if ii == len(z_conv) - 1:
                freq_conv[ii] = np.sum(num_trees[tree_heights <= z_conv[ii]])
            else:
                mask = (tree_heights <= z_conv[ii]) & (tree_heights > z_conv[ii + 1])
                freq_conv[ii] = np.sum(num_trees[mask])
        
        # 归一化
        if np.sum(freq_conv) > 0:
            freq_conv = freq_conv / np.sum(freq_conv)
            
        return freq_conv
    
    def _calculate_waveform(self, Fs: np.ndarray, 
                           layers_v: int,
                           layers_vb: int,
                           freq_conv: np.ndarray) -> np.ndarray:
        """
        计算LiDAR波形
        
        Parameters:
        -----------
        Fs : np.ndarray
            路径长度分布
        layers_v : int
            树冠层数
        layers_vb : int
            总层数（包括地面）
        freq_conv : np.ndarray
            高度分布卷积核
            
        Returns:
        --------
        R_l : np.ndarray
            模拟的LiDAR波形
        """
        cu = self.config.layer_height
        midLocat = self.config.midlocat
        u = self.canopy.leaf_area_density
        
        # 计算LAI
        LAI = np.dot(Fs[0, :], Fs[1, :]) * u
        print(f"Calculated LAI: {LAI:.3f}")
        
        # 初始化波形 (MATLAB: R_l = zeros(1,layers_vb))
        R_l = np.zeros(layers_vb)
        
        # z数组 (MATLAB: z = 0:1:layers_vb)
        z = np.arange(0, layers_vb + 1)
        
        # 计算衰减系数
        theta = self.G * u
        j0 = 1.0  # 入射强度
        
        # 预计算指数衰减
        exp_decay = np.exp(-theta * cu)
        
        # 计算树冠上部的贡献（对称部分）
        # MATLAB: for i = 1:ceil(layers_v*midLocat)
        mid_layer = int(np.ceil(layers_v * midLocat))
        
        for i in range(1, mid_layer + 1):
            # MATLAB: kesi = Fs(2, end:-1:end-i+1) .* exp_decay.^(z(i) - z(1:i))
            # MATLAB的Fs(2, end:-1:end-i+1)是倒序取最后i个元素
            # MATLAB的z(i)是z数组第i个元素（MATLAB索引从1开始）
            #   z = [0, 1, 2, ...], z(1)=0, z(2)=1, z(i)=i-1
            # MATLAB的z(1:i)是z数组前i个元素 = [0, 1, 2, ..., i-1]
            # 所以z(i) - z(1:i) = (i-1) - [0,1,2,...,i-1] = [i-1, i-2, i-3, ..., 0]
            z_diff = (i-1) - np.arange(0, i)  # [i-1, i-2, i-3, ..., 0]
            kesi = Fs[1, -1:-(i+1):-1] * (exp_decay ** z_diff)  # 倒序取最后i个
            R_l[i-1] = self.w_leaf * j0 * np.sum(kesi)
        
        # 计算树冠下部的贡献
        # MATLAB: for i = ceil(layers_v*midLocat) + 1 : 1 : layers_v
        for i in range(mid_layer + 1, layers_v + 1):
            kesi = np.zeros(layers_v)
            # MATLAB: for m = 1 : 1 : layers_v - i + 1
            for m in range(1, layers_v - i + 2):
                kesi[m-1] = Fs[1, -(m)] * np.exp(-theta * (z[i] - z[m]) * cu)
            R_l[i-1] = self.w_leaf * j0 * np.sum(kesi)
        
        # 卷积前先添加地面贡献（MATLAB顺序）
        # MATLAB: R_l(end) = R_l(end) + wb.*j0.* sum(Fs(2,:).*exp(-theta.*Fs(1,:)))
        ground_contribution = (self.w_ground * j0 * 
                              np.sum(Fs[1, :] * np.exp(-theta * Fs[0, :])))
        R_l[-1] += ground_contribution
        
        # 与树高分布卷积
        # MATLAB: R_l_C = conv(freConv, R_l(1:layers_v))
        # MATLAB: R_l(1:length(R_l_C)) = R_l_C
        if len(freq_conv) > 1:
            R_l_crown = R_l[:layers_v]
            R_l_conv = np.convolve(freq_conv, R_l_crown, mode='full')
            
            # MATLAB的逻辑：尝试将R_l_conv赋值到R_l，如果超出则扩展R_l
            # 但Python中R_l已经固定长度，需要特殊处理
            if len(R_l_conv) > len(R_l):
                # 卷积结果超过R_l长度，需要扩展R_l
                R_l_new = np.zeros(len(R_l_conv))
                R_l_new[:len(R_l)] = R_l
                R_l = R_l_new
            
            # 用卷积结果覆盖R_l的前面部分
            R_l[:len(R_l_conv)] = R_l_conv
        
        # 如果R_l长度超过layers_vb，截断并重新添加地面回波
        # MATLAB: if length(R_l) > layers_vb
        #           R_l = R_l(1:layers_vb);
        #           R_l(end) = R_l(end) + wb.*j0.* sum(Fs(2,:).*exp(-theta.*Fs(1,:)));
        #         end
        if len(R_l) > layers_vb:
            R_l = R_l[:layers_vb]
            R_l[-1] += ground_contribution
        
        return R_l
    
    def simulate(self, plot: bool = True) -> Tuple[np.ndarray, np.ndarray]:

        # 检查是否提供了manual参数（通用Fs计算方式）
        if self.config.manual_Fs is not None:
            print("\n使用用户提供的Fs分布（通用树高分布方法）")
            self.Fs = self.config.manual_Fs
            self.layers_v = self.config.manual_layers_v
            self.layers_vb = self.config.manual_layers_vb
            freq_conv = self.config.manual_freq_conv if self.config.manual_freq_conv is not None else np.array([1.0])
            
            # 从Fs推断crown_length和crown_top
            crown_length = self.Fs[0, -1]  # 最大路径长度
            crown_base, crown_top = self.canopy.crown_height_range
            crown_radius = np.mean(self.canopy.crown_radius) if isinstance(self.canopy.crown_radius, (list, np.ndarray)) else self.canopy.crown_radius
            
            print(f"  layers_v = {self.layers_v}, layers_vb = {self.layers_vb}")
            print(f"  Fs shape: {self.Fs.shape}, freq_conv length: {len(freq_conv)}")
            
            # 直接计算波形
            waveform = self._calculate_waveform(self.Fs, self.layers_v, self.layers_vb, freq_conv)
            
            # 处理脉冲卷积
            if self.config.use_pulse_shape:
                if self.config.pulse_waveform is not None:
                    waveform = np.convolve(waveform, self.config.pulse_waveform, mode='full')
                    pulse_center = len(self.config.pulse_waveform) // 2
                    height_offset = pulse_center * self.config.layer_height
                    height = crown_top + height_offset - np.arange(len(waveform)) * self.config.layer_height
                else:
                    pulse_width_ns = 3.0
                    c = 0.3
                    pulse_width_m = pulse_width_ns * c / 2
                    sigma = pulse_width_m / 2.355
                    n_sigma = 4
                    n_samples = int(2 * n_sigma * sigma / self.config.layer_height)
                    if n_samples % 2 == 0:
                        n_samples += 1
                    pulse_center = n_samples // 2
                    t = (np.arange(n_samples) - pulse_center) * self.config.layer_height
                    gaussian_pulse = np.exp(-0.5 * (t / sigma) ** 2)
                    gaussian_pulse = gaussian_pulse / np.sum(gaussian_pulse)
                    waveform = np.convolve(waveform, gaussian_pulse, mode='full')
                    height_offset = pulse_center * self.config.layer_height
                    height = crown_top + height_offset - np.arange(len(waveform)) * self.config.layer_height
            else:
                height = crown_top - np.arange(len(waveform)) * self.config.layer_height
            
            if plot:
                self.plot_waveform(height, waveform)
            
            return height, waveform
        
        # 原有的模拟流程
        # 获取树冠参数
        # 如果提供了真实树数据，使用最大树冠长度计算标准波形
        # 否则使用用户设置的crown_height_range
        if (self.config.tree_length is not None):
            # 使用最大树冠长度作为标准波形的树冠长度
            crown_length = np.max(self.config.tree_length)
            # crown_base可以是任意参考高度，这里使用用户设置或默认值
            crown_base, crown_top = self.canopy.crown_height_range
            crown_base = crown_top - crown_length  # 确保crown_length一致
            print(f"\n使用最大树冠长度计算标准波形: {crown_length:.2f} m")
            print(f"  参考高度范围: {crown_base:.2f} - {crown_top:.2f} m")
        else:
            crown_base, crown_top = self.canopy.crown_height_range
            crown_length = crown_top - crown_base
        
        if isinstance(self.canopy.crown_radius, (list, np.ndarray)):
            crown_radius = np.mean(self.canopy.crown_radius)
        else:
            crown_radius = self.canopy.crown_radius
        
        # RAMI模式：使用meanLength而不是crown_length来初始化Fs
        # MATLAB: Fs = FsfunIni(layers_v, cu, midLocat)，其中layers_v = ceil(meanLength/cu)
        if self.config.rami_mode and self.config.rami_layers_v is not None:
            # 从rami_layers_v反推meanLength
            rami_crown_length = self.config.rami_layers_v * self.config.layer_height
            print(f"[RAMI Mode] Using meanLength-based crown_length: {rami_crown_length:.3f} m for Fs initialization")
            self.Fs, self.layers_v = self._initialize_path_length_distribution(rami_crown_length)
        else:
            # 初始化路径长度分布
            self.Fs, self.layers_v = self._initialize_path_length_distribution(crown_length)
        
        # 计算形状相关的分布
        self.Fs = self._calculate_shape_distribution(self.Fs, crown_length, crown_radius)
        
        # 计算总层数
        # RAMI模式使用指定的layers_vb和layers_v（从max(treeHeight)和meanLength计算）
        if self.config.rami_mode and self.config.rami_layers_vb is not None:
            self.layers_vb = self.config.rami_layers_vb
            print(f"[RAMI Mode] Using specified layers_vb = {self.layers_vb}")
        else:
            self.layers_vb = int(np.ceil((crown_length + crown_base) / self.config.layer_height))
        
        if self.config.rami_mode and self.config.rami_layers_v is not None:
            self.layers_v = self.config.rami_layers_v
            print(f"[RAMI Mode] Using specified layers_v = {self.layers_v}")
        
        # 处理树高分布
        if self.config.deltaz is not None:
            # 树冠在deltaz范围内随机均匀分布
            cu = self.config.layer_height
            n_bins = int(np.ceil(self.config.deltaz / cu)) + 1
            freq_conv = np.ones(n_bins) / n_bins  # 均匀分布，归一化
            print(f"Using uniform distribution for tree heights within deltaz={self.config.deltaz:.2f}m")
        elif self.config.tree_heights is not None:
            # 使用tree_weights（如果提供），否则默认为1
            if self.config.tree_weights is not None:
                num_trees = self.config.tree_weights
            else:
                num_trees = np.ones(len(self.config.tree_heights))
            # MATLAB使用树基部高度（treeHeight）计算freConv，不是树顶高度！
            # MATLAB: freConv = calFreconv(treeHeight, numOftrees, cu)
            # RAMI场景: freConv = calFreconv(treeHeight, numOftreesWeibyla, cu)
            freq_conv = self._calculate_height_distribution(
                self.config.tree_heights, num_trees)
        else:
            freq_conv = np.array([1.0])
        
        # 计算波形
        waveform = self._calculate_waveform(self.Fs, self.layers_v, 
                                            self.layers_vb, freq_conv)
        
        # 生成高度数组
        # 标准波形从crown_top向下，卷积后会延伸到更低位置
        # 波形数组的第0个元素对应crown_top，最后一个元素对应最低点
        if self.config.use_pulse_shape:
            if self.config.pulse_waveform is not None:
                # 使用完整卷积，保留脉冲形状的完整信息
                pulse_length = len(self.config.pulse_waveform)
                pulse_center = pulse_length // 2
                
                # 卷积前记录原始波形长度
                original_length = len(waveform)
                waveform = np.convolve(waveform, self.config.pulse_waveform, mode='full')
                
                # 卷积后的高度数组需要考虑脉冲中心偏移
                # mode='full'输出长度 = len(waveform) + len(pulse) - 1
                # 脉冲中心位置对应原始波形的第0个元素
                # 因此高度数组起点应该向上偏移 pulse_center 个层
                height_offset = pulse_center * self.config.layer_height
                height = crown_top + height_offset - np.arange(len(waveform)) * self.config.layer_height
            else:
                # 使用高斯脉冲（确保峰值在中心）
                # ============================================================
                # GEDI脉冲配置参考 (Hancock et al. 2019b):
                # - FWHM = 15.6 ns (Full Width at Half Maximum)
                # - FWHM = 2.35 × σ  (σ: standard deviation)
                # - 有效区间: [μ - 3σ, μ + 3σ]  (μ: mean value)
                # 
                # 当前默认配置 (用户可修改):
                # - pulse_width_ns = 3.0 ns (对应FWHM)
                # - n_sigma = 4 (使用±4σ范围，比GEDI的±3σ更宽)
                # 
                # 如需匹配GEDI，请设置: pulse_width_ns = 15.6, n_sigma = 3
                # ============================================================
                pulse_width_ns = 15.6  # 脉冲宽度 (ns)，即FWHM
                c = 0.3  # 光速 (m/ns)
                pulse_width_m = pulse_width_ns * c / 2  # 转换为距离（往返）
                sigma = pulse_width_m / 2.355  # 标准差 σ (m): FWHM = 2.355 × σ
                
                # 脉冲长度
                n_sigma = 3  # 覆盖±4σ范围（GEDI使用±3σ）
                n_samples = int(2 * n_sigma * sigma / self.config.layer_height)
                if n_samples % 2 == 0:  # 确保是奇数，使峰值在中心
                    n_samples += 1
                
                pulse_center = n_samples // 2
                
                t = (np.arange(n_samples) - pulse_center) * self.config.layer_height
                gaussian_pulse = np.exp(-0.5 * (t / sigma) ** 2)
                gaussian_pulse = gaussian_pulse / np.sum(gaussian_pulse)
                
                # 卷积前记录原始波形长度
                original_length = len(waveform)
                waveform = np.convolve(waveform, gaussian_pulse, mode='full')
                
                # 调整高度数组，考虑脉冲中心偏移
                height_offset = pulse_center * self.config.layer_height
                height = crown_top + height_offset - np.arange(len(waveform)) * self.config.layer_height
        else:
            # 不使用脉冲卷积时，从标准波形的crown_top开始向下生成高度
            height = crown_top - np.arange(len(waveform)) * self.config.layer_height
        
        # 绘图
        if plot:
            self.plot_waveform(height, waveform)
        
        return height, waveform
    
    def plot_waveform(self, height: np.ndarray, waveform: np.ndarray):
        """
        绘制波形结果（使用双子图：完整波形 + 放大植被回波）
        
        Parameters:
        -----------
        height : np.ndarray
            高度数组
        waveform : np.ndarray
            波形数据
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        # 归一化波形
        waveform_norm = waveform / np.max(waveform) if np.max(waveform) > 0 else waveform
        
        # 左图：完整波形
        ax1.plot(waveform_norm, height, 'b-', linewidth=2, label='Complete Waveform')
        ax1.set_xlabel('Normalized Return Signal', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Height (m)', fontsize=12, fontweight='bold')
        ax1.set_title('Complete LiDAR Waveform', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 右图：放大植被回波（截断地面强回波）
        # 找到地面回波位置（通常是最低点且强度最大）
        ground_idx = np.argmax(waveform)
        ground_value = waveform_norm[ground_idx]
        
        # 计算植被回波的合理显示范围（排除地面峰值）
        # 使用地面回波的5-10%作为截断阈值
        vegetation_threshold = ground_value * 0.05
        vegetation_mask = waveform_norm < vegetation_threshold
        
        if np.any(vegetation_mask):
            veg_waveform = waveform_norm.copy()
            veg_waveform[~vegetation_mask] = vegetation_threshold  # 截断地面回波
            
            ax2.plot(veg_waveform, height, 'g-', linewidth=2, label='Vegetation Return (Zoomed)')
            ax2.set_xlabel('Normalized Return Signal (Truncated)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Height (m)', fontsize=12, fontweight='bold')
            ax2.set_title('Vegetation Return (Ground Truncated)', fontsize=14, fontweight='bold')
            ax2.set_xlim(0, vegetation_threshold * 1.2)  # 限制x轴范围
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 添加截断说明
            ax2.text(0.98, 0.02, f'Ground return truncated at {vegetation_threshold:.4f}',
                    transform=ax2.transAxes, fontsize=9,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        else:
            # 如果没有明显的地面回波，直接绘制
            ax2.plot(waveform_norm, height, 'g-', linewidth=2, label='Waveform')
            ax2.set_xlabel('Normalized Return Signal', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Height (m)', fontsize=12, fontweight='bold')
            ax2.set_title('Waveform (No Truncation)', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # 添加参数信息到左图
        info_text = (f"Crown: {self.canopy.crown_shape}\n"
                    f"LAD: {self.canopy.leaf_area_density:.2f}\n"
                    f"G: {self.G:.2f}\n"
                    f"ρ: {self.canopy.leaf_reflectance:.2f}\n"
                    f"τ: {self.canopy.leaf_transmittance:.2f}\n"
                    f"Fs0: {self.Fs0:.3f}")
        ax1.text(0.98, 0.98, info_text, transform=ax1.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def save_waveform(self, filename: str, height: np.ndarray, waveform: np.ndarray):
        """
        保存波形到文件
        
        Parameters:
        -----------
        filename : str
            输出文件名
        height : np.ndarray
            高度数组
        waveform : np.ndarray
            波形数据
        """
        data = np.column_stack([height, waveform])
        np.savetxt(filename, data, delimiter='\t', 
                  header='Height(m)\tReturn_Signal',
                  comments='')
        print(f"Waveform saved to {filename}")


class CHMWaveformSimulator:
    """
    基于冠层高度模型(CHM)的LiDAR波形模拟器
    
    从CHM栅格数据提取高度分布，转换为路径长度分布(PLD)，
    模拟LiDAR回波波形。
    
    Parameters:
    -----------
    chm_file : str
        CHM栅格文件路径(.tif格式)
    canopy_params : CanopyParameters
        树冠参数（用于提供叶片光学特性等）
    layer_height : float, optional
        层高 (m)，默认0.15m
    height_threshold : float, optional
        高度阈值 (m)，CHM < threshold 判定为间隙，默认1.0m
    branch_height_ratio : float, optional
        枝下高占平均树高的比例，用于计算crown length，默认1/3
    manual_Fs0 : float, optional
        手动设置空隙率 (0-1)，如果为None则根据CHM自动计算
    manual_tree_top_height : float, optional
        手动设置当前树顶高度 (m)，如果为None则使用max(CHM)
        用于CHM是早期数据但树已长高的情况
        注意：Fs分布不变，layers_v不变，但layers_vb会增加
    use_pulse_shape : bool, optional
        是否使用脉冲卷积，默认True
    pulse_waveform : np.ndarray, optional
        自定义脉冲波形，如果为None则使用高斯脉冲
    """
    
    def __init__(self, 
                 chm_file: str,
                 canopy_params: CanopyParameters,
                 layer_height: float = 0.15,
                 height_threshold: float = 1.0,
                 branch_height_ratio: float = 1/3,
                 manual_Fs0: Optional[float] = None,
                 manual_tree_top_height: Optional[float] = None,
                 use_pulse_shape: bool = True,
                 pulse_waveform: Optional[np.ndarray] = None):
        
        self.chm_file = chm_file
        self.canopy = canopy_params
        self.cu = layer_height
        self.height_threshold = height_threshold
        self.branch_height_ratio = branch_height_ratio
        self.manual_Fs0 = manual_Fs0
        self.manual_tree_top_height = manual_tree_top_height
        self.use_pulse_shape = use_pulse_shape
        self.pulse_waveform = pulse_waveform
        
        # 加载CHM数据
        self.chm_data = self._load_chm()
        
        # 计算参数
        self._calculate_parameters()
        
        # 创建模拟器实例（用于计算G函数和后向散射系数）
        config = SimulationConfig(
            layer_height=layer_height,
            use_pulse_shape=use_pulse_shape,
            pulse_waveform=pulse_waveform
        )
        self.simulator = LiDARWaveformSimulator(canopy_params, config)
        self.G = self.simulator.G
        self.w_leaf = self.simulator.w_leaf
        self.w_ground = self.simulator.w_ground
    
    def _load_chm(self) -> np.ndarray:
        """
        加载CHM数据
        
        Returns:
        --------
        chm_data : np.ndarray
            CHM高度数组 (m)
        """
        try:
            from osgeo import gdal
            gdal.UseExceptions()
            
            dataset = gdal.Open(self.chm_file)
            if dataset is None:
                raise IOError(f"无法打开CHM文件: {self.chm_file}")
            
            band = dataset.GetRasterBand(1)
            chm_data = band.ReadAsArray()
            
            # 处理NoData值
            nodata = band.GetNoDataValue()
            if nodata is not None:
                chm_data = np.ma.masked_equal(chm_data, nodata)
                chm_data = chm_data.compressed()  # 移除NoData
            else:
                chm_data = chm_data.flatten()
            
            # 移除负值和无效值
            chm_data = chm_data[chm_data >= 0]
            
            print(f"\n成功加载CHM: {self.chm_file}")
            print(f"  像元数量: {len(chm_data)}")
            print(f"  高度范围: {np.min(chm_data):.2f} - {np.max(chm_data):.2f} m")
            print(f"  平均高度: {np.mean(chm_data):.2f} m")
            
            return chm_data
            
        except ImportError:
            print("警告: GDAL未安装，尝试使用rasterio...")
            try:
                import rasterio
                with rasterio.open(self.chm_file) as src:
                    chm_data = src.read(1)
                    
                    # 处理NoData
                    if src.nodata is not None:
                        chm_data = np.ma.masked_equal(chm_data, src.nodata)
                        chm_data = chm_data.compressed()
                    else:
                        chm_data = chm_data.flatten()
                    
                    chm_data = chm_data[chm_data >= 0]
                    
                    print(f"\n成功加载CHM: {self.chm_file}")
                    print(f"  像元数量: {len(chm_data)}")
                    print(f"  高度范围: {np.min(chm_data):.2f} - {np.max(chm_data):.2f} m")
                    print(f"  平均高度: {np.mean(chm_data):.2f} m")
                    
                    return chm_data
            except ImportError:
                raise ImportError("需要安装 GDAL 或 rasterio 来读取栅格数据")
    
    def _calculate_parameters(self):
        """
        根据CHM数据计算模拟参数
        
        计算:
        - layers_vb: 基于最大CHM高度（或手动指定的树顶高度）
        - layers_v: 基于平均冠层长度（CHM - 枝下高）
        - Fs: 路径长度分布
        - Fs0: 空隙率
        """
        # 统计参数
        max_chm_height = np.max(self.chm_data)
        mean_height = np.mean(self.chm_data)
        
        # 确定当前树顶高度（手动指定或使用CHM最大值）
        if self.manual_tree_top_height is not None:
            current_tree_top = self.manual_tree_top_height
            print(f"\n使用手动指定的树顶高度: {current_tree_top:.2f} m")
            print(f"  CHM最大高度（历史数据）: {max_chm_height:.2f} m")
            print(f"  树木生长高度差: {current_tree_top - max_chm_height:.2f} m")
        else:
            current_tree_top = max_chm_height
        
        # 保存当前树顶高度供后续使用
        self.current_tree_top_height = current_tree_top
        
        # 计算枝下高（假设为平均树高的1/3）
        branch_height = mean_height * self.branch_height_ratio
        
        # 计算平均冠层长度（基于CHM的结构，不变）
        mean_crown_length = mean_height - branch_height
        
        # 计算layers
        # layers_vb基于当前树顶高度（如果树长高了，这个值会增加）
        self.layers_vb = int(np.ceil(current_tree_top / self.cu))
        # layers_v基于冠层长度（不变，因为冠层结构没变）
        self.layers_v = int(np.ceil(mean_crown_length / self.cu))
        
        print(f"\n计算的参数:")
        print(f"  当前树顶高度: {current_tree_top:.2f} m")
        print(f"  CHM最大高度: {max_chm_height:.2f} m")
        print(f"  平均CHM高度: {mean_height:.2f} m")
        print(f"  假设枝下高: {branch_height:.2f} m ({self.branch_height_ratio:.1%})")
        print(f"  平均冠层长度: {mean_crown_length:.2f} m (基于CHM结构)")
        print(f"  layers_vb: {self.layers_vb} (基于当前树顶)")
        print(f"  layers_v: {self.layers_v} (基于冠层结构)")
        
        # 计算路径长度分布 Fs（基于CHM结构，不变）
        self._calculate_path_length_distribution(mean_crown_length)
    
    def _calculate_path_length_distribution(self, crown_length: float):
        """
        从CHM数据计算路径长度分布(PLD)
        
        Parameters:
        -----------
        crown_length : float
            平均冠层长度 (m)
        """
        # 计算枝下高
        mean_height = np.mean(self.chm_data)
        branch_height = mean_height * self.branch_height_ratio
        
        # 计算每个像元的冠层长度（CHM - 枝下高）
        # 对于CHM < height_threshold，路径长度为0（间隙）
        crown_lengths = np.maximum(0, self.chm_data - branch_height)
        crown_lengths[self.chm_data < self.height_threshold] = 0
        
        # 统计空隙率
        n_gap = np.sum(self.chm_data < self.height_threshold)
        n_total = len(self.chm_data)
        calculated_Fs0 = n_gap / n_total if n_total > 0 else 0.5
        
        # 如果用户手动设置了Fs0，使用手动值
        if self.manual_Fs0 is not None:
            self.Fs0 = self.manual_Fs0
            print(f"\n使用手动设置的空隙率 Fs0: {self.Fs0:.4f}")
        else:
            self.Fs0 = calculated_Fs0
            print(f"\n从CHM计算的空隙率 Fs0: {self.Fs0:.4f}")
        
        print(f"  间隙像元数: {n_gap} / {n_total} ({calculated_Fs0:.2%})")
        print(f"  高度阈值: {self.height_threshold:.2f} m")
        
        # 创建路径长度bins
        # 使用平均冠层长度作为最大路径长度
        max_path_length_absolute = np.max(crown_lengths)
        mean_crown_length_actual = np.mean(crown_lengths[crown_lengths > 0]) if np.any(crown_lengths > 0) else crown_length
        
        max_path_length = max_path_length_absolute
        
        # 重新计算layers_v：基于实际使用的最大路径长度
        # 当mean_crown_length_actual被人为修改后，layers_v也需要相应调整
        layers_v_original = self.layers_v
        self.layers_v = int(np.ceil(max_path_length / self.cu))
        
        print(f"\n路径长度统计（基于平均冠层长度）:")
        print(f"  绝对最大路径长度: {max_path_length_absolute:.2f} m")
        print(f"  平均冠层长度: {mean_crown_length_actual:.2f} m")
        print(f"  使用平均冠层长度作为最大路径: {max_path_length:.2f} m")
        
        if self.layers_v != layers_v_original:
            print(f"  layers_v更新: {layers_v_original} -> {self.layers_v} (基于实际Fs最大路径)")
        
        # n_bins至少要等于layers_v+1，以确保波形计算时有足够的bins
        n_bins_min = self.layers_v + 1
        n_bins_from_path = int(np.ceil(max_path_length / self.cu)) + 1
        n_bins = max(n_bins_min, n_bins_from_path)
        
        if n_bins > n_bins_from_path:
            print(f"  注意: n_bins调整为 {n_bins} (>= layers_v+1={n_bins_min})，以避免波形计算时索引越界")
        
        # 初始化Fs数组: Fs[0, :] = 路径长度, Fs[1, :] = 概率密度
        self.Fs = np.zeros((2, n_bins))
        self.Fs[0, :] = np.arange(n_bins) * self.cu
        
        # 统计路径长度频率分布
        # bin 0: 路径长度为0（间隙）
        self.Fs[1, 0] = self.Fs0
        
        # bin 1-end: 根据CHM统计路径长度分布
        for i in range(1, n_bins):
            bin_min = self.Fs[0, i] - self.cu / 2
            bin_max = self.Fs[0, i] + self.cu / 2
            
            # 统计落在此bin内的像元数
            # 最后一个bin包含所有 >= bin_min 的值（避免遗漏超出百分位数的数据）
            if i == n_bins - 1:
                mask = crown_lengths > bin_min
            else:
                mask = (crown_lengths > bin_min) & (crown_lengths <= bin_max)
            count = np.sum(mask)
            
            # 转换为概率密度（归一化到1-Fs0）
            self.Fs[1, i] = count / n_total * (1 - self.Fs0) / (1 - self.Fs0) if n_total > n_gap else 0
        
        # 重新归一化Fs[1, 1:]（确保sum(Fs[1, 1:]) = 1 - Fs0）
        non_gap_sum = np.sum(self.Fs[1, 1:])
        if non_gap_sum > 0:
            self.Fs[1, 1:] = self.Fs[1, 1:] / non_gap_sum * (1 - self.Fs0)
        
        print(f"\n路径长度分布 (Fs):")
        print(f"  Bins数量: {n_bins}")
        print(f"  最大路径长度: {max_path_length:.2f} m")
        print(f"  Fs[1, :] 总和: {np.sum(self.Fs[1, :]):.6f} (应≈1.0)")
        print(f"  Fs[1, 0] (间隙): {self.Fs[1, 0]:.4f}")
        print(f"  Fs[1, 1:] 总和 (植被): {np.sum(self.Fs[1, 1:]):.4f}")
    
    def simulate(self, plot: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        模拟LiDAR波形
        
        Parameters:
        -----------
        plot : bool, optional
            是否绘制波形，默认True
        
        Returns:
        --------
        height : np.ndarray
            高度数组 (m)
        waveform : np.ndarray
            模拟的波形数据
        """
        print("\n" + "="*70)
        print("基于CHM的LiDAR波形模拟")
        print("="*70)
        
        # 计算LAI
        u = self.canopy.leaf_area_density
        LAI = np.dot(self.Fs[0, :], self.Fs[1, :]) * u
        print(f"\n计算的LAI: {LAI:.3f}")
        
        # 调用波形计算函数（不使用树高分布卷积）
        waveform = self._calculate_waveform(np.array([1.0]))
        
        # 生成高度数组（使用当前树顶高度）
        max_chm = self.current_tree_top_height
        
        if self.use_pulse_shape:
            if self.pulse_waveform is not None:
                # 使用自定义脉冲
                pulse_center = len(self.pulse_waveform) // 2
                waveform = np.convolve(waveform, self.pulse_waveform, mode='full')
                height_offset = pulse_center * self.cu
                height = max_chm + height_offset - np.arange(len(waveform)) * self.cu
            else:
                # 使用高斯脉冲（GEDI配置）
                pulse_width_ns = 15.6
                c = 0.3
                pulse_width_m = pulse_width_ns * c / 2
                sigma = pulse_width_m / 2.355
                n_sigma = 3
                n_samples = int(2 * n_sigma * sigma / self.cu)
                if n_samples % 2 == 0:
                    n_samples += 1
                
                pulse_center = n_samples // 2
                t = (np.arange(n_samples) - pulse_center) * self.cu
                gaussian_pulse = np.exp(-0.5 * (t / sigma) ** 2)
                gaussian_pulse = gaussian_pulse / np.sum(gaussian_pulse)
                
                waveform = np.convolve(waveform, gaussian_pulse, mode='full')
                height_offset = pulse_center * self.cu
                height = max_chm + height_offset - np.arange(len(waveform)) * self.cu
        else:
            height = max_chm - np.arange(len(waveform)) * self.cu
        
        print(f"\n波形长度: {len(waveform)}")
        print(f"高度范围: {np.min(height):.2f} - {np.max(height):.2f} m")
        print("="*70 + "\n")
        
        if plot:
            self._plot_waveform(height, waveform)
        
        return height, waveform
    
    def _calculate_waveform(self, freq_conv: np.ndarray) -> np.ndarray:
        """
        计算LiDAR波形（复用LiDARWaveformSimulator的逻辑）
        
        Parameters:
        -----------
        freq_conv : np.ndarray
            高度分布卷积核
            
        Returns:
        --------
        R_l : np.ndarray
            模拟的LiDAR波形
        """
        u = self.canopy.leaf_area_density
        theta = self.G * u
        j0 = 1.0
        
        # 初始化波形
        R_l = np.zeros(self.layers_vb)
        z = np.arange(0, self.layers_vb + 1)
        
        # 预计算指数衰减
        exp_decay = np.exp(-theta * self.cu)
        
        # 计算树冠波形（与LiDARWaveformSimulator相同的逻辑）
        midLocat = 1.0  # CHM使用默认值
        mid_layer = int(np.ceil(self.layers_v * midLocat))
        
        # 树冠上部
        for i in range(1, mid_layer + 1):
            z_diff = (i-1) - np.arange(0, i)
            kesi = self.Fs[1, -1:-(i+1):-1] * (exp_decay ** z_diff)
            R_l[i-1] = self.w_leaf * j0 * np.sum(kesi)
        
        # 树冠下部
        for i in range(mid_layer + 1, self.layers_v + 1):
            kesi = np.zeros(self.layers_v)
            for m in range(1, self.layers_v - i + 2):
                kesi[m-1] = self.Fs[1, -(m)] * np.exp(-theta * (z[i] - z[m]) * self.cu)
            R_l[i-1] = self.w_leaf * j0 * np.sum(kesi)
        
        # 添加地面贡献
        ground_contribution = (self.w_ground * j0 * 
                              np.sum(self.Fs[1, :] * np.exp(-theta * self.Fs[0, :])))
        R_l[-1] += ground_contribution
        
        # 卷积
        if len(freq_conv) > 1:
            R_l_crown = R_l[:self.layers_v]
            R_l_conv = np.convolve(freq_conv, R_l_crown, mode='full')
            
            if len(R_l_conv) > len(R_l):
                R_l_new = np.zeros(len(R_l_conv))
                R_l_new[:len(R_l)] = R_l
                R_l = R_l_new
            
            R_l[:len(R_l_conv)] = R_l_conv
        
        if len(R_l) > self.layers_vb:
            R_l = R_l[:self.layers_vb]
            R_l[-1] += ground_contribution
        
        return R_l
    
    def _plot_waveform(self, height: np.ndarray, waveform: np.ndarray):
        """
        绘制波形结果（双子图：完整波形 + 放大植被回波）
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        waveform_norm = waveform / np.max(waveform) if np.max(waveform) > 0 else waveform
        
        # 左图：完整波形
        ax1.plot(waveform_norm, height, 'b-', linewidth=2, label='Complete Waveform')
        ax1.set_xlabel('Normalized Return Signal', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Height (m)', fontsize=12, fontweight='bold')
        ax1.set_title('CHM-based LiDAR Waveform', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 右图：放大植被回波
        ground_idx = np.argmax(waveform)
        ground_value = waveform_norm[ground_idx]
        vegetation_threshold = ground_value * 0.05
        vegetation_mask = waveform_norm < vegetation_threshold
        
        if np.any(vegetation_mask):
            veg_waveform = waveform_norm.copy()
            veg_waveform[~vegetation_mask] = vegetation_threshold
            
            ax2.plot(veg_waveform, height, 'g-', linewidth=2, label='Vegetation Return (Zoomed)')
            ax2.set_xlabel('Normalized Return Signal (Truncated)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Height (m)', fontsize=12, fontweight='bold')
            ax2.set_title('Vegetation Return (Ground Truncated)', fontsize=14, fontweight='bold')
            ax2.set_xlim(0, vegetation_threshold * 1.2)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            ax2.text(0.98, 0.02, f'Ground return truncated at {vegetation_threshold:.4f}',
                    transform=ax2.transAxes, fontsize=9,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        else:
            ax2.plot(waveform_norm, height, 'g-', linewidth=2, label='Waveform')
            ax2.set_xlabel('Normalized Return Signal', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Height (m)', fontsize=12, fontweight='bold')
            ax2.set_title('Waveform (No Truncation)', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # 添加参数信息
        info_text = (f"CHM File: {self.chm_file.split('/')[-1]}\n"
                    f"LAD: {self.canopy.leaf_area_density:.2f}\n"
                    f"G: {self.G:.2f}\n"
                    f"ρ: {self.canopy.leaf_reflectance:.2f}\n"
                    f"Fs0: {self.Fs0:.3f}\n"
                    f"Threshold: {self.height_threshold:.1f}m")
        ax1.text(0.98, 0.98, info_text, transform=ax1.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def save_waveform(self, filename: str, height: np.ndarray, waveform: np.ndarray):
        """
        保存波形到文件
        """
        data = np.column_stack([height, waveform])
        np.savetxt(filename, data, delimiter='\t', 
                  header='Height(m)\tReturn_Signal',
                  comments='')
        print(f"Waveform saved to {filename}")
