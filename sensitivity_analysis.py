"""
Sensitivity Analysis for LiDAR Waveform Simulator
==================================================

Parameter sensitivity analysis for cylinder, sphere, and cone crown shapes.
Generates publication-quality figures for supplementary materials.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from lidar_simulator_core import (
    CanopyParameters,
    SimulationConfig,
    LiDARWaveformSimulator
)

# Set publication-quality defaults
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 10
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 100  # Lower DPI for screen display (was 300)


def calculate_max_crown_density(crown_radius):
    """
    Calculate maximum crown density to avoid coverage > 1
    Coverage = crown_density * π * crown_radius²
    For coverage < 1: crown_density < 1 / (π * crown_radius²)
    Use safety factor of 0.9
    """
    max_density = 0.9 / (np.pi * crown_radius**2)
    return max_density


def sensitivity_analysis(crown_shape='cylinder', save_prefix='cylinder'):
    """
    Perform sensitivity analysis for different parameters
    
    Parameters:
    -----------
    crown_shape : str
        'cylinder', 'sphere', or 'cone'
    save_prefix : str
        Prefix for saved figure filename
    """
    print(f"\n{'='*70}")
    print(f"Sensitivity Analysis: {crown_shape.upper()} Crown")
    print(f"{'='*70}")
    
    # Default parameters
    default_params = {
        'leaf_area_density': 0.5,
        'leaf_angle_distribution': 6,
        'crown_radius': 3.0,
        'crown_density': 0.03,
        'deltaz': 5,
        'crown_height_range': (15.0, 25.0),
        'leaf_reflectance': 0.5,
        'ground_reflectance': 0.2
    }
    
    # Parameter ranges for sensitivity analysis
    param_ranges = {
        'leaf_area_density': np.arange(0.1, 1.0, 0.2),  # [0.1, 0.3, 0.5, 0.7, 0.9]
        'leaf_angle_distribution': [1, 2, 3, 4, 5, 6],  # 6 leaf angle distributions
        'crown_radius': [1.0, 1.5, 2.0, 2.5, 3.0],  # Max coverage = 0.03*π*9 = 0.85 < 1
        'crown_density': np.arange(0.01, 0.051, 0.01),  # [0.01, 0.02, 0.03, 0.04, 0.05]
        'deltaz': np.arange(1, 11, 2)  # [1, 3, 5, 7, 9]
    }
    
    # Parameter labels for plots
    param_labels = {
        'leaf_area_density': 'FAVD (m²/m³)',
        'leaf_angle_distribution': 'LAD Type',
        'crown_radius': 'Crown Radius (m)',
        'crown_density': 'Crown Density (trees/m²)',
        'deltaz': 'Height range (m)'
    }
    
    # LAD type full names
    lad_names = {
        1: 'Planophile',
        2: 'Erectophile',
        3: 'Plagiophile',
        4: 'Extremophile',
        5: 'Uniform',
        6: 'Spherical'
    }
    
    # Create figure with 5 subplots (one per parameter) - horizontal layout
    # A4 width = 21cm, minus 5cm = 16cm = 6.3 inches
    # Height fixed by y-axis range (-3 to 28m), aspect ratio should be reasonable
    fig_width = 12  # A4 width minus 5cm
    fig_height = 4.5  # Height to accommodate y-axis range
    
    fig, axes = plt.subplots(1, 5, figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.12, wspace=0.25)
    
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
    
    # Iterate through parameters
    for idx, (param_name, param_values) in enumerate(param_ranges.items()):
        ax = axes[idx]
        print(f"\nAnalyzing parameter: {param_name}")
        
        # Store waveforms for this parameter
        waveforms = []
        heights = None
        
        for value in param_values:
            # Create parameter set
            params = default_params.copy()
            params[param_name] = value
            
            # Special handling for crown_density - skip if coverage would exceed 1
            if param_name == 'crown_density':
                max_density = calculate_max_crown_density(params['crown_radius'])
                if value > max_density:
                    print(f"  Skipping density {value:.3f} (exceeds max {max_density:.3f})")
                    continue
            
            # Special handling for sphere crown height range
            if crown_shape == 'sphere':
                diameter = 2 * params['crown_radius']
                params['crown_height_range'] = (25.0 - diameter, 25.0)
            
            # Create canopy and config
            canopy = CanopyParameters(
                crown_shape=crown_shape,
                crown_density=params['crown_density'],
                crown_radius=params['crown_radius'],
                crown_height_range=params['crown_height_range'],
                leaf_area_density=params['leaf_area_density'],
                leaf_angle_distribution=int(params['leaf_angle_distribution']),
                leaf_reflectance=params['leaf_reflectance'],
                ground_reflectance=params['ground_reflectance']
            )
            
            # Create config with appropriate midlocat for each shape
            # Sphere: midlocat=0.5 (symmetric)
            # Cylinder: midlocat=1.0 (default, all mass at maximum path length)
            # Cone: midlocat=1.0 (default, linear decrease distribution)
            if crown_shape == 'sphere':
                config = SimulationConfig(
                    deltaz=params['deltaz'],
                    use_pulse_shape=True,
                    midlocat=0.5
                )
            else:
                # For cylinder and cone, use default midlocat=1.0
                config = SimulationConfig(
                    deltaz=params['deltaz'],
                    use_pulse_shape=True
                )
            
            # Run simulation
            try:
                simulator = LiDARWaveformSimulator(canopy, config)
                height, waveform = simulator.simulate(plot=False)
                
                # Normalize waveform
                if np.max(waveform) > 0:
                    waveform = waveform / np.max(waveform)
                
                waveforms.append(waveform)
                if heights is None:
                    heights = height
                
                print(f"  {param_name} = {value}: Success")
            except Exception as e:
                print(f"  {param_name} = {value}: Failed - {e}")
                continue
        
        # Plot waveforms
        if len(waveforms) > 0 and heights is not None:
            # Use tab10 color palette for higher contrast
            colors = plt.cm.tab10(np.linspace(0, 0.9, len(waveforms)))
            
            # Apply non-linear x-axis transform to all shapes
            # 0-0.5 expanded (vegetation), 0.5-1.0 compressed (ground)
            use_transform = True
            
            for i, (wf, val) in enumerate(zip(waveforms, param_values[:len(waveforms)])):
                # Interpolate to common height if needed
                if len(wf) != len(heights):
                    from scipy.interpolate import interp1d
                    h_temp = np.linspace(heights[0], heights[-1], len(wf))
                    f = interp1d(h_temp, wf, kind='linear', fill_value='extrapolate')
                    wf = f(heights)
                
                # Apply x-axis transform
                if use_transform:
                    # Transform: 0-0.5 maps to 0-0.75, 0.5-1.0 maps to 0.75-1.0
                    wf_transformed = np.where(wf <= 0.5, 
                                             wf * 1.5,  # 0-0.5 -> 0-0.75
                                             0.75 + (wf - 0.5) * 0.5)  # 0.5-1.0 -> 0.75-1.0
                    plot_wf = wf_transformed
                else:
                    plot_wf = wf
                
                # Create label with full name for LAD
                if param_name == 'leaf_angle_distribution':
                    label = lad_names.get(int(val), f'{int(val)}')
                elif param_name in ['leaf_area_density', 'crown_density', 'crown_radius']:
                    label = f'{val:.2f}'
                else:
                    label = f'{int(val)}'
                ax.plot(plot_wf, heights, color=colors[i], linewidth=1.0, label=label, alpha=0.9)
            
            # Format subplot
            ax.set_xlabel('Relative signal', fontweight='bold', fontsize=10)
            # Only show ylabel on the leftmost subplot
            if idx == 0:
                ax.set_ylabel('Height (m)', fontweight='bold', fontsize=10)
            
            # Add subplot label and title together, centered and bold
            title_text = f"{subplot_labels[idx]} {param_labels[param_name]}"
            ax.set_title(title_text, fontweight='bold', fontsize=10, pad=8)
            
            ax.grid(True, alpha=0.25, linewidth=0.4, linestyle=':')
            # Place legend in lower right, slightly above ground return
            ax.legend(loc='lower right', bbox_to_anchor=(0.98, 0.15), framealpha=0.85, ncol=1, fontsize=9)
            
            # Adjust x-axis limits to show near-zero returns
            if use_transform:
                # For transformed axis
                ax.set_xlim([-0.02, 1.05])
                # Add custom x-tick labels to show actual values
                transform_ticks = [0, 0.375, 0.75, 0.875, 1.0]  # Transformed positions
                actual_values = [0, 0.25, 0.5, 0.75, 1.0]  # Actual intensity values
                ax.set_xticks(transform_ticks)
                ax.set_xticklabels([f'{v:.2f}' for v in actual_values])
            else:
                ax.set_xlim([-0.02, 1.05])
            
            ax.set_ylim([-3, 28])
    
    # Save figure
    filename = f'sensitivity_{save_prefix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n{'='*70}")
    print(f"Figure saved: {filename}")
    print(f"{'='*70}\n")
    
    plt.show()


def run_all_sensitivity_analyses():
    """
    Run sensitivity analysis for all three crown shapes
    """
    crown_shapes = [
        ('cylinder', 'cylinder'),
        ('sphere', 'sphere'),
        ('cone', 'cone')
    ]
    
    for shape, prefix in crown_shapes:
        try:
            sensitivity_analysis(crown_shape=shape, save_prefix=prefix)
        except Exception as e:
            print(f"\nError in {shape} analysis: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("LiDAR Waveform Simulator - Sensitivity Analysis")
    print("="*70)
    print("\nGenerating sensitivity analysis figures for:")
    print("1. Cylinder crown")
    print("2. Sphere crown")
    print("3. Cone crown")
    print("\nThis may take several minutes...")
    print("="*70 + "\n")
    
    run_all_sensitivity_analyses()
