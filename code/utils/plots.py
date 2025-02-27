# -*- coding: utf-8 -*-
"""
This file is part of the EStreams catalogue/dataset. See https://github.com/ 
for details.

Coded by: Thiago Nascimento
"""

from matplotlib.lines import Line2D
import matplotlib as mpl
import math
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=Warning)

def add_circular_legend(ax, color_mapping, legend_labels, legend_title):
    """
    Add a circular legend to the specified axes.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to which the legend will be added.
        color_mapping (dict): A dictionary mapping legend labels to colors.
        legend_labels (list): List of legend labels.
        legend_title (str): Title for the legend.

    Returns:
        None
    """
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=color_mapping[key], markeredgecolor='none', markersize=7) for key in color_mapping]
    legend = ax.legend(handles, legend_labels, loc='upper left', bbox_to_anchor=(1, 1), title=legend_title)
    legend.get_frame().set_linewidth(0)  # Remove legend frame
    legend.get_frame().set_facecolor('none')  # Remove legend background
    legend.set_bbox_to_anchor((-0.05, 0.8))  # Adjust legend position


def plot_num_measurementsmap_subplot(ax, plotsome: pd.DataFrame, xcoords="lon", ycoords="lat", column_labels="num_yearly_complete",
                                     crsproj='epsg:4326', showcodes=False, markersize_map=3, north_arrow=True, 
                                     set_map_limits=False, minx=0, miny=0, maxx=1, maxy=1, color_categories=None, color_mapping=None,
                                     legend_title=None, legend_labels=None, legend_loc='upper left', show_legend = True, 
                                     legend_outside=True, legend_bbox_to_anchor=(0.5, 1)):  # Add legend_outside and legend_bbox_to_anchor parameters:
    """
    Plot data on a subplot with additional options.

    Parameters:
        ax (matplotlib.axes.Axes): The subplot where the data will be plotted.
        plotsome (pd.DataFrame): The data to be plotted.
        xcoords (str): The name of the column containing x-coordinates.
        ycoords (str): The name of the column containing y-coordinates.
        column_labels (str): The name of the column containing data for coloring.
        crsproj (str): The coordinate reference system (CRS) for the data.
        showcodes (bool): Whether to show data labels.
        markersize_map (int): Size of the markers.
        north_arrow (bool): Whether to include a north arrow.
        set_map_limits (bool): Whether to set specific map limits.
        minx (float): Minimum x-axis limit.
        miny (float): Minimum y-axis limit.
        maxx (float): Maximum x-axis limit.
        maxy (float): Maximum y-axis limit.
        color_categories (list): List of color categories for data bins.
        color_mapping (dict): Mapping of color categories to colors.
        legend_title (str): Title for the legend.
        legend_labels (list): Labels for the legend items.
        legend_loc (str): Location of the legend.
        show_legend (bool): Whether to display the legend.
        legend_outside (bool): Whether to place the legend outside the plot.
        legend_bbox_to_anchor (tuple): Position of the legend (x, y).

    Returns:
        None
    """
    # Prepare the data for plotting
    crs = {'init': crsproj}
    geometry = plotsome.apply(lambda row: Point(row[xcoords], row[ycoords]), axis=1)
    geodata = gpd.GeoDataFrame(plotsome, crs=crs, geometry=geometry)
    geodatacond = geodata

    if color_categories is not None and color_mapping is not None:
        geodatacond['color_category'] = pd.cut(geodatacond[column_labels], bins=[c[0] for c in color_categories] + [np.inf], labels=[f'{c[0]}-{c[1]}' for c in color_categories])
    else:
        raise ValueError("Both color_categories and color_mapping must be provided.")

    # Plotting and legend:
    for category, group in geodatacond.groupby('color_category'):
        #group.plot(ax=ax, color=color_mapping[category], markersize=markersize_map, legend=False, label=category)
        group.plot(ax=ax, marker='o', color=color_mapping[category], markersize=markersize_map, legend=False, label=category, edgecolor='none')
    
    if showcodes == True:
        geodatacond["Code"] = geodatacond.index
        geodatacond.plot(column='Code', ax=ax)
        for x, y, label in zip(geodatacond.geometry.x, geodatacond.geometry.y, geodatacond.index):
            ax.annotate(label, xy=(x, y), xytext=(1, 1), textcoords="offset points")
        plt.rcParams.update({'font.size': 12})

    if set_map_limits == False:
        total_bounds = geodatacond.total_bounds
        minx, miny, maxx, maxy = total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    # Plot the legend
    if legend_labels is None:
        legend_labels = [f'{c[0]}-{c[1]}' for c in color_categories]
        
    if show_legend:
        if legend_outside:
            legend = ax.legend(title=legend_title, labels=legend_labels, loc='upper left', bbox_to_anchor=legend_bbox_to_anchor,
                               bbox_transform=ax.transAxes, frameon=False)  # Use bbox_transform to position the legend
        else:
            legend = ax.legend(title=legend_title, labels=legend_labels, loc=legend_loc, frameon=False)
            
        if legend_outside:
            ax.add_artist(legend)
            
    # Plot the north arrow:
    if north_arrow == True:
        x, y, arrow_length = 0.975, 0.125, 0.1

        ax.annotate('N', xy=(x, y), xytext=(x, y - arrow_length),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15),
                    ha='center', va='center', fontsize=18,
                    xycoords='axes fraction')
  
    # Set font family and size using rcParams
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 8  # You can adjust this value as needed

def generate_legend_and_color_mapping(variable_name, color_categories, xmin_hist, xmax_hist, base_hist, hist_bins, color_palette='default'):
    """
    Generate legend labels and color mapping based on the provided color categories.

    Parameters:
        variable_name (str): Name of the variable.
        color_categories (list): List of tuples defining color categories.
        xmin_hist (float): Minimum value for histogram.
        xmax_hist (float): Maximum value for histogram.
        base_hist (float): Base value for histogram.
        hist_bins (int): Number of bins for histogram.
        color_palette (str): Name of the color palette to use. Options: 'default', 'custom', 'vegetation', 'grassland', 'blues'.

    Returns:
        tuple: Tuple containing variable name, color categories, legend labels, color mapping, histogram parameters.
    """
    # Adjusting the first and last legend labels
    legend_labels = [f"<{color_categories[0][1]}"]
    for i in range(len(color_categories) - 1):
        legend_labels.append(f"{color_categories[i][1]}-{color_categories[i+1][1]}")
    legend_labels[-1] = f">{color_categories[-1][0]}"  # Correcting the last legend label

    color_mapping = {}
    if color_palette == 'default':
        color_mapping = {
            f"{category[0]}-{category[1]}": ['#a6cee3', '#1f78b4', '#6a3d9a', '#b2df8a', '#33a02c'][i] for i, category in enumerate(color_categories)
        }
    elif color_palette == 'custom':
        # Define your custom color palette here
        color_mapping = {
            f"{category[0]}-{category[1]}": ['bisque', '#fdcc8a', '#fc8d59', '#d7301f', '#990000'][i] for i, category in enumerate(color_categories)
             #color_mapping[f"{category[0]}-{category[1]}"] = ['#ffbb78', '#ff7f0e', '#aec7e8', '#1f77b4', '#9467bd'][i] # This is from orange to purple

        }

    elif color_palette == 'vegetation':
        # Define your custom color palette here
        color_mapping = {
            f"{category[0]}-{category[1]}": ['#B2DF8A', '#66C2A5', '#238B45', '#006D2C', '#00441B'][i] for i, category in enumerate(color_categories)
        }

    elif color_palette == 'grassland':
        # Define your custom color palette here
        color_mapping = {
            f"{category[0]}-{category[1]}": ['#FFD699', '#FF9900', '#FF6600', '#FF3300', '#FF0000'][i] for i, category in enumerate(color_categories)
        }

    elif color_palette == 'blues':
        # Define your custom color palette here
        color_mapping = {
            f"{category[0]}-{category[1]}": ['#FFD699', '#b2df8a', '#a6cee3', '#1f78b4', '#6a3d9a'][i] for i, category in enumerate(color_categories)
        }

    else:
        raise ValueError("Invalid color_palette. Choose 'default' or 'custom'.")

    return variable_name, color_categories, legend_labels, color_mapping, xmin_hist, xmax_hist, base_hist, hist_bins

def plot_variable_subplot(ax, variable, estreams_attributes, color_mapping_list, gdf):
    """
    Plot a variable on a subplot along with its legend and histogram.

    Parameters:
        ax (matplotlib Axes): Subplot axes.
        variable (str): Name of the variable.
        estreams_attributes (DataFrame): DataFrame containing attribute data.
        color_mapping_list (dict): Dictionary containing color mapping information.
        gdf (GeoDataFrame): GeoDataFrame for plotting shapefile.

    Returns:
        None
    """
    # Extract color mapping information from the color mapping list
    legend_title = color_mapping_list[variable][0]
    color_categories = color_mapping_list[variable][1]
    legend_labels = color_mapping_list[variable][2]
    color_mapping = color_mapping_list[variable][3]
    xmin_hist = color_mapping_list[variable][4]
    xmax_hist = color_mapping_list[variable][5]
    base_hist = color_mapping_list[variable][6]
    hist_bins = color_mapping_list[variable][7]

    # Set the background color to white
    ax.set_facecolor('white')

    # Plot the shapefile with white facecolor and black boundaries
    gdf.plot(ax=ax, facecolor='whitesmoke', edgecolor='black', linewidth=0.2)
    ax.set_xlim(-24, 40) 
    ax.set_ylim(35, 70)  

    # Plot the data on the map
    plot_num_measurementsmap_subplot(plotsome=estreams_attributes, xcoords="lon", ycoords="lat", column_labels=variable,
                                     color_categories=color_categories, color_mapping=color_mapping, 
                                     legend_title=legend_title, legend_labels=legend_labels, legend_loc='lower left', ax=ax, 
                                     set_map_limits=True, minx=-24, miny=35, maxx=40, maxy=70, show_legend=False, 
                                     legend_outside=False, north_arrow=False, markersize_map=1)

    # Turn off both x-axis and y-axis
    ax.set_axis_off()

    # Create a histogram inset axis within the subplot
    hist_ax = ax.inset_axes([0.05, 0.05, 0.15, 0.175])  # Adjust the values as needed

    # Extract the data for the histogram
    hist_data = estreams_attributes[variable].dropna()

    # Plot the histogram within the inset axis
    hist_ax.hist(hist_data, bins=hist_bins, color='white', edgecolor='black', alpha=0.7, linewidth=0.5)
    
    # Hide the axis spines and ticks for the inset axis
    hist_ax.spines['top'].set_visible(False)
    hist_ax.spines['right'].set_visible(False)
    hist_ax.spines['left'].set_visible(False)
    hist_ax.spines['bottom'].set_visible(True)
    hist_ax.set_facecolor('none')
    hist_ax.set_yticklabels(hist_ax.get_yticks(), rotation=90, fontsize=8)

    # Remove y-axis ticks and labels
    hist_ax.set_yticks([])
    hist_ax.set_yticklabels([])

    # Call the function to add a circular legend
    add_circular_legend(ax, color_mapping, legend_labels, legend_title)

    # Adjust aspect ratio
    ax.set_aspect('equal')

    # Hide spines for main plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

# Function to create histograms inside subplots
def add_hist(axes, data: pd.DataFrame, axes_loc=[0.05, 0.05, 0.15, 0.175], alpha_hist=0.7,
             num_bins=10, x_ticks=[0, 5], base_xaxis=1, xlim_i=0, xlim_f=5):
    """
    Add a histogram to a subplot.

    Parameters:
        data (pandas.Series): Data for the histogram.
        axes (matplotlib.axes.Axes): The subplot where the histogram will be added.
        axes_loc (list): Location and size of the inset axis.
        alpha_hist (float): Alpha value for histogram transparency.
        num_bins (int): Number of histogram bins.
        x_ticks (list): Specific x-axis tick values.
        base_xaxis (int): Minor locator base for x-axis ticks.
        xlim_i (float): Minimum x-axis limit.
        xlim_f (float): Maximum x-axis limit.

    Returns:
        None
    """
    # Create a histogram inset axis within the subplot
    hist_ax = axes.inset_axes(axes_loc)  # Adjust the values as needed
    # Extract the data for the histogram (replace 'column_name' with the actual column you want to plot)
    hist_data = data.dropna()

    # Plot the histogram within the inset axis
    hist_ax.hist(hist_data, bins=num_bins, color='gray', alpha=alpha_hist)
    hist_ax.set_xlabel('')  # Replace with an appropriate label
    hist_ax.set_ylabel('')  # Replace with an appropriate label

    # Hide the axis spines and ticks for the inset axis
    hist_ax.spines['top'].set_visible(False)
    hist_ax.spines['right'].set_visible(False)
    hist_ax.spines['left'].set_visible(False)
    hist_ax.spines['bottom'].set_visible(True)
    hist_ax.set_facecolor('none')
    hist_ax.set_yticklabels(hist_ax.get_yticks(), rotation=90, fontsize=5)

    # Adjust y-tick label alignment for the right y-axis
    hist_ax.yaxis.tick_right()  # Move the y-tick labels to the right side
    hist_ax.yaxis.set_label_position("right")  # Move the y-axis label to the right side

    # Define the specific y-axis tick values you want to show
    hist_ax.set_xticks(x_ticks)

    # Remove y-axis ticks and labels
    hist_ax.set_yticks([])
    hist_ax.set_yticklabels([])

    hist_ax.xaxis.set_minor_locator(plt.MultipleLocator(base=base_xaxis))  # Adjust the base as needed
    # Set x-axis limits (adjust the values as needed)
    hist_ax.set_xlim(xlim_i, xlim_f)


# Histograms for different catchments groups:
def plot_histograms_by_group(df, col_to_classify, num_cols=3):
    """
    Plot histograms for each class within each variable in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        col_to_classify (str): Name of the column containing the classes.
        num_cols (int): Number of columns in each row of subplots. Default is 3.
    """
    # Get unique classes
    classes = df[col_to_classify].unique()
    classes = classes.astype(int)
    num_classes = len(classes)
    
    # Determine number of rows and columns for subplots
    num_rows = (df.shape[1] - 1) // num_cols + 1

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows*3))

    # Flatten the axes array
    axes = axes.flatten()

    # Plot histograms for each variable
    for i, (col, ax) in enumerate(zip(df.columns, axes)):
        if col != col_to_classify:
            for cls in classes:
                ax.hist(df[df[col_to_classify].astype(int) == cls][col], bins=10, alpha=0.5, label=cls)
            ax.set_title(col)
            ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_histograms_by_group_two_subsets(df, df2, col_to_classify, num_cols=3):
    """
    Plot histograms for each class within each variable in the DataFrame using only two different subsets.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        col_to_classify (str): Name of the column containing the classes.
        num_cols (int): Number of columns in each row of subplots. Default is 3.
    """
    # Get unique classes
    classes = df[col_to_classify].unique()
    classes = classes.astype(int)
    num_classes = len(classes)
    
    # Determine number of rows and columns for subplots
    num_rows = (df.shape[1] - 1) // num_cols + 1

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows*3))

    # Flatten the axes array
    axes = axes.flatten()

    # Plot histograms for each variable
    for i, (col, ax) in enumerate(zip(df.columns, axes)):
        if col != col_to_classify:
            for cls in classes:
                ax.hist(df[df[col_to_classify].astype(int) == cls][col], bins=10, alpha=0.5, label=cls, density=True)
                ax.hist(df2[df2[col_to_classify].astype(int) == cls][col], bins=10, alpha=0.5, label=cls, density=True)
            ax.set_title(col)
            ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def plot_watershed_groups_from_dict(estreams_attributes_used, title_plot="Plot title", figsize=(8, 6), ax=None, 
                                    add_legend=False, size_symbol=2, limits_europe=True):
    """
    Plot watershed groups on a map using GeoPandas.

    Parameters:
        estreams_attributes_used (dict): A dictionary containing watershed groups dataframes.
        title_plot (str): Title of the plot.
        figsize (tuple): Figure size in inches (width, height).
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new figure and axes will be created.
        add_legend (bool): Whether to add a legend to the plot.
        size_symbol (int): Size of the symbols in the plot.
        limits_europe (bool): Whether to set the plot limits to Europe.

    Returns:
        None
    """
    
    # Load the world shapefile dataset provided by GeoPandas
    gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Set font properties using rcParams
    mpl.rcParams['font.family'] = 'arial'  # Change the font family
    mpl.rcParams['font.size'] = 8            # Change the font size
    mpl.rcParams['font.weight'] = 'normal'   # Change font weight
    mpl.rcParams['axes.labelweight'] = 'bold'  # Change label font weight

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Clear the subplot
    ax.clear()

    # Set the background color to white
    ax.set_facecolor('white')

    # Plot the shapefile with white facecolor and black boundaries
    gdf.plot(ax=ax, facecolor='whitesmoke', edgecolor='black', linewidth=0.1)

    # Calculate the bounds of the data if limits_europe is False
    if not limits_europe:
        all_lons = []
        all_lats = []
        for data in estreams_attributes_used.values():
            all_lons.extend(data['lon'])
            all_lats.extend(data['lat'])
        
        if all_lons and all_lats:
            min_lon, max_lon = min(all_lons), max(all_lons)
            min_lat, max_lat = min(all_lats), max(all_lats)
            margin = 0.05  # Add some margin to the bounds
            ax.set_xlim(min_lon - margin, max_lon + margin)
            ax.set_ylim(min_lat - margin, max_lat + margin)

    else:
        ax.set_xlim(-24, 45)
        ax.set_ylim(35, 70)

    # Define markers for each watershed group
    markers = ['o', 's', '^', 'D', '*']  # You can extend this list if needed

    # Plot the gauges for each desired watershed group
    colors = plt.cm.tab10(np.linspace(0, 1, len(estreams_attributes_used.keys())))  # Get distinct colors
    i = 0
    for catchment, data in estreams_attributes_used.items():
        marker = markers[i % len(markers)]  # Choose marker cyclically
        color = colors[i]  # Choose color from the colormap
        ax.scatter(data['lon'], data['lat'], color=color, edgecolor='black',
                    linewidth=0.01, marker=marker, s=size_symbol, label=catchment)
        i += 1

    ax.set_aspect('equal')  # Adjust aspect ratio as needed
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_axis_off()  # Turn off both x-axis and y-axis

    # Set title for the plot
    ax.set_title(title_plot)
    # Add legend (if needed)
    if add_legend:
        ax.legend()

    # Adjust layout
    plt.tight_layout()
