# -*- coding: utf-8 -*-
"""
This file is part of the EStreams catalogue/dataset. See https://github.com/ 
for details.

Coded by: Thiago Nascimento
"""


import numpy as np
from collections import defaultdict



def find_unique_nested_catchments(df):
    """
    Return a list with unique nested catchemnts within the given initial list. When dealing with nested catchments
    groups that have intersection between each other, this code assumes the nested group with the higher number of 
    nested catchments. 

    Parameters:
        df (DataFrame): The DataFrame containing rows with lists of values.

    Returns:
        list: A list of indices of rows with the maximum number of unique values.
    """
    max_unique_rows = []  # List to store the indices of rows with maximum number of values
    col_name = df.columns[0]  # Get the name of the column containing the lists of values
    
    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        #current_row_values = row[col_name].split(', ')  # Convert string to list of values
        current_row_values = row[col_name]
        max_unique_count = len(current_row_values)  # Initialize the maximum count to the number of values in the current row
        
        # Loop through other rows to find potential overlapping rows
        for other_index, other_row in df.iterrows():
            if index != other_index:  # Skip the current row
                other_row_values = other_row[col_name]
                #other_row_values = other_row[col_name].split(', ')  # Convert string to list of values
                
                # Find the intersection of values between the current row and the other row
                intersection = set(current_row_values).intersection(other_row_values)
                if intersection:  # If there's any intersection
                    other_row_count = len(other_row_values)
                    if other_row_count > max_unique_count:  # If the other row has more values than the current maximum
                        max_unique_count = other_row_count  # Update the maximum count
        
        # Add the index of the current row to the list if it has the maximum count of values
        if len(current_row_values) == max_unique_count:
            max_unique_rows.append(index)
    
    return max_unique_rows

def find_directly_connected_catchments(df, starting_catchment):
    """
    Return a list of indices of rows with catchments that are somehow directly connected to the given starting catchment.
    For example, you can go from the selected headwater (starting_catchment) until the outlet of the watershed. 

    Parameters:
        df (DataFrame): The DataFrame containing rows with lists of values.
        starting_catchment (str): The starting catchment.

    Returns:
        list: A list of indices of rows with catchments that are somehow directly connected to the starting catchment.
    """
    connected_rows = []  # List to store indices of connected rows
    col_name = df.columns[0]  # Get the name of the column containing the lists of values
    
    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        #catchments = row[col_name].split(', ')  # Convert string to list of values
        catchments = row[col_name]

        # Check if the starting catchment is in the current row
        if starting_catchment in catchments:
            connected_rows.append(index)  # Add the index of the current row to the list
    
    return connected_rows

def find_max_unique_rows(df):
    """
    Identify rows with the maximum number of values that are unique, i.e., not repeated in other rows.

    Parameters:
        df (DataFrame): The DataFrame containing rows with lists of values.

    Returns:
        list: A list of indices of rows with the maximum number of unique values.
    """
    max_unique_rows = []  # List to store indices of rows with the maximum unique values
    col_name = df.columns[0]  # Assume the values are in the first column

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        current_row_values = row[col_name]  # Extract values for the current row
        max_unique_count = len(current_row_values)  # Initialize max count with the current row's values count
        
        # Loop through other rows to find overlaps
        for other_index, other_row in df.iterrows():
            if index != other_index:  # Skip comparing the row to itself
                other_row_values = other_row[col_name]  # Extract values for the other row

                # Find common values between current and other rows
                intersection = set(current_row_values).intersection(other_row_values)
                
                # Update max count if the other row has more values and overlaps with the current row
                if intersection:
                    other_row_count = len(other_row_values)
                    if other_row_count > max_unique_count:
                        max_unique_count = other_row_count
        
        # Add current row index if its value count matches the maximum found
        if len(current_row_values) == max_unique_count:
            max_unique_rows.append(index)
    
    return max_unique_rows


def find_iterative_immediate_downstream(df, catchments):
    """
    Finds the immediate downstream connection for each basin using an iterative approach,
    starting from the largest (end-point) basins and moving backward.
    """
    # Step 0:
    # Filter the dataframe to include only rows where 'basin_id' is in the selected_catchments list
    #filtered_df = df[df['basin_id'].isin(catchments)]
    #df = filtered_df

    # Step 1: Identify the largest basins (those not in the 'basin_id' column but in 'connected_basin_id')
    all_basins = set(df['basin_id'])
    all_connections = set(df['connected_basin_id'])
    largest_basins = all_connections - all_basins  # Basins that are only in the 'connected_basin_id' column

    # Step 2: Create mapping of connections
    downstream_map = defaultdict(set)
    for _, row in df.iterrows():
        downstream_map[row['basin_id']].add(row['connected_basin_id'])

    # Step 3: Reverse mapping for upstream tracking
    upstream_map = defaultdict(set)
    for basin, downstreams in downstream_map.items():
        for d in downstreams:
            upstream_map[d].add(basin)

    # Step 4: Iteratively determine the immediate downstream basin for each catchment
    immediate_downstream = {}
    processing_order = sorted(all_basins, key=lambda x: x in largest_basins, reverse=True)  # Start from largest

    for basin in processing_order:
        if basin in downstream_map:
            possible_downstreams = downstream_map[basin] & set(catchments)
            if possible_downstreams:
                # Select the downstream basin that is already assigned, or the one with least upstreams
                chosen_downstream = min(possible_downstreams, key=lambda b: len(upstream_map[b]))
                immediate_downstream[basin] = chosen_downstream

    return immediate_downstream