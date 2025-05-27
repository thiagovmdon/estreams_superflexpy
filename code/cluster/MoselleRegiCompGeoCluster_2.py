import pandas as pd
import datetime as datetime
import numpy as np
import spotpy
import tqdm as tqdm
from superflexpy.framework.unit import Unit
from superflexpy.framework.node import Node
from superflexpy.framework.network import Network
from superflexpy.implementation.elements.hbv import UnsaturatedReservoir, PowerReservoir
from superflexpy.implementation.numerical_approximators.implicit_euler import ImplicitEulerPython
from superflexpy.implementation.root_finders.pegasus import PegasusPython
from superflexpy.implementation.root_finders.pegasus import PegasusNumba
from superflexpy.implementation.numerical_approximators.implicit_euler import ImplicitEulerNumba
from superflexpy.implementation.elements.hbv import PowerReservoir
from superflexpy.framework.unit import Unit
from superflexpy.implementation.elements.thur_model_hess import SnowReservoir, UnsaturatedReservoir, PowerReservoir, HalfTriangularLag
from superflexpy.implementation.elements.structure_elements import Transparent, Junction, Splitter
from superflexpy.framework.element import ParameterizedElement
from collections import defaultdict
import matplotlib.pyplot as plt
import os 

# Define the functions
def obj_fun_nsee(observations, simulation, expo=0.5):
    """
    Calculate the Normalized Squared Error Efficiency (NSEE) while ensuring that
    NaNs in simulation are NOT masked (only NaNs in observations are masked).

    Parameters:
        observations (array-like): Observed values (with fixed NaNs).
        simulation (array-like): Simulated values (can contain NaNs).
        expo (float, optional): Exponent applied to observations and simulations. Default is 1.0.

    Returns:
        float: NSEE score (higher values indicate worse performance).
    """
    observations = np.asarray(observations)
    simulation = np.asarray(simulation)

    # Mask only NaNs in observations
    mask = ~np.isnan(observations)
    obs = observations[mask]
    sim = simulation[mask]  # Keep all simulated values, even NaNs

    # If simulation contains NaNs after masking observations, return penalty
    if np.isnan(sim).any():
        return 10.0  # Large penalty if NaNs appear in the simulation

    metric = np.sum((sim**expo - obs**expo)**2) / np.sum((obs**expo - np.mean(obs**expo))**2)
    
    return float(metric)

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

path_inputs = 'data/models/input/subset_1988_2001'

inputs = np.load(path_inputs+'//inputs.npy', allow_pickle=True).item()
observations = np.load(path_inputs+'//observations.npy', allow_pickle=True).item()
areas = np.load(path_inputs+'//areas.npy', allow_pickle=True).item()
perm_areas = np.load(path_inputs+'//perm_areas.npy', allow_pickle=True).item()
perm_areasglobal = np.load(path_inputs+'//perm_areasglobal.npy', allow_pickle=True).item()
quality_masks = np.load(path_inputs+'//quality_masks.npy', allow_pickle=True).item()
rootdepth_mean = np.load(path_inputs+'//rootdepth_mean.npy', allow_pickle=True).item()
waterdeficit_mean = np.load(path_inputs+'//waterdeficit_mean.npy', allow_pickle=True).item()
prec_mean= np.load(path_inputs+'//prec_mean.npy', allow_pickle=True).item()

#catchments_ids = ['DERP2017',
# 'DERP2033',
# 'DERP2007',
# 'DERP2024',
# 'FR003253',
# 'FR003308',
# 'FR003283',
# 'FR003301',
# 'DERP2003',
# 'FR003265',
# 'FR003272',
# 'DEBU1958',
# ]
catchments_ids = ['FR000184',
 'DERP2017',
 'DERP2011',
 'DERP2013',
 'DERP2007',
 'DERP2024',
 'FR003253',
 #'FR003308',
 'FR003283',
 'FR003301',
 'DERP2003',
 'FR003265',
 'FR003272',
 'DEBU1958']

print("version-18.03.2025")

# Here we retrieve the conectivity (from EStreams computation)
df = pd.read_excel("data/nested_catchments.xlsx")
# Rename columns for clarity
df = df.rename(columns={df.columns[1]: "basin_id", df.columns[2]: "connected_basin_id"})
df = df.drop(columns=[df.columns[0]])  # Drop the unnamed index column

# Load combined_df from CSV (already has group labels)
combined_df = pd.read_csv("data/network_estreams_moselle_108_gauges.csv")

# Loop over groups
group_names = combined_df['group'].unique()
for group in group_names[2:]:
    print(f"\n Running calibration for {group}...")

    # Select catchments in this group and remove LU gauges
    catchments_df = combined_df[(combined_df['group'] == group) & (~combined_df['basin_id'].str.contains("LU"))]
    catchments_ids = catchments_df['basin_id'].tolist()

    print(catchments_ids)

    # Run the iterative function
    iterative_immediate_downstream = find_iterative_immediate_downstream(df, catchments_ids)

    # Convert results to a DataFrame for display
    iterative_downstream_df = pd.DataFrame(iterative_immediate_downstream.items(), 
                                        columns=['basin_id', 'immediate_downstream_basin'])


    # Assuming the DataFrame has columns 'basin_id' and 'downstream_id'
    topology_list = {basin: None for basin in catchments_ids}  # Default to None

    # Filter DataFrame for relevant basin_ids and update topology
    for _, row in iterative_downstream_df.iterrows():
        if row['basin_id'] in topology_list:
            topology_list[row['basin_id']] = row['immediate_downstream_basin']


    root_finder = PegasusNumba()
    num_app = ImplicitEulerNumba(root_finder=root_finder)

    class ParameterizedSingleFluxSplitter(ParameterizedElement):
        _num_downstream = 2
        _num_upstream = 1
        
        def set_input(self, input):

            self.input = {'Q_in': input[0]}

        def get_output(self, solve=True):

            split_par = self._parameters[self._prefix_parameters + 'splitpar']

            output1 = [self.input['Q_in'] * split_par]
            output2 = [self.input['Q_in'] * (1 - split_par)]
            
            return [output1, output2]   
        
        
    lower_splitter = ParameterizedSingleFluxSplitter(
        parameters={'splitpar': 0.5},
        id='lowersplitter'
    )

    lower_splitter_medium = ParameterizedSingleFluxSplitter(
        parameters={'splitpar': 0.6},
        id='lowersplitter'
    )

    lower_splitter_high = ParameterizedSingleFluxSplitter(
        parameters={'splitpar': 0.7},
        id='lowersplitter'
    )

    # Fluxes in the order P, T, PET
    upper_splitter = Splitter(
        direction=[
            [0, 1, None],    # P and T go to the snow reservoir
            [2, None, None]  # PET goes to the transparent element
        ],
        weight=[
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        id='upper-splitter'
    )

    snow = SnowReservoir(
        parameters={'t0': 0.0, 'k': 0.01, 'm': 2.0},
        states={'S0': 0.0},
        approximation=num_app,
        id='snow'
    )

    upper_transparent = Transparent(
        id='upper-transparent'
    )

    upper_junction = Junction(
        direction=[
            [0, None],
            [None, 0]
        ],
        id='upper-junction'
    )


    unsaturated = UnsaturatedReservoir(
        parameters={'Smax': 150.0, 'Ce': 1.0, 'm': 0.01, 'beta': 2.0},
        states={'S0': 10.0},
        approximation=num_app,
        id='unsaturated'
    )

    fast = PowerReservoir(
        parameters={'k': 0.01, 'alpha': 2.0},
        states={'S0': 0.0},
        approximation=num_app,
        id='fast'
    )

    slow = PowerReservoir(
        parameters={'k': 1e-4, 'alpha': 1.0},
        states={'S0': 0.0},
        approximation=num_app,
        id='slow'
    )

    slowhigh = PowerReservoir(
        parameters={'k': 1e-4, 'alpha': 2.0},
        states={'S0': 0.0},
        approximation=num_app,
        id='slowhigh'
    )


    lower_junction = Junction(
        direction=[
            [0, 0]
        ],
        id='lower-junction'
    )

    lag_fun = HalfTriangularLag(
        parameters={'lag-time': 4.0},
        states={'lag': None},
        id='lag-fun'
    )

    lower_transparent = Transparent(
        id='lower-transparent'
    )

    lower_transparent2 = Transparent(
        id='lower-transparent2'
    )

    general = Unit(
        layers=[
            [upper_splitter],
            [snow, upper_transparent],
            [upper_junction],
            [unsaturated],
            [lower_splitter],
            [slow, lag_fun],
            [lower_transparent, fast],
            [lower_junction],
        ],
        id='general'
    )

    low = Unit(
        layers=[
            [upper_splitter],
            [snow, upper_transparent],
            [upper_junction],
            [unsaturated],
            [lower_splitter],
            [slow, lag_fun],
            [lower_transparent, fast],
            [lower_junction],
        ],
        id='low'
    )

    high = Unit(
        layers=[
            [upper_splitter],
            [snow, upper_transparent],
            [upper_junction],
            [unsaturated],
            [lower_splitter],
            [slow, lag_fun],
            [lower_transparent, fast],
            [lower_junction],
        ],
        id='high'
    )


    # Generate Nodes dynamically and assign them as global variables
    catchments = [] # Dictionary to store nodes

    for cat_id in catchments_ids:
        node = Node(
            units=[high, general, low],  # Use unit from dictionary or default
            weights=perm_areas[cat_id],
            area=areas.get(cat_id),  # Use predefined area or default
            id=cat_id
        )
        catchments.append(node)  # Store in the list

        # Assign the node as a global variable
        globals()[cat_id] = node

    # Ensure topology only includes nodes that exist in `catchments_ids`
    topology = {
        cat_id: upstream if upstream in catchments_ids else None
        for cat_id, upstream in topology_list.items() if cat_id in catchments_ids
    }

    # Create the Network
    model = Network(
        nodes=catchments,  # Pass list of Node objects
        topology=topology  
    )


    # Set inputs for each node using the manually defined dictionary
    for cat in catchments:
        cat.set_input(inputs[cat.id])  # Correct way to set inputs


    def assign_parameter_values(parameters_name_model, parameter_names, parameters):
        """
        Assigns values from `parameters` to `parameters_name_model` where a match exists in `parameter_names`,
        but keeps any parameters that have three segments (`X_Y_Z`) unchanged.

        Args:
            parameters_name_model (list): List of full parameter names (e.g., "general_slow_k").
            parameter_names (list): List of unique parameter names (e.g., "slow_k", "high_slow_k").
            parameters (list): List of values corresponding to `parameter_names`.

        Returns:
            dict: Dictionary {parameter_name_model: assigned_value}, where:
                - `X_Y` parameters are updated from `parameter_names`.
                - `X_Y_Z` parameters are kept unchanged.
        """
        # Create a dictionary mapping parameter_names to their corresponding values
        param_value_dict = {param_name: value for param_name, value in zip(parameter_names, parameters)}

        # Build the output dictionary
        filtered_parameters = {}

        for param_name in parameters_name_model:
            parts = param_name.split("_")  # Split the name to check structure
            base_name = "_".join(parts[-2:])  # Extract last two parts (X_Y)
            
            if base_name in param_value_dict:  # If X_Y is in parameter_names
                filtered_parameters[param_name] = param_value_dict[base_name]
            elif param_name in parameter_names:  # Direct match in parameter_names (X_Y)
                filtered_parameters[param_name] = param_value_dict[param_name]
        
        return filtered_parameters  # Return dictionary of matched parameters


    class spotpy_model(object):

        def __init__(self, model, catchments, dt, observations, parameters, parameter_names, parameter_names_model, output_index, warm_up, prec_mean):

            """
            Spotpy model for multi-node calibration in SuperflexPy.

            Args:
                model (Network): SuperflexPy network containing multiple nodes.
                catchments (list): List of Node objects.
                inputs (dict): Dictionary with inputs for each node.
                dt (float): Time step.
                observations (dict): Observed discharge data for each node.
                parameters (list): List of parameter distributions for calibration.
                parameter_names (list): Names of the parameters.
                output_index (str/int): The output key for extracting model results.
                warm_up (int): Number of time steps to ignore in the evaluation.
            """
            self._model = model  # The SuperflexPy network
            self._catchments = catchments  # List of catchments
            self._dt = dt  # Time step

            # Store shared calibration parameters
            self._parameters = parameters
            self._parameter_names = parameter_names
            self._parameter_names_model = parameter_names_model  # Store full parameter names

            # Store inputs and observations for each node
            self._observations = observations  # Dictionary {node_id: observed_data}
            self._output_index = output_index  # Output key (e.g., 'Q_out')
            self._warm_up = int(warm_up)  # Warm-up period

        def parameters(self):
            """Generate parameter samples for calibration."""
            return spotpy.parameter.generate(self._parameters)

        def simulation(self, parameters):
            """Runs the entire network using the same parameter set and collects per-node outputs."""

            # Convert parameter list into a dictionary
            #named_parameters = assign_parameter_values(self._parameter_names_model, self._parameter_names, parameters)
            
            # Check if parameters have changed (avoid unnecessary computations)
            if not hasattr(self, "_cached_params") or not np.array_equal(self._cached_params, parameters):
                self._cached_params = np.array(parameters)  # Store the current parameters
                named_parameters = assign_parameter_values(self._parameter_names_model, self._parameter_names, parameters)
                self._model.set_parameters(named_parameters)  # Apply shared parameters

            # Apply shared parameters to the whole network (this is due to the way we set Csumax)
            for key in model._content_pointer.keys():
                i = model._content_pointer[key] 
                self._model._content[i].set_parameters(named_parameters)
            #self._model.set_parameters(named_parameters)

            # Set timestep and reset the network
            self._model.set_timestep(self._dt)
            self._model.reset_states()

            # Run the full network
            output = self._model.get_output()  # Get outputs for all nodes

            # Return outputs as a list (one per node)
            return [output[cat.id][self._output_index] for cat in self._catchments]

        def evaluation(self):
            """Returns the observed data for all nodes."""
            return self._observations

        def objectivefunction(self, simulation, evaluation):
            """Computes the average NSE (or another metric) across all nodes."""

            obj_values = []  # Store individual NSE values for each node

            for sim, cat in zip(simulation, self._catchments):
                node_id = cat.id
                obs = evaluation[node_id]

                # Apply warm-up period
                sim = sim[self._warm_up + 1:]
                obs = obs[self._warm_up + 1:]

                # Compute NSE (or another metric like KGE)
                obj_value = obj_fun_nsee(observations=obs, simulation=sim, expo=0.5)
                obj_values.append(obj_value)

            # Compute the average objective function across all nodes
            return np.mean(obj_values)  # Minimize the average error

    spotpy_hyd_mod = spotpy_model(
        model=model,  # The entire SuperflexPy network
        catchments=catchments,  # Use predefined catchments list
        dt=1.0,  # Time step
        observations=observations,  # Observed data per node
        parameters=[
            spotpy.parameter.Uniform("general_fast_k", 0.0001, 1.0), #1e-5, 1.0
            spotpy.parameter.Uniform("low_fast_k", 0.0001, 1.0),
            spotpy.parameter.Uniform("high_fast_k", 0.0001, 1.0),

            spotpy.parameter.Uniform("high_slow_k", 1e-7, 0.1),
            spotpy.parameter.Uniform("general_slow_k", 1e-7, 0.1),
            spotpy.parameter.Uniform("low_slow_k", 1e-7, 0.1),

            spotpy.parameter.Uniform("unsaturated_Ce", 0.1, 3.0),
            spotpy.parameter.Uniform("snow_k", 0.01, 10.0),
            spotpy.parameter.Uniform("unsaturated_Smax", 100.0, 600.0),

            spotpy.parameter.Uniform("general_lowersplitter_splitpar", 0.1, 0.9),
            spotpy.parameter.Uniform("high_lowersplitter_splitpar", 0.1, 0.9),
            spotpy.parameter.Uniform("low_lowersplitter_splitpar", 0.1, 0.9),

            spotpy.parameter.Uniform("unsaturated_beta", 0.01, 10.0),
            spotpy.parameter.Uniform("lag-fun_lag-time", 1.0, 10.0),
        ],
        parameter_names=[
            "general_fast_k", "low_fast_k", "high_fast_k",
            "high_slow_k", "general_slow_k", "low_slow_k", "unsaturated_Ce", "snow_k", "unsaturated_Smax", "general_lowersplitter_splitpar", "high_lowersplitter_splitpar", "low_lowersplitter_splitpar",
            "unsaturated_beta", "lag-fun_lag-time",
        ],
        parameter_names_model = model.get_parameters_name(),
        output_index=0,  # Assumes all nodes have the same output variable
        warm_up=365,  # Warm-up period
        prec_mean=prec_mean

    )

    #sampler = spotpy.algorithms.sceua(spotpy_hyd_mod, dbname=None, dbformat='ram')
    sampler = spotpy.algorithms.sceua(spotpy_hyd_mod, dbname='sceua_results_regi2', dbformat='csv')

    sampler.sample(repetitions=50000)

    #results = sampler.getdata()                                                  # Load the results
    results = spotpy.analyser.load_csv_results('sceua_results_regi2')

    spotpy.analyser.plot_parametertrace(results)                                 # Show the results

    bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results)               # Get the best indexes and objective function


    spotpy.analyser.get_parameters(results)[bestindex]

    best_params_dict = dict(zip(spotpy.analyser.get_parameternames(results), spotpy.analyser.get_parameters(results)[bestindex]))

    #if 'splitpar' in best_params_dict:
    #    best_params_dict['general_lowersplitter_splitpar'] = best_params_dict.pop('splitpar')

    # Remove spaces and replace with underscores (or any other transformation)
    best_params_dict = {key.replace(" ", ""): value for key, value in best_params_dict.items()}

    parameter_names = list(best_params_dict.keys())
    parameters = list(best_params_dict.values())
    parameter_names_model = model.get_parameters_name()
    best_params_dict_model = assign_parameter_values(parameter_names_model, parameter_names, parameters)

    save_path = f"results/groups/moselle_best_params_regicomp_{group}_2.csv"

    # Convert dictionary to DataFrame and save
    pd.DataFrame.from_dict(best_params_dict_model, orient='index').to_csv(save_path)

    print(f"Saved best parameters for {group} to {save_path}")