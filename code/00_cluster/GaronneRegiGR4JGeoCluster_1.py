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


from superflexpy.implementation.root_finders.pegasus import PegasusNumba
from superflexpy.implementation.numerical_approximators.implicit_euler import ImplicitEulerNumba
from superflexpy.implementation.elements.structure_elements import Transparent, Splitter, Junction
from superflexpy.implementation.elements.gr4j import (
    InterceptionFilter,
    ProductionStore,
    UnitHydrograph1,
    UnitHydrograph2,
    RoutingStore,
    FluxAggregator,
)
from superflexpy.framework.unit import Unit


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

path_inputs = 'data/models/inputgaronne/subset_2001_2015'

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
combined_df = pd.read_csv("data/network_estreams_garonne_44_gauges.csv")

# Loop over groups
group_names = combined_df['group'].unique()
for group in group_names:
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

    x1, x2, x3, x4 = (836.9511, -0.67713785, 38.518562, 1.9999999)


    class UnitHydrograph2Modified(UnitHydrograph2):
        def __init__(self, parameters, states, id):
            lag_keys = [key for key in parameters if key.endswith('lag-time')]
            if not lag_keys:
                raise ValueError(f"Cannot find any parameter ending with 'lag-time'. Keys received: {list(parameters.keys())}")

            lag_key = lag_keys[0]
            x4_value = float(parameters[lag_key])

            # Adjust lag-time to be 2x the provided value. This is proposed so we can calibrate a single X4, and not two as it was before.
            adjusted_parameters = dict(parameters)
            adjusted_parameters[lag_key] = float(2.0 * x4_value)

            # Calculate lag array length BEFORE calling super()
            array_length = int(np.ceil(adjusted_parameters[lag_key]))

            # Initialize base class
            super().__init__(adjusted_parameters, states, id)

            ## Initialize lag state AFTER base class is initialized
            #self._states[self._prefix_states + 'lag'] = np.zeros(array_length)


    # Root finder / approximation
    class PegasusNumbaSafe(PegasusNumba):
        def __init__(self):
            super().__init__()
            self._iter_max = 50  # allow up to 50 iterations
            self._tol = 1e-6     # stricter tolerance


    root_finder = PegasusNumbaSafe()
    numerical_approximation = ImplicitEulerNumba(root_finder)


    # Fluxes in the order P, T, PET
    upper_splitter = Splitter(
        direction=[
            [0, None, None],  # PET goes to the transparent element
            [1, 2, None]    # P and T go to the snow reservoir
            ],
        weight=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0]
        ],
        id='upper-splitter'
        )

    snow = SnowReservoir(
        parameters={'t0': 0.0, 'k': 0.01, 'm': 2.0},
        states={'S0': 0.0},
        approximation=numerical_approximation,
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


    interception_filter = InterceptionFilter(id='ir')

    production_store = ProductionStore(parameters={'x1': x1, 'alpha': 2.0,
                                                'beta': 5.0, 'ni': 4/9},
                                    states={'S0': 10.0},
                                    approximation=numerical_approximation,
                                    id='ps')

    splitter = Splitter(weight=[[0.9], [0.1]],
                        direction=[[0], [0]],
                        id='spl')

    unit_hydrograph_1 = UnitHydrograph1(parameters={'lag-time': x4},
                                        states={'lag': None},
                                        id='uh1')

    unit_hydrograph_2 = UnitHydrograph2Modified(parameters={'lag-time': x4},
                                        states={'lag': None},
                                        id='uh2')

    routing_store = RoutingStore(parameters={'x2': x2, 'x3': x3,
                                            'gamma': 5.0, 'omega': 3.5},
                                states={'S0': 10.0},
                                approximation=numerical_approximation,
                                id='rs')

    transparent = Transparent(id='tr')

    junction = Junction(direction=[[0, None],  # First output
                                [1, None],  # Second output
                                [None, 0]], # Third output
                        id='jun')

    flux_aggregator = FluxAggregator(id='fa')


    general = Unit(layers=[
                        [upper_splitter],
                        [upper_transparent, snow],
                        [upper_junction],
                        [interception_filter],
                        [production_store],
                        [splitter],
                        [unit_hydrograph_1, unit_hydrograph_2],
                        [routing_store, transparent],
                        [junction],
                        [flux_aggregator]],
                id='general')
    
    low = Unit(layers=[
                        [upper_splitter],
                        [upper_transparent, snow],
                        [upper_junction],
                        [interception_filter],
                        [production_store],
                        [splitter],
                        [unit_hydrograph_1, unit_hydrograph_2],
                        [routing_store, transparent],
                        [junction],
                        [flux_aggregator]],
                id='low')
    
    high = Unit(layers=[
                        [upper_splitter],
                        [upper_transparent, snow],
                        [upper_junction],
                        [interception_filter],
                        [production_store],
                        [splitter],
                        [unit_hydrograph_1, unit_hydrograph_2],
                        [routing_store, transparent],
                        [junction],
                        [flux_aggregator]],
                id='high')
    
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


    ## Set inputs for each node using the manually defined dictionary
    #for cat in catchments:
    #    cat.set_input(inputs[cat.id])  # Correct way to set inputs
    
    for cat in catchments:
        P = inputs[cat.id][0]
        T = inputs[cat.id][1]
        E = inputs[cat.id][2]
        inputs_correct = [E, P, T]  

        cat.set_input(inputs_correct)

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

    def assign_parameter_valuesgr4j(parameters_name_model, parameter_names, parameters):
        """
        Handles GR4J-style multi-unit parameters (e.g., general_ps_x1, high_rs_x2)
        and shared parameters like snow_t0, snow_k, and lag-time.
        """
        param_value_dict = {param_name: value for param_name, value in zip(parameter_names, parameters)}
        filtered_parameters = {}

        for param_name in parameters_name_model:
            parts = param_name.split("_")

            #Shared snow parameters across all units
            if "snow_t0" in param_name and "snow_t0" in param_value_dict:
                filtered_parameters[param_name] = param_value_dict["snow_t0"]
                continue
            if "snow_k" in param_name and "snow_k" in param_value_dict:
                filtered_parameters[param_name] = param_value_dict["snow_k"]
                continue


            #Handle lag-time sharing between uh1 and uh2
            if parts[-1] == "lag-time":
                group_name = parts[0]  # e.g., 'high', 'low', 'general'
                shared_key = f"{group_name}_lag-time"
                if shared_key in param_value_dict:
                    filtered_parameters[param_name] = param_value_dict[shared_key]
                    continue

            #Direct match (unit-specific parameter like general_rs_x3)
            if param_name in param_value_dict:
                filtered_parameters[param_name] = param_value_dict[param_name]
                continue

            #Fallback: match by last two segments (e.g., rs_x3)
            base_name = "_".join(parts[-2:])
            if base_name in param_value_dict:
                filtered_parameters[param_name] = param_value_dict[base_name]
                continue

        return filtered_parameters

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
                named_parameters = assign_parameter_valuesgr4j(self._parameter_names_model, self._parameter_names, parameters)
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
            spotpy.parameter.Uniform("snow_t0", -5.0, 5.0), 
            spotpy.parameter.Uniform("snow_k", 0.01, 10.0),
            spotpy.parameter.Uniform("general_ps_x1", 50.0, 4000.0),
            spotpy.parameter.Uniform("general_rs_x2", -15.0, 15.0),
            spotpy.parameter.Uniform("general_rs_x3", 20.0, 3500.0),
            spotpy.parameter.Uniform("general_lag-time", 0.5, 30.0),
            spotpy.parameter.Uniform("high_ps_x1", 50.0, 4000.0),
            spotpy.parameter.Uniform("high_rs_x2", -15.0, 15.0),
            spotpy.parameter.Uniform("high_rs_x3", 20.0, 3500.0),
            spotpy.parameter.Uniform("high_lag-time", 0.5, 30.0),
            spotpy.parameter.Uniform("low_ps_x1", 50.0, 4000.0),
            spotpy.parameter.Uniform("low_rs_x2", -15.0, 15.0),
            spotpy.parameter.Uniform("low_rs_x3", 20.0, 3500.0),
            spotpy.parameter.Uniform("low_lag-time", 0.5, 30.0),
        ],

        parameter_names=[
            "snow_t0", "snow_k", "general_ps_x1",
            "general_rs_x2", "general_rs_x3", "general_lag-time", "high_ps_x1", "high_rs_x2", "high_rs_x3", "high_lag-time", "low_ps_x1", "low_rs_x2",
            "low_rs_x3", "low_lag-time",
        ],

        parameter_names_model = model.get_parameters_name(),
        output_index=0,  # Assumes all nodes have the same output variable
        warm_up=365,  # Warm-up period
        prec_mean=prec_mean

    )

    #sampler = spotpy.algorithms.sceua(spotpy_hyd_mod, dbname=None, dbformat='ram')
    sampler = spotpy.algorithms.sceua(spotpy_hyd_mod, dbname='sceua_results_regigr4j1', dbformat='csv')

    sampler.sample(repetitions=25000)

    #results = sampler.getdata()
    results = spotpy.analyser.load_csv_results('sceua_results_regigr4j1')
    
    # Load the results
    spotpy.analyser.plot_parametertrace(results)                                 # Show the results

    bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results)               # Get the best indexes and objective function

    spotpy.analyser.get_parameters(results)[bestindex]

    best_params_dict = dict(zip(spotpy.analyser.get_parameternames(results), spotpy.analyser.get_parameters(results)[bestindex]))

    #if 'splitpar' in best_params_dict:
    #    best_params_dict['general_lowersplitter_splitpar'] = best_params_dict.pop('splitpar')

    best_params_dict['lag-fun_lag-time'] = best_params_dict.pop('lagfun_lagtime')

    # Remove spaces and replace with underscores (or any other transformation)
    best_params_dict = {key.replace(" ", ""): value for key, value in best_params_dict.items()}

    parameter_names = list(best_params_dict.keys())
    parameters = list(best_params_dict.values())
    parameter_names_model = model.get_parameters_name()
    best_params_dict_model = assign_parameter_valuesgr4j(parameter_names_model, parameter_names, parameters)

    save_path = f"results/groups/garonne_best_params_regigr4j_{group}.csv"

    # Convert dictionary to DataFrame and save
    pd.DataFrame.from_dict(best_params_dict_model, orient='index').to_csv(save_path)

    print(f"Saved best parameters for {group} to {save_path}")