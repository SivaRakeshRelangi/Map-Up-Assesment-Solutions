import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    # Extract unique toll locations
    locations = pd.concat([df['id_start'], df['id_end']]).unique()
    num_locations = len(locations)
    
    # Initialize distance matrix with infinities
    distance_matrix = pd.DataFrame(np.inf, index=locations, columns=locations)
    
    # Fill in known distances
    for _, row in df.iterrows():
        distance_matrix.loc[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.loc[row['id_end'], row['id_start']] = row['distance']  # Ensure symmetry

    # Set diagonal to 0
    np.fill_diagonal(distance_matrix.values, 0)

    # Apply Floyd-Warshall algorithm to compute cumulative distances
    for k in locations:
        for i in locations:
            for j in locations:
                if distance_matrix.loc[i, j] > distance_matrix.loc[i, k] + distance_matrix.loc[k, j]:
                    distance_matrix.loc[i, j] = distance_matrix.loc[i, k] + distance_matrix.loc[k, j]

    return distance_matrix
# Example usage:
#df = pd.read_csv('dataset-2.csv')
#distance_matrix = calculate_distance_matrix(df)
#print (distance_matrix)


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
     # Prepare an empty list to hold the unrolled data
    unrolled_data = []

    # Get the index (locations) of the distance matrix
    locations = df.index

    # Iterate over all combinations of locations
    for i in locations:
        for j in locations:
            if i != j:  # Exclude same id_start and id_end
                distance = df.loc[i, j]
                unrolled_data.append({'id_start': i, 'id_end': j, 'distance': distance})

    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

# Example usage:
#distance_matrix = calculate_distance_matrix(df)  
#unrolled_df = unroll_distance_matrix(distance_matrix)
#print (unrolled_df)



def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    # Calculate the average distance for the reference ID
    reference_distances = df[df['id_start'] == reference_id]['distance']
    
    if reference_distances.empty:
        raise ValueError(f"No distances found for reference ID: {reference_id}")
    
    average_reference_distance = reference_distances.mean()
    
    # Calculate the 10% threshold values
    lower_bound = average_reference_distance * 0.9
    upper_bound = average_reference_distance * 1.1

    # Calculate average distances for all IDs
    average_distances = df.groupby('id_start')['distance'].mean()

    # Filter IDs within the threshold
    within_threshold = average_distances[
        (average_distances >= lower_bound) & (average_distances <= upper_bound)
    ].index

    # Create a sorted DataFrame of results
    result_df = pd.DataFrame(within_threshold, columns=['id_start'])
    result_df['average_distance'] = average_distances[within_threshold].values
    result_df = result_df.sort_values(by='average_distance').reset_index(drop=True)

    return result_df

# Example usage:
#unrolled_df = unroll_distance_matrix(distance_matrix)  
#result_df = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id=1001400)
#print (result_df)



def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type
    for vehicle_type, rate in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate
    df.pop('distance')
    return df

# Example usage:
#unrolled_df = unroll_distance_matrix(distance_matrix)  
#toll_rate_df = calculate_toll_rate(unrolled_df)
#print(toll_rate_df)



def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    # Define days of the week
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Prepare a list to hold new rows
    new_rows = []
    
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']
        
        for day in days:
            for hour in range(24):  # Iterate through each hour of the day
                start_time = time(hour)
                end_time = time(hour)  # Same start and end time for simplicity

                if day in ["Saturday", "Sunday"]:
                    discount_factor = 0.7  # Weekends
                else:
                    # Weekdays time ranges
                    if start_time < time(10, 0):
                        discount_factor = 0.8  # 00:00 to 10:00
                    elif start_time < time(18, 10):
                        discount_factor = 1.2  # 10:00 to 18:00
                    else:
                        discount_factor = 0.8  # 18:00 to 23:59

                # Calculate adjusted toll rates for each vehicle type
                adjusted_rates = {
                    'moto': row['moto'] * discount_factor,
                    'car': row['car'] * discount_factor,
                    'rv': row['rv'] * discount_factor,
                    'bus': row['bus'] * discount_factor,
                    'truck': row['truck'] * discount_factor,
                }

                # Append the new row with all details
                new_rows.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance':distance,
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    **adjusted_rates
                })
    
    # Create a new DataFrame from the collected rows
    expanded_df = pd.DataFrame(new_rows)

    return expanded_df

# Example usage:
#toll_rate_df = calculate_toll_rate(unrolled_df)  
#time_based_toll_df = calculate_time_based_toll_rates(toll_rate_df)
#print(time_based_toll_df)
