from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    result = []
    length = len(lst)
    
    for i in range(0, length, n):
        # Get the current group of n elements
        group = []
        for j in range(n):
            if i + j < length:  # Check if the index is within the list bounds
                group.append(lst[i + j])
        
        # Reverse the group manually and add to the result
        for k in range(len(group) - 1, -1, -1):
            result.append(group[k])
    
    return result
    


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    length_dict = {}

    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []  # Initialize a new list for this length
        length_dict[length].append(string)  # Add the string to the appropriate length list

    # Sort the dictionary by its keys
    sorted_length_dict = dict(sorted(length_dict.items()))

    return sorted_length_dict
    

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    def flatten_helper(current_dict: Dict, parent_key: str = '') -> Dict[str, Any]:
        items = {}
        for key, value in current_dict.items():
            # Construct new key
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                # Recursively flatten the nested dictionary
                items.update(flatten_helper(value, new_key))
            elif isinstance(value, list):
                # Handle lists by their indices
                for i, item in enumerate(value):
                    # If the item is a dict, flatten it
                    if isinstance(item, dict):
                        items.update(flatten_helper(item, f"{new_key}[{i}]"))
                    else:
                        # Add the item directly if it's not a dict
                        items[f"{new_key}[{i}]"] = item
            else:
                # If the value is neither a dict nor a list, just add it
                items[new_key] = value
        return items

    return flatten_helper(nested_dict)
    

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])  # Append a copy of the current permutation
            return
        seen = set()
        for i in range(start, len(nums)):
            if nums[i] not in seen:  # Skip duplicates
                seen.add(nums[i])  # Mark this number as seen
                nums[start], nums[i] = nums[i], nums[start]  # Swap
                backtrack(start + 1)  # Recur
                nums[start], nums[i] = nums[i], nums[start]  # Swap back

    nums.sort()  # Sort the list to handle duplicates
    result = []
    backtrack(0)
    return result
    


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
     # Define the regex pattern for matching dates
    date_pattern = r'\b(\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'
    
    # Use re.findall to extract all matching dates
    dates = re.findall(date_pattern, text)
    
    return dates


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline string into a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)
    
    # Create a DataFrame from the coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Calculate distances
    distances = [0]  # First point has distance 0
    for i in range(1, len(df)):
        lat1, lon1 = df.iloc[i - 1]
        lat2, lon2 = df.iloc[i]
        distance = haversine(lat1, lon1, lat2, lon2)
        distances.append(distance)
    
    df['distance'] = distances
    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
     n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    # Step 2: Transform the rotated matrix
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])  # Sum of the entire row
            col_sum = sum(rotated_matrix[k][j] for k in range(n))  # Sum of the entire column
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]  # Exclude itself
    
    return final_matrix


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    # Combine date and time into a single datetime column
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    # Group by (id, id_2)
    grouped = df.groupby(['id', 'id_2'])

    def check_completeness(group):
        # Get unique days of the week from start timestamps
        group['start_day_of_week'] = group['start_timestamp'].dt.day_name()
        unique_days = group['start_day_of_week'].unique()

        # Check for all 7 days
        all_days = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}
        if not all_days.issubset(unique_days):
            return False
        
        # Check for 24-hour coverage
        for day in all_days:
            day_group = group[group['start_day_of_week'] == day]
            if day_group.empty:
                return False
            
            start_min = day_group['start_timestamp'].min()
            end_max = day_group['end_timestamp'].max()
            
            # Check if the day covers from 12:00 AM to 11:59:59 PM
            if not (start_min.normalize() == start_min and end_max.normalize() == end_max + pd.Timedelta(seconds=86399)):
                return False
            
        return True

    # Apply completeness check
    result = grouped.apply(check_completeness)
    
    # Convert the result to a boolean Series with MultiIndex
    return result
