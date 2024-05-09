import numpy as np
import pandas as pd

def topsis(a, b, c):
    # Convert inputs to numpy arrays for easy manipulation
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    # Normalize the decision matrix nm=normalized matrix
    nm = a / np.sqrt(np.sum(a**2, axis=0))

    # Weighted normalized decision matrix wnm = weighted normalized matrix
    wnm = nm * b

    # Ideal and anti-ideal solutions i is idealsolution ai is anti ideal
    isolution = np.max(wnm, axis=0) if c else np.min(wnm, axis=0)
    aisolution = np.min(wnm, axis=0) if c else np.max(wnm, axis=0)

    # Calculate separation measures
    separationfromi = np.sqrt(np.sum((wnm - isolution)**2, axis=1))
    separationfromai = np.sqrt(np.sum((wnm - aisolution)**2, axis=1))

    # Relative closeness to the ideal solution
    closeness = separationfromai / (separationfromi + separationfromai)

    # Rank the alternatives based on the closeness
    rankings = np.argsort(closeness)[::-1] + 1  # Add 1 to make the rankings start from 1

    return closeness, rankings

# Read data from a local CSV file with exception handling
file_path = r'C:\Users\91896\Desktop\predictive analysis\102103047-data.csv'  # Replace with the actual file path

try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found at this path")
    # Handle the error as needed, e.g., exit the program or provide a default DataFrame
    exit()

# Extract names and matrix from the loaded data
names = data['Fund Name'].tolist()
matrix = data[['P1', 'P2', 'P3', 'P4', 'P5']].values.tolist()

# Example weights
weights = [2, 2, 3, 3, 4]  # Adjust the weights as needed

# Example benefit/cost criteria
impacts = [ False, True, False, True,False]

# Perform TOPSIS analysis
scores, rankings = topsis(matrix, weights, impacts)

# Display results
result_df = pd.DataFrame({'Score': scores, 'Rank': rankings})
result_df = pd.concat([data, result_df], axis=1)

print(result_df)
