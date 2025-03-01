import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from UQpy.distributions import Uniform, JointIndependent
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from UQpy.surrogates import *

#import file and extract inputs and outputs
import csv

##############################################################
# All matrices for MC & surrogates comparison

def r_squared(y_true, y_pred):
    """Calculate the Coefficient of Determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def rmse(y_true, y_pred):
    """Calculate the Root Mean Square Error (RMSE)."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """Calculate the Mean Absolute Error (MAE)."""
    return np.mean(np.abs(y_true - y_pred))

def mae_std(y_true, y_pred):
    """Calculate the Mean Absolute Error (MAE)."""
    return np.std(np.abs(y_true - y_pred))

def relative_error(y_true, y_pred):
    """Calculate the Relative Error (RE)."""
    return np.mean((y_true - y_pred) / y_true)

def nrmse(y_true, y_pred):
    """Calculate the Normalized Root Mean Square Error (NRMSE)."""
    rmse_value = rmse(y_true, y_pred)
    return rmse_value / (np.max(y_true) - np.min(y_true))

def max_error(y_true, y_pred):
    """Calculate the Maximum Error."""
    return np.max(np.abs(y_true - y_pred))

##############################################################
#import MC data

# Define the input CSV file path
#input_file = 'processed_data_30Yr_3D_MC.csv'
input_file = 'sorted_extracted_data_30Yr_3D_MC.csv'

# Lists to store the column values
haughton_mc = []
pitcaple_mc = []
shift_mc = []
area_mc = []

# Open and read the CSV file
with open(input_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        haughton_mc.append(float(row[1]))
        pitcaple_mc.append(float(row[2]))
        shift_mc.append(float(row[3]))
        area_mc.append(float(row[4]))

# Pair values in haughton_values, pitcaple_values, shift_values
xmc = [[haughton, pitcaple, shift] for haughton, pitcaple, shift in zip(haughton_mc, pitcaple_mc, shift_mc)]
xmc = np.array(xmc)
ymc = np.array(area_mc)

# Sort ymc values and create corresponding indices
ymc_sorted = np.sort(ymc)
indices = np.arange(len(ymc_sorted))

"""# Plot ymc values against their indices
plt.figure(figsize=(10, 6))
plt.plot(indices, ymc_sorted, marker='o', linestyle='-', color='blue')

# Add labels and title
plt.xlabel('Index (Sorted)')
plt.ylabel('ymc Values')
plt.title('Plot of Sorted ymc Values')
plt.grid(True)
plt.show()"""

##############################################################
# Section 0.9: Gauss Quad Order 5
##############################################################

input_file = 'sorted_extracted_data_30Yr_3D_halton_27.csv'

# Lists to store the column values
haughton_values = []
pitcaple_values = []
shift_values = []
areas = []

# Open and read the CSV file
with open(input_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        haughton_values.append(float(row[1]))
        pitcaple_values.append(float(row[2]))
        shift_values.append(float(row[3]))
        areas.append(float(row[4]))

# Pair values in haughton_values and pitcaple_values
xgauss_o5 = [[haughton, pitcaple, shift] for haughton, pitcaple, shift in zip(haughton_values, pitcaple_values, shift_values)]
xgauss_o5 = np.array(xgauss_o5)
ygauss_o5 = np.array(areas)

########################################################
#Section 1: Kriging
########################################################

import shutil

from UQpy import PythonModel
from UQpy.surrogates.gaussian_process.regression_models import ConstantRegression
from UQpy.surrogates.gaussian_process.regression_models import LinearRegression
from UQpy.sampling import RectangularStrata
from UQpy.sampling import TrueStratifiedSampling
from UQpy.run_model.RunModel import RunModel
from UQpy.distributions import Uniform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from UQpy.surrogates import GaussianProcessRegression
from UQpy.surrogates.gaussian_process.regression_models.QuadraticRegression import QuadraticRegression

# Create a distribution object.

# %%
from UQpy.utilities import Matern, RBF

marginals = [Uniform(loc=208.23, scale=427.90), Uniform(loc=47.37, scale=112.88),Uniform(-43200,43200)]

# %% md
#
# Create a strata object.

regression_model = QuadraticRegression()
#regression_model = LinearRegression()
kernel = Matern(nu=0.5)
#kernel = RBF()

from UQpy.utilities.MinimizeOptimizer import MinimizeOptimizer

optimizer = MinimizeOptimizer(method="L-BFGS-B")

K = GaussianProcessRegression(regression_model=regression_model, optimizer=optimizer,
                              kernel=kernel,
                              optimizations_number=20, noise=True,
                              hyperparameters=[80, 30, 80000, 10000, 10000])

np.random.seed(1)

K.fit(samples=xgauss_o5, values=ygauss_o5)
krig_predict27 = K.predict(xmc)


print('Process Variance: ', K.hyperparameters[3])
print('Output Noise: ', K.hyperparameters[4])


# Function to plot approx2 against ymc
def plot_approx_vs_ymc(ymc, approx, title):
    """
    Generate a scatter plot to compare ymc and approximations.

    Parameters:
    ymc: Actual data (ymc)
    approx: Approximated data (e.g., approx2)
    title: Title of the plot (string)
    """
    plt.figure(figsize=(8, 6))

    # Scatter plot of approx2 against ymc
    plt.scatter(ymc, approx, color='blue', label='Approximated', alpha=0.7)

    # Plot reference line (ideal approximation)
    plt.plot(ymc, ymc, color='red', linestyle='--', label='Ideal')

    plt.title(title)
    plt.xlabel('Actual FMC')
    plt.ylabel('Approximated Values')
    plt.legend()
    plt.grid(True)
    plt.show()


# Call the plot function for approx2
plot_approx_vs_ymc(ymc, krig_predict27, 'Approx vs Actual')

print(krig_predict27.shape)

# Flatten the arrays if necessary, as we're working with 1D arrays for plotting
krig_predict27_flat = krig_predict27.flatten()
ymc_flat = ymc.flatten()

# Sort the krig_predict27 array and sort ymc in the same order
sorted_indices1 = np.argsort(krig_predict27_flat)
sorted_indices2 = np.argsort(ymc)
krig_predict27_sorted = krig_predict27_flat[sorted_indices1]
ymc_sorted = ymc_flat[sorted_indices2]

# Create a single plot with two lines
plt.figure(figsize=(10, 6))

# Plot sorted krig_predict27
plt.plot(krig_predict27_sorted, label='Sorted GRP Prediction', color='b')

# Plot sorted ymc
plt.plot(ymc_sorted, label='Sorted FMC', color='r')

# Add labels and title
plt.title('Comparison of Sorted GPR Predictions and Sorted FMC Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)

# Add a legend to differentiate between the two lines
plt.legend()

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Function to calculate R-squared
def r_squared(y_true, y_pred):
    #Calculate the Coefficient of Determination (R²).
    ss_res = np.sum((y_true - y_pred) ** 2)
    print(ss_res)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    print(ss_tot)
    return 1 - (ss_res / ss_tot)

# Calculate R-squared for all approximations
#r2_approx0 = r_squared(ymc, krig_predict27)
from sklearn.metrics import r2_score
r2_approx0 = r2_score(ymc, krig_predict27)
print(r2_approx0)


# Store the results in a list
r2_values = [r2_approx0]  # Assuming r2_approx0 is already defined
approx_labels = ['PCE approx']

# Plot the R-squared values in a bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(approx_labels, r2_values, color='skyblue')

# Add labels and title
plt.xlabel('Approximations')
plt.ylabel('R-squared')
plt.title('R-squared of the approximation')
plt.ylim(0, 1)  # Set the y-axis limit between 0 and 1
plt.grid(True, axis='y')

# Add R-squared values on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}',
             ha='center', va='bottom')

# Display the bar chart
plt.show()

####### violin plot
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming y_true is your true data from full MC run and y_pred is the corresponding estimates from a surrogate
y_true = ymc_flat
y_pred = krig_predict27_flat

# Combine the datasets into a single DataFrame
data = pd.DataFrame({
    'Value': np.concatenate([y_true, y_pred]),
    'Type': ['True MC Data'] * len(y_true) + ['Surrogate Estimate'] * len(y_pred),
    'Category': ['Data'] * (len(y_true) + len(y_pred))  # Use a single category to plot them together
})

# Create a violin plot with split sides
plt.figure(figsize=(10, 6))

sns.violinplot(x='Category', y='Value', hue='Type', data=data, palette="Set2", split=True, inner="quart", gap=0.01)

# Customize the plot
plt.title('Comparison of True MC Data vs Surrogate Estimates', fontsize=16)
plt.ylabel('Values', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

# Add some visual enhancements
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

nrmse_approx0 = nrmse(ymc, krig_predict27)
print(nrmse_approx0)

# Additional Imports
from scipy.stats import entropy  # For KL divergence


# Calculate the error between the max values
def max_value_error(y_true, y_pred):
    return  np.max(y_pred) - np.max(y_true)


# Calculate the error between the min values
def min_value_error(y_true, y_pred):
    return  np.min(y_pred) - np.min(y_true)


# KL Divergence
def kl_divergence(y_true, y_pred, bins=30):
    """
    Calculate the Kullback-Leibler (KL) divergence between the distributions of y_true and y_pred.
    We'll use histograms to estimate the distributions.

    Parameters:
    - y_true: Ground truth data
    - y_pred: Predicted data
    - bins: Number of bins for histogram estimation (default is 30)
    """
    # Estimate the probability distribution of y_true and y_pred using histograms
    hist_true, bin_edges = np.histogram(y_true, bins=bins, density=True)
    hist_pred, _ = np.histogram(y_pred, bins=bin_edges, density=True)

    # To avoid division by zero, add a small epsilon to the histograms
    hist_true += 1e-10
    hist_pred += 1e-10

    # Calculate KL divergence
    kl_div = entropy(hist_true, hist_pred)
    return kl_div


# Compute Errors and Metrics
max_error_value = max_value_error(ymc, krig_predict27)
min_error_value = min_value_error(ymc, krig_predict27)
rmse_value = rmse(ymc, krig_predict27)
kl_divergence_value = kl_divergence(ymc, krig_predict27)

# Print results
print(f"Max Value Error: {max_error_value}")
print(f"Min Value Error: {min_error_value}")
print(f"RMSE: {rmse_value}")
print(f"KL Divergence: {kl_divergence_value}")

# Store the results in a dictionary
error_metrics = {
    'R-squared': r2_approx0,
    'Max Value Error': max_error_value,
    'Min Value Error': min_error_value,
    'RMSE': rmse_value,
    'NRSME': nrmse_approx0,
    'KL Divergence': kl_divergence_value
}

# Display the results
for metric, value in error_metrics.items():
    print(f"{metric}: {value:.4f}")

import pandas as pd

# Combine the datasets into a single DataFrame for violin plots
data = pd.DataFrame({
    'Value': np.concatenate([y_true, y_pred]),
    'Type': ['True MC Data'] * len(y_true) + ['Surrogate Estimate'] * len(y_pred),
    'Category': ['Data'] * (len(y_true) + len(y_pred))  # Use a single category to plot them together
})

# Save the violin plot data to a CSV file
output_file = 'violin_plot_data_GPR_3D_halton.csv'
#data.to_csv(output_file, index=False)

#print(f"Violin plot data saved to {output_file}")

