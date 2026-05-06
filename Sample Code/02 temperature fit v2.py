# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 08:17:23 2023

@author: Personal
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import pandas as pd

# Replace 'your_file.csv' with the actual path to your CSV file
file_path = 'E:/02 Cathy/09 math contest/10 math modelling/2023 high school/01 problem A/02 Data/temperature.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
# Print all column names
print(df.columns.tolist())
# Replace 'column_name' with the name of the column you want to extract
column_name = 'Mean Temp (°C)'

# Extract the column as a NumPy array
Temperature = df[column_name].to_numpy()



def sin_model(t, A, omega, phi, offset):
    """Sinusoidal model."""
    return A * np.sin(omega * t + phi) + offset

def fit_sin_to_data(t, data):
    """Fit sinusoidal model to data."""
    # Provide a better initial guess based on data characteristics
    initial_guess = [np.max(data) - np.min(data), 2 * np.pi / len(t), 0, np.mean(data)]

    # Fit the model to the data
    params, covariance = curve_fit(sin_model, t, data, p0=initial_guess)

    # Extract the fitted parameters
    A, omega, phi, offset = params

    return A, omega, phi, offset, covariance

def main():
    # Generate synthetic data
    
    # Extract the column as a NumPy array
    temperature = df[column_name].to_numpy()

    # Fit the sinusoidal model to the data
    A_fit, omega_fit, phi_fit, offset_fit, covariance = fit_sin_to_data(days, temperature)

    # Generate the fitted curve
    fitted_curve = sin_model(days, A_fit, omega_fit, phi_fit, offset_fit)
    
    # Calculate R-squared
    r_squared = r2_score(temperature, fitted_curve)

    # Plot the original data, fitted curve, and residuals
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(days, temperature, label='Original Data')
    plt.plot(days, fitted_curve, label='Fitted Curve', linestyle='--')
    plt.errorbar(days, temperature, yerr=1, fmt='o', alpha=0.5, label='Error Bars')
    plt.title('Daily Temperature Fitting with Sinusoidal Model')
    plt.xlabel('Day of the Year')
    plt.ylabel('Temperature (°C)')
    plt.legend()

    residuals = temperature - fitted_curve
    plt.subplot(2, 1, 2)
    plt.plot(days, residuals, label='Residuals')
    plt.axhline(0, color='r', linestyle='--', label='Zero Residuals')
    plt.title('Residuals Plot')
    plt.xlabel('Day of the Year')
    plt.ylabel('Residuals')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Display the fitted parameters and their uncertainties
    print(f'Amplitude (A): {A_fit:.2f} ± {np.sqrt(covariance[0, 0]):.2f}')
    print(f'Angular Frequency (omega): {omega_fit:.2f} ± {np.sqrt(covariance[1, 1]):.2f}')
    print(f'Phase (phi): {phi_fit:.2f} ± {np.sqrt(covariance[2, 2]):.2f}')
    print(f'Offset (B): {offset_fit:.2f} ± {np.sqrt(covariance[3, 3]):.2f}')

if __name__ == '__main__':
    main()