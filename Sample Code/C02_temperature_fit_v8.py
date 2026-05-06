
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import pandas as pd
from scipy.integrate import odeint

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

def odes(x, t):
    global A_fit, omega_fit, phi_fit, offset_fit
    # parameters
    temperature = A_fit * np.sin(omega_fit * t + phi_fit) + offset_fit
    a1 = para1(temperature)
    a2 = para2(temperature)
    a3 = para3(temperature)
    a4 = para4(temperature)
    a5 = para5(t,temperature)
    a6 = para6(temperature)

    # assign each ODE to a vector element
    S = x[0]
    F = x[1]
    P = x[2]

    # define each ODE
    dSdt = a1*P + a2*S
    dFdt = a3*S + a4*F
    dPdt = a5*F + a6*P

    return [dSdt, dFdt, dPdt]

def para1(temperature):
    Gsi = 3.0
    alpha = 250
    if ((temperature <= 25.0) and (temperature >= 10.0)) :
        x = 1.0
    else:
        x = 0.0
    Fg = gemrate(temperature)/100.0
    para = x*alpha*Fg/Gsi
    return para

def para2(temperature):
    # temperature = A * np.sin(omega * t + phi) + offset
    Gs = 10.0
    mus = 0.25
    Fd = devrate(temperature)/100.0
    para = - (Fd/Gs + mus)
    return para

def para3(temperature):
    # temperature = A * np.sin(omega * t + phi) + offset
    Gs = 10.0
    Fd = devrate(temperature)/100.0
    para = Fd/Gs
    return para

def para4(temperature):
    # temperature = A * np.sin(omega * t + phi) + offset
    Gf = 10.0
    muf = 0.0 #0.25
    r = 1 #1-NSO(t)/K*alpha
    Fl = lndrate(temperature)/100.0
    para = - (Fl/Gf *r + muf)
    return para

def para5(t,temperature):
    # temperature = A * np.sin(omega * t + phi) + offset
    Gf = 10.0
    Fl = lndrate(temperature)/100.0
    r = 1.0 #1-Nso(t)/(K*alpha)
    if (t >= 90.0) :
        y = 1.0
    else:
        y = 0.0
    para = Fl*y*r/Gf
    return para

def para6(temperature):
    # temperature = A * np.sin(omega * t + phi) + offset
    Lp = 10.0
    mup = 0.0 #0.25
    para = - (1.0/Lp + mup)
    return para

def gemrate(temperature):
    if temperature < 5.0:
        rate = 0.
    elif temperature>=5 and  temperature< 10:
        rate = 5. * temperature - 20.
    elif temperature>=10 and temperature < 15:
        rate = 9. * temperature - 60.
    elif temperature>=15 and temperature< 25:
        rate = 1. * temperature + 60.
    elif temperature>=25 and temperature < 35:
        rate = -7.5 * temperature + 272.5
    elif temperature >= 35 and temperature < 40:
        rate = -2. * temperature + 80.        
    else:
        rate = 0.
    
    return rate

def devrate(temperature):
    if temperature < 5.0:
        rate = 0.
    elif temperature>=5 and  temperature< 10:
        rate = 5. * temperature - 20.
    elif temperature>=10 and temperature < 15:
        rate = 9. * temperature - 60.
    elif temperature>=15 and temperature< 25:
        rate = 1. * temperature + 60.
    elif temperature>=25 and temperature < 35:
        rate = -7.5 * temperature + 272.5
    elif temperature >= 35 and temperature < 40:
        rate = -2. * temperature + 80.        
    else:
        rate = 0.
    rate = 0.5*rate
    
    return rate

def lndrate(temperature):
    if temperature < 5.0:
        rate = 0.
    elif temperature>=5 and  temperature< 10:
        rate = 5. * temperature - 20.
    elif temperature>=10 and temperature < 15:
        rate = 9. * temperature - 60.
    elif temperature>=15 and temperature< 25:
        rate = 1. * temperature + 60.
    elif temperature>=25 and temperature < 35:
        rate = -7.5 * temperature + 272.5
    elif temperature >= 35 and temperature < 40:
        rate = -2. * temperature + 80.        
    else:
        rate = 0.
    
    return rate


# Replace 'your_file.csv' with the actual path to your CSV file
file_path = 'E:/02 Cathy/09 math contest/10 math modelling/2023 high school/01 problem A/02 Data/temperature.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
# Print all column names
print(df.columns.tolist())
# Replace 'column_name' with the name of the column you want to extract
column_name = 'Mean Temp (°C)'

# Extract the column as a NumPy array
temperature = df[column_name].to_numpy()




# def main():
# Generate synthetic data
days = np.arange(365)
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

print("R square",r_squared)


# initial conditions
x0 = [0,0,1]

# declare a time vector (time window)
t = np.linspace(0,365,365)
x = odeint(odes,x0,t)

y1 = x[:,0]
y2 = x[:,1]
y3 = x[:,2]


# plot results
plt.plot(t,y1,'r-',linewidth=2,label='Seed')
plt.plot(t,y2,'b--',linewidth=2,label='Flower')
plt.plot(t,y3,'g:',linewidth=2,label='Puffball')
plt.xlabel('time')
plt.ylabel('y(t)')
plt.legend()
plt.show()
    
    
    
    
    
# if __name__ == '__main__':
#     main()



