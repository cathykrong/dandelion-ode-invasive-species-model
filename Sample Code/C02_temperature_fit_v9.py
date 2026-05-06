import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import pandas as pd
from scipy.integrate import odeint
import matplotlib.dates as mdates
import datetime as dt

def sin_model(t, A, omega, phi, offset):
    """Sinusoidal model."""
    return A * np.sin(omega * t + phi) + offset

def fit_sin_to_data(t, data):
    """Fit sinusoidal model to data."""
    # Provide a better initial guess based on data characteristics
    initial_guess = [np.max(data) - np.min(data), 2 * np.pi / 365, -np.pi/2, np.mean(data)]

    # Fit the model to the data
    params, covariance = curve_fit(sin_model, t, data, p0=initial_guess)

    # Extract the fitted parameters
    A, omega, phi, offset = params

    return A, omega, phi, offset, covariance

# Function for linear interpolation
def linear_interpolation(x_val, x_data, y_data):
    if x_val <= x_data[0]:
        return y_data[0]
    elif x_val >= x_data[-1]:
        return y_data[-1]
    else:
        # Find the index of the interval containing x_val
        index = np.searchsorted(x_data, x_val)
        
        # Calculate the interpolation
        x0, x1 = x_data[index - 1], x_data[index]
        y0, y1 = y_data[index - 1], y_data[index]
        
    return y0 + (y1 - y0) * (x_val - x0) / (x1 - x0)

def odes(x, t):
    global A_fit, omega_fit, phi_fit, offset_fit
    global ARH_fit, omegaRH_fit, phiRH_fit, offsetRH_fit
    # parameters
    temperature = A_fit * np.sin(omega_fit * t + phi_fit) + offset_fit
    RH = (ARH_fit * np.sin(omegaRH_fit * t + phiRH_fit) + offsetRH_fit)/100.0
    
    a1 = para1(temperature,RH)
    a2 = para2(temperature,RH)
    a3 = para3(temperature,RH)
    a4 = para4(temperature,RH)
    a5 = para5(t,temperature,RH)
    a6 = para6(temperature,RH)
    # a7 = para7(temperature,RH)
    # a8 = para8(t,temperature,RH)

    # assign each ODE to a vector element
    S = x[0]
    F = x[1]
    P = x[2]

    # define each ODE
    dSdt = a1*P + a2*S
    dFdt = a3*S + a4*F #+ a7*(P+F)*F
    dPdt = a5*F + a6*P #+ a8*(P+F)*F

    return [dSdt, dFdt, dPdt]

def para1(temperature,RH):
    Gsi = 3.0
    alpha = 250
    if ((temperature <= 35.0) and (temperature >= 5.0)) : #
        x = 1.0
    else:
        x = 0.0
    Fg = gemrate(temperature)/100.0
    para = x*alpha*Fg*RH/Gsi
    return para

def para2(temperature,RH):
    # temperature = A * np.sin(omega * t + phi) + offset
    Gs = gemtime(temperature)
    mus = 0.35
    Fd = devrate(temperature)/100.0
    para = - (Fd/Gs + mus)*RH
    return para

def para3(temperature,RH):
    Gs = gemtime(temperature)
    Fd = devrate(temperature)/100.0
    para = Fd*RH/Gs
    return para

def para4(temperature,RH):
    Gf = devtime(temperature)
    muf = 0.15 #0.25
    r = 1 #1-NSO(t)/K*alpha
    Fl = lndrate(temperature)/100.0
    para = - (Fl/Gf *r + muf)*RH
    return para

def para5(t,temperature,RH):
    Gf = devtime(temperature)
    Fl = lndrate(temperature)/100.0
    # K = 1.0
    # alpha = 40
    # r = 1.0 - Nso(t)/(K*alpha)
    if (t >= 80.0) :
        y = 1.0
    else:
        y = 0.0	
    para = Fl*y*RH/Gf
    return para

def para6(temperature,RH):
    Lp = 10.5
    mup = 0.01 #0.25
    para = - (1.0/Lp + mup)*RH
    return para

# def para7(temperature,RH):
#     Gf = devtime(temperature)
#     # muf = 0.01 #0.25
#     rd = 0.15/2        #single dandelion radius
#     Ad = np.pi * rd*rd  #single dandelion area
#     Atotal = 10000     #research area
#     K = 1            #environmental carrying capacity
#     # r = 1 #1-NSO(t)/K*alpha
#     Fl = lndrate(temperature)/100.0
#     para = (Fl/Gf)*RH*(Ad/K*Atotal)
#     return para

# def para8(t,temperature,RH):
#     Gf = devtime(temperature)
#     # muf = 0.01 #0.25
#     rd = 0.5/2
#     Ad = np.pi * rd*rd
#     Atotal = 10000
#     K = 1
#     Fl = lndrate(temperature)/100.0
#     if (t >= 90.0) :
#         y = 1.0
#     else:
#         y = 0.0	
#     para = - Fl*y*RH*(Ad/K*Atotal)/Gf
#     return para

def gemrate(temperature):
    # if temperature < 5.0:
    #     rate = 0.
    # elif temperature>=5 and  temperature< 10:
    #     rate = 5. * temperature - 20.
    # elif temperature>=10 and temperature < 15:
    #     rate = 9. * temperature - 60.
    # elif temperature>=15 and temperature< 25:
    #     rate = 1. * temperature + 60.
    # elif temperature>=25 and temperature < 35:
    #     rate = -7.5 * temperature + 272.5
    # elif temperature >= 35 and temperature < 40:
    #     rate = -2. * temperature + 80.        
    # else:
    #     rate = 0.
    y_germ = [0, 5, 30, 75, 85, 10, 0]
    x = [0, 5, 10, 15, 25, 35, 40]
    rate = linear_interpolation(temperature, x, y_germ)

    return rate

def devrate(temperature):
    y_dev = [0, 5, 60, 25, 0]
    x = [0, 5, 25, 35, 40]

    rate = linear_interpolation(temperature, x, y_dev)
    
    return rate


def lndrate(temperature):
    y_land = [0,25,55,65,55,5,0]
    x = [5,10,20,25,35,40,60]

    rate = linear_interpolation(temperature, x, y_land)
     
    return rate

def gemtime(temperature):
    # if temperature < 5.0:
    #     rate = 0.
    # elif temperature>=5 and  temperature< 10:
    #     rate = 5. * temperature - 20.
    # elif temperature>=10 and temperature < 15:
    #     rate = 9. * temperature - 60.
    # elif temperature>=15 and temperature< 25:
    #     rate = 1. * temperature + 60.
    # elif temperature>=25 and temperature < 35:
    #     rate = -7.5 * temperature + 272.5
    # elif temperature >= 35 and temperature < 40:
    #     rate = -2. * temperature + 80.        
    # else:
    #     rate = 0.
    y_germ = [360, 15, 5, 10, 360]
    x = [0, 10, 20, 30, 40]
    rate = linear_interpolation(temperature, x, y_germ)

    return rate

def devtime(temperature):
    y_dev = [360, 10, 5, 8, 18, 360]
    x = [0, 10, 20, 30, 40, 50]

    rate = linear_interpolation(temperature, x, y_dev)
    
    return rate


# def lndrate(temperature):
#     y_land = [0,25,55,65,55,5,0]
#     x = [5,10,20,25,35,40,60]

#     rate = linear_interpolation(temperature, x, y_land)
     
#     return rate
def annot_max(x,y,posx=50,posy=-1000,txnote='max', ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= txnote + ": {:.0f} (".format(ymax) + dt.datetime.utcfromtimestamp(xmax* 86400.0).strftime("%m-%d)")
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="data",
              arrowprops=arrowprops, bbox=bbox_props, ha="left", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(xmax+posx,ymax+posy), **kw)

def annot_min(x,y,posx=50,posy=-1000,txnote='min', ax=None):
    xmax = x[np.argmin(y)]
    ymax = y.min()
    text= txnote + ": {:.0f} (".format(ymax) + dt.datetime.utcfromtimestamp(xmax* 86400.0).strftime("%m-%d)")
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="data",
              arrowprops=arrowprops, bbox=bbox_props, ha="left", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(xmax+posx,ymax+posy), **kw)

def annot(x,y,idx=30,posx=0,posy=0,txnote='', ax=None):
    xmax = x[idx]
    ymax = y[idx]
    text= txnote + "{:.0f} (".format(ymax) + dt.datetime.utcfromtimestamp(xmax* 86400.0).strftime("%m-%d)")
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="data",
              arrowprops=arrowprops, bbox=bbox_props, ha="left", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(xmax+posx,ymax+posy), **kw)

def plotfitcrv(temperature, days, fitted_curve, str1, r_squared):

    # Plot the original data, fitted curve, and residuals
    plt.figure(figsize=(20, 12))
    plt.subplot(2, 1, 1)
    plt.plot(days, temperature, label='Original Data')
    plt.plot(days, fitted_curve, label='Fitted Curve', linestyle='-',linewidth=3)
    date_form = mdates.DateFormatter("%m-%d")
    plt.gca().xaxis.set_major_formatter(date_form)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))    
    plt.errorbar(days, temperature, yerr=1, fmt='o', alpha=0.5, label='Error Bars')
    plt.title('Daily ' + str1 + ' Fitting with Sinusoidal Model')
    plt.xlabel('Day of the Year')
    plt.ylabel(str1)
    plt.legend()

    residuals = temperature - fitted_curve
    plt.subplot(2, 1, 2)
    plt.plot(days, residuals, label='Residuals')
    plt.axhline(0, color='r', linestyle='--', label='Zero Residuals')
    plt.title('Residuals Plot - ' + f'R square: {r_squared:.4f}')
              # str(r_squared))
    plt.xlabel('Day of the Year')
    plt.ylabel('Residuals')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Replace 'your_file.csv' with the actual path to your CSV file
# file_path = 'E:/02 Cathy/09 math contest/10 math modelling/2023 high school/01 problem A/02 Data/temperature.csv'
# file_path = 'C:/_LOCALdata/rongf/Personal/Cathy/HiMCM2013/temperature.csv'
file_path = 'C:/_LOCALdata/rongf/Personal/Cathy/HiMCM2013/tempRH.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
# Print all column names
# print(df.columns.tolist())
# Replace 'column_name' with the name of the column you want to extract
# column_name = 'Mean Temp (°C)'
column_name = ['date', 'max_relative_humidity_v', 'min_relative_humidity_v', 'max_temperature_v', 'min_temperature_v']
# day_column_name = 'Date/Time'
df1 = df.copy()
df1.drop(df1.columns.difference(column_name), 1, inplace=True)
# df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
mindf=dt.datetime.strptime(min(df1['date'].to_numpy()), '%Y-%m-%d')
df1['avg_temperature'] = df1.apply(lambda row: (row.max_temperature_v + row.min_temperature_v)/2, axis=1)
df1['avg_relative_humidity'] = df1.apply(lambda row: (row.max_relative_humidity_v + row.min_relative_humidity_v)/2, axis=1)
df1['date1'] = df1.apply(lambda row: dt.datetime.strptime(row.date, '%Y-%m-%d'), axis=1)
df1['dayidx'] = df1.apply(lambda row: (row.date1 - mindf).days, axis=1)

# Extract the column as a NumPy array
temperature = df1['avg_temperature'].to_numpy()
RH = df1['avg_relative_humidity'].to_numpy()
# def main():
# Generate synthetic data
# days = np.arange(365)
days = df1['dayidx'].to_numpy()
date = df1['date1'].to_numpy()
# Extract the column as a NumPy array
# temperature = df[column_name].to_numpy()

# Fit the sinusoidal model to the data
A_fit, omega_fit, phi_fit, offset_fit, covariance = fit_sin_to_data(days, temperature)

# Generate the fitted curve
fitted_curve = sin_model(days, A_fit, omega_fit, phi_fit, offset_fit)

# Calculate R-squared
r_squared = r2_score(temperature, fitted_curve)

# plotfitcrv(temperature, days, fitted_curve, str1, r_squared):
plotfitcrv(temperature, date, fitted_curve,'Temperature (°C)', r_squared)

# Display the fitted parameters and their uncertainties
print('Temperature fitting:')
print(f'Amplitude (A): {A_fit:.2f} ± {np.sqrt(covariance[0, 0]):.2f}')
print(f'Angular Frequency (omega): {omega_fit:.2f} ± {np.sqrt(covariance[1, 1]):.2f}')
print(f'Phase (phi): {phi_fit:.2f} ± {np.sqrt(covariance[2, 2]):.2f}')
print(f'Offset (B): {offset_fit:.2f} ± {np.sqrt(covariance[3, 3]):.2f}')
print(f'Offset (B): {offset_fit:.2f} ± {np.sqrt(covariance[3, 3]):.2f}')
print(f'R square: {r_squared:.4f}')

# print("R square",r_squared)

ARH_fit, omegaRH_fit, phiRH_fit, offsetRH_fit, covarianceRH = fit_sin_to_data(days, RH)

# Generate the fitted curve
fitted_curve = sin_model(days, ARH_fit, omegaRH_fit, phiRH_fit, offsetRH_fit)

# Calculate R-squared
r_squared = r2_score(RH, fitted_curve)

print('RH fitting:')
print(f'Amplitude (A): {ARH_fit:.2f} ± {np.sqrt(covarianceRH[0, 0]):.2f}')
print(f'Angular Frequency (omega): {omegaRH_fit:.2f} ± {np.sqrt(covarianceRH[1, 1]):.2f}')
print(f'Phase (phi): {phiRH_fit:.2f} ± {np.sqrt(covarianceRH[2, 2]):.2f}')
print(f'Offset (B): {offsetRH_fit:.2f} ± {np.sqrt(covarianceRH[3, 3]):.2f}')
print(f'R square: {r_squared:.4f}')

plotfitcrv(RH, date, fitted_curve, 'Relative Humidity (%)', r_squared)

desirdt = dt.datetime.fromisoformat('2022-05-01')
addphi = (desirdt-mindf).days*2*np.pi/365
phi_fit = phi_fit + addphi
phiRH_fit = phiRH_fit + addphi
totalday = 365
maxdt = desirdt + dt.timedelta(days=totalday)
daysaxis = mdates.drange(desirdt,maxdt,dt.timedelta(days=1))

# initial conditions
x0 = [0,0,1]

# declare a time vector (time window)
t = np.linspace(0,totalday,totalday)
x = odeint(odes,x0,t)

y1 = x[:,0]
y2 = x[:,1]
y3 = x[:,2]


# plot results
plt.figure(figsize=(20, 12))
plt.subplot(3, 1, 1)
# ax = fig.add_subplot(2, 1, 1)
# ax.set_yscale('log')
plt.semilogy(daysaxis,y1,'r-',linewidth=2,label='Seed')
date_form = mdates.DateFormatter("%m-%d")
plt.gca().xaxis.set_major_formatter(date_form)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1)) 
annot_max(daysaxis[0:60],y1[0:60],posx=0,posy=+80000,txnote='localmax')   
annot_max(daysaxis,y1)   
annot_min(daysaxis[60:90],y1[60:90],posx=11,posy=-1.1,txnote='localmin')   
annot(daysaxis,y1,idx=31,posx=0,posy=+5000)
annot(daysaxis,y1,idx=61,posx=0,posy=+5000)
annot(daysaxis,y1,idx=92,posx=10)
annot(daysaxis,y1,idx=184,posx=10)
plt.gca().set_ylim([1e-5, 0])
# plt.figure(figsize=(10, 6))
plt.ylabel('Seed')
plt.legend()
plt.subplot(3, 1, 2)
plt.semilogy(daysaxis,y2,'b--',linewidth=2,label='Flower')
date_form = mdates.DateFormatter("%m-%d")
plt.gca().xaxis.set_major_formatter(date_form)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))    
annot_max(daysaxis[0:60],y2[0:60],posx=0,posy=+2000,txnote='localmax')   
annot_max(daysaxis,y2)   
annot_min(daysaxis[60:90],y2[60:90],posx=11,posy=-0.5,txnote='localmin')   
# plt.figure(figsize=(10, 6))
plt.ylabel('Flower')
plt.legend()
plt.subplot(3, 1, 3)
plt.semilogy(daysaxis,y3,'g:',linewidth=2,label='Puffball')
date_form = mdates.DateFormatter("%m-%d")
plt.gca().xaxis.set_major_formatter(date_form)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))    
annot_max(daysaxis[0:60],y3[0:60],posx=0,posy=+1000,txnote='localmax')   
annot_max(daysaxis,y3)   
annot_min(daysaxis[60:90],y3[60:90],posx=11,posy=0,txnote='localmin')   
plt.xlabel('Date')
plt.ylabel('Puffball')
plt.legend()
plt.show()
    
    
    
    
    
# if __name__ == '__main__':
#     main()



