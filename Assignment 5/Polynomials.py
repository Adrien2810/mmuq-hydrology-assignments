'''Perturbing and reaclibrating'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NSim = 10
xs = 430.5

# Read CSV file
csv_file = r"c:\MMUQ\Assignment 4\data\time_series___24163005.csv"

# discharge over depth
df = pd.read_csv(csv_file, sep=";", index_col=False)
plt.scatter(df['ddho__ref'], df['diso__ref'],
            color='blue', label='Data points')
df.sort_values(by='ddho__ref', inplace=True)

# Splitting data
first_intervall = df[df['ddho__ref'] <= xs]
second_intervall = df[df['ddho__ref'] >= xs]

# Polynomial fit
d = np.linspace(df['ddho__ref'].min(),
                df['ddho__ref'].max(), 1000)


fit_first = np.polyfit(first_intervall['ddho__ref'],
                       first_intervall['diso__ref'], deg=6)
print(fit_first)
a_first, b_first, c_first, d_first, e_first, f_first, g_first = fit_first
Q_first = a_first * d**6 + b_first * d**5 + c_first * d**4 + \
    d_first * d**3 + e_first * d**2 + f_first * d + g_first
plt.plot(d, Q_first, color='red', label='1. Polynomial')

fit_second = np.polyfit(second_intervall['ddho__ref'],
                        second_intervall['diso__ref'], deg=6)
print(fit_second)
Q_second = fit_second[0] * d**6 + fit_second[1] * d**5 + fit_second[2] * d**4 + \
    fit_second[3] * d**3 + fit_second[4] * \
    d**2 + fit_second[5] * d + fit_second[6]
plt.plot(d, Q_second, color='green', label='2. Polynomial')

# Smoothing the polynomials
p1 = np.poly1d(fit_first)
p2 = np.poly1d(fit_second)

dp1 = np.polyder(p1)
dp2 = np.polyder(p2)

den = dp2(xs)
if np.isclose(den, 0.0, atol=1e-12):
    raise ValueError(
        "Derivative at the joining point is zero, cannot recalibrate.")

a = dp1(xs) / den
c = p1(xs) - a * p2(xs)


def Q(x):
    x = np.asarray(x, dtype=float)
    return np.where(x <= xs, p1(x), a * p2(x) + c)


Qd = Q(d)


# Plotting
plt.plot(d, Qd, label="Piecewise polynomial", color="orange")
plt.axvline(xs, linestyle="--")
plt.xlim(0, 500)
plt.ylim(df['diso__ref'].min()*0.9, df['diso__ref'].max()*1.1)
plt.xlabel('Depth')
plt.ylabel('Discharge')
plt.legend()
plt.show()
