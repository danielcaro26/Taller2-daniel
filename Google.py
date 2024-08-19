
# Funciones numéricas adicionales
import numpy as np

# Lectura de datos y manejo de Data-sets
import pandas as pd

# Datos
import yfinance as yf

# Gráficos
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import seaborn as sns

#analisis tecnico
import mplfinance as mpf

# Probabilidad y estadística
import math
from scipy.stats import norm, chi2, jarque_bera,shapiro
from scipy.optimize import brentq
from scipy import stats

# ARCH model
from arch import arch_model

# Engel test for ARCH effects
from statsmodels.stats.diagnostic import het_arch

# Descargamos datos historicos de la acción de Google
df = yf.download('GOOGL', start='2005-01-01')

#Hacer un gráfico de velas japonesas 
mpf.plot(df,type='candle', volume=True,figratio=(19,8),style='yahoo',title='Google')

# Calcular retornos logaritmicos en una nueva columna.
df['Log Returns'] = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(1))
df['Log Returns'][0] = 0

# Calcular retornos anuales
LogReturns = np.mean(df["Log Returns"])*252
print(LogReturns)

#Graficar retornos logaritmicos
plt.figure(figsize=(15,8))
plt.plot(df['Log Returns'], color = 'red')
plt.title('Retornos Logarítmicos de Apple [$AAPL]')
plt.xlabel('Fecha')
plt.show()

# Calculamos la volatilidad diaria con los retornos logaritmicos.
vol_d = np.std(df['Log Returns'])

# Anualizamos la volatilidad diaria.
vol_a = vol_d * np.sqrt(252)

print("Volatilidad diaria: {:.4f} %".format(100*vol_d))
print("Volatilidad anualizada: {:.4f} %".format(100*vol_a))

# calculate the rolling standard deviation using the Risk Metrics model
window = 252
lambda_param = 0.94
variance = [df['Log Returns'][:window].var()]
for i in range(window, len(df['Log Returns'])):
    variance.append(lambda_param * variance[-1] + (1 - lambda_param) * df['Log Returns'][i-window:i].var())
std = pd.Series(np.sqrt(variance)*np.sqrt(252))

# plot the results
import matplotlib.pyplot as plt
plt.plot(std)
plt.title("Risk Metrics Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.show()

def black_scholes_call(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    N = norm.cdf
    return S*N(d1) - K*np.exp(-r*T)*N(d2)

def implied_volatility(S, K, r, T, market_price, sigma_guess=0.30):
    """
    Calculate the implied volatility of a European call option using the Black-Scholes formula.

    Parameters:
    S (float): underlying asset price
    K (float): option strike price
    r (float): risk-free interest rate
    T (float): time to maturity in years
    market_price (float): observed market price of the option
    sigma_guess (float, optional): initial guess for the volatility

    Returns:
    float: the implied volatility
    """
    def f(sigma):
        return black_scholes_call(S, K, r, sigma, T) - market_price
    return brentq(f, 0.0001, 1, xtol=1e-10, rtol=1e-10, maxiter=100000)