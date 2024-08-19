
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