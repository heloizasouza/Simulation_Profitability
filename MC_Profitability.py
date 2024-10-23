# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:57:43 2024

@author: 08560273409
"""

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

# Parâmetros de entrada
n_years = 10  # Número de anos
n_scenarios = 1000  # Número de cenários de Monte Carlo

# Parâmetros dos ativos (exemplo: renda fixa, renda variável, exterior, estruturado)
mean_fixed_income = 0.05  # Média de rentabilidade da renda fixa
std_fixed_income = 0.04  # Desvio padrão da renda fixa

mean_equity = 0.08  # Média de rentabilidade da renda variável
std_equity = 0.15  # Desvio padrão da renda variável

mean_international = 0.07  # Média de rentabilidade no exterior
std_international = 0.12  # Desvio padrão no exterior

mean_structured = 0.06  # Média de rentabilidade estruturada
std_structured = 0.05  # Desvio padrão estruturado

# Ponderações
weights = np.array([0.90, 0.05, 0.04, 0.01])  # Renda fixa, renda variável, exterior, estruturado
mean_returns = np.array([mean_fixed_income, mean_equity, mean_international, mean_structured])
std_devs = np.array([std_fixed_income, std_equity, std_international, std_structured])

# Cálculo da média e desvio padrão ponderados
weighted_mean = np.sum(weights * mean_returns)
weighted_std = np.sqrt(np.sum((weights * std_devs) ** 2))  # Assumindo independência

# Simulação de Monte Carlo
np.random.seed(42)  # Para reprodutibilidade
scenarios = np.random.normal(weighted_mean, weighted_std, (n_scenarios, n_years))

# Cálculo da rentabilidade cumulativa
cumulative_returns = (1 + scenarios).cumprod(axis=1)

# Resultados: rentabilidade final após 10 anos
final_returns = cumulative_returns[:, -1]

# Cálculo da média e percentis
mean_final_return = np.mean(final_returns)
percentiles = np.percentile(final_returns, [25, 50, 75])  # 25%, 50%, 75%

# Exibição dos resultados
print(f'Média da rentabilidade final: {mean_final_return:.2f}')
print(f'Percentis de rentabilidade final: 25%: {percentiles[0]:.2f}, 50%: {percentiles[1]:.2f}, 75%: {percentiles[2]:.2f}')

# Gráfico da distribuição dos retornos finais
plt.figure(figsize=(10, 6))
plt.hist(final_returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribuição dos Retornos Finais após 10 anos')
plt.xlabel('Retorno Final')
plt.ylabel('Frequência')
plt.axvline(mean_final_return, color='red', linestyle='dashed', linewidth=1, label='Média')
plt.axvline(percentiles[1], color='yellow', linestyle='dashed', linewidth=1, label='Mediana (50%)')
plt.legend()
plt.show()
