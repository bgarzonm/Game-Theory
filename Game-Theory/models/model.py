import pandas as pd
import numpy as np
import re

df_A = pd.read_csv(r"C:\Users\esnei\Desktop\game_theory\playerAcomplete.csv")
df_B = pd.read_csv(r"C:\Users\esnei\Desktop\game_theory\playerBcomplete.csv")

df_A.info()

def analisis(df):
    # Número de veces que se eligió la estrategia de halcón o paloma
    conteo_halcon = df.filter(regex='^NC_x').sum().sum() # suma de todos los conteos de no cooperación para el agente X
    conteo_paloma = df.filter(regex='^NC_y').sum().sum() # suma de todos los conteos de no cooperación para el agente Y
    print('-'*50)
    print(f'La estrategia de Halcón fue elegida {conteo_halcon} veces.')
    print('-'*50)
    print(f'La estrategia de Paloma fue elegida {conteo_paloma} veces.')
    print('-'*50)


    col_pattern = r"^(C|NC)_"
    relevant_cols = [col for col in df.columns if re.match(col_pattern, col)]
    
    # Calculate the payoffs for each agent
    df["Pago_X"] = df[relevant_cols].apply(lambda row: max(row["NC_x"], row["C_x"]), axis=1)
    df["Pago_Y"] = df[relevant_cols].apply(lambda row: max(row["NC_y"], row["C_y"]), axis=1)
    
    # Calculate the number of times each agent won and the number of ties
    agent_x_wins = (df["Pago_X"] > df["Pago_Y"]).sum()
    agent_y_wins = (df["Pago_X"] < df["Pago_Y"]).sum()
    ties = (df["Pago_X"] == df["Pago_Y"]).sum()
    
    print(f'El agente X ganó {agent_x_wins} veces.')
    print('-'*50)
    print(f'El agente Y ganó {agent_y_wins} veces.')
    print('-'*50)
    print(f'Hubo {ties} empates.')
    print('-'*50)

    # Promedio del número de agentes que utilizan cada estrategia después de 10,000 iteraciones
    ultimas_10000 = df.iloc[-10000:]
    num_juegos = 20
    prom_C_X = ((ultimas_10000.filter(regex='^C_x').sum().sum()) / (10000* num_juegos))
    prom_NC_X = (ultimas_10000.filter(regex='^NC_x').sum().sum()) / (10000 * num_juegos)
    prom_C_Y = (ultimas_10000.filter(regex='^C_y').sum().sum()) / (10000 * num_juegos)
    prom_NC_Y = (ultimas_10000.filter(regex='^NC_y').sum().sum()) / (10000 * num_juegos)

    print(f'Después de 10,000 iteraciones,\nla probabilidad promedio de\ncooperación para el agente X es {prom_C_X}.')
    print('-'*50)
    print(f'Después de 10,000 iteraciones,\nla probabilidad promedio de\nno cooperación para el agente X es {prom_NC_X}.')
    print('-'*50)
    print(f'Después de 10,000 iteraciones,\nla probabilidad promedio de\ncooperación para el agente Y es {prom_C_Y}.')
    print('-'*50)
    print(f'Después de 10,000 iteraciones,\nla probabilidad promedio de\nno cooperación para el agente Y es {prom_NC_Y}.')


analisis(df_A)
analisis(df_B)



import seaborn as sns
import matplotlib.pyplot as plt

# calculate the correlation matrix
corr_matrix = df.corr()

# plot the correlation matrix using a heatmap
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)

# set plot parameters
plt.title("Correlation Matrix")
plt.xticks(rotation=45)
plt.tight_layout()

# show the plot
plt.show()
