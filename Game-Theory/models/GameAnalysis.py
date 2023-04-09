import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class GameAnalysis:
    def __init__(self, df_A, df_B):
        self.df_A = df_A
        self.df_B = df_B
        
    def boxplot(self, df):
        fig, ax = plt.subplots(figsize=(15, 10))
        # Crear el boxplot con los datos
        boxplot = ax.boxplot(df, notch=True, patch_artist=True)
        # Definir colores para los boxes de "Coperar" y "No coperar"
        coperar_color = 'blue'
        no_coperar_color = 'orange'

        # Iterar sobre los boxes y asignarles un color según su posición en la lista de columnas
        for i, box in enumerate(boxplot['boxes']):
            if i % 2 == 0: # Si la columna es de "Coperar"
                box.set(facecolor=coperar_color)
            else: # Si la columna es de "No coperar"
                box.set(facecolor=no_coperar_color)
        # Define color order
        column_order = ['C', 'NC', 'C', 'NC','C', 'NC' ,'C','NC','C', 'NC',
                        'C', 'NC', 'C', 'NC','C', 'NC' ,'C','NC','C', 'NC',
                        'C', 'NC', 'C', 'NC','C', 'NC' ,'C','NC','C', 'NC',
                        'C', 'NC', 'C', 'NC','C', 'NC' ,'C','NC','C', 'NC']
        
        if df is self.df_A:
            # Agregar título y etiquetas de ejes
            ax.set_title(f'BoxPlot jugador A', fontsize=20)
            ax.set_xticklabels(column_order)
            # Agregar etiquetas en el gráfico
            ax.text(0.05, 0.95, 'Cooperativos', transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor=coperar_color, alpha=0.5))

            ax.text(0.05, 0.9, 'No Cooperativos', transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor=no_coperar_color, alpha=0.5))
            plt.show()
        elif df is self.df_B:
            # Agregar título y etiquetas de ejes
            ax.set_title(f'BoxPlot jugador B', fontsize=20)

            ax.set_xticklabels(column_order)
            # Agregar etiquetas en el gráfico            
            ax.text(0.05, 0.95, 'Cooperativos', transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor=coperar_color, alpha=0.5))

            ax.text(0.05, 0.9, 'No Cooperativos', transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor=no_coperar_color, alpha=0.5))
            plt.show()
            
            
    def correlation_matrix(self, df):

        corr_matrix = df.corr()
        column_order = ['C', 'NC', 'C', 'NC', 'C', 'NC' ,'C', 'NC', 'C', 'NC',
                    'C', 'NC', 'C', 'NC', 'C', 'NC' ,'C', 'NC', 'C', 'NC',
                    'C', 'NC', 'C', 'NC', 'C', 'NC' ,'C', 'NC', 'C', 'NC',
                    'C', 'NC', 'C', 'NC', 'C', 'NC' ,'C', 'NC', 'C', 'NC']

        f = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, cmap='viridis', vmin=-1, vmax=1, xticklabels=column_order, yticklabels=column_order)
        plt.title("Matriz de correlación")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


    def analisis(self,df):
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


    # -----------------------------------------------------
    # K-Means Clustering
    # -----------------------------------------------------


    def cluster_analysis(self,df):
        # Seleccionar columnas que comienzan con "C" y "NC"
        c_cols = [col for col in df.columns if col.startswith('C')]
        nc_cols = [col for col in df.columns if col.startswith('NC')]

        # Crear dataframes separados para estrategias cooperativas y no cooperativas
        c_df = df[c_cols]
        nc_df = df[nc_cols]

        # Normalizar los datos usando StandardScaler
        scaler = StandardScaler()
        c_scaled = scaler.fit_transform(c_df.T)
        nc_scaled = scaler.fit_transform(nc_df.T)

        # Convertir los numpy arrays de vuelta a dataframes de pandas
        c_scaled_df = pd.DataFrame(c_scaled.T, columns=c_cols)
        nc_scaled_df = pd.DataFrame(nc_scaled.T, columns=nc_cols)

        # Realizar agrupamiento de series de tiempo en los datos de estrategias cooperativas
        sse_c = []
        for k in range(1, 11):
            kmeans_c = KMeans(n_clusters=k, random_state=42)
            kmeans_c.fit(c_scaled_df)
            sse_c.append(kmeans_c.inertia_)

        # Graficar la curva del codo para los datos de estrategias cooperativas
        plt.plot(range(1, 11), sse_c)
        plt.title('Curva del Codo para Estrategias Cooperativas')
        plt.xlabel('Número de Agrupaciones')
        plt.ylabel('SSE')
        plt.show()

        # Realizar agrupamiento de series de tiempo en los datos de estrategias no cooperativas
        sse_nc = []
        for k in range(1, 11):
            kmeans_nc = KMeans(n_clusters=k, random_state=42)
            kmeans_nc.fit(nc_scaled_df)
            sse_nc.append(kmeans_nc.inertia_)

        # Graficar la curva del codo para los datos de estrategias no cooperativas
        plt.plot(range(1, 11), sse_nc)
        plt.title('Curva del Codo para Estrategias No Cooperativas')
        plt.xlabel('Número de Agrupaciones')
        plt.ylabel('SSE')
        plt.show()



    def kmeans(self, df, n_clusters=int):
        c_cols = [col for col in df.columns if col.startswith('C')]
        nc_cols = [col for col in df.columns if col.startswith('NC')]

        # Realizar clustering k-means en las columnas seleccionadas
        c_df = df[c_cols]
        nc_df = df[nc_cols]

        # Establecer el número de clusters
        n_clusters = n_clusters

        # Ajustar KMeans a los datos de estrategias cooperativas
        kmeans_c = KMeans(n_clusters=n_clusters)
        kmeans_c.fit(c_df)

        # Ajustar KMeans a los datos de estrategias no cooperativas
        kmeans_nc = KMeans(n_clusters=n_clusters)
        kmeans_nc.fit(nc_df)

        # Añadir etiquetas de cluster al dataframe original
        df['C_Cluster'] = kmeans_c.labels_
        df['NC_Cluster'] = kmeans_nc.labels_

        if df.equals(self.df_A):
            # Para el jugador A
            plt.title('Resultados del jugador A')
            plt.scatter(self.df_A['C_Cluster'], c_df.mean(axis=1), c='blue', label='Cooperativo')
            plt.scatter(self.df_A['NC_Cluster'] , nc_df.mean(axis=1), c='orange', label='No Cooperativo')
            plt.legend()
            plt.xlabel('Cluster')
            plt.ylabel('Pago promedio')
            plt.show()  
        elif df.equals(self.df_B):
            # Para el jugador B
            plt.title('Resultados del jugador B')
            plt.scatter(self.df_B['C_Cluster'], c_df.mean(axis=1), c='blue', label='Cooperativo')
            plt.scatter(self.df_B['NC_Cluster'] , nc_df.mean(axis=1), c='orange', label='No Cooperativo')
            plt.legend()
            plt.xlabel('Cluster')
            plt.ylabel('Pago promedio')
            plt.show()
