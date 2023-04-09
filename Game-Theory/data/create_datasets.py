import pandas as pd

# Definimos la ruta donde se encuentran los archivos
ruta = 'C:/Users/esnei/Desktop/game_theory/'

# Creamos una lista vacía para almacenar los dataframes
dfs = []

# Iteramos sobre los archivos usando un ciclo for
for ciclo in range(1, 21):
    nombre_archivo = f'simulacion{str(ciclo)}B.csv'
    ruta_archivo = ruta + nombre_archivo
    df = pd.read_csv(ruta_archivo)
    # Si es el primer DataFrame, establecerlo como el DataFrame final
    if len(dfs) == 0:
        dfs.append(df)
    # Si no es el primer DataFrame, combinarlo con el DataFrame final anterior
    else:
        dfs.append(pd.merge(dfs[-1], df, on='ciclo'))

# Mostramos el resultado final
df_final_B = dfs[-1]
df_final_B.to_csv('C:/Users/esnei/Desktop/game_theory/playerBcomplete.csv', index=False)

