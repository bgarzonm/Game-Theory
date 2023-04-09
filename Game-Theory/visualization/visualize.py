import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("..")
from models.GameAnalysis import GameAnalysis


# import datasets from features folder

df_A = pd.read_csv("../features/playerAcomplete.csv")
df_B = pd.read_csv("../features/playerBcomplete.csv")

del df_A['ciclo']
del df_B['ciclo']
# Create an instance of the GameAnalysis class
game_analysis = GameAnalysis(df_A, df_B)

# Call the boxplot method for both players dataframe
game_analysis.boxplot(df_A)
game_analysis.boxplot(df_B)


#Call the correlation_matrix method for both players dataframe
game_analysis.correlation_matrix(df_A)
game_analysis.correlation_matrix(df_A)


# Call the analisis method for both players dataframe
game_analysis.analisis(df_A)
game_analysis.analisis(df_B)


# Call the cluster_analysis method for both players dataframe
game_analysis.cluster_analysis(df_A)
game_analysis.cluster_analysis(df_B)

# Call the kmeans method for both players dataframe
game_analysis.kmeans(df_A, 3)
game_analysis.kmeans(df_B,3)