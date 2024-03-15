import pandas as pd
import glob
import os

dataFolder = ""

def load_data(dataFolder, listSeriesGrouped):
    # list all csv files under data folder
    all_files = glob.glob(os.path.join(dataFolder, "*.csv"))

    # list columns to only needed
    fields = ["din_instante","id_subsistema","id_estado","nom_tipousina","nom_tipocombustivel","val_geracao"]
    
    # import data to pandas
    series = pd.concat((pd.read_csv(file, sep=";", skipinitialspace=True, usecols=fields) for file in all_files), ignore_index=True)
    
    # convert din_instante to date with monthly periods
    series["din_instante"] = pd.to_datetime(series["din_instante"]).dt.to_period("M")

    # # group by month and year
    series_grouped = series.groupby(listSeriesGrouped, as_index=False)["val_geracao"].sum()
    # series_grouped = series.groupby(["din_instante","nom_tipocombustivel","nom_tipousina","id_estado","id_subsistema"], as_index=False)["val_geracao"].sum()

        
    #return grouped concatenated series to main
    return series_grouped