# Ignore the seaborn warnings.
import warnings
from pprint import pprint

warnings.filterwarnings("ignore");

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

# kandydat dla ktorego bedzie tworzona siec
candidate = "Ted Cruz"
# import danych z pliku .csv
data = pd.read_csv('data/primary_results.csv')
candidate = data.loc[data.candidate == candidate]
details = pd.read_csv('data/county_facts.csv')
# uzywane kategorie
facts_dictionary = ["area_name", "state", "PST045214", "SEX255214", "POP645213", "EDU635213", "EDU685213", "VET605213",
                    "HSG445213", "INC910213", "SBO001207", "SBO315207", "SBO215207", "SBO015207", "LND110210"]
columns = [col for col in details.columns if col in facts_dictionary]
columns.append('state')
details = details[columns]

# candidat_info : ( state, county, votes)
candidat_info = []
for state, county, votes in zip(candidate.state, candidate.county, candidate.fraction_votes):
    candidat_info.append((state, county, votes))
# county detail : ( state, name, edu, edu, hsg, inc, lnd, pop, pst, vet, sex, sbo, sbo, sbo, sbo )
county_details = []
for ar, st, edu1, edu2, hsg, inc, lnd, pop, pst, vet, sex, sbo1, sbo2, sbo3, sbo4 in zip(details.area_name,
    details.state, details.EDU635213, details.EDU685213, details.HSG445213, details.INC910213, details.LND110210,
    details.POP645213, details.PST045214, details.VET605213, details.SEX255214, details.SBO001207, details.SBO315207,
    details.SBO215207, details.SBO015207):
    county_details.append(ar, st, edu1, edu2, hsg, inc, lnd, pop, pst, vet, sex, sbo1, sbo2, sbo3, sbo4)
print(len(candidate))

# import wynikow