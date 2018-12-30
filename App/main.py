import warnings
from pprint import pprint
import pandas as pd
import math
warnings.filterwarnings("ignore")
import numpy as np
from App.NeuralNetwork import *

# Candidate prediction
# ------------------------
candidate = "Donald Trump"
party = "Republican"
# ------------------------
# import from .csv files
data = pd.read_csv('data/primary_results.csv')
candidate = data.loc[data.candidate == candidate]
details = pd.read_csv('data/county_facts.csv')
# used details
details = details[['area_name', 'state_abbreviation', 'area_name', 'PST045214', 'SEX255214', 'POP645213', 'EDU635213',
                   'EDU685213', 'VET605213', 'HSG445213', 'INC910213', 'SBO001207', 'SBO315207', 'SBO215207',
                   'SBO015207',
                   'LND110210']]
# candidat_info : ( state, county, votes)
candidat_info = []
for state, county, votes in zip(candidate.state, candidate.county, candidate.votes):
    candidat_info.append((state, county, votes))
# county detail : ( state, name, edu, edu, hsg, inc, lnd, pop, pst, vet, sex, sbo, sbo, sbo, sbo )
stats = []
for (i, st, edu1, edu2, hsg, inc, lnd, pop, pst, vet, sex, sbo1, sbo2, sbo3, sbo4) in zip(
        details.area_name.iterrows(),
        details.state_abbreviation,
        details.EDU635213, details.EDU685213, details.HSG445213, details.INC910213,
        details.LND110210, details.POP645213, details.PST045214, details.VET605213, details.SEX255214,
        details.SBO001207,
        details.SBO315207, details.SBO215207, details.SBO015207):
    stats.append((str.split(str(i[1]), '\n')[0].replace('area_name ', '').lstrip(),
                  st, edu1, edu2, hsg, inc, lnd, pop, pst, vet, sex, sbo1, sbo2, sbo3, sbo4))

# prepare test set
# county = []
states = ["California", "Texas", "Florida", "Iowa", "New Jersey", "Virginia"]
# get sum of votes from state
sum_of_votes = []
for state in states:
    sum_state = 0
    v = data[(data.state == state) & (data.party == party)].votes
    sum_of_votes.append(pd.Series(v).sum())
# get votes from state
votes = []
for state in states:
    sum_state = 0
    for result in candidat_info:
        if result[0] == state:
            sum_state = sum_state + result[2]
    votes.append(sum_state)

# get state facts
facts = []
for state in states:
    for stat in stats:
        if state == stat[0] and math.isnan(stat[1]):
            det = stat[2:]
            facts.append(det)

# to predict
states_predict = ["Georgia", "Ohio", "New York", "Alaska"]
# get sum of votes from state to predict
sum_of_votes_predict = []
for state in states_predict:
    sum_state = 0
    v = data[(data.state == state) & (data.party == party)].votes
    sum_of_votes_predict.append(pd.Series(v).sum())
# get votes from state to predict
votes_to_predict = []
for state in states_predict:
    sum_state = 0
    for result in candidat_info:
        if result[0] == state:
            sum_state = sum_state + result[2]
    votes_to_predict.append(sum_state)
# get state facts to predict
facts_vtp = []
for state in states_predict:
    for stat in stats:
        if state == stat[0] and math.isnan(stat[1]):
            det = stat[2:]
            facts_vtp.append(det)
# convert to numpy array training set
votes = np.asanyarray(votes, dtype=np.float32).reshape(1, 6)
facts = np.asanyarray(facts, dtype=np.float32)
# convert to numpy array to predict
votes_to_predict = np.asarray(votes_to_predict, dtype=np.float32).reshape(1,4)
facts_vtp = np.asanyarray(facts_vtp, dtype=np.float32)
# normalize values 0..1
# normalize votes
votes = votes / sum_of_votes
votes_to_predict = votes_to_predict / sum_of_votes_predict
# normalize facts
norm = facts.max(axis=0)
norm_2 = facts_vtp.max(axis=0)
for i in range(0, len(norm)):
    if norm[i] < norm_2[i]:
        norm[i] = norm_2[i]
facts = facts / norm
facts_vtp = facts_vtp / norm

neural_network = ANN()
neural_network.train_model(facts, votes, facts_vtp, votes_to_predict)
