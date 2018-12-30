import warnings
import pandas as pd
import math
warnings.filterwarnings("ignore");
import numpy as np

# Candidate prediction
# ------------------------
candidate = "Ted Cruz"
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
states_predict = ["Georgia", "Ohio", "New York"]
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
votes = np.asanyarray(votes)
facts = np.asanyarray(facts)
# convert to numpy array to predict
votes_to_predict = np.asarray(votes_to_predict)
facts_vtp = np.asanyarray(facts_vtp)
# normalize values 0..1
# normalize votes
max_vote = np.max(votes)
if np.max(votes_to_predict) > max_vote:
    max_vote = np.max(votes_to_predict)
votes = votes / max_vote
votes_to_predict = votes_to_predict / max_vote
# normalize facts
print("XD")
