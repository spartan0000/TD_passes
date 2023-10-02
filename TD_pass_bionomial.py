import pandas as pd
import nfl_data_py as nfl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

seasons = [2019,2021,2022,2023]
df1 = nfl.import_pbp_data(seasons)

df2 = df1[df1['season_type'] == 'REG']
df2 = df1[df1['play_type'] == 'pass']
df2 = df1[df1['two_point_attempt'] == False ]

#weekly statistics for passers by game
#game date is included in the groupby line to get each game data as a unique row

df_agg = df2.groupby(['passer', 'season', 'week'], as_index = False).agg({'pass_attempt':'sum', 'complete_pass':'sum','air_yards':'sum', 'passing_yards':'sum', 'pass_touchdown':'sum', 'total_line':'max', 'spread_line':'max'})
df_agg = df_agg[df_agg['pass_attempt'] > 5]

#create columns for completion percentage as well as the implied total for each game
df_agg['comp_pct'] = df_agg['complete_pass']/df_agg['pass_attempt']
df_agg['off_implied_total'] = (df_agg['total_line'] - df_agg['spread_line'])/2

#create df of medians of all the above values for each passer


qb_median = df_agg.groupby(['passer'], as_index = False)[['pass_attempt', 'complete_pass', 'comp_pct', 'air_yards', 'passing_yards', 'pass_touchdown', 'off_implied_total']].median()
print(qb_median)


qb_median = qb_median.rename(columns = {'pass_attempt':'pass_att_med', 'complete_pass':'comp_pass_med', 'comp_pct':'comp_pct_med', 'air_yards':'air_yards_med', 'passing_yards':'passing_yards_med', 'pass_touchdown':'passTD_med', 'off_implied_total':'off_imp_tot_med'})

print(qb_median)

#do the same for means
qb_mean = df_agg.groupby(['passer'], as_index = False)[['pass_attempt', 'complete_pass', 'comp_pct', 'air_yards', 'passing_yards', 'pass_touchdown', 'off_implied_total']].mean()
print(qb_mean)

qb_mean = qb_mean.rename(columns = {'pass_attempt':'pass_att_mean', 'complete_pass':'comp_pass_mean', 'comp_pct':'comp_pct_mean', 'air_yards':'air_yards_mean', 'passing_yards':'passing_yards_mean', 'pass_touchdown':'passTD_mean', 'off_implied_total':'off_imp_tot_mean'})

print(qb_mean)

#merge the two df together
#reorder the columns so that the mean and medians of each stat are next to eachother

df_med_mean = qb_mean.merge(qb_median, how = 'inner', left_on = 'passer', right_on = 'passer')

df_med_mean['td_pct'] = df_med_mean['passTD_mean']/df_med_mean['pass_att_mean']
cols = ['passer', 'pass_att_mean', 'pass_att_med', 'comp_pass_mean', 'comp_pass_med', 'comp_pct_mean', 'comp_pct_med', 'air_yards_mean', 'air_yards_med', 'passing_yards_mean', 'passing_yards_med', 'passTD_mean', 'passTD_med', 'td_pct','off_imp_tot_med', 'off_imp_tot_mean']
df_med_mean = df_med_mean[cols]

from scipy.stats import binom
startingQB = ['D.Ridder',
             'J.Dobbs',
             'B.Young',
             'J.Fields',
             'D.Prescott',
             'J.Goff',
             'J.Love',
             'M.Stafford',
             'K.Cousins',
             'D.Carr',
             'D.Jones',
             'J.Hurts',
             'B.Purdy',
             'G.Smith',
             'B.Mayfield',
             'S.Howell',
             'L.Jackson',
             'J.Allen',
             'J.Burrow',
             'D.Watson',
             'R.Wilson',
             'C.Stroud',
             'A.Richardson',
             'T.Lawrence',
             'P.Mahomes',
             'J.Garoppolo',
             'J.Herbert',
             'T.Tagovailoa',
             'M.Jones',
             'Z.Wilson',
             'K.Pickett',
             'R.Tannehill',]

#create a df with only the 32 starting QBs
#pull only data on current starting QBs
df_start_QB = (df_med_mean[df_med_mean['passer'].isin(startingQB)])

#round pass attempt mean to whole number so that we can use it in the binomial
df_start_QB['pass_att_round'] = round(df_start_QB['pass_att_mean'])

#try binomial distribution using the pass attempts as the number of trials and td rate as the p
#round pass attempts to get a whole number

#reset the index to be able to iterate through the DF

df_start_QB = df_start_QB.reset_index(drop = True)

#create new columns for result of bionimal distribution predictions
for i in range(len(df_start_QB)):
    df_start_QB['0 TD'] = 0
    df_start_QB['1 TD'] = 0
    df_start_QB['2 TD'] = 0
    df_start_QB['3 TD'] = 0
    df_start_QB['4 TD'] = 0
    df_start_QB['5 TD'] = 0

#then iterate and fill the columns with results

for i in range(len(df_start_QB)):
    n = df_start_QB['pass_att_round'][i]
    p = df_start_QB['td_pct'][i]
    
    df_start_QB['0 TD'][i] = binom.pmf(0,n,p)
    df_start_QB['1 TD'][i] = binom.pmf(1,n,p)
    df_start_QB['2 TD'][i] = binom.pmf(2,n,p)
    df_start_QB['3 TD'][i] = binom.pmf(3,n,p)
    df_start_QB['4 TD'][i] = binom.pmf(4,n,p)
    df_start_QB['5 TD'][i] = binom.pmf(5,n,p)
