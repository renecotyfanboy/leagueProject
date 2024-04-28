This section is dedicated to the dataset used in the project. It is available 
[on HuggingFace](https://huggingface.co/datasets/renecotyfanboy/leagueData)

# Data gathering 

All the data gathered here comes from the Riot Games API. I used a personal API key and the awesome 
[`pulsefire`](https://github.com/iann838/pulsefire) Python package. 

<div class="annotate" markdown>
1. I collected all the SoloQ players in the `EUW` server in the evening of 9 April 2024. 
2. I randomly picked 100 players in each division, with at least 200 games played in the corresponding split. I assume 
that $\sim$ 100 games are enough for the summoner to be near its true rank. (1) So I gathered the last 100 games played
in SoloQ for each of these players.
3. I collected the data for each of these games, and added each player's individual statistics provided by the 
[MATCH-V5](https://developer.riotgames.com/apis#match-v5) API as an individual row in the dataset.
</div>

1. ![La source](https://risibank.fr/cache/medias/0/14/1420/142061/full.png){ align=left }

# Cookbook

Here I'll provide a few examples of how to use the dataset using [`polars`](https://docs.pola.rs/).

## Load the dataset

```python
import polars as pl

df = pl.read_csv("league_dataframe.csv")

print(df)
```

## Find the history of a player using its `puuid`

``` python
puuid = 'your_puuid' # (1)!
historic_of_random_player = df.filter(
    puuid=puuid, is_in_reference_sample=True # (2)!
    ).sort(by='gameStartTimestamp') 

```

1. `b3fhGxFuV-hCD3B5Vvj9nrD--8YwlFACxvAIox_sOq2aNUtmkcsmem8NFufjdZd79L49I9spnh7LQg` is a valid `puuid`.
2. `is_in_reference_sample=True` indicates that we only keep the match history collected initially. Sometimes, the player
can appear in the others matches, but for history analysis it would include matches that were not initially selected.

## Build the win/loss curve of a player

``` python
import matplotlib.pyplot as plt
```