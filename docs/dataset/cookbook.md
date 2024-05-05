# Cookbook

Here I'll provide a few examples of how to use the dataset using [`polars`](https://docs.pola.rs/).

### Load the dataset

```python
import polars as pl

df = pl.read_csv("league_dataframe.csv")

print(df)
```

## Find the history of a player

``` python
puuid = 'your_puuid' # (1)!
historic_of_random_player = df.filter(
    puuid=puuid, is_in_reference_sample=True # (2)!
    ).sort(by='gameStartTimestamp') 

```

1. `b3fhGxFuV-hCD3B5Vvj9nrD--8YwlFACxvAIox_sOq2aNUtmkcsmem8NFufjdZd79L49I9spnh7LQg` is a valid `puuid`.
2. `is_in_reference_sample=True` indicates that we only keep the match history collected initially. Sometimes, the player
can appear in the others matches, but for history analysis it would include matches that were not initially selected.

## Build the win/loss curve

``` python
import matplotlib.pyplot as plt
```