# Cookbook

Here I'll provide a few examples of how to use the dataset using [`polars`](https://docs.pola.rs/).

## Load the dataset

Load the dataset from [`huggingface`](https://huggingface.co/datasets/renecotyfanboy/leagueData) and display all the available columns.

```python
from datasets import load_dataset

df = load_dataset("renecotyfanboy/leagueData", split="train").to_polars()
print(df.columns)
```

## Find the history of a player

```python
puuid = 'your_puuid' # (1)!
historic_of_random_player = df.filter(
    puuid=puuid, is_in_reference_sample=True # (2)!
    ).sort(by='gameStartTimestamp')
```

1. `b3fhGxFuV-hCD3B5Vvj9nrD--8YwlFACxvAIox_sOq2aNUtmkcsmem8NFufjdZd79L49I9spnh7LQg` is a valid `puuid`.
2. `is_in_reference_sample=True` indicates that we only keep the match history collected initially. Sometimes, the player
can appear in the others matches, but for history analysis it would include matches that were not initially selected.

## Lowest number of games
Remake games were removed from the dataset, so some players don't have 100 games. This is how we get the lowest number of game for a single player, which is 85.

```python
from datasets import load_dataset

columns = ['elo', 'puuid', 'gameStartTimestamp', 'is_in_reference_sample', 'win']
df = load_dataset("renecotyfanboy/leagueData", split="train").select_columns(columns).to_polars()
df = df.filter(is_in_reference_sample=True)

number_of_games = []

for puuid in df['puuid'].unique():
    player = df.filter(puuid=puuid)
    number_of_games.append(len(player.sort(by='gameStartTimestamp')['win'].to_numpy()))
    
min(number_of_games)
```

## History of the Gold III players

Display the history of Gold III players in the dataset as an image.

```python
import numpy as np 
import matplotlib.pyplot as plt 
from datasets import load_dataset

columns = ['elo', 'puuid', 'gameStartTimestamp', 'is_in_reference_sample', 'win']
df = load_dataset("renecotyfanboy/leagueData", split="train").select_columns(columns).to_polars()
df = df.filter(elo="GOLD_III", is_in_reference_sample=True)

history = []

for puuid in df['puuid'].unique():
    player = df.filter(puuid=puuid)
    history.append(player.sort(by='gameStartTimestamp')['win'].to_numpy()[-85:])
    
plt.matshow(np.asarray(history))
```

