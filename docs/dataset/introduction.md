# Introduction

This section is dedicated to the dataset used in the project. It is available 
[on HuggingFace](https://huggingface.co/datasets/renecotyfanboy/leagueData)

## Data gathering 

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

Let's explore a bit our dataset. In the next plot, I show the winrate of players in each division. The winrate is 
computed using the history list of each player.

```plotly
{"file_path": "dataset/assets/winrate_over_division.json"}
```