import polars as pl
import numpy as np
from datasets import load_dataset


def get_dataset(select_columns=None) -> pl.DataFrame:
    """
    Load the dataset from the Hugging Face Hub and return it as a Polars DataFrame.

    Parameters:
        select_columns (list): The columns to select from the dataset. If None, all columns are selected.

    Returns:
        pl.DataFrame: The dataset as a Polars DataFrame.
    """

    dataset = load_dataset("renecotyfanboy/leagueData", split="train")

    if select_columns is None:
        return dataset.to_polars()

    return dataset.select_columns(select_columns).to_polars()


def get_tier_sorted() -> list:
    """
    Return a list of all the tiers in League of Legends, sorted from the lowest to the highest.

    Returns:
        list: The list of all tiers in League of Legends, sorted from the lowest to the highest.
    """

    tier_list = ['IRON', 'BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND', 'MASTER', 'GRANDMASTER',
                 'CHALLENGER']
    division_list = ['I', 'II', 'III', 'IV'][::-1]
    tier_with_sub = []

    for tier in tier_list:

        if tier not in ['MASTER', 'GRANDMASTER', 'CHALLENGER']:
            for division in division_list:
                tier_with_sub.append(f'{tier}_{division}')

    return tier_with_sub + ['MASTER', 'GRANDMASTER', 'CHALLENGER']


def get_tier_batch() -> list:
    """
    Return batches of tiers in League of Legends, sorted from the lowest to the highest.
    """

    tier_list = ['IRON', 'BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND', 'MASTER', 'GRANDMASTER',
                 'CHALLENGER']
    division_list = ['I', 'II', 'III', 'IV'][::-1]
    

    for tier in tier_list:

        tier_with_sub = []

        if tier not in ['MASTER', 'GRANDMASTER', 'CHALLENGER']:
            for division in division_list:
                tier_with_sub.append(f'{tier}_{division}')
                
            yield tier_with_sub
        
        else:
            yield [tier]


def get_history_dict():
    """
    Return a two level dictionary containing the history of all players in the reference sample.
    Accessed by elo and then by puuid.
    """

    columns = ['elo', 'puuid', 'gameStartTimestamp', 'is_in_reference_sample', 'win']
    df = get_dataset(columns)
    unique_elo = df.filter(is_in_reference_sample=True)['elo'].unique()

    history = {}

    for elo in unique_elo:
        loc_df = df.filter(elo=elo, is_in_reference_sample=True)
        history[elo] = {}
        unique_puuid = loc_df['puuid'].unique()

        for puuid in unique_puuid:
            loc_history = loc_df.filter(puuid=puuid)
            history[elo][puuid] = np.asarray(loc_history.sort(by='gameStartTimestamp')['win'])

    return history