import polars as pl
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
