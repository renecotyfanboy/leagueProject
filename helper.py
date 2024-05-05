import numpy as np
import jax.numpy as jnp
import polars as pl
from tensorflow_probability.substrates.jax import distributions as tfd
from jax.random import PRNGKey
from bidict import bidict
from itertools import product
from scipy.linalg import eig
from datasets import load_dataset


def get_dataset(select_columns = None):
    dataset = load_dataset("renecotyfanboy/leagueData", split="train")

    if select_columns is None:
        dataset.to_polars()
    
    return dataset.select_columns(select_columns).to_polars()



def get_tier_sorted():

    tier_list = ['IRON', 'BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND', 'MASTER', 'GRANDMASTER',
                 'CHALLENGER']
    division_list = ['I', 'II', 'III', 'IV'][::-1]
    tier_with_sub = []

    for tier in tier_list:

        if tier not in ['MASTER', 'GRANDMASTER', 'CHALLENGER']:
            for division in division_list:
                tier_with_sub.append(f'{tier}_{division}')

    return tier_with_sub + ['MASTER', 'GRANDMASTER', 'CHALLENGER']

def get_history_dict(df_path="league_dataframe.csv"):

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
            history[elo][puuid] = list(loc_history.sort(by='gameStartTimestamp')['win'])

    return history


class StateTools:

    def __init__(self, n):
        self.is_bernoulli = n == 0
        self.n = max(n, 1)

    @property
    def ref_table(self):
        """Generates an iterator over all binary tuples of size n."""
        return bidict({state: i for i, state in enumerate(product([0, 1], repeat=self.n))})

    @property
    def uniform_prior(self):
        return jnp.ones((2 ** self.n,)) / 2 ** self.n

    def get_states(self):

        states = []

        for i in range(2 ** self.n):
            states.append(self.ref_table.inv[i])

        return states

    def build_transition_matrix(self, probs):

        transition_matrix = jnp.zeros((2 ** self.n, 2 ** self.n))

        for i in range(2 ** self.n):
            state_i = self.ref_table.inv[i]
            i_to_win = self.ref_table[state_i[1:self.n] + (1,)]
            i_to_lose = self.ref_table[state_i[1:self.n] + (0,)]
            win_prob = probs.at[i].get()
            transition_matrix = transition_matrix.at[i, i_to_win].set(win_prob)
            transition_matrix = transition_matrix.at[i, i_to_lose].set(1 - win_prob)

        return transition_matrix

    def binary_serie_to_categorical(self, serie):

        new_serie = np.empty((len(serie) - self.n + 1), dtype=int)

        for i in range(len(new_serie)):
            state = tuple(serie[i:i + self.n])
            new_serie[i] = self.ref_table[state]

        return new_serie

    def categorical_serie_to_binary(self, serie):

        new_serie = np.empty((len(serie) + self.n - 1), dtype=int)

        for i in range(len(serie)):
            new_serie[i:i + self.n] = self.ref_table.inv[int(serie[i])]

        return new_serie

    def build_process(self, steps, probs=None):

        if probs is None:
            transition_matrix = self.build_transition_matrix(jnp.ones(2 ** self.n) * 0.5)
        else:
            transition_matrix = self.build_transition_matrix(probs)

        # win_prob = numpyro.sample(f"{state_i}_to_win", dist.Uniform(0, 1))

        def transition_fn(_, x):
            return tfd.Categorical(probs=transition_matrix[x])

        return tfd.MarkovChain(
            initial_state_prior=tfd.Categorical(probs=self.uniform_prior),
            transition_fn=transition_fn,
            num_steps=steps
        )

    def stationary_distribution(self, probs):
        transition_matrix = self.build_transition_matrix(probs)
        eig_val, eig_ref = eig(transition_matrix, left=True, right=False)
        stat_distribution = eig_ref[:, np.argwhere(np.isclose(eig_val, 1))[0]]
        return np.abs(stat_distribution / stat_distribution.sum())

    def to_mermaid(self, probs):

        transition_matrix = self.build_transition_matrix(probs)
        states = self.get_states()

        graph_str = "graph LR \n"

        for i, state_i in enumerate(states):
            for j, state_j in enumerate(states):
                prob = transition_matrix[i, j]

                if prob > 0:
                    line_str = '\t'
                    line_str += f'{"".join(["W" if i else "L" for i in state_i])} --> |{int(prob * 100)}%| {"".join(["W" if i else "L" for i in state_j])}'
                    graph_str += line_str + '\n'

        return graph_str
