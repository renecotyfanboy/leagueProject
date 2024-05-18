import jax.random
import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.random import PRNGKey
from tensorflow_probability.substrates.jax import distributions as tfd
from bidict import bidict
from itertools import product
from scipy.linalg import eig


class DTMCModel:
    """
    Class used to define a Discrete Time Markov Chain for modelling game history.
    """

    def __init__(self, n):
        """
        Build a Discrete Markov Chain model.

        Parameters:
            n (int): The number of game in memory. If n = 0, the model is a Bernoulli process.
        """
        self.is_bernoulli = n == 0
        self.n = max(n, 1)

    @property
    def ref_table(self):
        """
        Mapping between binary and categorical representation of states.
        """
        return bidict({state: i for i, state in enumerate(product([0, 1], repeat=self.n))})

    @property
    def uniform_prior(self):
        """
        Define a uniform prior over the states.

        Returns:
            jnp.array: The uniform prior.
        """
        return jnp.ones((2 ** self.n,)) / 2 ** self.n

    def get_states(self):
        """
        Get all the states of the model, which are the combinations of n binary values.

        Returns:
            list: The list of all states.
        """

        states = []

        for i in range(2 ** self.n):
            states.append(self.ref_table.inv[i])

        return states

    def build_transition_matrix(self, probs):
        """
        Build the transition matrix of the model.

        Parameters:
            probs (jnp.array): The probabilities of winning at each state.

        Returns:
            jnp.array: The transition matrix.
        """

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
        """
        Convert a binary representation of states to a categorical representation.

        Parameters:
            serie (np.array): The binary serie to convert.

        Returns:
            np.array: The categorical serie.
        """

        new_serie = np.empty((len(serie) - self.n + 1), dtype=int)

        for i in range(len(new_serie)):
            state = tuple(serie[i:i + self.n])
            new_serie[i] = self.ref_table[state]

        return new_serie

    def categorical_serie_to_binary(self, serie):
        """
        Convert a categorical representation of states to a binary representation.

        Parameters:
            serie (np.array): The categorical serie to convert.

        Returns:
            np.array: The binary serie.
        """

        new_serie = np.empty((len(serie) + self.n - 1), dtype=int)

        for i in range(len(serie)):
            new_serie[i:i + self.n] = self.ref_table.inv[int(serie[i])]

        return new_serie

    def build_process(self, steps, probs=None):
        """
        Build a Markov Chain process with the given number of steps.

        Parameters:
            steps (int): The number of steps of the process.
            probs (jnp.array): The probabilities of winning at each state.

        Returns:
            tfd.MarkovChain: The Markov Chain process.
        """

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
        """
        Compute the stationary distribution of the model.

        Parameters:
            probs (jnp.array): The probabilities of winning at each state.

        Returns:
            jnp.array: The stationary distribution.
        """

        transition_matrix = self.build_transition_matrix(probs)
        eig_val, eig_ref = eig(transition_matrix, left=True, right=False)
        stat_distribution = eig_ref[:, np.argwhere(np.isclose(eig_val, 1))[0]]
        return np.abs(stat_distribution / stat_distribution.sum())

    def to_mermaid(self, probs):
        """
        Convert the model to a Mermaid graph.

        Parameters:
            probs (jnp.array): The probabilities of winning at each state.

        Returns:
            str: The Mermaid graph.
        """

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


def generate_obvious_loser_q(number_of_games=85, number_of_players=200, key=PRNGKey(42)):
    """
    Generate mock history of players using the obvious loserQ model.

    Parameters:
        number_of_games (int): The number of games in the mock history.
        number_of_players (int): The number of players.
        key (PRNGKey): The key to generate the mock history.
    """

    markov_util_ref = DTMCModel(4)

    probs = jnp.empty((2 ** 4))
    importance = 0.5

    probs_keys = {
        0.: 0.5 - 0.375 * importance,
        0.25: 0.5 - 0.125 * importance,
        0.5: 0.5,
        0.75: 0.5 + 0.125 * importance,
        1.: 0.5 + 0.375 * importance
    }

    for i, state in enumerate(markov_util_ref.get_states()):
        probs = probs.at[i].set(probs_keys[sum(state) / 4])

    mock_history_encoded = markov_util_ref.build_process(number_of_games - 3, probs=probs).sample(number_of_players, seed=key)
    mock_history = np.apply_along_axis(markov_util_ref.categorical_serie_to_binary, 1, mock_history_encoded)

    return mock_history


def generate_coinflip_history(number_of_games=85, number_of_players=200, key=PRNGKey(42)):
    """
    Generate mock history of players using the coinflip model.

    Parameters:
        number_of_games (int): The number of games in the mock history.
        number_of_players (int): The number of players.
        key (PRNGKey): The key to generate the mock history.
    """

    return np.asarray(jax.random.bernoulli(key, 0.5, shape=(number_of_players, number_of_games)))


def generate_nasty_loser_q(number_of_games=85, number_of_players=200, key=PRNGKey(42), return_importance=False):
    """
    Generate mock history of players using the nasty loserQ model.

    Parameters:
        number_of_games (int): The number of games in the mock history.
        number_of_players (int): The number of players.
        key (PRNGKey): The key to generate the mock history.
        return_importance (bool): Whether to return the importance of the loserQ for each player.
    """
    markov = DTMCModel(4)
    keys = jax.random.split(key, 2)

    importance = dist.Beta(1.2, 10).sample(keys[0], sample_shape=(number_of_players,))

    def single_history(key, importance, number_of_games):
        probs = jnp.empty((2 ** 4))

        probs_keys = {0.: 0.5 - 0.375 * importance,
                      0.25: 0.5 - 0.125 * importance,
                      0.5: 0.5,
                      0.75: 0.5 + 0.125 * importance,
                      1.: 0.5 + 0.375 * importance}

        for i, state in enumerate(markov.get_states()):
            probs = probs.at[i].set(probs_keys[sum(state) / 4])

        return markov.build_process(number_of_games -3, probs=probs).sample(1, seed=key)[0]

    keys = jax.random.split(keys[1], number_of_players)
    history_categorical = np.asarray(
        jax.vmap(lambda key, importance: single_history(key, importance, number_of_games)
                 )(keys, importance))

    history = np.apply_along_axis(markov.categorical_serie_to_binary, 1, history_categorical)

    if return_importance:
        return history, importance

    return history


def generate_simulation(number_of_games=85, number_of_players=200, key=PRNGKey(42), return_importance=False):
    """
    Generate mock history of players using the nasty loserQ model.

    Parameters:
        number_of_games (int): The number of games in the mock history.
        number_of_players (int): The number of players.
        key (PRNGKey): The key to generate the mock history.
        return_importance (bool): Whether to return the importance of the loserQ for each player.
    """
    markov = DTMCModel(4)
    keys = jax.random.split(key, 2)

    importance = dist.Beta(1.2, 10).sample(keys[0], sample_shape=(number_of_players,))

    def single_history(key, importance, number_of_games):
        probs = jnp.empty((2 ** 4))

        probs_keys = {0.: 0.5 - 0.375 * importance,
                      0.25: 0.5 - 0.125 * importance,
                      0.5: 0.5,
                      0.75: 0.5 + 0.125 * importance,
                      1.: 0.5 + 0.375 * importance}

        for i, state in enumerate(markov.get_states()):
            probs = probs.at[i].set(probs_keys[sum(state) / 4])

        return markov.build_process(number_of_games -3, probs=probs).sample(1, seed=key)[0]

    keys = jax.random.split(keys[1], number_of_players)
    history_categorical = np.asarray(
        jax.vmap(lambda key, importance: single_history(key, importance, number_of_games)
                 )(keys, importance))

    history = np.apply_along_axis(markov.categorical_serie_to_binary, 1, history_categorical)

    if return_importance:
        return history, importance

    return history
