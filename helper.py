import numpy as np
import jax.numpy as jnp
from bidict import bidict
from itertools import product
from scipy.linalg import eig


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
