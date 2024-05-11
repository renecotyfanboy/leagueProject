import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd


def numpyro_model(markov_model, observed_data):
    """
    Function that is used as a model in NumPyro to perform inference on the Discrete Markov Chain model.

    Parameters:
        markov_model (DMCModel): The Discrete Markov Chain model to use.
        observed_data (jnp.array): The observed data to use for inference.
    """

    if not markov_model.is_bernoulli:
        proba = numpyro.sample('proba',
                               dist.Uniform(low=jnp.zeros(2 ** markov_model.n), high=jnp.ones(2 ** markov_model.n)))
    else:
        proba = numpyro.sample('proba', dist.Uniform(low=0, high=1)) * jnp.ones(2 ** markov_model.n)

    transition_matrix = markov_model.build_transition_matrix(proba)

    def transition_fn(_, x):
        return tfd.Categorical(probs=transition_matrix[x])

    encoded_history = np.apply_along_axis(markov_model.binary_serie_to_categorical, 1, observed_data)

    likelihood_dist = tfd.MarkovChain(
        initial_state_prior=tfd.Categorical(probs=markov_model.uniform_prior),
        transition_fn=transition_fn,
        num_steps=encoded_history.shape[1]
    )

    numpyro.sample('likelihood', likelihood_dist, obs=encoded_history)