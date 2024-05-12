import arviz as az
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd
from .model import DTMCModel
from numpyro.infer import MCMC, NUTS
from jax.random import PRNGKey, fold_in


def numpyro_model(markov_model, observed_data):
    """
    Function that is used as a model in NumPyro to perform inference on the Discrete Markov Chain model.

    Parameters:
        markov_model (DTMCModel): The Discrete Markov Chain model to use.
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


def fit_history_with_dmc(
    history, 
    lowest_memory=0, 
    highest_memory=6, 
    num_warmup=1000, 
    num_samples=2000, 
    num_chains=5,
    key=PRNGKey(0),
    assert_convergence=True
    ):

    dict_of_id = {}

    for i in range(highest_memory, lowest_memory - 1, -1):
    
        markov_util = DTMCModel(i)
        kernel = NUTS(numpyro_model)
        mcmc = MCMC(
            kernel, 
            num_warmup=num_warmup, 
            num_samples=num_samples, 
            num_chains=num_chains, 
            chain_method='vectorized'
        )
        
        mcmc.run(fold_in(key, i), markov_util, history)
        
        dict_of_id[f'{i} games'] = az.from_numpyro(mcmc)

        if assert_convergence:
            assert np.all(az.rhat(dict_of_id[f'{i} games'])<1.01)

    return dict_of_id