# Validating the approach

!!! abstract "TL;DR"
    - I motivate a methodology to recover the underlying dynamics of a history of games based on DTMC.
    - This methodology is validated using various simulated dynamics, showing that it can accurately recover the 
    transition probabilities for simulated DTMC or point to correlations with a relatively low number of games.
    - A small digression about why do this instead of simply using `p-values`

## Defining the approach

In the previous section, I introduced the concept of Discrete-time Markov Chains (DTMC), which are powerful mathematical 
tools for describing the dynamics of a sequence of events. In the context of League of Legends, we can treat the history
of games as a sequence of states, where each state represents a game outcome (win or loss). By applying DTMC to these 
sequences, we can analyze the transition probabilities between states, revealing the underlying patterns in match 
outcomes. This methodology enables us to investigate whether losing streaks (or winning streaks) follow a predictable 
pattern, thus providing insights into the existence of "LoserQ," a phenomenon where players might be more likely to 
continue losing once they start. Using Bayesian inference and model comparison, we aim to identify the DTMC model that 
best captures these transition probabilities, offering a deeper understanding of any potential biases or streak 
tendencies in the game.

## What to do in practice ?

The previous page was an exhaustive description of the model I chose to describe the history of games in League of 
Legends. When trying to assess stuff with mathematical models, it is always good to check if the methodology works on 
mock data. To do this, I will show that this methodology is able to recover the parameters of three simulated samples.


1. A simulation where there is an obvious LoserQ mechanism, where your probability of winning is linked to
the four previous game you played. 
2. A simulation where there is a nasty LoserQ mechanism, where most of the players would not see significant 
patterns while some would be cursed by long streaks of wins and losses.

To generate mock history of players in this case, we simply flip a coin with a $50\%$ chance of winning.

I propose to generate an obvious LoserQ mechanism using the following logic. 
The probability of winning the next game is linked to the winrate of the four previous games, the values are 
highlighted in the following table. (I = 0.5)



The probability of winning the next game is linked to the winrate of the four previous games, but in this situation, 
a random variable is drawn from a given distribution to describe how much this player would be affected by the 
LoserQ. The values are highlighted in the following table.



| Winrate of the 4 previous games | Probability of winning next game |
| :-----------------------------: | :------------------------------: |
| $0\%$                           | $50\% - I\times 37.5\%$          |
| $25\%$                          | $50\% - I\times 12.5\%$          |
| $50\%$                          | $50\%$                           |
| $75\%$                          | $50\% + I\times 12.5\%$          |
| $100\%$                         | $75\% + I\times 37.5\%$          |

where $I$ is drawn from a $\beta$ random variable $\alpha = 1.2$ and $\beta=10$.

<div class="grid cards" markdown>

-   <p style='text-align: center;'> **True and simulated history of games for comparison** </p>
    
    === "Example history"
    
        ``` plotly
        {"file_path": "loserQ/assets/history_multiple_players.json"}
        ```
    
    === "Pure coin flips"
    
        ``` plotly
        {"file_path": "loserQ/assets/validation_coinflip.json"}
        ```
    
    === "Obvious LoserQ"
    
        ``` plotly
        {"file_path": "loserQ/assets/validation_obvious.json"}
        ```
    
    === "Nasty LoserQ"
    
        ``` plotly
        {"file_path": "loserQ/assets/validation_nasty.json"}
        ```

</div>

??? question annotate
    Can you distinguish a true player history compared to simulated ones ? Solution : (1)
    <table>
    <tr>
    <td> <center> History n°1  </center> </td>
    <td> <center> History n°2  </center> </td>
    <td> <center> History n°3  </center> </td>
    </tr>
    <tr>
    <td> 
    ``` plotly
    {"file_path": "loserQ/assets/simulation_0.json"}
    ```
    </td>
    <td>
    ``` plotly
    {"file_path": "loserQ/assets/simulation_1.json"}
    ```
    </td>
    <td> 
    ``` plotly
    {"file_path": "loserQ/assets/simulation_2.json"}
    ```
    </td>
    </tr>
    </table>


1.  All are simulated. The one below is not. It is sampled from summoners in Emerald I. Or is it ? (1)
    { .annotate }
    1.  It is. I swear. Prankex free.
    <div class="tenor-gif-embed" data-postid="27578414" data-share-method="host" data-aspect-ratio="1" data-width="50%"><a href="https://tenor.com/view/prankex-gif-27578414">Prankex Sticker</a>from <a href="https://tenor.com/search/prankex-stickers">Prankex Stickers</a></div> <script type="text/javascript" async src="https://tenor.com/embed.js"></script>

    <center>
    ``` plotly
    {"file_path": "loserQ/assets/simulation_4.json"}
    ```
    </center>


## Validation on mock data

So, the whole idea would be to recover the underlying dynamics from a given set of history of games. To do this, my take 
is to determine the best transition probabilities for a given DTMC. By determining these probabilities for various DTMC 
with increasing memory size, we obtain best-fit models for the underlying dynamics. By comparing these models using 
appropriate methodologies, we can determine the best model to describe the history of games. 

In practice, I'll determine the transition probabilities for DTMC with memory size $1$ to $10$. I will then compare these
models using state of the art Bayesian model comparison techniques. And by comparing these, we get the best model to 
describe the history of games.

For each memory size, we solve a Bayesian inference problem, using flat prior between $0$ and $1$ for the authorized 
transition probabilities. The likelihood is simply determined using the history of games, as we can unroll the chain 
an compute the likelihood of each player history in the dataset, given the transition probabilities of the chain. 
Summing these likelihoods gives us the likelihood of the observed history we gathered, and can be used to compute the 
posterior distribution of the transition probabilities. We sample this posterior distribution using MCMC sampler, in 
particular the NUTS sampler as implemented in the `numpyro` library.

We test the methodology on LoserQ-like **simulated** data. I show that we can recover the good parameters of the model
using mock data.


=== "Transitions (i)"
    <div class="grid cards" markdown>

    -   <p style='text-align: center;'> **(Loss, Loss, Loss, Loss) to Win** </p>
        
        ``` plotly
        {"file_path": "loserQ/assets/validation_transition_probas_0.json"}
        ```

    -   <p style='text-align: center;'> **(Loss, Loss, Loss, Win) to Win** </p>

        ``` plotly
        {"file_path": "loserQ/assets/validation_transition_probas_1.json"}
        ```

    -   <p style='text-align: center;'> **(Loss, Loss, Win, Loss) to Win** </p>

        ``` plotly
        {"file_path": "loserQ/assets/validation_transition_probas_2.json"}
        ```

    -   <p style='text-align: center;'> **(Loss, Loss, Win, Win) to Win** </p>

        ``` plotly
        {"file_path": "loserQ/assets/validation_transition_probas_3.json"}
        ```

    </div>

=== "Transitions (ii)"
    <div class="grid cards" markdown>

    -   <p style='text-align: center;'> **(Loss, Win, Loss, Loss) to Win** </p>
        
        ``` plotly
        {"file_path": "loserQ/assets/validation_transition_probas_4.json"}
        ```

    -   <p style='text-align: center;'> **(Loss, Win, Lose, Win) to Win** </p>

        ``` plotly
        {"file_path": "loserQ/assets/validation_transition_probas_5.json"}
        ```

    -   <p style='text-align: center;'> **(Loss, Win, Win, Loss) to Win** </p>

        ``` plotly
        {"file_path": "loserQ/assets/validation_transition_probas_6.json"}
        ```

    -   <p style='text-align: center;'> **(Loss, Win, Win, Win) to Win** </p>

        ``` plotly
        {"file_path": "loserQ/assets/validation_transition_probas_7.json"}
        ```

    </div>

=== "Transitions (iii)"
    <div class="grid cards" markdown>

    -   <p style='text-align: center;'> **(Win, Loss, Loss, Loss) to Win** </p>
        
        ``` plotly
        {"file_path": "loserQ/assets/validation_transition_probas_8.json"}
        ```

    -   <p style='text-align: center;'> **(Win, Loss, Loss, Win) to Win** </p>

        ``` plotly
        {"file_path": "loserQ/assets/validation_transition_probas_9.json"}
        ```

    -   <p style='text-align: center;'> **(Win, Loss, Win, Loss) to Win** </p>

        ``` plotly
        {"file_path": "loserQ/assets/validation_transition_probas_10.json"}
        ```

    -   <p style='text-align: center;'> **(Win, Loss, Win, Win) to Win** </p>

        ``` plotly
        {"file_path": "loserQ/assets/validation_transition_probas_11.json"}
        ```

    </div>

=== "Transitions (iv)"
    <div class="grid cards" markdown>

    -   <p style='text-align: center;'> **(Win, Win, Loss, Loss) to Win** </p>
        
        ``` plotly
        {"file_path": "loserQ/assets/validation_transition_probas_12.json"}
        ```

    -   <p style='text-align: center;'> **(Win, Win, Loss, Win) to Win** </p>

        ``` plotly
        {"file_path": "loserQ/assets/validation_transition_probas_13.json"}
        ```

    -   <p style='text-align: center;'> **(Win, Win, Win, Loss) to Win** </p>

        ``` plotly
        {"file_path": "loserQ/assets/validation_transition_probas_14.json"}
        ```

    -   <p style='text-align: center;'> **(Win, Win, Win, Win) to Win** </p>

        ``` plotly
        {"file_path": "loserQ/assets/validation_transition_probas_15.json"}
        ```

    </div>
    
*[DTMC]: Discrete-time Markov Chain
*[MCMC]: Markov Chain Monte Carlo
*[NUTS]: No-U-Turn Sampler
*[ELDP]: Expected log pointwise predictive density