# Validating the approach

!!! abstract "TL;DR"
    - I want to validate this methodology using mock data to show its ability to recover dynamics in match histories.
    - I propose two kind of simulations to do this : an obvious LoserQ mechanism and a nasty LoserQ
    - I show that the methodology can accurately recover arbitrary dynamics with a relatively small number of games ($\lesssim 10000$)
    for both the obvious and nasty mock LoserQ mechanism.

## Generating mock data

The previous page was an exhaustive description of the model I chose to describe the history of games in League of 
Legends. When trying to assess stuff with mathematical models, it is always good to check if the methodology works on 
mock data. To do this, I will show that this methodology is able to recover the parameters of two simulated samples.

1. A simulation where there is an obvious LoserQ mechanism, where your probability of winning is linked to
the four previous game you played. 
2. A simulation where there is a nasty LoserQ mechanism, where most of the players would not see significant 
patterns while some would be cursed by long streaks of wins and losses.

The probability of winning the next game is linked to the winrate of the four previous games (4-order DTMC), the values are 
highlighted in the following table.

??? info "Transition probabilities for mock LoserQ mechanisms"

    | Winrate of the 4 previous games | Probability of winning next game |
    | :-----------------------------: | :------------------------------: |
    | $0\%$                           | $50\% - I\times 37.5\%$          |
    | $25\%$                          | $50\% - I\times 12.5\%$          |
    | $50\%$                          | $50\%$                           |
    | $75\%$                          | $50\% + I\times 12.5\%$          |
    | $100\%$                         | $75\% + I\times 37.5\%$          |

    - $I = 0.5$ for the obvious loserQ, enabling huge streaks occuring for everyone.
    - $I$ is drawn from a $\beta$ random variable $\alpha = 1.2$ and $\beta=10$ for the nasty loserQ, so that most of the 
        player would experience no significant pattern, but some would be cursed by long streaks of wins and losses.
    

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

??? question annotate "Minigame"
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


1.  All are simulated (1). The one below is not. It is sampled from summoners in Emerald I.
    { .annotate }
    1. <center> <div class="tenor-gif-embed" data-postid="27578414" data-share-method="host" data-aspect-ratio="1" data-width="50%"><a href="https://tenor.com/view/prankex-gif-27578414">Prankex Sticker</a>from <a href="https://tenor.com/search/prankex-stickers">Prankex Stickers</a></div> </center> <script type="text/javascript" async src="https://tenor.com/embed.js"></script>

    <center>
    ``` plotly
    {"file_path": "loserQ/assets/simulation_4.json"}
    ```
    </center>


## Practical implementation

So, as a summary of the theory section,  the whole idea would be to recover the underlying dynamics from a given set of match history. 
To do this, my take is to determine the best transition probabilities for a given DTMC using MCMC methods. By determining these probabilities for various DTMC 
with increasing memory size, we obtain best-fit models for the underlying dynamics. By comparing these models using 
ELDP-LOO, we can determine the best model to describe the history of games. 

In practice, I'll determine the transition probabilities for DTMC with memory size $1$ to $6$. We sample this posterior 
distributions using the NUTS sampler as implemented in the `numpyro` library. Then, I compare these
models using the comparator implemented in the `arviz` library. All the code is available on the [Github repository](https://github.com/renecotyfanboy/leagueProject),
and the API of the helper package I wrote is detailed in the [documentation](../api/data.md).
As a mock dataset, I generated 85 games for 100 players, and applied the aforementioned methodology for both the obvious and nasty LoserQ mechanisms. 
Most of the computations where performed either on the [SSP Cloud data](https://datalab.sspcloud.fr/), which nicely and freely
providing GPUs to Fr*nch academics, or on my personal computer.

## Assessing the performance

Let's first observe the comparator plot for the two simulated datasets. We will discuss a bit on how to interpret them.

=== "Obvious LoserQ"

    <div class="grid cards" markdown>

    -   <p style='text-align: center;'> **Comparator plot** </p>
        
        ``` plotly
        {"file_path": "loserQ/assets/validation_obvious_compare.json"}
        ```

    </div>

=== "Nasty LoserQ"

    <div class="grid cards" markdown>

    -   <p style='text-align: center;'> **Comparator plot** </p>
        
        ``` plotly
        {"file_path": "loserQ/assets/validation_obvious_compare.json"}
        ```

    </div>


We can see first that the higher ELDP models in our comparator are the 4-order models. This is great, since this is the 
memory I used to generate the mock data. For the obvious LoserQ, we see that the 5-order model is also a contender for 
the first place. Since the true input is a 4-order dynamics, a 5-order dynamics can also reproduce the data, but with 
lower ELDP since it overfit a bit the data. Same comment for the 6-order model. For the nasty LoserQ, we see that the 
4-order model is also the best, and that the 2-order is the second best. This is pretty interesting since this mechanism 
was designed to be hard to detect, and should disguise as lower order dynamic for the people that are not violently cursed. 

Since the obvious LoserQ is a pure DTMC, we can also check the transition probabilities we recovered for the best model
and check that they are close to the one I used to run the simulations. This is what I show in the following plots, where
we see that our posterior distributions (in green) are coincident with the true values (in grey).

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