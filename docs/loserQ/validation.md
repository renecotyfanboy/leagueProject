# Validating the approach

!!! abstract "TL;DR"
    - I would like to validate this methodology using mock data to show its ability to identify dynamics in game histories.
    - I propose two types of simulations to do this: an obvious LoserQ mechanism and a nasty LoserQ mechanism.
    - I show that the methodology can accurately recover arbitrary dynamics with a relatively small number of games ($\lesssim 35000$) for both the obvious and the nasty mock LoserQ mechanism.

## Generating mock data

The previous page was an exhaustive description of the model I chose to describe the history of games in League of Legends. When trying to evaluate things with mathematical models, it is always good to check if the methodology works on simulated data. To do this, I will show that this methodology can recover the parameters of three simulated samples.

1. A pure coin flip simulation.
2. A simulation where there is an obvious LoserQ mechanism, where your probability of winning is linked to the four previous games you have played. 
3. A simulation where there is a nasty LoserQ mechanism, where most players would not see significant patterns, while some would be cursed by long streaks of wins and losses.

The probability of winning the next game is linked to the winning rate of the previous four games (4-order DTMC), the values are highlighted in the table below.

??? info "Transition probabilities for mock LoserQ mechanisms"

    | Win rate of the 4 previous games | Probability of winning next game |
    | :------------------------------: | :------------------------------: |
    | $0\%$                            | $50\% - I\times 37.5\%$          |
    | $25\%$                           | $50\% - I\times 12.5\%$          |
    | $50\%$                           | $50\%$                           |
    | $75\%$                           | $50\% + I\times 12.5\%$          |
    | $100\%$                          | $75\% + I\times 37.5\%$          |

    - $I = 0.5$ for the obvious LoserQ, enabling considerable streaks occurring for everyone.
    - $I$ is drawn from a $\beta$ random variable $\alpha = 1.2$ and $\beta=10$ for the nasty LoserQ, so that most of the players would experience no significant pattern, but some would be cursed by long streaks of wins and losses.

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

So, to summarise the theory section, the whole idea would be to recover the underlying dynamics from a given set of game histories. To achieve this, my approach is to determine the best transition probabilities for a given DTMC using MCMC methods. By determining these probabilities for different DTMCs with increasing memory size, we obtain best-fit models for the underlying dynamics. By comparing these models using ELDP-LOO, we can determine the best model to describe the history of the games. 

In practice, I'll be determining the transition probabilities for DTMC with memory sizes $1$ to $6$. We sample these posterior distributions using the NUTS sampler implemented in the `numpyro` library. I then compare these models using the comparator implemented in the `arviz` library. All the code is available on the [Github repository](https://github.com/renecotyfanboy/leagueProject), and the API of the helper package I wrote is detailed in the [documentation](../api/data.md). As a dummy dataset, I generated 85 games for 400 players, using the above methodology for both the obvious and the nasty LoserQ mechanisms. Such a number of games is equivalent to a single division in the dataset, such as Bronze or Gold, in terms of observed data. Most of the computations were done either on the [SSP Cloud data](https://datalab.sspcloud.fr/), which kindly and freely provides GPUs to Fr*nch academics, or on my personal computer.

## Assessing the performance

Let's first look at the comparator plot for the three simulated data sets. We will discuss how to interpret them.

=== "Coinflip history"

    <div class="grid cards" markdown>

    -   <p style='text-align: center;'> **Comparator plot** </p>
        
        ``` plotly
        {"file_path": "loserQ/assets/validation_coinflip_compare.json"}
        ```

    </div>

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
        {"file_path": "loserQ/assets/validation_nasty_compare.json"}
        ```

    </div>

This graph shows the ELDP calculated for different memory sizes, along with the difference to the best ELDP. Plotting these two helps to compare the models and see which are compatible with the best. First we see that the 0-order DTMC is the best model to describe our coinflip dataset, which is great because 0-order DTMC is basically a coinflip too. For the other two, we can see that the higher ELDP models in our comparator are the 4 order models. This is great because this is the memory size I used to generate the fake LoserQs. For the obvious LoserQ we see that the 5-order model is also a contender for first place. Since the true input is a 4-order dynamics, a 5-order dynamics can also reproduce the observed histories, but with lower ELDP since it overfits the data a bit. Same for the 6th order model. For the nasty LoserQ, we see that the 4-order model is also the best, and that the 2-order is the second best. This is quite interesting, since this mechanism is designed to be difficult to detect, and should be disguised as a lower order dynamic for the people who are not violently cursed. 

Since the apparent LoserQ is a pure DTMC, we can also check the transition probabilities we obtained for the best model and see if they are close to the one I used to run the simulations. I show this in the following plots, where we see that our posterior distributions (in green) agree with the true values (in grey).

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

We find that we get most of the input parameters back. Some of them are a bit off due to the sampling variance. This is because we are working with a finite number of matches, generated by a random process. In this situation, some transitions are (by pure luck) slightly over- or under-represented, which can lead to a slight deviation in the posterior distribution. Adding a larger number of games would reduce this variance, but this is not necessary for the purposes of this project, since we have shown that we can find the good dynamics with just a few games.

To conclude on validation, I'd say that this approach works pretty well on dummy data, and only needs $\sim 34 000$ matches to show that something is happening or not. It can catch random behaviour and obvious or much more subtle mechanisms that would otherwise be difficult to see. When we apply it to the dataset of real matches, which is five times larger, we will be able to find the best way to describe the history of matches with even more confidence. 

*[DTMC]: Discrete-time Markov Chain
*[MCMC]: Markov Chain Monte Carlo
*[NUTS]: No-U-Turn Sampler
*[ELDP]: Expected log pointwise predictive density