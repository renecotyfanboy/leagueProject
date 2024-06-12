# Application to true data

!!! abstract "TL;DR"
    - When applied to true data, the best model to describe the history of games is a 1st order DTMC, where the outcome of a game *weakly* depends on the previous game.
    - We show that this model is performant at describing the expected correlations, streak lengths and others 

## Best-fit model 

It's time to see what we obtain with our dataset of true matches. Let's fit our model and look at the ELDP-LOO comparison plot.

<div class="grid cards" markdown>

-   <p style='text-align: center;'> **Comparison of DTMC Models on True Data** </p>

    ``` plotly
    {"file_path": "loserQ/assets/true_data_compare_results_balanced.json"}
    ```
    
    <p style='text-align: justify; width: 80%; margin-left: auto; margin-right: auto;'>
    Comparison of the ELPD-LOO metric for different depth of DTMC model, along with the difference with the best ELPD-LOO value. 
    </p>

</div>

The previous comparison shows that the best model to describe the history we observe is a 1st order DTMC, where the outcome of a game *weakly* depends on the previous game. In other words, to best predict the outcome of a game, you only need to know the outcome of the previous game. This is the most significant dependence we can find using the full dataset. Note that the second order and third order model are also compatible, but this is probably because they can also reproduce the first order model, but with a bit of overfitting, which reduces the ELPD-LOO metric. This behaviour is similar to what we observed in the validation of the model on the simulated data. So we will focus on the first order model. We can plot the associated transition probabilities for this model.

<div class="grid cards" markdown>

-   <p style='text-align: center;'> **Winning after winning** </p>
    ``` plotly
    {"file_path": "loserQ/assets/true_data_transition_probas_0.json"}
    ```


-   <p style='text-align: center;'> **Losing after winning** </p>
    ``` plotly
    {"file_path": "loserQ/assets/true_data_transition_probas_1.json"}
    ```


-   <p style='text-align: center;'> **Winning after losing** </p>
    ``` plotly
    {"file_path": "loserQ/assets/true_data_transition_probas_2.json"}
    ```
  

-   <p style='text-align: center;'> **Losing after losing** </p>
    ``` plotly
    {"file_path": "loserQ/assets/true_data_transition_probas_3.json"}
    ```

</div>

The first thing that stands out is that the parameters are very well constrained. These posterior distributions are clearly Gaussian and can be interpreted pretty easily. With this dataset, you get a probability of winning the next game of $({{"{:,.2f}".format((100*true_data.win_after_win_mean)| abs)}} \pm {{"{:,.2f}".format(100*true_data.win_after_win_std)}}) \%$ if the previous is a win, and a probability of losing the next game of $({{"{:,.2f}".format((100*true_data.lose_after_lose_mean)| abs)}} \pm {{"{:,.2f}".format(100*true_data.lose_after_lose_std)}}) \%$ if the previous is a loss. There is a small but significant difference between the two, which is consistent with the hypothesis that the outcome of a game is not completely independent of the previous. 

## Streak lengths

To check whether our best fit model is indeed descriptive of our data or not, we can perform what is called posterior predictive checks. This is a simple procedure where you generate a lot of data using the best fit model, and compare the distribution of the observables to the real data. This is a good way to check if the model can reproduce the data. One issue we have with directly comparing the true game history to the model is that it is challenging to compare two random processes directly. Imagine you have to compare two series of coin flips, checking one by one if the outcomes are the same is a nonsense since this is stochastic. However, you can easily compare averaged quantities, such as the mean of the two series, their standard deviation etc. In the following, I compare the distribution of streak lengths from the true dataset to what we expected to measure using the best-fit model. Below is the distribution of streak lengths in our true dataset of match histories. 

<div class="grid cards" markdown>

-   <p style='text-align: center;'> **Streak length distribution** </p>
    ``` plotly
    {"file_path": "loserQ/assets/history_streak_histogram.json"}
    ```
    <p style='text-align: justify; width: 80%; margin-left: auto; margin-right: auto;'>
    Distribution of the lengths of the streak of wins and losses in the dataset. Note that the y-axis is in log scale.
    </p>

</div>

This kind of looks like the results I got in my previous post, and that there are some players experiencing really long streaks of wins or losses (up to 16 wins and 17 losses in a row (1)).
{ .annotate }

1.  <div class="tenor-gif-embed" data-postid="27295909" data-share-method="host" data-aspect-ratio="1.77778" data-width="100%"><a href="https://tenor.com/view/%C3%A7a-fait-beaucoup-la-non-gif-27295909">ça Fait Beaucoup La Non GIF</a>from <a href="https://tenor.com/search/%C3%A7a+fait+beaucoup+la+non-gifs">ça Fait Beaucoup La Non GIFs</a></div> <script type="text/javascript" async src="https://tenor.com/embed.js"></script>

In the next graph, I simulate 100 datasets and compute the streak lengths for each of them to propagate the uncertainties coming from the sample variance and the model. I do the same for the obvious LoserQ model I used in the validation to show what we would expect in this case.

<div class="grid cards" markdown>

-   <p style='text-align: center;'> **Streak length distribution** </p>
    ``` plotly
    {"file_path": "loserQ/assets/mock_streak_histogram_win.json"}
    ```
    
    ``` plotly
    {"file_path": "loserQ/assets/mock_streak_histogram_loss.json"}
    ```
    <p style='text-align: justify; width: 80%; margin-left: auto; margin-right: auto;'>
    Distribution of the lengths of the streak of wins and losses in the dataset. Note that the y-axis is in log scale.
    </p>

</div>

As the last time, the real data is consistent with the simulated data. This is indicated by the fact that the real data falls between the $95\%$ confidence interval of the simulated data. This means that the streak lengths we see in the player dataset are compatible with what we expect from the best-fit model. 

## Auto-correlation

The auto-correlation of a sequence is a measure of the similarity between the sequence and a delayed version of itself. This is another kind of measurement we can use to both check the correlation within the dataset and the validity of our model. This is also the most visual way to see if the outcome of a game depends on the previous games. It can be computed with the following formula:

$$
R_X(k) = \mathbb{E}\left[ X_i \times X_{i-k} \right]
$$

This is the measure of the average of the product between two games that are $k$ games apart. If the sequence is random, then the auto-correlation should be close to 0. If the sequence is *positively*-correlated (if you win a game, you have more chance to win the game after $k$ other ones), then the auto-correlation should be close to 1. Conversely, if the sequence is *negatively*-correlated (if you win a game, you have more chance to lose the game after $k$ other ones), then the auto-correlation should be close to -1. The following graph shows the auto-correlation of a sequence of games, where the outcome of a game depends on the previous game only. 

``` plotly
{"file_path": "loserQ/assets/simulated_correlation.json"}
```

??? note 
    The graph of the DTMC model used to draw the previous plot are the following : 

    <div class="grid cards" markdown>

    - **Random model**
        ``` mermaid
        graph LR
            Win -->|50%| Loss
            Loss -->|50%| Win
            Win -->|50%| Win
            Loss -->|50%| Loss
        ``` 
    
    - **Correlated model**
        ``` mermaid
        graph LR
            Win -->|20%| Loss
            Loss -->|20%| Win
            Win -->|80%| Win
            Loss -->|80%| Loss
        ``` 

    - **Anti-correlated model**
        ``` mermaid
        graph LR
            Win -->|80%| Loss
            Loss -->|80%| Win
            Win -->|20%| Win
            Loss -->|20%| Loss
        ```

    </div>

We compute the autocorrelation for a simulated dataset and compare it to the autocorrelation of the true dataset. The bands represent the 95% spread of autocorrelation within the simulated dataset, and we overlap samples from the true dataset to show that it is consistent. We also add the autocorrelation of the obvious LoserQ model to show that we would expect higher values up to a lag of 4, which is the order of the underlying DTMC model. 

``` plotly
{"file_path": "loserQ/assets/true_data_correlation.json"}
```

Once again, we see that this quantity is well predicted by our best-fit DTMC model, while significant deviation from zero should be visible if there were an efficient LoserQ at act. However, the 1st order dynamic we showed is so low that it would have been hard to detect without using DTMC, which motivates a bit more this approach. 

## Conclusion

So, did we show that there is not LoserQ in League of Legend? Finally? At last? Well, no. In fact, it's impossible to establish that unicorns, dragons, and good smelling league players don't exist. This statement is true for a lot of other stuff including LoserQ. I ontologically cannot disprove the existence of without looking at the matchmaking source code. However, we can still get good interpretation from these results. What is shown here is that matches can be modelled very well by a DTMC which only needs the previous game to define its transitions. In short, using only win/loss information, the best way to predict the outcome of a player's match is to look at his last game. He will then have a $({{"{:,.2f}".format((100*true_data.win_after_win_mean)| abs)}} \pm {{"{:,.2f}".format(100*true_data.win_after_win_std)}}) \%$  chance of winning if his previous game was won, and $({{"{:,.2f}".format((100*true_data.win_after_lose_mean)| abs)}} \pm {{"{:,.2f}".format(100*true_data.win_after_lose_std)}}) \%$ if his previous game was lost. This behaviour was confirmed by studying the series lengths or auto-correlation that such dynamics would induce, and comparing them with what is observed in the real data.

I wouldn't interpret such low values as emerging from an engineered process to increase player involvement. If this is indeed the case, then Riot's competence is questionable, since the effect of this LoserQ would only be visible once or twice out of several hundred games. In general, it's healthier to come up with simpler, more reasoned interpretations when you're in this kind of situation. Typically, we know that players are more tilted after defeats[^1] and that players' tilt reduces their overvall probability to win[^2][^3][^4]. This elementary explanation isn't necessarily the right one, but given that it's simple, it should be favoured until proven otherwise. In the end, even using a more robust methodology and better quality data, we still find the same results as those I presented in 2023 on reddit.

!!! tip "Takeaways"
    Players lower their win rate to $({{"{:,.2f}".format((100*true_data.win_after_lose_mean - 50)| abs)}} \pm {{"{:,.2f}".format(100*true_data.win_after_lose_std)}}) \%$ after a loss and increase it by $({{"{:,.2f}".format((100*true_data.win_after_win_mean - 50)| abs)}} \pm {{"{:,.2f}".format(100*true_data.win_after_win_std)}}) \%$ after a win. This is a significant deviation from randomness, but you can see its effect once or twice for hundreds of games players. In this situation, either LoserQ does not exist, or if it exists, it is really ineffective. 

??? danger "Why people would still believe in LoserQ?"

    - Being placed too low/high early in the season, leading to large wins/losses streak until the hidden elo stabilize
    - Under damped MMR system, where the official rank would be much faster to move than the hidden elo
    - Confirmation bias, as many people think they notice patterns in the player they are matched with 
    - Coping, as lower players tend to overestimate their own skill and think they are held back by the matchmaking[^5]
    - And many other reasons I guess

*[ELPD]: Expected log predictive density
*[LOO]: Leave One Out
*[DTMC]: Discrete-time Markov Chain

[^1]: [**Analyzing the changes in the psychological profile of professional League of Legends players during competition**, *Mateo-Orcajada & al* (2022)](https://www.sciencedirect.com/science/article/abs/pii/S0747563221003538)
[^2]: [**Understanding Tilt in Esports: A Study on Young League of Legends Players**, *Wu & Lee* (2021)](https://dl.acm.org/doi/abs/10.1145/3411764.3445143)
[^3]: [**Exploring Stress in Esports Gaming: Physiological and Data-driven approach on Tilt**, *Lee* (2021)](https://escholarship.org/content/qt61p8c951/qt61p8c951.pdf)
[^4]: [**Effects of individual toxic behavior on team performance in League of Legends**, *Monge & O'Brien* (2017)](https://www.tandfonline.com/doi/abs/10.1080/15213269.2020.1868322)
[^5]: [**The psychology of esports players’ ELO Hell: Motivated bias in League of Legends and its impact on players’ overestimation of skill**, *Aeschbach & al* (2023)](https://www.sciencedirect.com/science/article/pii/S0747563223001796)