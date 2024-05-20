!!! abstract "TL;DR"
    - When applied to true data, the best model to describe the history of games is a 1st order DTMC, where 
    the outcome of a game *weakly* depends on the previous game.
    - We show that this model is performant at describing the expected correlations, streak lengths and others 

# Application to true data

This is the first approach I used in my [previous post about the LoserQ](https://www.reddit.com/r/leagueoflegends/comments/15k2nw4/existence_of_loser_queue_a_statistical_analysis/).
The idea is pretty simple : you can go through all the collected histories, and count the number of games in a row 
that were lost or won. By plotting the histogram of these streak lengths, you can get an idea of how they are distributed. 

## Best-fit model 

The below comparison shows that the best model to describe the history we observe is a 1st order DTMC, where the outcome
of a game *weakly* depends on the previous game. This is a good sign that the data is consistent with the hypothesis that
the streak lengths are drawn from a random process. 

<div class="grid cards" markdown>

-   <p style='text-align: center;'> **Comparison of DTMC Models on True Data** </p>

    ``` plotly
    {"file_path": "loserQ/assets/true_data_compare_results.json"}
    ```
    
    <p style='text-align: justify; width: 80%; margin-left: auto; margin-right: auto;'>
    Comparison of the ELPD-LOO metric for different depth of DTMC model, along with the difference with the best 
    ELPD-LOO value. 
    </p>

</div>

The model comparison displayed in the previous graph shows that the best model to describe the history we observe is a
1st order DTMC. In other words, to best predict the outcome of a game, you only need to know the outcome of the previous
game. This is the most significant dependence we can find using this dataset. Note that the second order and third order
model are also compatible, but this is probably due to the fact that they can also reproduce the first order model, but 
with a bit of overfitting, which reduces the ELPD-LOO metric. This behavior is similar as what we observed in the 
validation of the model on the simulated data. So we will focus on the first order model. We can plot the associated 
transition probabilities for this model.

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

These posterior distributions are clearly gaussian and can be interpreted pretty easily. With this dataset, you get a 
probability of winning the next game of $50.54\% \pm 0.14 \%$ is the previous is a win, and a probability of losing 
the next game of $50.19\% \pm 0.14 \%$ if the previous is a loss. There is a small but significant difference between
the two, which is consistent with the hypothesis that the outcome of a game is not completely independent of the 
previous. 

## Streak lengths

To check whether our best fit model is indeed descriptive of our data or not, we can perform what is called a 
posterior predictive check. This is a simple procedure where you generate a lot of data using the best fit model, and
compare the distribution of the observables to the real data. This is a good way to check if the model is able to 
reproduce the data, and if it is consistent with the hypothesis that the data is drawn from the model. One issue we 
have with directly comparing the true game history to the model is that it is hard to compare two random processes 
directly. Imagine you have to compare two series of coinflip, checking one by one if the outcomes are the same is a 
non-sense since this is stochastic. However, you can easily compare averaged quantities, such as the mean of the two 
series, their standard deviation etc. In my previous analysis, I chose to compare the distribution of streak lengths, 
which should be comparable (up to the intrinsic uncertainties) when comparing the true data to the model. Below is the 
distribution of streak lengths in our sample of game histories. 

<div class="grid cards" markdown>

-   <p style='text-align: center;'> **Streak length distribution** </p>
    ``` plotly
    {"file_path": "loserQ/assets/history_streak_histogram.json"}
    ```
    <p style='text-align: justify; width: 80%; margin-left: auto; margin-right: auto;'>
    Distribution of the lengths of the streak of wins and losses in the dataset. Note that 
    the y-axis is in log scale.
    </p>

</div>

You can easily check that it is already consistent with the results I get in my previous post, and that there are some
players experiencing really long streaks of wins or losses (up to 16 wins and 17 losses in a row (1)).
{ .annotate }

1.  <div class="tenor-gif-embed" data-postid="27295909" data-share-method="host" data-aspect-ratio="1.77778" data-width="100%"><a href="https://tenor.com/view/%C3%A7a-fait-beaucoup-la-non-gif-27295909">ça Fait Beaucoup La Non GIF</a>from <a href="https://tenor.com/search/%C3%A7a+fait+beaucoup+la+non-gifs">ça Fait Beaucoup La Non GIFs</a></div> <script type="text/javascript" async src="https://tenor.com/embed.js"></script>

The approach I had was to compare it to a simulated sample of coin flips. I simply generated a lot of sequences of coin 
flips to mock players histories, using a winrate drawn from the winrate distribution in our sample. I then computed 
the distribution of streak lengths for these sequences, and compared it to the real data. Redoing this with the new dataset 
yield the following graph :

<div class="grid cards" markdown>

-   <p style='text-align: center;'> **Streak length distribution** </p>
    ``` plotly
    {"file_path": "loserQ/assets/mock_streak_histogram_win.json"}
    ```
    
    ``` plotly
    {"file_path": "loserQ/assets/mock_streak_histogram_loss.json"}
    ```
    <p style='text-align: justify; width: 80%; margin-left: auto; margin-right: auto;'>
    Distribution of the lengths of the streak of wins and losses in the dataset. Note that 
    the y-axis is in log scale.
    </p>

</div>


As the last time, the real data is consistent with the simulated data, which is hinted by the fact that the real data 
fall between the $90\%$ confidence interval of the simulated data. This is a good sign that the streak lengths we 
observe in our dataset are consistent with what you would expect from random coin flips.

## Auto-correlation

The auto-correlation of a sequence is a measure of the similarity between the sequence and a delayed version of itself.
This is the most visual way to see if the outcome of a game depends on the previous games. It can be computed with the 
following formula:
 
$$
R_X(k) = \mathbb{E}\left[ X_i \times X_{i-k} \right]
$$

This is the measure of the average of the product between two games that are $k$ games apart. If the sequence is random, 
then the auto-correlation should be close to 0. If the sequence is *positively*-correlated (if you win a game, you have
more chance to win the game after $k$ other ones), then the auto-correlation 
should be close to 1. If the sequence is *negatively*-correlated (if you win a game, you have more chance to lose the game
after $k$ other ones), then the auto-correlation should be close to -1. The following graph shows the auto-correlation of
a sequence of games where the outcome of a game depends on the previous game only. 

``` plotly
{"file_path": "loserQ/assets/simulated_correlation.json"}
```

??? note 
    The graph of the markov models used to draw the previous plot are the following : 

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

The envelope of the previous graph represents the spread on this measure due to computing this auto-correlation for a 
set of individual game histories. This auto-correlation varies from one player to another, and this envelopes gives us
an idea of how much this quantity varies.

``` plotly
{"file_path": "loserQ/assets/true_data_correlation.json"}
```

We see that our model can also reproduce the observed auto-correlation. 

*[ELPD]: Expected log predictive density
*[LOO]: Leave One Out
*[DTMC]: Discrete-time Markov Chain