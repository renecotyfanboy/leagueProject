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

``` plotly
{"file_path": "loserQ/assets/true_data_compare_results.json"}
```

Now we can plot the associated transition probabilities for this model.

``` plotly
{"file_path": "loserQ/assets/true_data_transition_probas.json"}
```

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

``` plotly
{"file_path": "loserQ/assets/history_streak_histogram.json"}
```

You can easily check that it is already consistent with the results I get in my previous post, and that there are some
players experiencing really long streaks of wins or losses (up to 16 wins and 17 losses in a row (1)).
{ .annotate }

1.  <div class="tenor-gif-embed" data-postid="27295909" data-share-method="host" data-aspect-ratio="1.77778" data-width="100%"><a href="https://tenor.com/view/%C3%A7a-fait-beaucoup-la-non-gif-27295909">ça Fait Beaucoup La Non GIF</a>from <a href="https://tenor.com/search/%C3%A7a+fait+beaucoup+la+non-gifs">ça Fait Beaucoup La Non GIFs</a></div> <script type="text/javascript" async src="https://tenor.com/embed.js"></script>

The approach I had was to compare it to a simulated sample of coin flips. I simply generated a lot of sequences of coin 
flips to mock players histories, using a winrate drawn from the winrate distribution in our sample. I then computed 
the distribution of streak lengths for these sequences, and compared it to the real data. Redoing this with the new dataset 
yield the following graph :

``` plotly
{"file_path": "loserQ/assets/mock_streak_histogram.json"}
```

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

## Statistics :nerd:

!!! Quote "Comment from [Matos3001](https://www.reddit.com/r/leagueoflegends/comments/15k2nw4/comment/jvlq50c/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) on my previous post"
    [...] while [you] might understand a lot about balls in the sky, [you] are no statistician. [...]

After my first post, I was really surprised by the number of people how disclaimed the results, saying that I was not a
statistician. 
C'est vrai que je suis resté en surface, ça manquait le pas entre un simple truc pour s'amuser et des résultats de qualité scientifique.
While I am perfectly aware and transparent about the bias of the previous and current study, I am still an 
astrophysicist, and I know a bit about statistics. When doing evidence based science, statistic is the backbone of any 
solid result you can get. I am no researched in applied statistics, but hell I know how to use a $\chi^2$ test. And I 
know [how bad it is](https://en.wikipedia.org/wiki/Misuse_of_p-values). An issue when trying to vulgarize scientific result is that most of the people are not aware about 
what and how you tell "my results are credible". Last time, I focused on visual proofs and I didn't go into the details
of how significant the results are. But since people are asking for p-values (1), I will provide them with p-values.
{ .annotate }

1.  In french, we would say that they are "autodiag zét"


This time, I accounted for the intrinsic spread of this distribution by simulating 100 dataset and plotting 
the associated $90\%$ confidence interval. The real data is well within this interval, which means that the distribution
of streak lengths is consistent with what you would expect from random coin flips. We can go a bit further and quantify
this agreement using a statistical test. To do this, we use the $\chi^2$ statistic. This is a straightforward approach which computes how much the 
true data is within the values expected from the simulated data. The $\chi^2$ of fitted data follows a known statistical 
distribution, which allows us to quantify the goodness of fit. 

!!! info
    
    The $\chi^2$ statistic is defined as : 
    $$ \chi^2 = \sum_n \frac{(x_n - \mu_n)^2}{\sigma_n^2}$$ 

    where $x_n$ is the observed frequency of streaks of length $n$, 
    $\mu_n$ is the expected frequency and $\sigma_n$ is its standard deviation 
    as computed from our simulations. The values measured on our sample are : 

    $$ \chi^2_{\text{win}} = {{ streak.chi2win }} \quad \text{and} \quad \chi^2_{\text{loss}} = {{ streak.chi2loss }} $$

    The lower the $\chi^2$, the better the fit! These are $\chi^2$ measured with 20 lengths, 
    which is the number of bins in our histogram, and a model with 2 parameters, which are 
    the winrate distribution mean and standard deviation. This means that the probability of 
    observing a $\chi^2$ greater than this value can be computed using the survival function
    of a $\chi^2$ random variable with 18 degrees of freedom, noted as $X\sim\chi^2(18)$ in the following : 

    $$ P(X > \chi^2_{\text{win}}) = {{ streak.p_value_win }} \quad \text{and} \quad P{X > \chi^2_{\text{loss}}} = {{ streak.p_value_loss }} $$

    In this situation, this probability is refered to as the p-value. 

For the whole dataset, we can therefore compute a p-value
of $\sim {{ streak.p_value_win }}$ for the winning streaks and $\sim {{ streak.p_value_loss }}$ for the losing streaks. This means that the data is
consistent with the hypothesis that the streak lengths are drawn from the same kind of process we used to make the 
simulations, which are simply coin flips with a probability of success drawn randomly from the winrate distribution
in the dataset. This is a {{ streak.z_score_win }}$\sigma$ and {{ streak.z_score_loss }}$\sigma$ significant result 
for the winning and losing streaks respectively. To give you an idea, 3$\sigma$ is an acceptable standard in physics, 
$5\sigma$ is the common threshold for a discovery in particle physics at the CERN, and $5.1\sigma$ is the 
significance of the first gravitational waves event detected by LIGO.


Don't get fooled by the small p-values. They are not a measure of the probability of the hypothesis 
being true, and have no meaning *per se* except for disclosing obviously wrong hypothesis. I want 
you to be aware that p-values only make sense if compared with other p-values. In any case, it is promising to see
that the data is consistent with the hypothesis that the streak lengths are drawn from a random process.
