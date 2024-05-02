# Streak lengths

This is the first approach I used in my [previous post about the LoserQ](https://www.reddit.com/r/leagueoflegends/comments/15k2nw4/existence_of_loser_queue_a_statistical_analysis/).
The idea is pretty simple : you can go through all the collected histories, and count the number of games in a row 
that were lost or won. By plotting the histogram of these streak lengths, you can get an idea of how they are distributed. 

## Whole dataset

I plot below the histogram for the whole dataset.

``` plotly
{"file_path": "loserQ/assets/history_streak_histogram.json"}
```

The approach I had was to compare it to a simulated sample of coin flips. I simply generated a lot of sequences of coin 
flips to mock players histories, using a winrate drawn from the winrate distribution in our sample. I then computed 
the distribution of streak lengths for these sequences, and compared it to the real data. Redoing this with the new dataset 
yield the following graph :

``` plotly
{"file_path": "loserQ/assets/mock_streak_histogram.json"}
```

## Hypothesis testing

This time, I also accounted for the intrinsic spread of this distribution by simulating 100 dataset and plotting 
the associated $68\%$ confidence interval. The real data is well within this interval, which means that the distribution
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
in the dataset. Don't get fooled by the small p-values. They are not a measure of the probability of the hypothesis 
being true, and have no meaning *per se* except for disclosing obviously wrong hypothesis. I want 
you to be aware that p-values only make sense if compared with other p-values. In any case, it is promising to see
that the data is consistent with the hypothesis that the streak lengths are drawn from a random process.

## Subdivide by tier

To show that this trend is the very same across all elos, I performed the previous test on every tier separately. 