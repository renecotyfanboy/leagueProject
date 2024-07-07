# Why do I even bother?

## What's LoserQ?

By now, everyone who plays League of Legends has heard of the LoserQ. It is something that is praised by many streamers (at least in France, where I live), and it is surprisingly difficult to find a clear definition that satisfies everyone. Some people even rejected my previous work, saying that I didn't understand what LoserQ was. Here are some quotes from the internet that try to explain what it is:

=== "GhostCalib3r on [this Reddit post](https://www.reddit.com/r/leagueoflegends/comments/htginy/what_is_losers_queue/)"

    !!! Quote 
        [...] it's the tendency to lose 3–5 games in a row after winning 3-5 in a row; "losers queue into winners queue" 
        and vice versa. Some people refer to loser's queue as "forced 50% winrate". [...]

=== "LLander_ on [this Reddit post](https://www.reddit.com/r/leagueoflegends/comments/htginy/what_is_losers_queue/)"

    !!! Quote 
        It's when the matchmaking constantly puts you with people that you have a very low chance to win with

=== "AcrobaticApricot on [this Reddit post](https://www.reddit.com/r/leagueoflegends/comments/1at554j/comment/kquvwy4/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)"

    !!! Quote 
        I think the idea is that there are 5 losers, all in “loser’s queue,” who play against 5 winners. So everyone on a 
        loser’s queue player’s team is also in loser’s queue. 

=== "MattWolfTV on [this Reddit post](https://www.reddit.com/r/leagueoflegends/comments/1at554j/comment/kquwo2z/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)"

    !!! Quote 
        Often when people complain about losers queue it is more about the game being determined 
        from matchmaking aka loading screen. 

=== "Straight_Rule_535 on [this Reddit post](https://www.reddit.com/r/leagueoflegends/comments/1at554j/comment/kquxhr0/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)"

    !!! Quote 
        Idk went to masters with 60% wr, all in lobby has 60-70wr. I lose two matches in a row. Only get matched with 
        teammates ~45%wr and enemies wr is still 60-70. It might not exist but this is sussy 

So LoserQ is the kind of thing that everyone understands and feels, but no one can really explain or define. Some people share their personal game histories full of winning and losing streaks and complain about being stuck in LoserQ. Some complain about being matched in games they cannot win in the lobby. A lot of people claim that Riot can know the outcome of a single game from the start, and I can't argue with that! If you have a good proxy for the level of all the players, their mental state and so on, it would be easy to create a lobby where one team is *expected* to slightly underperform compared to the other team. Add to this the extra leverage of auto-filling players, and you can see why some are convinced that Riot is cheating with their matchmaking. 

But why would Riot do this? Well, the answer is simple, and it is the same for every company: **money**. The more you play, the more money you are likely to spend on the game. And the more you are likely to spend money on the game, the more money Riot is likely to make. Therefore, it is in their interest to keep you playing, and the best way to do that is to keep you engaged. And what is the best way to keep you engaged? Well, some people think that getting players into a series of winning and losing streaks is a good way to do this. There is a paper that shows that certain patterns of win/loss can increase player engagement[^1]. It is called Engagement Optimised Matchmaking (EOMM), and some people claim that it has been implemented in games like APEX-legend, even though the developers say the opposite. 

In the EOMM paper, they looked at how some patterns of winning and losing streaks were associated with players leaving the game for longer periods of time. They focused on the correlation between players' last 3 games and players not playing for a week, and found that in their sample, 3% of players are likely to quit if they have won their last 3 games, while 5% of players quit after patterns such as win-win-loss or complete losses. These results are quite anti-LoserQ, as triple losses are the pattern that Riot should avoid at all costs to keep players engaged. But I could argue for LoserQ by saying that these results are not directly exportable to LoL.

## Riot's take on LoserQ

Riot Games has always been clear on this issue, claiming that there is no LoserQ in League of Legends. Here is the infamous tweet from Rioter Phroxzon.

<div style="display: flex; justify-content: center;">
<blockquote class="twitter-tweet">
<p lang="en" dir="ltr">Losers queue doesn&#39;t exist<br><br>
We&#39;re not intentionally putting bad players on your team to make you lose more. <br><br>
(Even if we assumed that premise, wouldn&#39;t we want to give you good players so you stop losing?)
<br><br>For ranked, we match you on your rating and that&#39;s all. If you&#39;ve won a…</p>&mdash; Matt Leung-Harrison (@RiotPhroxzon) 
<a href="https://twitter.com/RiotPhroxzon/status/1756511358571643286?ref_src=twsrc%5Etfw">February 11, 2024</a></blockquote> 
</div>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

Many Rioters had similar takes on reddit and other sites, but a portion of the community is still convinced this is a lie. Just take a look at the comments on the tweet to get an idea.

## Why did I redo the analysis?

!!! Quote annotate "Comment from [Matos3001](https://www.reddit.com/r/leagueoflegends/comments/15k2nw4/comment/jvlq50c/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) on my previous post"

    [...] while [you] might understand a lot about balls in the sky, [you] are no statistician. [...] (1)

 1. <figure markdown="span">
  ![Relevant meme](https://i.kym-cdn.com/entries/icons/original/000/035/410/Screen_Shot_2020-10-05_at_11.51.58_AM.png){ width="400" }
</figure>

Back in the summer of 2023, I posted [some clues](https://www.reddit.com/r/leagueoflegends/comments/15k2nw4/existence_of_loser_queue_a_statistical_analysis/) on reddit showing that there was probably no LoserQ mechanism in League of Legends. This post was well received and I had a lot of feedback from the community. There was some criticism, the most constructive being that the sample I collected was only Master+ players, which is not necessarily representative of all players. To be honest, this first analysis was a 4Fun thing that I did while I had to write my manuscript with deadlines that were really tight. The whole process for the reddit post took me one day, from data collection to final publication. This was clearly rushed from a scientific point of view and I wanted to do better when I had more time. Now that I'm defending my PhD (and have the free time to do so), I want to provide a more robust analysis, something I wouldn't be ashamed to publish in a scientific journal.

The current analysis was started on 9 April 2024, and it took $\sim 2$ months of spare time here and there to put all the pieces together, get this website up with all the content, and get it reviewed by people. It took that long because I wanted it to be reproducible, well documented and peer reviewed, unlike any other analysis I have seen. You'll find the dataset on [HuggingFace](https://huggingface.co/datasets/renecotyfanboy/leagueData), and the code on the associated [GitHub repository](https://github.com/renecotyfanboy/leagueProject). Anyone is welcome to reproduce the analysis and to criticise the methodology, results or interpretation. I am open to any discussion and will update this website with the most relevant comments I receive.

## What can I show or not?

Riot may well know the outcome of a match (although I doubt it) and could use this information to match you with people who will make you lose. They could keep people in a loop of winning and losing, which would make them play more and spend more money on the game. They could autofill you when they want to, or use Tencent's almighty AI to know what colour your underwear is. My goal here is to investigate this and see if there is any evidence of such a mechanism in the data. Don't get me wrong:

1. I cannot show that Riot is matching you with people who are already losing. That is too much recursive calling of the API, and my poor personal API key would take eons to collect what I need. Also, I generally can't predict the outcome of a game with the data I collect through the API (or at least I can't collect enough data in a short time frame to do so). This is something I might take a look at when I play around with the amazing `trueskill2` algorithm.
2. I cannot write about in-game feel. I don't care if games feel unwinnable or unloseable, only the results in terms of wins and losses matter in this situation. In particular, I cannot prove or disprove that Riot has perfect control over the outcome of games. But if they do, and if they make it deviate from randomness, I could see it in the dataset.
3. I cannot deny that players go on winning or losing streaks, because they do. Especially in the early seasons, when people are ranked too high or too low, they will experience winning and losing streaks, and this is to be expected as the algorithm is not perfect at predicting your level with a low game count, and it may take some time for you to reach your true rank. This is why I focus on players who are supposedly close to their true rank. The dataset contains the last 100 games of players who played at least 200 games in the first split of 2024, from which remakes are removed, resulting in at least 85 games per player.

However, there are a lot of things you can do with the data from match histories. But as I said in the reddit post, I cannot disprove the existence of the LoserQ. The best I can say is "if it does exist, it either works or it does not". We'll discuss this a bit more in the conclusion.

*[EOMM]: Engagement Optimized Matchmaking

[^1]: [**EOMM: An Engagement Optimized Matchmaking
Framework**, *Chen & al.* (2017)](https://web.cs.ucla.edu/~yzsun/papers/WWW17Chen_EOMM] frameworks)