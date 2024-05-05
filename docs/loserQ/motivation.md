# Motivations

Back in summer 2023, I posted on reddit [some hints](https://www.reddit.com/r/leagueoflegends/comments/15k2nw4/existence_of_loser_queue_a_statistical_analysis/). 
showing that there were probably no LoserQ mechanism in League of Legends. This post was well received, and I got a lot 
of feedback from the community. However, some people were still skeptical about the results, and I wanted to go further
in the analysis, due to multiple factors. 

## What's LoserQ?

At this point, everyone playing League of Legends has heard about the LoserQ at least once. It is something which is 
praised by many streamers (at least in France, where I live), and it is suprisingly hard to find a clear definition 
that satisfies everyone. Here are a few quotes I found on the internet :

!!! Quote "[GhostCalib3r](https://www.reddit.com/user/GhostCalib3r/) on [this Reddit post](https://www.reddit.com/r/leagueoflegends/comments/htginy/what_is_losers_queue/)"
    [...] it's the tendency to lose 3-5 games in a row after winning 3-5 in a row; "losers queue into winners queue" 
    and vice-versa. Some people refer to loser's queue as "forced 50% winrate". [...]

!!! Quote "[LLander_](https://www.reddit.com/user/LLander_/) on [the same post](https://www.reddit.com/r/leagueoflegends/comments/htginy/what_is_losers_queue/)"
    It's when the matchmaking constantly puts you with people that you have a very low chance to win with


<blockquote class="reddit-embed-bq" data-embed-height="432"><a href="">Comment</a><br> by<a href="https://www.reddit.com/user/wazaaup/">u/wazaaup</a> from discussion<a href="https://www.reddit.com/r/leagueoflegends/comments/htginy/what_is_losers_queue/"><no value=""></no></a><br> in<a href="https://www.reddit.com/r/leagueoflegends/">leagueoflegends</a></blockquote><script async="" src="https://embed.reddit.com/widgets.js" charset="UTF-8"></script>
<blockquote class="reddit-embed-bq" data-embed-height="240"><a href="https://www.reddit.com/r/leagueoflegends/comments/htginy/comment/fygt7ie/">Comment</a><br> by<a href="https://www.reddit.com/user/wazaaup/">u/wazaaup</a> from discussion<a href="https://www.reddit.com/r/leagueoflegends/comments/htginy/what_is_losers_queue/"><no value=""></no></a><br> in<a href="https://www.reddit.com/r/leagueoflegends/">leagueoflegends</a></blockquote><script async="" src="https://embed.reddit.com/widgets.js" charset="UTF-8"></script>
<blockquote class="reddit-embed-bq" data-embed-height="240"><a href="https://www.reddit.com/r/leagueoflegends/comments/ypkhis/comment/ivjgpkm/">Comment</a><br> by<a href="https://www.reddit.com/user/wazaaup/">u/wazaaup</a> from discussion<a href="https://www.reddit.com/r/leagueoflegends/comments/htginy/what_is_losers_queue/"><no value=""></no></a><br> in<a href="https://www.reddit.com/r/leagueoflegends/">leagueoflegends</a></blockquote><script async="" src="https://embed.reddit.com/widgets.js" charset="UTF-8"></script>

I tried to come up with an exhaustive formulation of what the LoserQ is, why it would exist, and how it would work. 
Here is a summary of what I found :

!!! Quote

    - What? Loser queue is a mechanism in matchmaking that improves player engagement by artificially enabling win and lose streaks.
    - How? When losing, you get a higher probability of being matched with people that are themselves in lose streak and against players on win streaks, thus reducing your probability of winning the game.
    - Why? Improving player's engagement is always good for business, and since League is a game which is hard to start to play, it is easier to retain old players to keep a good player base.
    - Hints? Other companies such as EA are using Engagement Optimized Matchmaking frameworks is their competitive games such as APEX.


## Past hints 

The main concern I was facing is that I only focused on Master+ players, and
this is not necessarily representative of the whole player base. Moreover, I didn't go deep in the details of the 
methodology I used, and maybe was too quick on some points. Therefore, I decided to collect a cleaner dataset and redo 
this analysis, but this time going deeper in the details. I wanted to provide a more robust analysis, that would be 
paper-worthy.

The idea was to investigate the existence of a "Loser Queue" in League of Legends. The Loser Queue is a concept that has been around for a long time in the community. It is the idea that Riot Games would match you with bad players if you have been winning too much, in order to make you lose more games. The goal of this analysis was to investigate this concept using a dataset of match histories.

RIOT is cheating with the fairness -> investigate history of games -> prove that losers queue exists
RIOT is matching you with filled people or bad players -> investigate the performance of players

<blockquote class="twitter-tweet">
<p lang="en" dir="ltr">Losers queue doesn&#39;t exist<br><br>
We&#39;re not intentionally putting bad players on your team to make you lose more. <br><br>
(Even if we assumed that premise, wouldn&#39;t we want to give you good players so you stop losing?)
<br><br>For ranked, we match you on your rating and that&#39;s all. If you&#39;ve won a…</p>&mdash; Matt Leung-Harrison (@RiotPhroxzon) 
<a href="https://twitter.com/RiotPhroxzon/status/1756511358571643286?ref_src=twsrc%5Etfw">February 11, 2024</a></blockquote> 
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
