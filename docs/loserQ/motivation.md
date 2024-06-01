# Why do I even bother ?

Back in summer 2023, I posted on reddit [some hints](https://www.reddit.com/r/leagueoflegends/comments/15k2nw4/existence_of_loser_queue_a_statistical_analysis/). 
showing that there were probably no LoserQ mechanism in League of Legends. This post was well received, and I got a lot 
of feedback from the community. However, some people were still skeptical about the results, and I wanted to go further
in the analysis, due to multiple factors. 

https://towardsdatascience.com/analyzing-tilt-to-win-more-games-league-of-legends-347de832a5b1

## What's LoserQ?

At this point, everyone playing League of Legends has heard about the LoserQ at least once. It is something which is 
praised by many streamers (at least in France, where I live), and it is suprisingly hard to find a clear definition 
that satisfies everyone. Here are a few quotes I found on the internet :

!!! Quote "GhostCalib3r on [this Reddit post](https://www.reddit.com/r/leagueoflegends/comments/htginy/what_is_losers_queue/)"
    [...] it's the tendency to lose 3-5 games in a row after winning 3-5 in a row; "losers queue into winners queue" 
    and vice-versa. Some people refer to loser's queue as "forced 50% winrate". [...]

!!! Quote "LLander_ on [this Reddit post](https://www.reddit.com/r/leagueoflegends/comments/htginy/what_is_losers_queue/)"
    It's when the matchmaking constantly puts you with people that you have a very low chance to win with

!!! Quote "AcrobaticApricot on [this Reddit post](https://www.reddit.com/r/leagueoflegends/comments/1at554j/comment/kquvwy4/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)"
    I think the idea is that there are 5 losers, all in “loser’s queue,” who play against 5 winners. So everyone on a 
    loser’s queue player’s team is also in loser’s queue. 

!!! Quote "MattWolfTV on [this Reddit post](https://www.reddit.com/r/leagueoflegends/comments/1at554j/comment/kquwo2z/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)"
    Often when people complain about losers queue it is more about the game being determined 
    from matchmaking aka loading screen. 

!!! Quote "Straight_Rule_535 on [this Reddit post](https://www.reddit.com/r/leagueoflegends/comments/1at554j/comment/kquxhr0/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)"
    Idk went to masters with 60% wr, all in lobby has 60-70wr. I lose two matches in a row. Only get matched with 
    teammates ~45%wr and enemies wr is still 60-70. It might not exist but this is sussy 

So, LoserQ is the kind of stuff which everyone get and feel, but no one can really explain or define. Back in the day, 
people where just sharing their personal game histories full of win-streaks and loss-streaks, but nowadays, I think this 
is more about getting matched in games that are unwinnable from the lobby. Many people are stating that Riot can now 
from the start the outcome of a single game, and they are right! If you have a good proxy of the level of all the players, 
their mental state and others, it would be easy to create lobby where a team is *expected* to slightly under-perform 
when compared to the other team. Add this the extra leverage of autofilling players, and you can see why some are 
convinced that Riot cheats with the matchmaking. 

But why would Riot do that? Well, the answer is simple, and it is the same for every company : **money**. The more you
play, the more you are likely to spend money on the game. And the more you are likely to spend money on the game, the more
Riot is likely to make money. Therefore, it is in their interest to keep you playing, and the best way to do this is to
keep you engaged. And what is the best way to keep you engaged? Well, some thinks that getting the players into 
successions of win and loss streaks is a good way to do this. There is few studies on the subject, but some companies
such as EA are using Engagement Optimized Matchmaking[^1] is their competitive games such as APEX. In the associated 
publication, they studied how some patterns of win and loss streaks are linked to players stop playing the game for 
longer periods of time. They focused on the correlation between the 3 last games players and the players not playing the 
game for a week, and they found that in their sample, 3% of the players are likely to abandon the game if they won 
their 3 last games, while 5% of the players will stop after patterns such as win-win-loss or full losses. These results 
are quite anti-LoserQ, the triple losses is the pattern that Riot should avoid at all cost to keep the players engaged.
But their results are not directly exportable to the case of LoL, as the game they used can result in draws. 

## Riot's take on LoserQ

Riot Game has always been very clear about this topic : they claim that there is no LoserQ in League of Legends.
Here is the infamous tweet from Riot Phroxzon on this topic :

<blockquote class="twitter-tweet">
<p lang="en" dir="ltr">Losers queue doesn&#39;t exist<br><br>
We&#39;re not intentionally putting bad players on your team to make you lose more. <br><br>
(Even if we assumed that premise, wouldn&#39;t we want to give you good players so you stop losing?)
<br><br>For ranked, we match you on your rating and that&#39;s all. If you&#39;ve won a…</p>&mdash; Matt Leung-Harrison (@RiotPhroxzon) 
<a href="https://twitter.com/RiotPhroxzon/status/1756511358571643286?ref_src=twsrc%5Etfw">February 11, 2024</a></blockquote> 
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

## Why did I redo the analysis?

!!! Quote annotate "Comment from [Matos3001](https://www.reddit.com/r/leagueoflegends/comments/15k2nw4/comment/jvlq50c/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) on my previous post"

    [...] while [you] might understand a lot about balls in the sky, [you] are no statistician. [...] (1)

 1. <figure markdown="span">
  ![Relevant meme](https://i.kym-cdn.com/entries/icons/original/000/035/410/Screen_Shot_2020-10-05_at_11.51.58_AM.png){ width="400" }
</figure>

First analysis was a 4Fun stuff I did during my manuscript writing in a day. 
I do not know convincing analysis that prove the existence of loserQ, all I read 
was either irreproducible, erroneous or anectodical. 

After my first post, I was really surprised by the number of people how disclaimed the results, saying that I was not a
statistician. 
C'est vrai que je suis resté en surface, ça manquait le pas entre un simple truc pour s'amuser et des résultats de qualité scientifique.
While I am perfectly aware and transparent about the bias of the previous and current study, I am still an 
astrophysicist, and I know a bit about statistics. When doing evidence based science, statistic is the backbone of any 
solid result you can get.

The main concern I was facing is that I only focused on Master+ players, and
this is not necessarily representative of the whole player base. Moreover, I didn't go deep in the details of the 
methodology I used, and maybe was too quick on some points. Therefore, I decided to collect a cleaner dataset and redo 
this analysis, but this time going deeper in the details. I wanted to provide a more robust analysis, that would be 
paper-worthy.

The idea was to investigate the existence of a "Loser Queue" in League of Legends. The Loser Queue is a concept that has been around for a long time in the community. It is the idea that Riot Games would match you with bad players if you have been winning too much, in order to make you lose more games. The goal of this analysis was to investigate this concept using a dataset of match histories.

RIOT is cheating with the fairness -> investigate history of games -> prove that losers queue exists
RIOT is matching you with filled people or bad players -> investigate the performance of players
I want to make an analysis which is paper-worthy, and that would be a good starting point for a publication.
I must be well documented, peer-validated and reproducible. 


[^1]: [**EOMM: An Engagement Optimized Matchmaking
Framework**, *Chen & al.* (2017)](https://web.cs.ucla.edu/~yzsun/papers/WWW17Chen_EOMM] frameworks)