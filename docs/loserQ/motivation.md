# Why do I even bother?

## What's LoserQ?

At this point, everyone playing League of Legends has heard about the LoserQ at least once. It is something which is praised by many streamers (at least in France, where I live), and it is surprisingly difficult to find a clear definition that satisfies everyone. Some people even rejected my previous work only stating that I didn't understand what LoserQ is. Here are quotes from the internet trying to explain what it is precisely:

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

So, LoserQ is the kind of stuff which everyone gets and feels, but no one can really explain or define. Some people share their personal game histories full of win-streaks and loss-streaks and complain that they are stuck in LoserQ. Some are bothered about getting matched in games that are unwinnable from the lobby. Many people are stating that Riot can know from the start the outcome of a single game, and I can't argue! If you have a good proxy of the level of all the players, their mental state and others, it would be easy to create a lobby where a team is *expected* to slightly under-perform when compared to the other team. Add this the extra leverage of autofilling players, and you can see why some are convinced that Riot cheats with the matchmaking. 

But why would Riot do that? Well, the answer is simple, and it is the same for every company : **money**. The more you play, the more you are likely to spend money on the game. And the more you are likely to spend money on the game, the more Riot is likely to make money. Therefore, it is in their interest to keep you playing, and the best way to do this is to keep you engaged. And what is the best way to keep you engaged? Well, some think that getting the players into successions of win and loss streaks is a good way to do this. There is a paper showing that specific patterns of win/loss could increase the player engagement[^1]. It is referred to as Engagement Optimized Matchmaking (EOMM), and some people claim that it has been implemented in games like APEX-legend, even if the developers say the opposite. 

In the EOMM publication, they studied how some patterns of win and loss streaks are linked to players stopping the game for longer periods of time. They focused on the correlation between the 3 last games players and the players not playing the game for a week, and they found that in their sample, 3% of the players are likely to abandon the game if they won their 3 last games, while 5% of the players will stop after patterns such as win-win-loss or full losses. These results are quite anti-LoserQ, as the triple losses is the pattern that Riot should avoid at all cost to keep the players engaged. But I could advocate for the LoserQ by saying that these results are not directly exportable to LoL.

## Riot's take on LoserQ

Riot Game has always been clear about this topic : they claim that there is no LoserQ in League of Legends. Here is the infamous tweet from the Rioter Phroxzon.

<div style="display: flex; justify-content: center;">
<blockquote class="twitter-tweet">
<p lang="en" dir="ltr">Losers queue doesn&#39;t exist<br><br>
We&#39;re not intentionally putting bad players on your team to make you lose more. <br><br>
(Even if we assumed that premise, wouldn&#39;t we want to give you good players so you stop losing?)
<br><br>For ranked, we match you on your rating and that&#39;s all. If you&#39;ve won a…</p>&mdash; Matt Leung-Harrison (@RiotPhroxzon) 
<a href="https://twitter.com/RiotPhroxzon/status/1756511358571643286?ref_src=twsrc%5Etfw">February 11, 2024</a></blockquote> 
</div>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

Many Rioters had takes on reddit and other, but a portion of the community is still convinced this is a lie. Just have a look at the comments on the tweet to get an idea.

## Why did I redo the analysis?

!!! Quote annotate "Comment from [Matos3001](https://www.reddit.com/r/leagueoflegends/comments/15k2nw4/comment/jvlq50c/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) on my previous post"

    [...] while [you] might understand a lot about balls in the sky, [you] are no statistician. [...] (1)

 1. <figure markdown="span">
  ![Relevant meme](https://i.kym-cdn.com/entries/icons/original/000/035/410/Screen_Shot_2020-10-05_at_11.51.58_AM.png){ width="400" }
</figure>

Back in summer 2023, I posted on reddit [some hints](https://www.reddit.com/r/leagueoflegends/comments/15k2nw4/existence_of_loser_queue_a_statistical_analysis/) showing that there was probably no LoserQ mechanism in League of Legends. This post was well received, and I got a lot of feedback from the community. Some criticism rose, the most constructive were about the sample I gathered, which was only composed of Master+ players, which is not necessarily representative of all the players. To be honest, this first analysis was a true 4Fun stuff I did while I was supposed to write my manuscript with deadlines that were really close. The whole process for the reddit post took me a day, from data gathering to the final publication. This was clearly rushed from a scientific point of view, and I wanted to do better work around this once I had more time. Now that I defended my PhD (and the associated spare time), I want to provide a more robust analysis, something that I wouldn't be ashamed to publish in a scientific journal.

The current analysis work was started on the 9 April 2024, and it took $\sim 2$ months of spare time here and there to get all the pieces together, provide this website with all the content, and get it reviewed by external people. It took that long because I wanted it to be reproducible, well documented and peer reviewed, unlike any other analysis I have seen. You'll find the dataset on [HuggingFace](https://huggingface.co/datasets/renecotyfanboy/leagueData), and the code on the associated [GitHub repository](https://github.com/renecotyfanboy/leagueProject). Anyone can reproduce the analysis and emits critics on the methodology, the results or the interpretation. I am open to any discussion, and I will update this website with the most relevant critics I receive.

*[EOMM]: Engagement Optimized Matchmaking
[^1]: [**EOMM: An Engagement Optimized Matchmaking
Framework**, *Chen & al.* (2017)](https://web.cs.ucla.edu/~yzsun/papers/WWW17Chen_EOMM] frameworks)