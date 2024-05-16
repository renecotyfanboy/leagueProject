# Auto-correlation

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
--8<-- "loserQ/assets/simulated_correlation.json"
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

## Instantaneous winrate analysis? 