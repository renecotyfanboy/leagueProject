import jax.numpy as jnp
import jax
from jax import lax 


@jax.jit
def find_streaks(array):
    
    array = jnp.asarray(array, dtype=bool)
    win_streaks = jnp.zeros_like(array, dtype=jnp.int32)
    loss_streaks = jnp.zeros_like(array, dtype=jnp.int32)
    previous = ~array[0]
    current_length = 1
    
    def body_fn(carry, current):
        
        win_streaks, loss_streaks, previous, current_length = carry 
        
        win_streaks = lax.select(
            (previous == current), 
            win_streaks, 
            win_streaks.at[current_length-1].add(previous)
        )

        loss_streaks = lax.select(
            (previous == current), 
            loss_streaks, 
            loss_streaks.at[current_length-1].add(~previous)
        )
        
        current_length = lax.select(previous == current, current_length + 1, 1)
        previous = current
        
        return (win_streaks, loss_streaks, previous, current_length), None 

    (win_streaks, loss_streaks, previous, current_length), _ = lax.scan(body_fn, (win_streaks, loss_streaks, previous, current_length), array)

    return win_streaks, loss_streaks