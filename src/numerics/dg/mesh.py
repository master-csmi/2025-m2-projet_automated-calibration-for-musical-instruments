import jax.numpy as jnp

def create_uniform_nodes_with_ghosts(N_intervals, x_min=0.0, x_max=1.0):
    dx = (x_max - x_min) / N_intervals
    x_nodes = jnp.linspace(x_min, x_max, N_intervals + 1)
    left_ghost  = jnp.array([x_nodes[0] - dx])
    right_ghost = jnp.array([x_nodes[-1] + dx])
    ghost_cells = jnp.concatenate([left_ghost, right_ghost])
    return x_nodes, ghost_cells

def cell_edges_from_nodes(x_nodes):
    return x_nodes[:-1], x_nodes[1:]