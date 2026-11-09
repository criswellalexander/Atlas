
import jax

# Used to enable 64-bit mode in JAX for better precision in calculations.
if not jax.config.read('jax_enable_x64'):
    
    # Enable 64-bit mode in JAX for better precision in calculations
    jax.config.update('jax_enable_x64', True)

    # Check the flag rather than creating an array: an array would initialize
    # a device (the GPU, if present) on every import, including in joblib
    # workers that never use JAX.
    if not jax.config.read('jax_enable_x64'):
        msg = "JAX default dtype is still not float64 even after enabling. " + \
              "You may lose needed precision for many calculations."
        print(msg)
    else:
        msg = "JAX 64-bit mode automatically enabled successfully."
        print(msg)

