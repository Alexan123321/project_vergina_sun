import tenseal as ts
from dataclasses import dataclass
import numpy as np

#TODO: COMMENTING:
def _vector_sum(X: np.array([])) -> np.array([]):
    n = len(X[0])
    __sum = np.zeros(n)
    for i in range(n):
        __sum += X[i]
    return __sum

# Setup TenSEAL context
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
          )
context.generate_galois_keys()
context.global_scale = 2**40

v1 = np.array([0, 1, 2, 3, 4])
v2 = np.array([4, 3, 2, 1, 0])

# encrypted vectors
enc_v1 = ts.ckks_vector(context, v1)
enc_v2 = ts.ckks_vector(context, v2)

result = _vector_sum(enc_v1)

print(result.decrypt())
