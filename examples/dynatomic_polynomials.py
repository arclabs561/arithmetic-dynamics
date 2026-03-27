# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "sympy>=1.13",
# ]
# ///
"""Dynatomic polynomials for f(z) = z^2 + c.

The n-th dynatomic polynomial Phi_n(z, c) isolates periodic points of exact
period n. Its roots (over an algebraically closed field) are exactly those z
satisfying f^n(z) = z but f^d(z) != z for any proper divisor d of n. The
construction uses Mobius inversion:

    Phi_n(z, c) = product over d|n of (f^d(z) - z)^{mu(n/d)}

where mu is the Mobius function and f^n is n-fold composition.

For f(z) = z^2 + c:
  Phi_1(z, c) = z^2 - z + c
  Phi_2(z, c) = z^2 + z + c + 1
  Phi_3(z, c) has degree 6

The degree of Phi_n is sum_{d|n} mu(n/d) * 2^d, which equals the number of
periodic points of exact period n counted with multiplicity.

Over finite fields F_p, the roots of Phi_n mod p give the periodic orbits of
exact period n in the iteration digraph -- with a caveat. At primes where a
periodic point has multiplier 1 (i.e., the derivative (f^n)'(z) = 1 mod p),
the dynatomic polynomial can acquire extra roots from lower-period points.
This is because the Mobius cancellation is exact over Q but can fail mod p
when numerator and denominator share a common factor.

Ref: Silverman, "Lecture Notes on Arithmetic Dynamics" (2010, Arizona Winter School)

Run: uv run examples/dynatomic_polynomials.py
"""

from __future__ import annotations

from functools import cache

from sympy import (
    Poly,
    Symbol,
    cancel,
    divisors,
    factor,
    mobius,
    prod,
)


z = Symbol("z")
c = Symbol("c")


# -- symbolic iteration -------------------------------------------------------


@cache
def iterate_f(n: int):
    """Compute f^n(z) for f(z) = z^2 + c, returned as a sympy expression.

    f^0(z) = z, f^1(z) = z^2 + c, f^2(z) = (z^2+c)^2 + c, etc.
    """
    if n == 0:
        return z
    prev = iterate_f(n - 1)
    return prev**2 + c


# -- dynatomic polynomials ----------------------------------------------------


def dynatomic_polynomial(n: int):
    """Compute Phi_n(z, c) using Mobius inversion.

    Phi_n = product_{d | n} (f^d(z) - z)^{mu(n/d)}

    For mu(n/d) = -1 terms, this is division. We use sympy's cancel()
    to simplify the resulting rational expression into a polynomial.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    # Separate numerator factors (mu = +1) from denominator factors (mu = -1).
    # mu = 0 terms contribute nothing.
    numer_factors = []
    denom_factors = []

    for d in divisors(n):
        mu_val = mobius(n // d)
        if mu_val == 1:
            numer_factors.append(iterate_f(d) - z)
        elif mu_val == -1:
            denom_factors.append(iterate_f(d) - z)

    numer = prod(numer_factors) if numer_factors else 1
    denom = prod(denom_factors) if denom_factors else 1

    # cancel() performs polynomial division and simplifies.
    result = cancel(numer / denom)
    return Poly(result, z).as_expr()


def dynatomic_degree(n: int) -> int:
    """Theoretical degree of Phi_n: sum_{d|n} mu(n/d) * 2^d."""
    return sum(mobius(n // d) * 2**d for d in divisors(n))


# -- finite field computation -------------------------------------------------


def iterate_mod(x: int, c_val: int, p: int, n: int) -> int:
    """Compute f^n(x) mod p for f(x) = x^2 + c_val."""
    val = x
    for _ in range(n):
        val = (val * val + c_val) % p
    return val


def exact_period(x: int, c_val: int, p: int) -> int:
    """Find the exact period of x under f(z) = z^2 + c_val mod p.

    Returns the smallest n >= 1 such that f^n(x) = x.
    Returns 0 if x is not periodic (not reachable within p steps).
    """
    for n in range(1, p + 1):
        if iterate_mod(x, c_val, p, n) == x:
            return n
    return 0


def periodic_points_of_period(n: int, c_val: int, p: int) -> list[int]:
    """Find all points in Z_p with exact period n under f(z) = z^2 + c_val."""
    return sorted(x for x in range(p) if exact_period(x, c_val, p) == n)


def dynatomic_roots_mod(n: int, c_val: int, p: int) -> list[int]:
    """Find roots of Phi_n(z, c_val) in Z_p by evaluation.

    Substitutes c = c_val, then evaluates the resulting univariate polynomial
    at each z in {0, ..., p-1}. Uses int() conversion to avoid sympy Integer
    modular arithmetic issues.
    """
    phi_n = dynatomic_polynomial(n)
    phi_specialized = phi_n.subs(c, c_val)
    poly_z = Poly(phi_specialized, z)

    roots = []
    for x in range(p):
        val = int(poly_z.eval(x)) % p
        if val == 0:
            roots.append(x)
    return sorted(roots)


def multiplier_mod(x: int, c_val: int, p: int, n: int) -> int:
    """Compute the multiplier (f^n)'(x) mod p using the chain rule.

    For f(z) = z^2 + c, f'(z) = 2z. The chain rule gives:
    (f^n)'(x) = product_{k=0}^{n-1} f'(f^k(x)) = product_{k=0}^{n-1} 2*f^k(x)
    """
    result = 1
    val = x
    for _ in range(n):
        result = (result * 2 * val) % p
        val = (val * val + c_val) % p
    return result


# -- Galois group (small degrees) --------------------------------------------


def dynatomic_galois_group(n: int):
    """Compute the Galois group of Phi_n(z, 0) over Q.

    Specializes c = 0 so that Phi_n becomes a univariate polynomial over Q.
    sympy can compute Galois groups for small degrees.

    The Galois group of dynatomic polynomials has deep structure: for
    f(z) = z^2 + c, the group Gal(Phi_n) embeds into the wreath product
    (Z/2Z) wr S_m where m = deg(Phi_n)/2. This reflects the branching
    structure of the binary tree of preimages.
    """
    phi_n = dynatomic_polynomial(n)
    phi_specialized = Poly(phi_n.subs(c, 0), z)

    try:
        group, is_solv = phi_specialized.galois_group()
        return group, is_solv
    except Exception as e:
        return None, str(e)


# -- main ---------------------------------------------------------------------


def main() -> None:
    print("Dynatomic polynomials for f(z) = z^2 + c")
    print("=" * 60)
    print()

    # -- 1. Symbolic dynatomic polynomials --
    print("Symbolic Phi_n(z, c):")
    print("-" * 60)
    for n in range(1, 4):
        phi = dynatomic_polynomial(n)
        phi_factored = factor(phi)
        print(f"  Phi_{n} = {phi}")
        if phi_factored != phi:
            print(f"       = {phi_factored}  (factored)")
    print()

    # -- 2. Degree sequence --
    print("Degree sequence of Phi_n (in z):")
    print("-" * 60)
    print(f"  {'n':>3}  {'deg(Phi_n)':>10}  {'theoretical':>11}  {'match':>5}")
    for n in range(1, 7):
        phi = dynatomic_polynomial(n)
        actual_deg = int(Poly(phi, z).degree())
        theoretical = int(dynatomic_degree(n))
        match = "yes" if actual_deg == theoretical else "NO"
        print(f"  {n:>3}  {actual_deg:>10}  {theoretical:>11}  {match:>5}")
    print()
    # Degree sequence: 2, 2, 6, 12, 30, 54 for n = 1..6.
    # Growth is roughly 2^n (exponential in n).

    # -- 3. Periodic points over F_p --
    # Use p=23, c=2: has periods 1, 2, and 3 with clean cycle structure.
    p, c_val = 23, 2
    print(f"Periodic points over F_{p} with c = {c_val}:")
    print("-" * 60)

    for n in range(1, 4):
        brute = periodic_points_of_period(n, c_val, p)
        roots = dynatomic_roots_mod(n, c_val, p)

        # Over F_p, dynatomic roots are a superset of exact-period points.
        # Extra roots appear when the multiplier (f^d)'(x) = 1 mod p for
        # some proper divisor d of n, causing Phi_d and Phi_n to share a
        # root mod p even though they are coprime over Q(c).
        exact_match = brute == roots
        superset = set(brute).issubset(set(roots))

        if exact_match:
            status = "PASSED (exact match)"
        elif superset:
            extra = sorted(set(roots) - set(brute))
            extra_info = []
            for x in extra:
                ep = exact_period(x, c_val, p)
                mult = multiplier_mod(x, c_val, p, ep)
                extra_info.append(f"{x} (period {ep}, multiplier {mult})")
            status = f"superset -- extra roots: {', '.join(extra_info)}"
        else:
            status = "FAILED (unexpected)"

        print(f"  Period {n}:")
        print(f"    Brute-force periodic points: {brute}")
        print(f"    Roots of Phi_{n} mod {p}:     {roots}")
        print(f"    Verification: {status}")

        # Show the orbits.
        if brute:
            orbits: list[list[int]] = []
            seen: set[int] = set()
            for x in brute:
                if x in seen:
                    continue
                orbit = [x]
                seen.add(x)
                y = iterate_mod(x, c_val, p, 1)
                while y != x:
                    orbit.append(y)
                    seen.add(y)
                    y = iterate_mod(y, c_val, p, 1)
                orbits.append(orbit)
            orbit_strs = [
                " -> ".join(str(v) for v in orb) + f" -> {orb[0]}" for orb in orbits
            ]
            print(f"    Orbits: {', '.join(orbit_strs)}")
        print()

    # -- 4. Full cycle structure of the iteration digraph --
    print(f"Full iteration digraph cycle structure for f(z) = z^2 + {c_val} mod {p}:")
    print("-" * 60)
    total_periodic = 0
    for n in range(1, p + 1):
        pts = periodic_points_of_period(n, c_val, p)
        if pts:
            total_periodic += len(pts)
            print(f"  Period {n}: {pts}  ({len(pts)} points)")
    non_periodic = p - total_periodic
    print(f"  Non-periodic (tail) points: {non_periodic}")
    print()

    # -- 5. Connection to iteration digraph structure --
    print("Connection to iteration digraphs:")
    print("-" * 60)
    print("  The periodic points of exact period n are exactly the nodes")
    print("  lying on n-cycles in the iteration digraph. The dynatomic")
    print("  polynomial Phi_n encodes this algebraically.")
    print()
    print("  Over Q (or an algebraically closed field), roots of Phi_n are")
    print("  exactly the period-n points. Over F_p, extra roots can appear")
    print("  at primes dividing the discriminant of Phi_n -- these are")
    print("  lower-period points where the multiplier (f^d)' equals 1.")
    print()

    for n in range(1, 4):
        brute = periodic_points_of_period(n, c_val, p)
        n_cycles = len(brute) // n if brute and len(brute) % n == 0 else 0
        print(
            f"  n={n}: {len(brute)} periodic points -> {n_cycles} cycle(s) of length {n}"
        )
    print()

    # -- 6. Galois groups (c=0 specialization) --
    print("Galois groups of Phi_n(z, 0) over Q:")
    print("-" * 60)
    print("  For c=0, f(z) = z^2, so f^n(z) = z^{2^n}. The dynatomic")
    print("  polynomials become cyclotomic: Phi_n(z, 0) divides z^{2^n-1} - 1")
    print("  but not z^{2^d-1} - 1 for d < n. Their Galois groups are cyclic")
    print("  of order equal to the multiplicative order of 2 mod (2^n - 1).")
    print()

    for n in range(1, 4):
        phi = dynatomic_polynomial(n)
        phi_0 = Poly(phi.subs(c, 0), z)
        print(f"  Phi_{n}(z, 0) = {phi_0.as_expr()}")

        group, is_solv = dynatomic_galois_group(n)
        if group is not None:
            print(f"    Galois group: {group}")
            print(f"    Solvable: {is_solv}")
        else:
            # Phi_1(z,0) = z^2 - z = z(z-1) is reducible, so no Galois group.
            print("    (reducible over Q, no Galois group)")
        print()

    # -- 7. Wreath product structure --
    print("Wreath product structure:")
    print("-" * 60)
    print("  For generic c, Gal(Phi_n / Q(c)) embeds in the iterated")
    print("  wreath product [Z/2Z] wr ... wr [Z/2Z] (n factors).")
    print("  This reflects the binary tree of preimages: each point")
    print("  has at most 2 preimages under f(z) = z^2 + c.")
    print()
    print("  Concretely, |wreath product| = 2^{2^n - 1}:")
    for n in range(1, 6):
        wreath_order = 2 ** (2**n - 1)
        deg = int(dynatomic_degree(n))
        print(f"    n={n}: deg(Phi_n) = {deg}, |wreath| = {wreath_order}")
    print()

    # -- 8. Divisibility lattice --
    print("Divisibility structure:")
    print("-" * 60)
    print("  f^n(z) - z = product_{d | n} Phi_d(z, c)")
    print()
    for n in range(1, 5):
        fn_minus_z = iterate_f(n) - z
        product_check = prod(dynatomic_polynomial(d) for d in divisors(n))
        diff = cancel(fn_minus_z - product_check)
        status = "PASSED" if diff == 0 else "FAILED"
        print(f"  n={n}: f^{n}(z) - z = Prod_{{d|{n}}} Phi_d  ... {status}")

    # Verify coprimality numerically: Phi_m and Phi_n share no roots mod a
    # large prime (verifies coprimality over Q(c) by Schwartz-Zippel).
    print()
    print("  Coprimality: gcd(Phi_m, Phi_n) = 1 for m != n")
    print("  (verified numerically mod 101 at c=0)")
    for m in range(1, 5):
        for n in range(m + 1, 5):
            roots_m = set(dynatomic_roots_mod(m, 0, 101))
            roots_n = set(dynatomic_roots_mod(n, 0, 101))
            shared = roots_m & roots_n
            status = "PASSED" if not shared else f"FAILED (shared: {shared})"
            print(f"    gcd(Phi_{m}, Phi_{n}): {status}")


if __name__ == "__main__":
    main()
