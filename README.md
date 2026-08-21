I have decided to change the topic of the paper. Instead of continuing to sharpen the localization bound on the Taylor intermediate point, I am now focusing on a rigidity question that came directly out of that work.

Why the change

The Taylor localization branch produced a classification result:
the Taylor–Lagrange intermediate point satisfies θₙ = 1/(n+2) on every interval if and only if f is a polynomial of degree ≤ n+2.
At n = 0 this is the classical statement that the mean value point is the midpoint of every interval exactly when f is a quadratic.
That raised the obvious follow-up: what if the condition is only imposed on intervals of one fixed length instead of all intervals? The expected answer is that quadratics remain the only solutions.
They do not. Rigidity fails, and there is an infinite family of non-polynomial solutions. That failure is more interesting than the bound I was originally improving, so the paper is now about this.

The new question

If the midpoint of [x, x+1] is a mean value point of f for every real x, must f be a quadratic?

Equivalently, characterize all solutions of

f(x+1) - f(x) = f'(x + 1/2)

Substituting f = e^(λx) reduces this to the transcendental equation 2 sinh(λ/2) = λ, which is entire of order 1 and therefore has infinitely many complex roots. Each root yields a genuine non-polynomial solution; the quadratics turn out to be the triple root at λ = 0.



Contents
WMVT - "A Localization Theorem for the Weighted Mean Value Theorem" (finished, under peer review)
rigidity - characteristic root solver, counterexample verification, figures (in progress)
paper -  the new paper (in progress)


Timeline
10.07.2026 — Started writing the Taylor localization paper
26.07.2026 — Topic changed to the rigidity, believe it is more interesting then Taylor localiztion
Expected finish date: 01.10.2026
