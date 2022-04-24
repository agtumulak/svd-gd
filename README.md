# Singular Value Decomposition with Monotonic Constraint

Traditional methods of performing Singular Value Decomposition typically use an
iterative method which factorizes a matrix into an eigenvalue-revealing form by
applying a sequence of unitary transformations. The resulting factorization can
be used to find an optimal rank-r approximation of the original matrix with
respect to the Frobenius norm. However, if additional constraints on the form
of the SVD are required, this constrained optimization method might yield the
desired results.
