% NEXT: Revert to a true normal form and implement that first, 
% perhaps simplify later (making u a free variable)

function [tspmupm,B,lpiter] = lp_solver(AV,r0,tspmupm,B,options)
% This function provides a customized simplex solver for the following LP:
% 
% if options.norm == 'l1':
% 
%   Minimize e^\top t 
%   with variables (t,sp,sm,up,um) \in \R^n \times \R^n \times \R^n \times R^k \times R^k
%   s.t.   t - sp + AV * up - AV * um = r0
%   and  - t + sm + AV * up - AV * um = r0
%   and  t, sp, sm, up, um \ge 0 
%
% tspmupm is an initial guess and it must be provided. It must be a vertex
% associated with the basis B.
%
% if options.norm == 'linf':
%
%   NOT YET IMPLEMENTED
% 
% The function returns a solution tspmupm (a vertex) and the corresponding
% basis B, as well as the number of simplex steps lpiter.

% Determine the problem size
k = size(AV,2);
n = length(r0);
l = 3*n+2*k;

% Check initial guess dimension
assert(length(tspmupm) == l,'lp_solver: initial guess has incorrect length.');

% Initialize some counters
lpiter = 0;
done = 0;

% Set the termination tolerance 
tol = 1e-6;

% Set some problem data
e = [ones(n,1); zeros(2*n+2*k,1)];
Aeq = [[speye(n), -speye(n), sparse(n,n), AV, -AV]; [-speye(n), sparse(n,n), speye(n), AV, -AV]];
beq = [r0; r0];
assert(rank(full(Aeq)) == 2*n,'lp_solver: Aeq does not have full rank.')

% Determine the non-basic indices
N = setdiff([1:l]',B);

% Check feasibility of initial guess and corresponding basis
assert(~any(tspmupm(N)),'lp_solver: initial non-basic elements are not all zero.');
assert(max(abs(Aeq*tspmupm-beq)) < 1e-6,'lp_solver: initial guess does not satisfy equality constraints.');
assert(all(tspmupm >= 0),'lp_solver: initial guess is not non-negative.' );
assert(rank(full(Aeq(:,B))) == length(B),'lp_solver: initial basis matrix is rank-deficient.');

% Output the initial objective
% e' * tspmupm

% Enter simplex loop
while (~done)

	% Compute the reduced cost vector
	Delta = e - Aeq' * (Aeq(:,B)' \ e(B));

	% Round it to prevent errors from just slightly
	% negative entries (which should be zero)
	Delta(find(abs(Delta) < sqrt(eps))) = 0;

	% Check for optimality
	if (all(Delta(N) >= 0))
		done = 1;
	end

	% Perform a simplex step if not done
	if (~done)

		% Pricing
		k = N(min(find(Delta(N) < 0)));

		% Determine the step direction
		d = zeros(l,1);
		d(B) = Aeq(:,B) \ (Aeq(:,k));

		% Round near-zero elements to zero
		ix = find(abs(d) < sqrt(eps));
		d(ix) = 0;

		% Perform the ratio test and update the vector of unknowns
		q = tspmupm ./ d;
		q(N) = NaN;
		q(find(d <= 0)) = NaN;
		[t,r] = min(q);
		tspmupm(B) = tspmupm(B) - t * d(B);
		tspmupm(k) = t;
		tspmupm(r) = 0;

		% Output the new objective
		% fprintf('[Delta, tspmupm, d, q]\n');
		% [Delta, tspmupm, d, q]
		% fprintf('[lpiter, k, r, t, e'' * tspmupm]\n');
		% [lpiter, k, r, t, e' * tspmupm]

		% Update the basis 
		B = setdiff(union(B,k),r);
		N = setdiff(union(N,r),k);

		% Check new basis for full rank
		assert(rank(full(Aeq(:,B))) == length(B) ,'lp_solver: basis matrix is rank-deficient.');

		% Round near-zero basic and non-basic elements to zero
		ix = find(abs(tspmupm) < sqrt(eps));
		tspmupm(ix) = 0;

		% Increase the iteration counter
		lpiter = lpiter + 1;
		if (lpiter > inf)
			done = -1;
		end

	end % if (~done)

end % while (~done)

