% Decisions
?::d1.

% State Variables
state_variables(s1, s2, s3).

% Transition
0.02::x(s1) :- s1, d1.
0.65::x(s1) :- \+s1, d1.
0.01::x(s1) :- s1, \+d1.
0.88::x(s1) :- \+s1, \+d1.

0.69::x(s2) :- s2, d1.
0.97::x(s2) :- \+s2, d1.
0.73::x(s2) :- s2, \+d1.
0.53::x(s2) :- \+s2, \+d1.

0.76::x(s3) :- s3, d1.
0.94::x(s3) :- \+s3, d1.
0.55::x(s3) :- s3, \+d1.
0.35::x(s3) :- \+s3, \+d1.

% Structure
0.68::x(s2) :- s1, x(s1).
0.76::x(s2) :- \+s1, x(s1).
0.95::x(s2) :- s1, \+x(s1).
0.93::x(s2) :- \+s1, \+x(s1).

0.42::x(s3) :- s2, x(s2).
0.92::x(s3) :- \+s2, x(s2).
0.92::x(s3) :- s2, \+x(s2).
0.1::x(s3) :- \+s2, \+x(s2).

% Rewards
utility(d1, 1).
utility(s1, 5).
utility(s2, -2).
utility(s3, -5).
