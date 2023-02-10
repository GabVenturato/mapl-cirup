% Decisions
?::d1.

% State Variables
state_variables(s1, s2, s3).

% Transition
0.32::x(s1) :- s1, d1.
0.6::x(s1) :- \+s1, d1.
0.46::x(s1) :- s1, \+d1.
0.25::x(s1) :- \+s1, \+d1.

0.79::x(s2) :- s2, d1.
0.78::x(s2) :- \+s2, d1.
0.89::x(s2) :- s2, \+d1.
0.87::x(s2) :- \+s2, \+d1.

0.47::x(s3) :- s3, d1.
0.36::x(s3) :- \+s3, d1.
0.18::x(s3) :- s3, \+d1.
0.21::x(s3) :- \+s3, \+d1.

% Structure
0.2::x(s1) :- s2, x(s2).
0.36::x(s1) :- \+s2, x(s2).
0.82::x(s1) :- s2, \+x(s2).
0.09::x(s1) :- \+s2, \+x(s2).

0.75::x(s2) :- s1.
0.09::x(s2) :- \+s1.
0.57::x(s2) :- s3.
0.34::x(s2) :- \+s3.

0.23::x(s3) :- s2, x(s2).
0.97::x(s3) :- \+s2, x(s2).
0.04::x(s3) :- s2, \+x(s2).
0.19::x(s3) :- \+s2, \+x(s2).

% Rewards
utility(d1, 1).
utility(s1, -2).
utility(s2, -3).
utility(s3, -1).
