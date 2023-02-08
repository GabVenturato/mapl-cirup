% Decisions
?::d1.

% State Variables
state_variables(s1, s2, s3).

% Transition
0.97::x(s1) :- s1, d1.
0.5::x(s1) :- \+s1, d1.
0.97::x(s1) :- s1, \+d1.
0.51::x(s1) :- \+s1, \+d1.

0.91::x(s2) :- s2, d1.
0.19::x(s2) :- \+s2, d1.
0.28::x(s2) :- s2, \+d1.
0.97::x(s2) :- \+s2, \+d1.

0.5::x(s3) :- s3, d1.
0.94::x(s3) :- \+s3, d1.
0.39::x(s3) :- s3, \+d1.
0.85::x(s3) :- \+s3, \+d1.

% Structure
0.48::x(s2) :- s1, x(s1).
0.74::x(s2) :- \+s1, x(s1).
0.4::x(s2) :- s1, \+x(s1).
0.66::x(s2) :- \+s1, \+x(s1).

0.37::x(s3) :- s2, x(s2).
0.88::x(s3) :- \+s2, x(s2).
0.78::x(s3) :- s2, \+x(s2).
0.74::x(s3) :- \+s2, \+x(s2).

% Rewards
utility(d1, -1).
utility(s1, 1).
utility(s2, 4).
utility(s3, 2).
