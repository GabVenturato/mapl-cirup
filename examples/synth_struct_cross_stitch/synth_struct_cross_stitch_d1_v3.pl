% Decisions
?::d1.

% State Variables
state_variables(s1, s2, s3).

% Transition
0.91::x(s1) :- s1, d1.
0.19::x(s1) :- \+s1, d1.
0.28::x(s1) :- s1, \+d1.
0.97::x(s1) :- \+s1, \+d1.

0.5::x(s2) :- s2, d1.
0.94::x(s2) :- \+s2, d1.
0.39::x(s2) :- s2, \+d1.
0.85::x(s2) :- \+s2, \+d1.

0.48::x(s3) :- s3, d1.
0.74::x(s3) :- \+s3, d1.
0.4::x(s3) :- s3, \+d1.
0.66::x(s3) :- \+s3, \+d1.

% Structure
0.37::x(s1) :- s2, x(s2).
0.88::x(s1) :- \+s2, x(s2).
0.78::x(s1) :- s2, \+x(s2).
0.74::x(s1) :- \+s2, \+x(s2).

0.09::x(s2) :- s1.
0.66::x(s2) :- \+s1.
0.11::x(s2) :- s3.
0.16::x(s2) :- \+s3.

0.84::x(s3) :- s2, x(s2).
0.37::x(s3) :- \+s2, x(s2).
0.73::x(s3) :- s2, \+x(s2).
0.47::x(s3) :- \+s2, \+x(s2).

% Rewards
utility(d1, 0).
utility(s1, 5).
utility(s2, 3).
utility(s3, 3).
