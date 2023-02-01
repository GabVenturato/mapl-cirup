% Decisions
?::d1.

% State Variables
state_variables(s1, s2, s3).

% Transition
0.13::x(s1) :- s1, d1.
0.85::x(s1) :- \+s1, d1.
0.76::x(s1) :- s1, \+d1.
0.26::x(s1) :- \+s1, \+d1.

0.5::x(s2) :- s2, d1.
0.45::x(s2) :- \+s2, d1.
0.65::x(s2) :- s2, \+d1.
0.79::x(s2) :- \+s2, \+d1.

0.09::x(s3) :- s3, d1.
0.03::x(s3) :- \+s3, d1.
0.84::x(s3) :- s3, \+d1.
0.43::x(s3) :- \+s3, \+d1.

% Structure
0.76::x(s2) :- s1, x(s1).
0.0::x(s2) :- \+s1, x(s1).
0.45::x(s2) :- s1, \+x(s1).
0.72::x(s2) :- \+s1, \+x(s1).

0.23::x(s3) :- s2, x(s2).
0.95::x(s3) :- \+s2, x(s2).
0.9::x(s3) :- s2, \+x(s2).
0.03::x(s3) :- \+s2, \+x(s2).

% Rewards
utility(d1, -1).
utility(s1, 4).
utility(s2, 2).
utility(s3, -6).
