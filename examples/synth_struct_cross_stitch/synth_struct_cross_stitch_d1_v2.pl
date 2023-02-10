% Decisions
?::d1.

% State Variables
state_variables(s1, s2).

% Transition
0.13::x(s1) :- s1, d1.
0.85::x(s1) :- \+s1, d1.
0.76::x(s1) :- s1, \+d1.
0.26::x(s1) :- \+s1, \+d1.

0.5::x(s2) :- s2, d1.
0.45::x(s2) :- \+s2, d1.
0.65::x(s2) :- s2, \+d1.
0.79::x(s2) :- \+s2, \+d1.

% Structure
0.09::x(s1) :- s2, x(s2).
0.03::x(s1) :- \+s2, x(s2).
0.84::x(s1) :- s2, \+x(s2).
0.43::x(s1) :- \+s2, \+x(s2).

0.76::x(s2) :- s1.
0.0::x(s2) :- \+s1.

% Rewards
utility(d1, 0).
utility(s1, 0).
utility(s2, -1).
