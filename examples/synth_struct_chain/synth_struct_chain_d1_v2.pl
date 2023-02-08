% Decisions
?::d1.

% State Variables
state_variables(s1, s2).

% Transition
0.89::x(s1) :- s1, d1.
0.39::x(s1) :- \+s1, d1.
0.61::x(s1) :- s1, \+d1.
0.77::x(s1) :- \+s1, \+d1.

0.7::x(s2) :- s2, d1.
0.27::x(s2) :- \+s2, d1.
0.8::x(s2) :- s2, \+d1.
0.59::x(s2) :- \+s2, \+d1.

% Structure
0.1::x(s2) :- s1, x(s1).
0.32::x(s2) :- \+s1, x(s1).
0.02::x(s2) :- s1, \+x(s1).
0.65::x(s2) :- \+s1, \+x(s1).

% Rewards
utility(d1, -1).
utility(s1, 2).
utility(s2, -1).
