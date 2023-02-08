% Decisions
?::d1.

% State Variables
state_variables(s1).

% Transition
0.13::x(s1) :- s1, d1.
0.85::x(s1) :- \+s1, d1.
0.76::x(s1) :- s1, \+d1.
0.26::x(s1) :- \+s1, \+d1.

% Structure
% Rewards
utility(d1, 0).
utility(s1, 1).
