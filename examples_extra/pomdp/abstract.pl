% TODO: IS THIS A POMDP? Final policy is on states, not observations (?)
?::a.

state_variables(x1, x2).

% Transition
0.5::x(x1) :- x1, a.
0.2::x(x1) :- \+x1, a.

0.1::x(x2) :- x1, x2, a.
0.9::x(x2) :- x1, \+x2, a.
0.75::x(x2) :- \+x1, x2, a.

% Observation
0.5::o1 :- x2, x(x2).
0.2::o1 :- x2, \+x(x2).
0.7::o1 :- \+x2.

0.3::o2 :- x(x1), x(x2).

% Reward
utility(x1, 3).