?::a.

% state vars
state_variables(x1, x2, x3, x4).

% model
0.5::x(x2) :- x2.
0.7::x(x3) :- x3.
0.2::x(x4) :- x4.

% transition
0.2::x(x1) :- x1.
0.5::x(x1) :- a, \+x1.
0.8::x(x1) :- \+a, \+x1.

0.9::x(x2) :- x1, x(x1).
0.5::x(x2) :- x1, \+x(x1).
0.8::x(x2) :- \+x1, x(x1).
0.1::x(x2) :- \+x1, \+x(x1).

0.75::x(x3) :- x2, x(x2), a.
0.35::x(x3) :- x2, \+x(x2), a.
0.2::x(x3) :- \+x2, x(x2), a.
0.15::x(x3) :- \+x2, \+x(x2), \+a.

0.75::x(x4) :- x3, x(x3), x(x1).
0.2::x(x4) :- \+x3, x(x3), \+x(x1).

% rewards
utility(a, -1).
utility(x1, -10).
utility(x2, -4).
utility(x3, 5).
utility(x4, 2).
