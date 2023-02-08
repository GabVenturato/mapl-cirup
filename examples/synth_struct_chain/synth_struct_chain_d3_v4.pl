% Decisions
?::d1; ?::d2; ?::d3.

% State Variables
state_variables(s1, s2, s3, s4).

% Transition
0.83::x(s1) :- s1, d1, d2, d3.
0.74::x(s1) :- \+s1, d1, d2, d3.
0.97::x(s1) :- s1, \+d1, d2, d3.
0.61::x(s1) :- \+s1, \+d1, d2, d3.
0.92::x(s1) :- s1, d1, \+d2, d3.
0.58::x(s1) :- \+s1, d1, \+d2, d3.
0.35::x(s1) :- s1, \+d1, \+d2, d3.
0.67::x(s1) :- \+s1, \+d1, \+d2, d3.
0.53::x(s1) :- s1, d1, d2, \+d3.
0.37::x(s1) :- \+s1, d1, d2, \+d3.
0.68::x(s1) :- s1, \+d1, d2, \+d3.
0.0::x(s1) :- \+s1, \+d1, d2, \+d3.
0.16::x(s1) :- s1, d1, \+d2, \+d3.
0.82::x(s1) :- \+s1, d1, \+d2, \+d3.
0.36::x(s1) :- s1, \+d1, \+d2, \+d3.
0.35::x(s1) :- \+s1, \+d1, \+d2, \+d3.

0.23::x(s2) :- s2, d1, d2, d3.
0.35::x(s2) :- \+s2, d1, d2, d3.
0.11::x(s2) :- s2, \+d1, d2, d3.
0.79::x(s2) :- \+s2, \+d1, d2, d3.
0.23::x(s2) :- s2, d1, \+d2, d3.
0.48::x(s2) :- \+s2, d1, \+d2, d3.
0.98::x(s2) :- s2, \+d1, \+d2, d3.
0.93::x(s2) :- \+s2, \+d1, \+d2, d3.
0.83::x(s2) :- s2, d1, d2, \+d3.
0.1::x(s2) :- \+s2, d1, d2, \+d3.
0.03::x(s2) :- s2, \+d1, d2, \+d3.
0.71::x(s2) :- \+s2, \+d1, d2, \+d3.
0.75::x(s2) :- s2, d1, \+d2, \+d3.
0.29::x(s2) :- \+s2, d1, \+d2, \+d3.
0.54::x(s2) :- s2, \+d1, \+d2, \+d3.
0.12::x(s2) :- \+s2, \+d1, \+d2, \+d3.

0.4::x(s3) :- s3, d1, d2, d3.
0.18::x(s3) :- \+s3, d1, d2, d3.
0.27::x(s3) :- s3, \+d1, d2, d3.
0.19::x(s3) :- \+s3, \+d1, d2, d3.
0.12::x(s3) :- s3, d1, \+d2, d3.
0.82::x(s3) :- \+s3, d1, \+d2, d3.
0.57::x(s3) :- s3, \+d1, \+d2, d3.
0.74::x(s3) :- \+s3, \+d1, \+d2, d3.
0.87::x(s3) :- s3, d1, d2, \+d3.
0.67::x(s3) :- \+s3, d1, d2, \+d3.
0.15::x(s3) :- s3, \+d1, d2, \+d3.
0.44::x(s3) :- \+s3, \+d1, d2, \+d3.
0.83::x(s3) :- s3, d1, \+d2, \+d3.
0.76::x(s3) :- \+s3, d1, \+d2, \+d3.
0.48::x(s3) :- s3, \+d1, \+d2, \+d3.
0.48::x(s3) :- \+s3, \+d1, \+d2, \+d3.

0.33::x(s4) :- s4, d1, d2, d3.
0.7::x(s4) :- \+s4, d1, d2, d3.
0.63::x(s4) :- s4, \+d1, d2, d3.
0.65::x(s4) :- \+s4, \+d1, d2, d3.
0.63::x(s4) :- s4, d1, \+d2, d3.
0.91::x(s4) :- \+s4, d1, \+d2, d3.
0.04::x(s4) :- s4, \+d1, \+d2, d3.
0.39::x(s4) :- \+s4, \+d1, \+d2, d3.
0.38::x(s4) :- s4, d1, d2, \+d3.
0.58::x(s4) :- \+s4, d1, d2, \+d3.
0.48::x(s4) :- s4, \+d1, d2, \+d3.
0.47::x(s4) :- \+s4, \+d1, d2, \+d3.
0.62::x(s4) :- s4, d1, \+d2, \+d3.
0.15::x(s4) :- \+s4, d1, \+d2, \+d3.
1.0::x(s4) :- s4, \+d1, \+d2, \+d3.
0.29::x(s4) :- \+s4, \+d1, \+d2, \+d3.

% Structure
0.12::x(s2) :- s1, x(s1).
0.92::x(s2) :- \+s1, x(s1).
0.15::x(s2) :- s1, \+x(s1).
0.52::x(s2) :- \+s1, \+x(s1).

0.83::x(s3) :- s2, x(s2).
0.12::x(s3) :- \+s2, x(s2).
0.11::x(s3) :- s2, \+x(s2).
0.47::x(s3) :- \+s2, \+x(s2).

0.78::x(s4) :- s3, x(s3).
0.63::x(s4) :- \+s3, x(s3).
0.47::x(s4) :- s3, \+x(s3).
0.81::x(s4) :- \+s3, \+x(s3).

% Rewards
utility(d1, 0).
utility(d2, -2).
utility(d3, 1).
utility(s1, -3).
utility(s2, 2).
utility(s3, -4).
utility(s4, 6).