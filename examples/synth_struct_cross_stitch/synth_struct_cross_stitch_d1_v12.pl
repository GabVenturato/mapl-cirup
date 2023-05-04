% Decisions
?::d1.

% State Variables
state_variables(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12).

% Transition
0.46::x(s1) :- s1, d1.
0.03::x(s1) :- \+s1, d1.
0.23::x(s1) :- s1, \+d1.
0.18::x(s1) :- \+s1, \+d1.

0.58::x(s2) :- s2, d1.
0.86::x(s2) :- \+s2, d1.
0.8::x(s2) :- s2, \+d1.
0.8::x(s2) :- \+s2, \+d1.

0.82::x(s3) :- s3, d1.
0.26::x(s3) :- \+s3, d1.
0.84::x(s3) :- s3, \+d1.
0.67::x(s3) :- \+s3, \+d1.

0.08::x(s4) :- s4, d1.
0.02::x(s4) :- \+s4, d1.
0.01::x(s4) :- s4, \+d1.
0.76::x(s4) :- \+s4, \+d1.

0.25::x(s5) :- s5, d1.
0.11::x(s5) :- \+s5, d1.
0.62::x(s5) :- s5, \+d1.
0.34::x(s5) :- \+s5, \+d1.

0.07::x(s6) :- s6, d1.
0.16::x(s6) :- \+s6, d1.
0.53::x(s6) :- s6, \+d1.
0.17::x(s6) :- \+s6, \+d1.

0.27::x(s7) :- s7, d1.
0.71::x(s7) :- \+s7, d1.
0.45::x(s7) :- s7, \+d1.
0.32::x(s7) :- \+s7, \+d1.

0.47::x(s8) :- s8, d1.
0.02::x(s8) :- \+s8, d1.
0.39::x(s8) :- s8, \+d1.
0.42::x(s8) :- \+s8, \+d1.

0.19::x(s9) :- s9, d1.
0.11::x(s9) :- \+s9, d1.
0.9::x(s9) :- s9, \+d1.
0.51::x(s9) :- \+s9, \+d1.

0.21::x(s10) :- s10, d1.
0.61::x(s10) :- \+s10, d1.
0.82::x(s10) :- s10, \+d1.
0.02::x(s10) :- \+s10, \+d1.

0.02::x(s11) :- s11, d1.
0.15::x(s11) :- \+s11, d1.
0.72::x(s11) :- s11, \+d1.
0.16::x(s11) :- \+s11, \+d1.

0.7::x(s12) :- s12, d1.
0.68::x(s12) :- \+s12, d1.
0.54::x(s12) :- s12, \+d1.
0.22::x(s12) :- \+s12, \+d1.

% Structure
0.98::x(s1) :- s2, x(s2).
0.8::x(s1) :- \+s2, x(s2).
0.52::x(s1) :- s2, \+x(s2).
0.22::x(s1) :- \+s2, \+x(s2).

0.65::x(s2) :- s1.
0.39::x(s2) :- \+s1.
0.58::x(s2) :- s3.
0.32::x(s2) :- \+s3.

0.63::x(s3) :- s2, x(s2).
0.06::x(s3) :- \+s2, x(s2).
0.3::x(s3) :- s2, \+x(s2).
0.97::x(s3) :- \+s2, \+x(s2).
0.88::x(s3) :- s4, x(s4).
0.31::x(s3) :- \+s4, x(s4).
0.86::x(s3) :- s4, \+x(s4).
0.31::x(s3) :- \+s4, \+x(s4).

0.94::x(s4) :- s3.
0.74::x(s4) :- \+s3.
0.42::x(s4) :- s5.
0.25::x(s4) :- \+s5.

0.01::x(s5) :- s4, x(s4).
0.88::x(s5) :- \+s4, x(s4).
0.04::x(s5) :- s4, \+x(s4).
0.82::x(s5) :- \+s4, \+x(s4).
0.96::x(s5) :- s6, x(s6).
0.57::x(s5) :- \+s6, x(s6).
0.17::x(s5) :- s6, \+x(s6).
0.87::x(s5) :- \+s6, \+x(s6).

0.97::x(s6) :- s5.
0.7::x(s6) :- \+s5.
0.51::x(s6) :- s7.
0.38::x(s6) :- \+s7.

0.35::x(s7) :- s6, x(s6).
0.21::x(s7) :- \+s6, x(s6).
0.67::x(s7) :- s6, \+x(s6).
0.43::x(s7) :- \+s6, \+x(s6).
0.19::x(s7) :- s8, x(s8).
0.1::x(s7) :- \+s8, x(s8).
0.67::x(s7) :- s8, \+x(s8).
0.3::x(s7) :- \+s8, \+x(s8).

0.5::x(s8) :- s7.
0.33::x(s8) :- \+s7.
0.87::x(s8) :- s9.
0.9::x(s8) :- \+s9.

0.02::x(s9) :- s8, x(s8).
0.2::x(s9) :- \+s8, x(s8).
0.33::x(s9) :- s8, \+x(s8).
0.99::x(s9) :- \+s8, \+x(s8).
0.78::x(s9) :- s10, x(s10).
0.34::x(s9) :- \+s10, x(s10).
0.21::x(s9) :- s10, \+x(s10).
0.67::x(s9) :- \+s10, \+x(s10).

0.84::x(s10) :- s9.
0.93::x(s10) :- \+s9.
0.34::x(s10) :- s11.
0.88::x(s10) :- \+s11.

0.69::x(s11) :- s10, x(s10).
0.48::x(s11) :- \+s10, x(s10).
0.99::x(s11) :- s10, \+x(s10).
0.23::x(s11) :- \+s10, \+x(s10).
0.73::x(s11) :- s12, x(s12).
0.08::x(s11) :- \+s12, x(s12).
0.17::x(s11) :- s12, \+x(s12).
0.91::x(s11) :- \+s12, \+x(s12).

0.21::x(s12) :- s11.
0.76::x(s12) :- \+s11.

% Rewards
utility(d1, 1).
utility(s1, 8).
utility(s2, -8).
utility(s3, -1).
utility(s4, -3).
utility(s5, -3).
utility(s6, -17).
utility(s7, -6).
utility(s8, -9).
utility(s9, 14).
utility(s10, 21).
utility(s11, 7).
utility(s12, -16).