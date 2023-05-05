% Decisions
?::d1.

% State Variables
state_variables(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18).

% Transition
0.43::x(s1) :- s1, d1.
0.16::x(s1) :- \+s1, d1.
0.71::x(s1) :- s1, \+d1.
0.67::x(s1) :- \+s1, \+d1.

0.25::x(s2) :- s2, d1.
0.06::x(s2) :- \+s2, d1.
0.96::x(s2) :- s2, \+d1.
0.81::x(s2) :- \+s2, \+d1.

0.55::x(s3) :- s3, d1.
0.54::x(s3) :- \+s3, d1.
0.85::x(s3) :- s3, \+d1.
0.45::x(s3) :- \+s3, \+d1.

0.4::x(s4) :- s4, d1.
0.34::x(s4) :- \+s4, d1.
0.26::x(s4) :- s4, \+d1.
0.02::x(s4) :- \+s4, \+d1.

0.65::x(s5) :- s5, d1.
0.42::x(s5) :- \+s5, d1.
0.57::x(s5) :- s5, \+d1.
0.06::x(s5) :- \+s5, \+d1.

0.35::x(s6) :- s6, d1.
0.14::x(s6) :- \+s6, d1.
0.13::x(s6) :- s6, \+d1.
0.26::x(s6) :- \+s6, \+d1.

0.83::x(s7) :- s7, d1.
0.4::x(s7) :- \+s7, d1.
0.4::x(s7) :- s7, \+d1.
0.61::x(s7) :- \+s7, \+d1.

0.23::x(s8) :- s8, d1.
0.01::x(s8) :- \+s8, d1.
0.53::x(s8) :- s8, \+d1.
0.5::x(s8) :- \+s8, \+d1.

0.65::x(s9) :- s9, d1.
0.44::x(s9) :- \+s9, d1.
0.69::x(s9) :- s9, \+d1.
0.73::x(s9) :- \+s9, \+d1.

0.24::x(s10) :- s10, d1.
0.5::x(s10) :- \+s10, d1.
0.48::x(s10) :- s10, \+d1.
0.23::x(s10) :- \+s10, \+d1.

0.41::x(s11) :- s11, d1.
0.56::x(s11) :- \+s11, d1.
0.91::x(s11) :- s11, \+d1.
0.92::x(s11) :- \+s11, \+d1.

0.28::x(s12) :- s12, d1.
0.65::x(s12) :- \+s12, d1.
0.05::x(s12) :- s12, \+d1.
0.07::x(s12) :- \+s12, \+d1.

0.51::x(s13) :- s13, d1.
0.88::x(s13) :- \+s13, d1.
0.16::x(s13) :- s13, \+d1.
0.77::x(s13) :- \+s13, \+d1.

0.88::x(s14) :- s14, d1.
0.31::x(s14) :- \+s14, d1.
0.69::x(s14) :- s14, \+d1.
0.85::x(s14) :- \+s14, \+d1.

0.37::x(s15) :- s15, d1.
0.7::x(s15) :- \+s15, d1.
0.74::x(s15) :- s15, \+d1.
0.59::x(s15) :- \+s15, \+d1.

0.86::x(s16) :- s16, d1.
0.9::x(s16) :- \+s16, d1.
0.96::x(s16) :- s16, \+d1.
0.57::x(s16) :- \+s16, \+d1.

0.18::x(s17) :- s17, d1.
0.25::x(s17) :- \+s17, d1.
0.22::x(s17) :- s17, \+d1.
0.57::x(s17) :- \+s17, \+d1.

0.76::x(s18) :- s18, d1.
0.05::x(s18) :- \+s18, d1.
0.68::x(s18) :- s18, \+d1.
0.72::x(s18) :- \+s18, \+d1.

% Structure
0.35::x(s1) :- s2, x(s2).
0.52::x(s1) :- \+s2, x(s2).
0.16::x(s1) :- s2, \+x(s2).
0.73::x(s1) :- \+s2, \+x(s2).

0.04::x(s2) :- s1.
0.98::x(s2) :- \+s1.
0.81::x(s2) :- s3.
0.63::x(s2) :- \+s3.

0.27::x(s3) :- s2, x(s2).
0.91::x(s3) :- \+s2, x(s2).
0.96::x(s3) :- s2, \+x(s2).
0.14::x(s3) :- \+s2, \+x(s2).
0.78::x(s3) :- s4, x(s4).
0.84::x(s3) :- \+s4, x(s4).
0.66::x(s3) :- s4, \+x(s4).
0.7::x(s3) :- \+s4, \+x(s4).

0.45::x(s4) :- s3.
0.92::x(s4) :- \+s3.
0.97::x(s4) :- s5.
0.38::x(s4) :- \+s5.

0.8::x(s5) :- s4, x(s4).
0.43::x(s5) :- \+s4, x(s4).
0.16::x(s5) :- s4, \+x(s4).
0.33::x(s5) :- \+s4, \+x(s4).
0.13::x(s5) :- s6, x(s6).
0.91::x(s5) :- \+s6, x(s6).
0.96::x(s5) :- s6, \+x(s6).
0.12::x(s5) :- \+s6, \+x(s6).

0.6::x(s6) :- s5.
0.41::x(s6) :- \+s5.
0.12::x(s6) :- s7.
0.3::x(s6) :- \+s7.

0.25::x(s7) :- s6, x(s6).
0.75::x(s7) :- \+s6, x(s6).
0.0::x(s7) :- s6, \+x(s6).
0.19::x(s7) :- \+s6, \+x(s6).
0.44::x(s7) :- s8, x(s8).
0.02::x(s7) :- \+s8, x(s8).
0.63::x(s7) :- s8, \+x(s8).
0.61::x(s7) :- \+s8, \+x(s8).

0.84::x(s8) :- s7.
0.21::x(s8) :- \+s7.
0.28::x(s8) :- s9.
0.54::x(s8) :- \+s9.

0.27::x(s9) :- s8, x(s8).
0.59::x(s9) :- \+s8, x(s8).
0.25::x(s9) :- s8, \+x(s8).
0.68::x(s9) :- \+s8, \+x(s8).
0.79::x(s9) :- s10, x(s10).
0.81::x(s9) :- \+s10, x(s10).
0.97::x(s9) :- s10, \+x(s10).
0.55::x(s9) :- \+s10, \+x(s10).

0.49::x(s10) :- s9.
0.86::x(s10) :- \+s9.
0.77::x(s10) :- s11.
0.57::x(s10) :- \+s11.

0.38::x(s11) :- s10, x(s10).
0.28::x(s11) :- \+s10, x(s10).
0.11::x(s11) :- s10, \+x(s10).
0.81::x(s11) :- \+s10, \+x(s10).
0.12::x(s11) :- s12, x(s12).
0.75::x(s11) :- \+s12, x(s12).
0.55::x(s11) :- s12, \+x(s12).
0.96::x(s11) :- \+s12, \+x(s12).

0.76::x(s12) :- s11.
0.97::x(s12) :- \+s11.
0.14::x(s12) :- s13.
0.5::x(s12) :- \+s13.

0.57::x(s13) :- s12, x(s12).
0.31::x(s13) :- \+s12, x(s12).
0.5::x(s13) :- s12, \+x(s12).
0.36::x(s13) :- \+s12, \+x(s12).
0.53::x(s13) :- s14, x(s14).
0.0::x(s13) :- \+s14, x(s14).
0.44::x(s13) :- s14, \+x(s14).
0.45::x(s13) :- \+s14, \+x(s14).

0.3::x(s14) :- s13.
0.4::x(s14) :- \+s13.
0.78::x(s14) :- s15.
0.68::x(s14) :- \+s15.

0.49::x(s15) :- s14, x(s14).
0.65::x(s15) :- \+s14, x(s14).
0.38::x(s15) :- s14, \+x(s14).
0.2::x(s15) :- \+s14, \+x(s14).
0.0::x(s15) :- s16, x(s16).
0.28::x(s15) :- \+s16, x(s16).
0.6::x(s15) :- s16, \+x(s16).
0.88::x(s15) :- \+s16, \+x(s16).

0.83::x(s16) :- s15.
0.51::x(s16) :- \+s15.
0.99::x(s16) :- s17.
0.46::x(s16) :- \+s17.

0.83::x(s17) :- s16, x(s16).
0.41::x(s17) :- \+s16, x(s16).
0.74::x(s17) :- s16, \+x(s16).
0.99::x(s17) :- \+s16, \+x(s16).
0.31::x(s17) :- s18, x(s18).
0.17::x(s17) :- \+s18, x(s18).
0.62::x(s17) :- s18, \+x(s18).
0.53::x(s17) :- \+s18, \+x(s18).

0.36::x(s18) :- s17.
0.0::x(s18) :- \+s17.

% Rewards
utility(d1, 0).
utility(s1, 18).
utility(s2, 15).
utility(s3, 7).
utility(s4, -28).
utility(s5, 27).
utility(s6, -5).
utility(s7, 1).
utility(s8, -34).
utility(s9, 16).
utility(s10, -17).
utility(s11, 14).
utility(s12, -2).
utility(s13, -14).
utility(s14, -27).
utility(s15, -35).
utility(s16, 8).
utility(s17, -3).
utility(s18, 16).
