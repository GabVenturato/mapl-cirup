% COFFEE
% TODO: description

%% Decisions
?::move; ?::delc; ?::getu; ?::buyc.

%% State variables
state_variables(huc, hrc, wet, raining, umbrella, office).

%% Transition
% move
x(wet) :- wet, move.
0.1::x(wet) :- \+wet, raining, umbrella, move.
0.9::x(wet) :- \+wet, raining, \+umbrella, move.
0.1::x(office) :- office, move.
0.9::x(office) :- \+office, move.

x(huc) :- huc, move.
x(hrc) :- hrc, move.
x(umbrella) :- umbrella, move.
x(raining) :- raining, move.

% delc
x(huc) :- huc, delc.
0.8::x(huc) :- \+huc, hrc, office, delc.
0.1::x(hrc) :- hrc, office, delc.
0.2::x(hrc) :- hrc, \+office, delc.

x(wet) :- wet, delc.
x(umbrella) :- umbrella, delc.
x(office) :- office, delc.
x(raining) :- raining, delc.

% getu
x(umbrella) :- umbrella, getu.
0.9::x(umbrella) :- \+umbrella, office, getu.

x(huc) :- huc, getu.
x(hrc) :- hrc, getu.
x(wet) :- wet, getu.
x(office) :- office, getu.
x(raining) :- raining, getu.

% buyc
x(hrc) :- hrc, buyc.
0.9::x(hrc) :- \+hrc, \+office, buyc.

x(huc) :- huc, buyc.
x(wet) :- wet, buyc.
x(umbrella) :- umbrella, buyc.
x(office) :- office, buyc.
x(raining) :- raining, buyc.

%% Reward
r0 :- huc, wet.
r1 :- huc, \+wet.
r3 :- \+huc, \+wet.

utility(r0, 0.9).
utility(r1, 1).
utility(r3, 0.1).