% COFFEE
% TODO: description

%% Decisions
?::move; ?::delc; ?::getu; ?::buyc.

%% State variables
state_variables(huc, hrc, wet, raining, umbrella, office).

%% Model
0.75::x(raining) :- raining.
0.3::x(raining) :- \+raining.

%% Transition
% move
0.99::x(wet) :- wet, move.
0.13::x(wet) :- \+wet, raining, umbrella, move.
0.93::x(wet) :- \+wet, raining, \+umbrella, move.
0.02::x(office) :- office, move.
0.95::x(office) :- \+office, move.

% delc
0.99::x(huc) :- huc, delc.
0.83::x(huc) :- \+huc, hrc, office, delc.
0.05::x(hrc) :- hrc, office, delc.
0.68::x(hrc) :- hrc, \+office, delc.

% getu
x(umbrella) :- umbrella, getu.
0.81::x(umbrella) :- \+umbrella, office, getu.

% buyc
x(hrc) :- hrc, buyc.
0.98::x(hrc) :- \+hrc, \+office, buyc.

%% Reward
r0 :- huc, wet.
r1 :- huc, \+wet.
r3 :- \+huc, \+wet.

utility(r0, 3).
utility(r1, 5).
utility(r3, -2).