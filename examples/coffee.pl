% COFFEE
% TODO: description

%% Decisions
?::move; ?::delc; ?::getu; ?::buyc.

%?::a1.
%?::a2.
%move :- a1, a2.
%delc :- \+a1, a2.
%getu :- a1, \+a2.
%buyc :- \+a1, \+a2.

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

x(huc) :- huc, move.
x(hrc) :- hrc, move.
x(umbrella) :- umbrella, move.

% delc
0.99::x(huc) :- huc, delc.
0.83::x(huc) :- \+huc, hrc, office, delc.
0.05::x(hrc) :- hrc, office, delc.
0.68::x(hrc) :- hrc, \+office, delc.

x(wet) :- wet, delc.
x(umbrella) :- umbrella, delc.
x(office) :- office, delc.

% getu
x(umbrella) :- umbrella, getu.
0.81::x(umbrella) :- \+umbrella, office, getu.

x(huc) :- huc, getu.
x(hrc) :- hrc, getu.
x(wet) :- wet, getu.
x(office) :- office, getu.

% buyc
x(hrc) :- hrc, buyc.
0.98::x(hrc) :- \+hrc, \+office, buyc.

x(huc) :- huc, buyc.
x(wet) :- wet, buyc.
x(umbrella) :- umbrella, buyc.
x(office) :- office, buyc.

%% Reward
r0 :- huc, wet.
r1 :- huc, \+wet.
r3 :- \+huc, \+wet.

utility(r0, 3).
utility(r1, 5).
utility(r3, -2).