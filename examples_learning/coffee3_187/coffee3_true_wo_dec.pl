% COFFEE
% There is a robot that has to deliver coffee. There is a user in an office.
% There is a shop where the robot can get coffee. When moving between the two
% places (shop and office) the robot has to go outsite the building and it might
% rain. The robot can get an umbrella from the office. The possible actions are:
% - move: the robot moves from one place to the other
% - delc: the robot delivers the coffee to the user
% - getu: the robot gets an umbrella (it can get it only from the office)
% - buyc: the robot buys a coffee (only from the shop)
% The state variables are:
% - huc: has user coffee?
% - hrc: has robot coffee?
% - wet: true if the robot is wet
% - raining: true if it's raining
% - umbrella: true if the robot has the umbrella
% - office: true is the robot is in the office, false if it's in the shop

%% Decisions
% ?::move; ?::delc; ?::getu; ?::buyc.

%?::a1.
%?::a2.
%move :- a1, a2.
%delc :- \+a1, a2.
%getu :- a1, \+a2.
%buyc :- \+a1, \+a2.

%% State variables
decisions(move, delc, getu, buyc).
state_variables(huc, hrc, wet, raining, umbrella, office).
reward_variables(r0, r1, r3, office, move, delc, getu, buyc).

%% Model
0.75::x(raining) :- raining.
0.3::x(raining) :- \+raining.

%huc :- raining.

%% Transition
% move
x(wet) :- wet, move.
0.13::x(wet) :- \+wet, raining, umbrella, move.
0.93::x(wet) :- \+wet, raining, \+umbrella, move.
x(office) :- \+office, move.

x(huc) :- huc, move.
x(hrc) :- hrc, move.
x(umbrella) :- umbrella, move.

% delc
x(huc) :- huc, delc.
0.83::x(huc) :- \+huc, hrc, office, delc.
0.05::x(hrc) :- hrc, office, delc.
0.68::x(hrc) :- hrc, \+office, delc.

x(wet) :- wet, delc.
x(umbrella) :- umbrella, delc.
x(office) :- office, delc.

% getu
x(umbrella) :- umbrella, getu.
x(umbrella) :- \+umbrella, office, getu.

x(huc) :- huc, getu.
x(hrc) :- hrc, getu.
x(wet) :- wet, getu.
x(office) :- office, getu.

% buyc
x(hrc) :- hrc, buyc.
x(hrc) :- \+hrc, \+office, buyc.

x(huc) :- huc, buyc.
x(wet) :- wet, buyc.
x(umbrella) :- umbrella, buyc.
x(office) :- office, buyc.

%% Reward
r0 :- huc, wet.
r1 :- huc, \+wet.
r3 :- \+huc, \+wet.



utility(buyc, 4).
utility(r1, 5).
utility(move, 6).
utility(getu, 6).
utility(r0, 1).
utility(delc, 6).
utility(r3, 3).
utility(office, 8).