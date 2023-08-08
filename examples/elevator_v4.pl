% ELEVATOR
% A person has to move from second to fourth floor.
% The only non-determinism here is in the person's actions.

% Decisions
?::move.
?::up.

%% State variables
% Floor <- floora | floorb
% 1        F        F
% 2        F        T
% 3        T        F
% 4        T        T
state_variables(floora, floorb, inside, destination).

%% Transition
% move up
x(floorb) :- \+floora, \+floorb, move, up.
x(floora) :- \+floora, floorb, move, up.
x(floora) :- floora, \+floorb, move, up.
x(floorb) :- floora, \+floorb, move, up.

x(inside) :- inside, move, up.
x(destination) :- destination, move, up.

% move down
x(floora) :- floora, floorb, move, \+up.
x(floorb) :- floora, \+floorb, move, \+up.

x(inside) :- inside, move, \+up.
x(destination) :- destination, move, \+up.

% stop
0.9::x(inside) :- inside, \+floora, floorb, \+move.
0.9::x(destination) :- destination, \+floora, \+floorb, \+move.

%% Reward
utility(destination, 1).
utility(move, -0.1).