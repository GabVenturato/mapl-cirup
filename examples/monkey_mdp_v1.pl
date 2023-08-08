% MONKEY
% A monkey is trying to hit you with some stones. You can decide to stay or to
% move. If you get hit you pay a cost.
% If you move you pay a cost (moving requires energies). If you stay and not get
% hit you get a positive reward. Also, when you get hit the monky celebrates and
% it's less likely it will hit you immediately afterwards. Finally, if you don't
% move, it is more likely you get hit.

% decisions
?::move.

% state vars
state_variables(hit).

% model
% ---

% transition
0.2::x(hit) :- hit.
0.5::x(hit) :- move, \+hit.
0.8::x(hit) :- \+move, \+hit.

% rewards
utility(move, -1).
utility(hit, -10).
