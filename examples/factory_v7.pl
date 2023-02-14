% FACTORY
% Two pieces (A and B) need to be connected. You can use glue or bolts (you need to own those first).
% To bolt the two pieces they have to be drilled frist. Gluing does not require drilling instead.
% When bolting you have more probability of having an high quality connection. When gluing, instead, it's more likely to
% have a low quality connection.
% You get paid more if an high quality connection is requested, and less for a low quality request. But of course you
% have to succeed in connecting at the corresponding quality.

%% Decisions
?::a1.
?::a2.
%drilla :- a1, a2.
%drillb :- \+a1, a2.
%bolt :- a1, \+a2.
%glue :- \+a1, \+a2.

%% State variables
state_variables(hglue, needhq, connected, connectedwell, hbolts, adrilled, bdrilled).

%% Transition
% drilla
0.85::x(adrilled) :- \+connected, \+adrilled, a1, a2.

x(hglue) :- hglue, a1, a2.
x(needhq) :- needhq, a1, a2.
x(connected) :- connected, a1, a2.
x(connectedwell) :- connectedwell, a1, a2.
x(hbolts) :- hbolts, a1, a2.
x(adrilled) :- adrilled, \+a1, a2.
x(bdrilled) :- bdrilled, a1, a2.

% drillb
0.91::x(bdrilled) :- \+connected, \+bdrilled, \+a1, a2.

x(hglue) :- hglue, \+a1, a2.
x(needhq) :- needhq, \+a1, a2.
x(connected) :- connected, \+a1, a2.
x(connectedwell) :- connectedwell, \+a1, a2.
x(hbolts) :- hbolts, \+a1, a2.
x(adrilled) :- adrilled, \+a1, a2.
x(bdrilled) :- bdrilled, a1, a2.

% bolt
x(connected) :- adrilled, bdrilled, hbolts, a1, \+a2.
0.95::x(connectedwell) :- adrilled, bdrilled, hbolts, a1, \+a2.

x(hglue) :- hglue, a1, \+a2.
x(needhq) :- needhq, a1, \+a2.
x(connected) :- connected, a1, \+a2.
x(connectedwell) :- connectedwell, a1, \+a2.
x(hbolts) :- hbolts, a1, \+a2.
x(adrilled) :- adrilled, a1, \+a2.
x(bdrilled) :- bdrilled, a1, \+a2.

% glue
x(connected) :- hglue, \+a1, \+a2.
0.2::x(connectedwell) :- hglue, \+a1, \+a2.

x(hglue) :- hglue, \+a1, \+a2.
x(needhq) :- needhq, \+a1, \+a2.
x(connected) :- connected, \+a1, \+a2.
x(connectedwell) :- connectedwell, \+a1, \+a2.
x(hbolts) :- hbolts, \+a1, \+a2.
x(adrilled) :- adrilled, \+a1, \+a2.
x(bdrilled) :- bdrilled, \+a1, \+a2.

%% Reward
r1 :- needhq, connected, connectedwell.
r2 :- \+needhq, connected, connectedwell.
r3 :- \+needhq, connected, \+connectedwell.

utility(r1, 10).
utility(r2, 1).
utility(r3, 3).