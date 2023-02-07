%% FACTORY

%% Decisions
?::shapea; ?::shapeb; ?::drilla; ?::drillb; ?::dipa; ?::dipb; ?::spraya; ?::sprayb; ?::handpainta; ?::handpaintb, ?::bolt, ?::glue, ?::polisha, ?::polishb.

%% State variables
state_variables(ashaped, bshaped, typeneeded, glue, apainted, apainted_aux, bpainted, bpainted_aux, connected, connected_aux, asmooth, bsmooth, bolts, spraygun, skilledlab, adrilled, bdrilled).
% for X in {apainted, bpainted, connected} we need to simulate a ternary variable with two binary variables as follows
% X  X_aux | X_origin
% T    T   |   good
% T    F   |   good
% F    T   |   poor
% F    F   |    f

%% Transition
% shapea
%0.8::x(ashaped) :- \+connected, \+connected_aux, shapea.
%x(bshaped) :-