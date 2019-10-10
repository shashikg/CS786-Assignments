
clear;

% the temporal context model assumes that the past becomes increasingly
% dissimilar to the future, so that memories become harder to retrieve the
% farther away in the past they are

N_WORLD_FEATURES = 5;
N_ITEMS = 10;
ENCODING_TIME = 500;
TEST_TIME = 20;

% we are going to model the world as a set of N continuous-valued features.
% we will model observations of states of the world as samples from N
% Gaussians with time-varying means and fixed variance. For simplicity,
% assume that agents change nothing in the world.

success = 0;
%% Averaging over 20 trials
for trial = 1:20
  % first fix the presentation schedule; I'm assuming its random
  
  schedule = [(ENCODING_TIME - N_ITEMS + (1:N_ITEMS))' (1:N_ITEMS)'];         % taking last 10 items
  schedule_load = ENCODING_TIME/median(diff(schedule(:,1)));                  % variable important for parts 2,3 of assignment
  encoding = zeros(N_ITEMS,N_WORLD_FEATURES + 1);

  world_m = [1 2 1 2 3];              % can generate randomly for yourself
  world_var = 1;
  delta = 0.05;                       % what does this parameter affect? Constant Drift in world states
  beta_param = 0.001;                 % what does this parameter affect? Used in for sampled drift to create gaussian mixture using the proportion as beta_param and 1-beta_param
  m = 1;
  
  % simulating encoding
  for time = 1:ENCODING_TIME
      world_m = world_m + delta;
      world = normrnd(world_m, world_var);
      % any item I want to encode in memory, I encode in association with the
      % state of the world at that time.
      if(m<(N_ITEMS+1))
          if(time==schedule(m,1))
              encoding(m,:) = [world m];                                              % encode into the encoding vector
              m =  m + 1;
          end;  
      end;
  end;
  
  while(time<ENCODING_TIME+TEST_TIME)
  % the state of the world is the retrieval cue
      world_m = world_m + delta;
      world = normrnd(world_m, world_var);                                             % model world evolution

      for m = 1:N_ITEMS
          % dot-product to find association
          soa(m) = encoding(m,:)*[world m]';               % finding association strengths
      end;
      soa = soa/sum(soa);                                                               % normalize
      
      out(time-ENCODING_TIME+1) = find(drawFromADist(soa));
      time = time + 1;       
  end;
  
  success = success + length(unique(out));
end

success = success/20  % success is number of unique retrievals
