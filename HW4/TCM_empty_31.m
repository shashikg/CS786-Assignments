clear;

% the temporal context model assumes that the past becomes increasingly
% dissimilar to the future, so that memories become harder to retrieve the
% farther away in the past they are

N_WORLD_FEATURES = 5;
N_ITEMS = 10;
ENCODING_TIME = 500;
TEST_TIME = 20;

success = 0;
%% Averaging over 20 trials
for trial = 1:20
  % first fix the presentation schedule; I'm assuming its random
  %sc1 = [(1:(N_ITEMS/2)), (N_ITEMS/2 + 99*(1:(N_ITEMS/2)))];
  
  schedule = [(ENCODING_TIME - N_ITEMS + (1:N_ITEMS))' (1:N_ITEMS)'];
  %schedule = [sort(round(rand(1,N_ITEMS)*ENCODING_TIME))' (1:N_ITEMS)'];
  schedule_load = ENCODING_TIME/median(diff(schedule(:,1)));                  % variable important for parts 2,3 of assignment
  encoding = zeros(N_ITEMS,N_WORLD_FEATURES + 1);

  world_m = [1 2 1 2 3];              % can generate randomly for yourself
  world_var = 1;
  delta = 0.05;                       % what does this parameter affect?
  beta_param = 0.001;                 % what does this parameter affect?
  m = 1;
  % simulating encoding
  
  gme = gmdistribution([1*delta;50*delta], 1, [beta_param; 1 - beta_param]);
  for time = 1:ENCODING_TIME
      drift = random(gme);
      world_m = world_m + drift;
      world = world_m;
      data(time,:) = drift;
      % any item I want to encode in memory, I encode in association with the
      % state of the world at that time.
      if(m<(N_ITEMS+1))
          if(time==schedule(m,1))
              encoding(m,:) = [world m];                                              % encode into the encoding vector
              m =  m + 1;
          end;  
      end;
  end;
  
  GMModel = fitgmdist(data,2);                                                  % fit GMM using inbuilt EM algo in Matlab
  gmr = gmdistribution(GMModel.mu, GMModel.Sigma, GMModel.ComponentProportion);
  while(time<ENCODING_TIME+TEST_TIME)
  % the state of the world is the retrieval cue
      drift = random(gmr);
      world_m = world_m + drift;
      world = world_m;                                                           % model world evolution

      for m = 1:N_ITEMS
          soa(m) = encoding(m,:)*[world m]';                                                             % finding association strengths
      end;
      soa = soa/sum(soa);                                                                 % normalize
      
      out(time-ENCODING_TIME+1) = find(drawFromADist(soa));
      time = time + 1;       
  end;
  
  success = success + length(unique(out));
end

success = success/20  % success is number of unique retrievals

% humans can retrieve about 7 items effectively from memory. get this model
% to behave like humans