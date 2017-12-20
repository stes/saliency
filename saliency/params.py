param = {

    "mapWidth" = 64,               # this controls the size of the 'Center' scale

    "useMultipleCenterScales" = 0, # classic Itti Algorithm uses multiple scales ( "c \in {2,3,4}" ). but here we default to just 1.

    "surroundSig" = [ 2, 8 ],    # this is the standard deviation of the Gaussian blur applied to obtain the 'surround' scale(s).
                                 # default : one surround scale works fine in my experience. with sigma = 5.
                                 # this can also be an array of surround sigmas, e.g. [ 4 6 ]
                                 # Note: in classic  Itti algorithm, we have ( "delta \in {3,4}\" ).
                                 # .. **I think** this should correspond to roughly surroundSig = [sqrt(2^3) sqrt(2^4)]

# %%%%%%%% normalize maps according to peakiness  %%%%%

    "useNormWeights" = 1,        % 0 = do not weight maps differently , 1 = weight according to peakiness
                                 % in classic Itti algorithm, this is used with local maxima normalization.

    "subtractMin" = 1,           % 1 => (subtract min, divide by max) ; 0 => (just divide by max)

# %%%%%%%%% channel parameters %%%%%%%%%%%%%

    "channels" = 'I',          % can include following characters: C (color), O (orientation), I (intensity), B (color bias)
                                  % e.g. use 'OICB' to include orientation, intensity, Color and color Bias

    "nGaborAngles" = 4,          % number of oriented gabors if there is an 'O' channel
                                 % as an example, 4 implies => 0 , 45 , 90 , 135 degrees


%%%%%%%%% final operations on saliency map %%%%

    "centerbias" = 0,            % apply global center bias (0 =no, 1=yes)
                                 % (using center bias tends to improve predictive performance)

    "blurRadius" = 0.04,         % blur final saliency map (sigma=0.04 works well in my experience).
                                 % NOTE: ROC and NSS Scores are VERY sensitive to saliency map blurring. This is
                                 % highly suggested.
}
