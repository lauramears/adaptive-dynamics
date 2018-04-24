# -*- coding: utf-8 -*-
import numpy as num
import copy as copy
import matplotlib.pyplot as plt
import seaborn as sns

#matrix of strategies
X = num.array([[0.0,0.0]]);
#matrix of 200 individuals
N = num.tile(X,(100,1));
#matrix of patches
K = num.tile(X, (50,2));
#fitness of strategies
S = num.tile(X, (1,1));
#generations
tMax = 150;
#plotting
plotInterval = 5;
count = 0;

def DrawPlots(N,t):
    '''
        Takes population matrix and timepoint and scatters:
        x against y for two-trait
        x against t for one-trait
    '''
    
    [n,m] = num.shape(N); #check dimensions of N
    x = N[:,0]; #trait 1 as x
    if m == 2: #two-trait scatter
        y = N[:,1]; #trait 2 as y
        
        sns.jointplot(x=x, y=y, kind="kde", stat_func = None, space=0, color="g")
        plt.show()

        
        plt.scatter(x,y);
        plt.show();
    else: #one-trait tree
        y = t; #time as y
        plt.scatter(x,y); #do not show unitl end of run (should all plot together)
    
    return

def PlotFitnessFunction():
    '''
        Populates a unit square and calls fitness function
        contour plot for overall fitness landscape     
        slice through the diagonal of the fitness landscape
    '''
    
    delta = 0.1;
    x = num.arange(-1.0, 1.0, delta);
    y = num.arange(1.0, -1.0, -delta);
    X, Y = num.meshgrid(x, y) #populate unit square

    for i in range (-10, 10):     #make arrays of unit square in format for fitness function
       for j in range (10, -10, -1):
           
           x = float(i*delta);
           y = float(j*delta);
           
           if x == -1. and y == 1.:
               #initialise unit
               unit = [x,y];
               
           else:
               #add to unit
               unit = num.vstack((unit, [x,y]));
    
               
    S1 = Fitness(unit, [-1,1]); #get fitness of unit square for patch 1
    S2 = Fitness(unit, [1,-1]); #get fitness of unit square for patch 2
    S = S1+S2; #overall fitness for each unit square strategy
    
    S = num.reshape(S, (20,20)).copy(); #reshape into square matrix for plots
    
    plt.figure()
    CS = plt.contour(X, Y, S) #plot  fitness landscape
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Fitness landscape')
    plt.show()
    
    
    line = num.diag(S); #take slice through fitness landscape at x+y = 0 and plot from the side
    plt.plot(line) #plot slice
    plt.ylabel('Fitness at x+y = 0')
    plt.show()
    
    return X
    
    return

def Reproduce(N):
    '''
        Takes population N and doubles it    
        Returns N
    '''
    N = num.tile(N,(2,1));
    return N

def AddMutant(N,X):
    '''
        Takes population N and current strategies X
        Selects two strategies from X
        Mutates one value in each by a random number drawn uniformly in the range +-muMax
        Adds a number of mutants (freq) to population N
        Return population N with new mutant strategies
    '''

    muMax = 0.1;
    freq = 10;  #Add mutants existing population at low frequency
    
    [n,m] = num.shape(X); #find number of current strategies and number of traits
    
    
    for i in range(0,m): #once for each trait
        idx = num.random.randint(n); #choose a random index
        mut = num.random.uniform(-muMax,muMax); #generate a random mutation +-muMax
        
        y = copy.deepcopy(X[idx, :]); #copy existing strategy
        y[i] = y[i] + mut; # add mut to trait
        
        N = num.vstack((N, num.tile(y,(freq,1)))); #add new individuals to population
    return N

def Disperse(N):
    '''
        Takes population N 
        Shuffles rows and splits randomly across two patches
        Store patch populations side-by-side in array K
        Return K
    '''
    
    n = int((num.size(N,0)/2)); #get half population size
    K = num.zeros((n,4));  #initialise patch array
    
    num.random.shuffle(N); #shuffle N
    K[:,:2] = N[:n,:]; #first half in patch 1
    K[:,2:] = N[n:,:]; #second half in patch2
    
    return K

def Select(X):
    '''
        Takes strategies X
        Calls fitness function to test against patch targets mu1 and mu2
        Returns fitness S
    '''
    
    mu1 = num.array([[-1,1]]); #target phenotype for patch 1
    mu2 = num.array([[1,-1]]); #target phenotype for patch 2
    
    S1 = Fitness(X, mu1); #fitness of current strategies at patch 1
    S2 = Fitness(X, mu2); #fitness of current strategies at patch 1
   
    S = num.vstack((S1,S2));
    S = num.transpose(S);
    return S

def Fitness(X,mu):
    '''
        Takes strategies X and target mu
        Calculates pre-competitive survival probability
        Frequency-independent
        Survival is a bell-shaped function of strategy
    '''
    alpha = 1; #try playing with this to alter the slope of the function
    
    n = int(num.size(X,0)); #get number of strategies
    MU = num.tile(mu, (n, 1)); #make an array of mu for ease of calculations
    
    DIST = X-MU; #calculate distance between strategy and target
    SQRMAG = num.sum(DIST**2, axis = 1); #square magnitude of distance (x*x + y*y)
    #would sqrt to find magnitude, but about to square again anyway

    #sigmasqr = num.sum(DIST**2)/n; #richard method
    sigmasqr = 0.5; #tom method
    
    S = alpha * num.exp(-((SQRMAG)/(2* (sigmasqr)))); #geritz 1998 e.22
    return S


def CountIndividualsUsingStrategyX(A,X):
    '''
        Takes population A
        Counts individuals using strategy X
        Returns count
    '''
    n = num.size(X,0); #get number of strategies
    count = num.zeros((n,1)); #set up array count individuals using each strategy

    for i in range(0,num.size(A,0)): #run through population
       for j in range(0,num.size(X,0)): #check against each strategy
          if (A[i,:] == X[j,:]).all():
              count[j] = count[j] +1;
    
    return count

def GenerateNewPopulation(POP,X):
    '''
        Takes POP distribution specifying number of individuals with each strategy X
        Returns newN population of 100 individuals
    '''

    endIdx = -1;
    for i in range (0, num.size(POP)):
        indiv = int(POP[i]); #find number of individuals
        startIdx = endIdx+1;  #calculate indices to update
        endIdx = startIdx + indiv-1;
        if i == 0:
            #initialise new population
            newN = num.tile(X[i,:],(indiv,1));
        else:
            #add to newN
            newN = num.vstack((newN, num.tile(X[i,:],(indiv,1))));
            
    return newN

def Contest(K,X,S):
    '''
        Takes population split by patch in K
        Gets count of individuals using each strategy X
        Calculates percentage of patch available to each strategy (according to fitness)
        Gets new population N
        Returns N
    '''
    
    C1 = CountIndividualsUsingStrategyX(K[:,:2],X); #patch1
    C2 =CountIndividualsUsingStrategyX(K[:,2:],X); #patch2
    COUNT = num.hstack((C1,C2));
    
    #calculate percentage of each patch available to each strategy
    #equation 18 from Geritz 1998
    P1 = (S[:,0]* COUNT[:,0])/num.sum(S[:,0] * COUNT[:,0]);
    P2 = (S[:,1] * COUNT[:,1])/num.sum(S[:,1] * COUNT[:,1]);
    
    POP = num.rint(((P1 + P2)/2)*100); #calculate abs individuals with each strategy for population of 100
    newN = GenerateNewPopulation(POP,X);
            
    return newN

def RemoveFailedStrategies(N,X):
    '''
        Takes population N and counts individuals using each strategy
        Removes strategies from X if in N below threshold
        Returns X
    '''

    threshold = 3;
    
    X = num.unique(N, axis = 0); #unique strategies currently in use
    count = CountIndividualsUsingStrategyX(N,X); #count individuals in N using each strategy
    idx = num.nonzero(count<threshold); #get indices of strategies below threshold
    X = num.delete(X, idx, axis = 0); #remove failed strategies from X
    return X

def Main(X,N,K,tMax,plotInterval):
    for t in range(0,tMax):
        N = Reproduce(N); #all established individuals have the same fecundity irrespective of strategy or patch
        N = AddMutant(N,X);  #an initially rare mutant with strategy y
        X = num.unique(N, axis = 0); #record new mutant strategies
        K = Disperse(N); #offspring are distributed randomly
        S = Select(X); #juveniles first undergo a period of frequency-independent selection
        N = Contest(K, X, S); #followed by a period of non-selective contest competition
        X = RemoveFailedStrategies(N,X); #when the frequency of a strategy drops below threshold, the strategy is extinct
        
        
        #plot
        if t == 0 or t % plotInterval == 0:
            DrawPlots(N,t);
        if t == tMax:
            plt.show();

    return X, N, K, S


PlotFitnessFunction();
[X,N,K,S] = Main(X,N,K,tMax,plotInterval);
