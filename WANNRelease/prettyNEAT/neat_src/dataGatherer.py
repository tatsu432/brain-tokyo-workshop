import os
import numpy as np
import copy
from .ann import exportNet
from ._speciate import Species
from .ind import Ind

class DataGatherer():
  """Data recorder for NEAT algorithm
    It keeps two kinds of "best" individuals:
    - elite: best individual of the current generation (one per generation).
    - best: best individual seen so far across all generations (also one per generation; it repeats the previous best if no improvement).

    And it keeps time-series arrays for plots:
    - x_scale: cumulative evaluation count (not generation index—more on that)
    - fit_med: median fitness in population per generation
    - fit_max: best fitness in population per generation (elite fitness)
    - fit_top: best-so-far fitness per generation (best fitness across history)
    - node_med: median number of nodes per generation
    - conn_med: median number of connections per generation

    Optional extras:
    - spec_fit: per-individual fitness labeled by species id (for NEAT speciation visualization)
    - objVals: if doing multi-objective optimization (MOO), it stores fitness and complexity together
    - action_dist_history: list of action distributions per generation
      """
  def __init__(self, filename: str, hyp: dict): 
    """
    Args:
      filename - (string) - path+prefix of file output destination
      hyp      - (dict)   - algorithm hyperparameters
    """
    self.filename = filename # File name path + prefix
    self.p = hyp
    
    # Initialize empty fields
    self.elite: list[Ind] = [] # List of elite individuals until the current generation
    self.best: list[Ind] = [] # List of best individuals until the current generation
    self.bestFitVec: list[float] = []
    self.spec_fit: list[tuple[int, float]] = []
    self.field: list[str] = ['x_scale','fit_med','fit_max','fit_top',\
                  'node_med','conn_med','num_species',\
                  'elite','best'  ]
    

    if self.p['alg_probMoo'] > 0:
      self.objVals = np.array([])

    for f in self.field[:-2]:  # FIX: exclude 'elite', 'best' (last 2)
      exec('self.' + f + ' = np.array([])')
      #e.g. self.fit_max   = np.array([]) 

    self.newBest = False
    
    # Action distribution tracking
    # Each entry is [nOutput x n_bins] array showing action distribution
    # Bins: [-inf, -0.5], (-0.5, 0], (0, 0.5], (0.5, inf]
    self.action_dist_history: list[np.ndarray] = []
    self.action_bin_labels = ['<-0.5', '-0.5~0', '0~0.5', '>0.5']
    self.current_action_dist = None

  def gatherData(self, pop: list[Ind], species: Species, action_dist: np.ndarray = None) -> None:
    """Collect and stores run data
    This is called once per generation (or once per "iteration" of the algorithm).
    
    Args:
      pop         - [Ind]      - list of individuals in population
      species     - (Species)  - current species
      action_dist - (np_array) - aggregated action distribution [nOutput x n_bins]
    """

    # Readability
    fitness = [ind.fitness for ind in pop]
    nodes = np.asarray([np.shape(ind.node)[1] for ind in pop]) # Number of nodes in the individual
    conns = np.asarray([ind.nConn for ind in pop]) # Number of connections in the individual
    
    # --- Evaluation Scale ---------------------------------------------------
    # it's not the generation index. It's like an x-axis for learning curves in terms of evaluations.
    if len(self.x_scale) == 0:
      self.x_scale = np.append(self.x_scale, len(pop))
    else:
      self.x_scale = np.append(self.x_scale, self.x_scale[-1]+len(pop))
    # ------------------------------------------------------------------------ 

    
    # --- Best Individual ----------------------------------------------------
    self.elite.append(pop[np.argmax(fitness)])
    if len(self.best) == 0:
      self.best.append(copy.deepcopy(self.elite[-1]))
    elif (self.elite[-1].fitness > self.best[-1].fitness):
      self.best.append(copy.deepcopy(self.elite[-1]))
      self.newBest = True
    else:
      self.best.append(copy.deepcopy(self.best[-1]))   
      self.newBest = False
    # ------------------------------------------------------------------------ 

    
    # --- Generation fit/complexity stats ------------------------------------ 
    self.node_med = np.append(self.node_med,np.median(nodes)) # Median number of nodes in the population
    self.conn_med = np.append(self.conn_med,np.median(conns)) # Median number of connections in the population
    self.fit_med  = np.append(self.fit_med, np.median(fitness)) # Median fitness in the population
    self.fit_max  = np.append(self.fit_max,  self.elite[-1].fitness) # Best fitness in the population
    self.fit_top  = np.append(self.fit_top,  self.best[-1].fitness) # Best fitness across history
    # ------------------------------------------------------------------------ 


    # --- MOO Fronts ---------------------------------------------------------
    if self.p['alg_probMoo'] > 0:
      if len(self.objVals) == 0:
        self.objVals = np.c_[fitness,conns]
      else:
        self.objVals = np.c_[self.objVals, np.c_[fitness,conns]]
    # ------------------------------------------------------------------------ 

    
    # --- Species Stats ------------------------------------------------------
    if self.p['alg_speciate'] == 'neat':
      specFit = np.empty((2,0))
      #print('# of Species: ', len(species))
      for iSpec in range(len(species)):
        for ind in species[iSpec].members:
          tmp = np.array((iSpec,ind.fitness))
          specFit = np.c_[specFit,tmp]
      self.spec_fit = specFit

    self.num_species = np.append(self.num_species, len(species))
    # ------------------------------------------------------------------------
    
    # --- Action Distribution ------------------------------------------------
    if action_dist is not None:
      self.action_dist_history.append(action_dist.copy())
      self.current_action_dist = action_dist
    # ------------------------------------------------------------------------


  def display(self):
    """Console output for each generation
    """
    # return    "|---| Elite Fit: " + '{:.2f}'.format(self.fit_max[-1]) \
    #      + " \t|---| Best Fit:  "  + '{:.2f}'.format(self.fit_top[-1]) \
    output = "Elite Fit: " + '{:.2f}'.format(self.fit_max[-1]) \
         + " Best Fit:"  + '{:.2f}'.format(self.fit_top[-1]) \
         + " #Species:"  + str(int(self.num_species[-1])) \
         + " Med #nodes:"  + str(int(self.node_med[-1])) \
         + " Med #conns:"  + str(int(self.conn_med[-1]))
    
    # Add action distribution to display
    if self.current_action_dist is not None:
      output += " | ActDist:"
      for i, row in enumerate(self.current_action_dist):
        # Format: Out0[25%,25%,25%,25%] for each output dimension
        pcts = ['{:.0f}%'.format(v*100) for v in row]
        output += " O{}[{}]".format(i, ','.join(pcts))
    
    return output


  def save(self, gen=(-1)):
    """Save algorithm stats to disk
    """
    ''' Save data to disk '''
    filename = self.filename
    pref = 'log/' + filename

    # --- Generation fit/complexity stats ------------------------------------ 
    gStatLabel = ['x_scale',\
                  'fit_med','fit_max','fit_top','node_med','conn_med']
    genStats = np.empty((len(self.x_scale),0))
    for i in range(len(gStatLabel)):
      #e.g.         self.    fit_max          [:,None]
      evalString = 'self.' + gStatLabel[i] + '[:,None]'
      genStats = np.hstack((genStats, eval(evalString)))
    lsave(pref + '_stats.out', genStats)
    # ------------------------------------------------------------------------ 


    # --- Best Individual ----------------------------------------------------
    wMat = self.best[gen].wMat
    aVec = self.best[gen].aVec
    exportNet(pref + '_best.out',wMat,aVec)
    
    if gen > 1:
      folder = 'log/' + filename + '_best/'
      if not os.path.exists(folder):
        os.makedirs(folder)
      exportNet(folder + str(gen).zfill(4) +'.out',wMat,aVec)
    # ------------------------------------------------------------------------


    # --- Species Stats ------------------------------------------------------
    if self.p['alg_speciate'] == 'neat':
      lsave(pref + '_spec.out', self.spec_fit)
    # ------------------------------------------------------------------------


    # --- MOO Fronts ---------------------------------------------------------
    if self.p['alg_probMoo'] > 0:
      lsave(pref + '_objVals.out',self.objVals)
    # ------------------------------------------------------------------------
    
    # --- Action Distribution History ----------------------------------------
    if len(self.action_dist_history) > 0:
      # Save action distribution history as a flattened array
      # Each row is: [gen, output_dim, bin0, bin1, bin2, bin3]
      action_dist_data = []
      for gen_idx, dist in enumerate(self.action_dist_history):
        for out_idx, row in enumerate(dist):
          action_dist_data.append([gen_idx, out_idx] + list(row))
      if action_dist_data:
        lsave(pref + '_action_dist.out', np.array(action_dist_data))
    # ------------------------------------------------------------------------

  def savePop(self,pop: list[Ind],filename: str) -> None:
    """Save all individuals in population as numpy arrays
    """
    folder = 'log/' + filename + '_pop/'
    if not os.path.exists(folder):
      os.makedirs(folder)

    for i in range(len(pop)):
      exportNet(folder+'ind_'+str(i)+'.out', pop[i].wMat, pop[i].aVec)

def lsave(filename: str, data: np.ndarray) -> None:
  """Short hand for numpy save with csv and float precision defaults
  """
  np.savetxt(filename, data, delimiter=',',fmt='%1.2e')
