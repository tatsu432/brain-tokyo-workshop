"""Data gathering and best individual management."""

import numpy as np
from neat_src.dataGatherer import DataGatherer
from neat_src.neat import Neat


def gather_data(
    data: DataGatherer,
    neat: Neat,
    gen: int,
    hyp: dict,
    save_pop: bool = False,
    action_dist=None,
    raw_fitness=None,
    shaped_stats=None,
    batch_eval_fn=None,
) -> DataGatherer:
    """Collects run data, saves it to disk, and exports pickled population.

    Args:
      data       - (DataGatherer)  - collected run data
      neat       - (Neat)          - neat algorithm container
        .pop     - [Ind]           - list of individuals in population
        .species - (Species)       - current species
      gen        - (int)           - current generation
      hyp        - (dict)          - algorithm hyperparameters
      save_pop   - (bool)          - save current population to disk?
      action_dist - (np_array)     - aggregated action distribution [nOutput x n_bins]
      raw_fitness - (np_array)     - raw fitness values (actual game reward) for each individual
      shaped_stats - (dict)        - aggregated shaped reward component statistics
      batch_eval_fn - (callable)   - function to evaluate population (for check_best)

    Return:
      data - (DataGatherer) - updated run data
    """
    data.gatherData(
        neat.pop,
        neat.species,
        action_dist=action_dist,
        raw_fitness=raw_fitness,
        shaped_stats=shaped_stats,
    )
    if (gen % hyp["save_mod"]) == 0:
        if batch_eval_fn is not None:
            data = check_best(data, hyp, batch_eval_fn)
        data.save(gen)

    if save_pop is True:  # Get a sample pop to play with in notebooks
        import pickle

        pref = "log/" + data.filename
        with open(pref + "_pop.obj", "wb") as fp:
            pickle.dump(neat.pop, fp)

    return data


def check_best(data: DataGatherer, hyp: dict, batch_eval_fn) -> DataGatherer:
    """Checks better performing individual if it performs over many trials.
    Test a new 'best' individual with many different seeds to see if it really
    outperforms the current best.

    Args:
      data - (DataGatherer) - collected run data
      hyp  - (dict)         - hyperparameters
      batch_eval_fn - (callable) - function to evaluate population

    Return:
      data - (DataGatherer) - collected run data with best individual updated

    * This is a bit hacky, but is only for data gathering, and not optimization
    """
    if data.newBest is True:
        n_worker = hyp.get("nWorker", 8)
        best_reps = max(hyp["bestReps"], (n_worker - 1))
        rep = np.tile(data.best[-1], best_reps)
        fit_vector, _, _, _ = batch_eval_fn(
            rep, n_worker, hyp, same_seed_for_each_individual=False, track_actions=False
        )
        true_fit = np.mean(fit_vector)
        if true_fit > data.best[-2].fitness:  # Actually better!
            data.best[-1].fitness = true_fit
            data.fit_top[-1] = true_fit
            data.bestFitVec = fit_vector
        else:  # Just lucky! Revert to previous best
            data.best[-1] = data.best[-2]
            data.fit_top[-1] = data.fit_top[-2]
            data.newBest = False
    return data
