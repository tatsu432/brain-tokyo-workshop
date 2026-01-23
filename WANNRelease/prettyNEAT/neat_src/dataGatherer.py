import copy
import os

import numpy as np

from ._speciate import Species
from .ann import exportNet
from .ind import Ind


class DataGatherer:
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

    Raw fitness (actual game reward without shaping):
    - fit_max_raw: elite raw fitness per generation
    - fit_top_raw: best-so-far raw fitness per generation

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
        self.filename = filename  # File name path + prefix
        self.p = hyp

        # Initialize empty fields
        self.elite: list[
            Ind
        ] = []  # List of elite individuals until the current generation
        self.best: list[
            Ind
        ] = []  # List of best individuals until the current generation
        self.bestFitVec: list[float] = []
        self.spec_fit: list[tuple[int, float]] = []
        self.field: list[str] = [
            "x_scale",
            "fit_med",
            "fit_max",
            "fit_top",
            "node_med",
            "conn_med",
            "num_species",
            "elite",
            "best",
        ]

        if self.p["alg_probMoo"] > 0:
            self.objVals = np.array([])

        for f in self.field[:-2]:  # FIX: exclude 'elite', 'best' (last 2)
            exec("self." + f + " = np.array([])")
            # e.g. self.fit_max   = np.array([])

        self.newBest = False

        # Raw fitness tracking (actual game reward without shaping)
        self.fit_max_raw = np.array([])  # Elite raw fitness per generation
        self.fit_top_raw = np.array([])  # Best-so-far raw fitness per generation
        self.elite_raw_fitness = []  # Raw fitness history for each elite
        self.best_raw_fitness = []  # Raw fitness history for each best
        self.current_elite_raw_fitness = None  # Current generation's elite raw fitness
        self.current_best_raw_fitness = None  # Current best raw fitness

        # Display configuration (can be toggled)
        self.display_config = {
            "show_elite_total": True,  # Show elite total fitness (shaped)
            "show_elite_raw": True,  # Show elite raw fitness (actual)
            "show_best_total": True,  # Show best total fitness (shaped)
            "show_best_raw": True,  # Show best raw fitness (actual)
            "show_species": True,  # Show species count
            "show_complexity": True,  # Show node/conn counts
            "show_actions": True,  # Show action distribution
            "show_shaped_details": False,  # Show detailed shaped reward components
        }

        # Shaped reward component tracking (population averages per generation)
        self.shaped_stats_history = []  # List of dicts, one per generation
        self.current_shaped_stats = None  # Current generation's shaped stats

        # Shaped component field names for display
        self.shaped_stat_labels = {
            "avg_touches": "Touch",
            "avg_rallies_won": "RallyW",
            "avg_rallies_lost": "RallyL",
            "avg_ball_time_opponent_side": "BallOpp",
            "avg_tracking_reward": "Track",
        }

        # Action distribution tracking
        # For continuous actions: [nOutput x n_bins] showing binned distribution
        # For discrete actions: [nOutput] showing count per action
        self.action_dist_history: list[np.ndarray] = []
        self.action_bin_labels = ["<-0.5", "-0.5~0", "0~0.5", ">0.5"]
        self.current_action_dist = None
        self.is_discrete_action = False  # Will be set by caller

        # Labels for 6 discrete SlimeVolley actions
        self.discrete_action_labels = ["L", "S", "R", "LJ", "SJ", "RJ"]
        # L=left, S=stay, R=right, J=jump

    def gatherData(
        self,
        pop: list[Ind],
        species: Species,
        action_dist: np.ndarray = None,
        raw_fitness: np.ndarray = None,
        shaped_stats: dict = None,
    ) -> None:
        """Collect and stores run data
        This is called once per generation (or once per "iteration" of the algorithm).

        Args:
          pop         - [Ind]      - list of individuals in population
          species     - (Species)  - current species
          action_dist - (np_array) - aggregated action distribution [nOutput x n_bins]
          raw_fitness - (np_array) - raw fitness (actual game reward) for each individual
                        If None, raw fitness is assumed to equal total fitness (no shaping)
          shaped_stats - (dict)    - aggregated shaped reward component statistics
                        Keys: avg_touches, avg_rallies_won, avg_rallies_lost,
                              avg_ball_time_opponent_side, avg_tracking_reward
        """

        # Readability
        fitness = [ind.fitness for ind in pop]
        nodes = np.asarray(
            [np.shape(ind.node)[1] for ind in pop]
        )  # Number of nodes in the individual
        conns = np.asarray(
            [ind.nConn for ind in pop]
        )  # Number of connections in the individual

        # Handle raw fitness - if not provided, use total fitness
        if raw_fitness is None:
            raw_fitness = np.array(fitness)

        # --- Evaluation Scale ---------------------------------------------------
        # it's not the generation index. It's like an x-axis for learning curves in terms of evaluations.
        if len(self.x_scale) == 0:
            self.x_scale = np.append(self.x_scale, len(pop))
        else:
            self.x_scale = np.append(self.x_scale, self.x_scale[-1] + len(pop))
        # ------------------------------------------------------------------------

        # --- Best Individual (total fitness) ------------------------------------
        elite_idx = np.argmax(fitness)
        self.elite.append(pop[elite_idx])
        if len(self.best) == 0:
            self.best.append(copy.deepcopy(self.elite[-1]))
        elif self.elite[-1].fitness > self.best[-1].fitness:
            self.best.append(copy.deepcopy(self.elite[-1]))
            self.newBest = True
        else:
            self.best.append(copy.deepcopy(self.best[-1]))
            self.newBest = False
        # ------------------------------------------------------------------------

        # --- Raw Fitness Tracking -----------------------------------------------
        # Track raw fitness for elite (best of current generation)
        elite_raw_fit = raw_fitness[elite_idx]
        self.elite_raw_fitness.append(elite_raw_fit)
        self.current_elite_raw_fitness = elite_raw_fit
        self.fit_max_raw = np.append(self.fit_max_raw, elite_raw_fit)

        # Track raw fitness for best (best across all generations)
        if len(self.best_raw_fitness) == 0:
            self.best_raw_fitness.append(elite_raw_fit)
            self.current_best_raw_fitness = elite_raw_fit
        elif self.newBest:
            # New best individual - update raw fitness
            self.best_raw_fitness.append(elite_raw_fit)
            self.current_best_raw_fitness = elite_raw_fit
        else:
            # Keep previous best's raw fitness
            self.best_raw_fitness.append(self.best_raw_fitness[-1])
            self.current_best_raw_fitness = self.best_raw_fitness[-1]

        self.fit_top_raw = np.append(self.fit_top_raw, self.current_best_raw_fitness)
        # ------------------------------------------------------------------------

        # --- Generation fit/complexity stats ------------------------------------
        self.node_med = np.append(
            self.node_med, np.median(nodes)
        )  # Median number of nodes in the population
        self.conn_med = np.append(
            self.conn_med, np.median(conns)
        )  # Median number of connections in the population
        self.fit_med = np.append(
            self.fit_med, np.median(fitness)
        )  # Median fitness in the population
        self.fit_max = np.append(
            self.fit_max, self.elite[-1].fitness
        )  # Best fitness in the population
        self.fit_top = np.append(
            self.fit_top, self.best[-1].fitness
        )  # Best fitness across history
        # ------------------------------------------------------------------------

        # --- MOO Fronts ---------------------------------------------------------
        if self.p["alg_probMoo"] > 0:
            if len(self.objVals) == 0:
                self.objVals = np.c_[fitness, conns]
            else:
                self.objVals = np.c_[self.objVals, np.c_[fitness, conns]]
        # ------------------------------------------------------------------------

        # --- Species Stats ------------------------------------------------------
        if self.p["alg_speciate"] == "neat":
            specFit = np.empty((2, 0))
            for iSpec in range(len(species)):
                for ind in species[iSpec].members:
                    tmp = np.array((iSpec, ind.fitness))
                    specFit = np.c_[specFit, tmp]
            self.spec_fit = specFit

        self.num_species = np.append(self.num_species, len(species))
        # ------------------------------------------------------------------------

        # --- Action Distribution ------------------------------------------------
        if action_dist is not None:
            self.action_dist_history.append(action_dist.copy())
            self.current_action_dist = action_dist
        # ------------------------------------------------------------------------

        # --- Shaped Reward Component Stats --------------------------------------
        if shaped_stats is not None:
            self.shaped_stats_history.append(shaped_stats.copy())
            self.current_shaped_stats = shaped_stats
        else:
            # Store empty dict if no stats provided
            self.shaped_stats_history.append({})
            self.current_shaped_stats = {}
        # ------------------------------------------------------------------------

    def set_display_config(self, **kwargs):
        """Configure which metrics to display.

        Args (all optional bool):
          show_elite_total: Show elite total fitness (shaped)
          show_elite_raw: Show elite raw fitness (actual)
          show_best_total: Show best total fitness (shaped)
          show_best_raw: Show best raw fitness (actual)
          show_species: Show species count
          show_complexity: Show node/conn counts
          show_actions: Show action distribution
        """
        for key, value in kwargs.items():
            if key in self.display_config:
                self.display_config[key] = value

    def display(self):
        """Console output for each generation.

        Output format depends on display_config settings:
        - Elite(T/R): total/raw fitness of current generation's best
        - Best(T/R): total/raw fitness of best across all generations
        - Species, node, conn counts
        - Action distribution

        Use set_display_config() to enable/disable specific outputs.
        """
        cfg = self.display_config
        parts = []

        # --- Elite fitness (current generation's best) ---
        if cfg["show_elite_total"] or cfg["show_elite_raw"]:
            elite_str = "Elite:"
            if cfg["show_elite_total"]:
                elite_str += " T={:>6.2f}".format(self.fit_max[-1])
            if cfg["show_elite_raw"] and self.current_elite_raw_fitness is not None:
                elite_str += " R={:>6.2f}".format(self.current_elite_raw_fitness)
            parts.append(elite_str)

        # --- Best fitness (best across all generations) ---
        if cfg["show_best_total"] or cfg["show_best_raw"]:
            best_str = "Best:"
            if cfg["show_best_total"]:
                best_str += " T={:>6.2f}".format(self.fit_top[-1])
            if cfg["show_best_raw"] and self.current_best_raw_fitness is not None:
                best_str += " R={:>6.2f}".format(self.current_best_raw_fitness)
            parts.append(best_str)

        # --- Species count ---
        if cfg["show_species"]:
            parts.append("#Sp:{:2d}".format(int(self.num_species[-1])))

        # --- Complexity (node/conn) ---
        if cfg["show_complexity"]:
            parts.append(
                "#n:{} #c:{}".format(int(self.node_med[-1]), int(self.conn_med[-1]))
            )

        output = " | ".join(parts)

        # --- Shaped reward component details ---
        if cfg.get("show_shaped_details", False) and self.current_shaped_stats:
            shaped_parts = []
            stats = self.current_shaped_stats

            # Ball touches
            if "avg_touches" in stats:
                shaped_parts.append("Touch:{:.1f}".format(stats["avg_touches"]))

            # Rallies won/lost
            if "avg_rallies_won" in stats:
                shaped_parts.append("RallyW:{:.1f}".format(stats["avg_rallies_won"]))
            if "avg_rallies_lost" in stats:
                shaped_parts.append("RallyL:{:.1f}".format(stats["avg_rallies_lost"]))

            # Ball time on opponent side (as percentage of steps)
            if "avg_ball_time_opponent_side" in stats:
                ball_opp = stats["avg_ball_time_opponent_side"]
                shaped_parts.append("BallOpp:{:.0f}".format(ball_opp))

            # Tracking reward (cumulative)
            if "avg_tracking_reward" in stats:
                shaped_parts.append("Track:{:.2f}".format(stats["avg_tracking_reward"]))

            if shaped_parts:
                output += " | Shp: " + " ".join(shaped_parts)

        # --- Action distribution ---
        if cfg["show_actions"] and self.current_action_dist is not None:
            if self.is_discrete_action:
                # Discrete actions: show distribution with action labels
                # Format: L:15% S:10% R:20% LJ:25% SJ:15% RJ:15%
                output += " | Act:"
                for i, pct in enumerate(self.current_action_dist):
                    label = (
                        self.discrete_action_labels[i]
                        if i < len(self.discrete_action_labels)
                        else f"A{i}"
                    )
                    output += " {}:{:2d}%".format(label, int(pct * 100))
            else:
                # Continuous actions: show binned distribution per output
                output += " | ActDist:"
                for i, row in enumerate(self.current_action_dist):
                    # Format: O0[25%,25%,25%,25%] for each output dimension
                    pcts = ["{:.0f}%".format(v * 100) for v in row]
                    output += " O{}[{}]".format(i, ",".join(pcts))

        return output

    def display_compact(self):
        """Compact display showing only raw (actual) fitness.

        Useful when you only care about actual game performance.
        """
        elite_raw = (
            self.current_elite_raw_fitness
            if self.current_elite_raw_fitness is not None
            else self.fit_max[-1]
        )
        best_raw = (
            self.current_best_raw_fitness
            if self.current_best_raw_fitness is not None
            else self.fit_top[-1]
        )

        return "Elite(raw):{:.2f} Best(raw):{:.2f}".format(elite_raw, best_raw)

    def display_full(self):
        """Full display showing all metrics including both total and raw fitness."""
        # Temporarily enable all displays
        old_config = self.display_config.copy()
        self.display_config = {k: True for k in self.display_config}
        output = self.display()
        self.display_config = old_config
        return output

    def save(self, gen=(-1)):
        """Save algorithm stats to disk"""
        """ Save data to disk """
        filename = self.filename
        pref = "log/" + filename

        # --- Generation fit/complexity stats ------------------------------------
        # Now includes both total and raw fitness
        gStatLabel = [
            "x_scale",
            "fit_med",
            "fit_max",
            "fit_top",
            "fit_max_raw",
            "fit_top_raw",
            "node_med",
            "conn_med",
        ]
        genStats = np.empty((len(self.x_scale), 0))
        for i in range(len(gStatLabel)):
            # e.g.         self.    fit_max          [:,None]
            evalString = "self." + gStatLabel[i] + "[:,None]"
            try:
                col = eval(evalString)
                # Ensure column has right length (pad with last value if needed)
                if len(col) < len(self.x_scale):
                    col = np.append(
                        col,
                        np.full(
                            len(self.x_scale) - len(col), col[-1] if len(col) > 0 else 0
                        ),
                    )[:, None]
                genStats = np.hstack((genStats, col))
            except:
                # If array doesn't exist or is empty, fill with zeros
                genStats = np.hstack((genStats, np.zeros((len(self.x_scale), 1))))
        lsave(pref + "_stats.out", genStats)
        # ------------------------------------------------------------------------

        # --- Best Individual ----------------------------------------------------
        wMat = self.best[gen].wMat
        aVec = self.best[gen].aVec
        exportNet(pref + "_best.out", wMat, aVec)

        if gen > 1:
            folder = "log/" + filename + "_best/"
            if not os.path.exists(folder):
                os.makedirs(folder)
            exportNet(folder + str(gen).zfill(4) + ".out", wMat, aVec)
        # ------------------------------------------------------------------------

        # --- Species Stats ------------------------------------------------------
        if self.p["alg_speciate"] == "neat":
            lsave(pref + "_spec.out", self.spec_fit)
        # ------------------------------------------------------------------------

        # --- MOO Fronts ---------------------------------------------------------
        if self.p["alg_probMoo"] > 0:
            lsave(pref + "_objVals.out", self.objVals)
        # ------------------------------------------------------------------------

        # --- Action Distribution History ----------------------------------------
        if len(self.action_dist_history) > 0:
            action_dist_data = []
            for gen_idx, dist in enumerate(self.action_dist_history):
                if dist.ndim == 1:
                    # Discrete actions: [gen, action0_pct, action1_pct, ...]
                    action_dist_data.append([gen_idx] + list(dist))
                else:
                    # Continuous actions: [gen, output_dim, bin0, bin1, bin2, bin3]
                    for out_idx, row in enumerate(dist):
                        action_dist_data.append([gen_idx, out_idx] + list(row))
            if action_dist_data:
                lsave(pref + "_action_dist.out", np.array(action_dist_data))
        # ------------------------------------------------------------------------

    def savePop(self, pop: list[Ind], filename: str) -> None:
        """Save all individuals in population as numpy arrays"""
        folder = "log/" + filename + "_pop/"
        if not os.path.exists(folder):
            os.makedirs(folder)

        for i in range(len(pop)):
            exportNet(folder + "ind_" + str(i) + ".out", pop[i].wMat, pop[i].aVec)


def lsave(filename: str, data: np.ndarray) -> None:
    """Short hand for numpy save with csv and float precision defaults"""
    np.savetxt(filename, data, delimiter=",", fmt="%1.2e")
