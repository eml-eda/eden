import numpy as np


class QWYC:
    def __init__(
        self, base_model, leaves_bits, alpha=0.1, tolerance=1e-03, cost=None, batch=1
    ):
        self.base_model = base_model
        self.alpha = alpha
        self.tolerance = tolerance
        self.batch = batch
        self.n_estimators = self.base_model.n_estimators
        self.pi = [i for i in range(self.n_estimators)]
        self.eps = [(None, None) for _ in range(self.n_estimators)]
        if cost is None:
            self.cost = [1.0 for _ in range(self.n_estimators)]
        else:
            self.cost = cost
        self.leaves_bits = int(leaves_bits)
        self.min_val = -(2 ** (int(leaves_bits) - 1))
        self.max_val = +(2 ** (int(leaves_bits) - 1)) - 1

    def fit(self, x):
        print("Starting the reordering")
        pred = self.raw_predictions(x)
        p_full = (
            self.base_model.predict(x, bitwidth=self.leaves_bits)
            .astype(bool)
            .reshape(-1)
        )
        for r in range(self.n_estimators):
            k_star = r
            j_star = None
            eps_star = (None, None)
            for k in range(r, self.n_estimators):
                self._swap(r, k)
                g = self.g(pred, r)
                eps = self.optimize_thresholds(g, p_full, r)
                j = self.j(g, r, eps)
                if j_star is None or j < j_star:
                    j_star = j
                    eps_star = eps
                    k_star = k
                self._swap(r, k)
            self._swap(r, k_star)
            self.eps[r] = eps_star
            print("R:", r, "Optimal k:", self.pi[r], "Optimal eps:", eps_star)

    def fit_no_ordering(self, x):
        pred = self.raw_predictions(x)
        p_full = (
            self.base_model.predict(x, bitwidth=self.leaves_bits)
            .astype(bool)
            .reshape(-1)
        )
        g = self.g(pred, self.n_estimators - 1)
        for i in range(self.n_estimators):
            eps = self.optimize_thresholds(g, p_full, i)
            self.eps[i] = eps
            print("Optimized thresholds for classifier {}: {}".format(i, eps))

    def predict(self, x, batch):
        pred = self.raw_predictions(x)
        g = self.g(pred, self.n_estimators - 1)
        label = np.zeros((x.shape[0]))
        complexities = np.zeros((x.shape[0]))

        pi = self.pi
        branches = pred["branches"][self.pi]
        count = np.zeros((x.shape[0]))
        branches = np.cumsum(branches, axis=0)
        for t in range(batch - 1, self.n_estimators - 1, batch):
            p_t = g[pi[t]] > self.eps[t][1]
            n_t = g[pi[t]] < self.eps[t][0]
            complexities[(count == 0) & (p_t | n_t)] = branches[t]
            label[(count == 0) & (p_t)] = 1
            count[(count == 0) & (p_t | n_t)] = t + 1
        classified = count == 0
        label[(g[pi[-1]] >= 0) & classified] = 1
        complexities[classified] = branches[self.n_estimators - 1]
        count[count == 0] = self.n_estimators

        return label, count, complexities

    def optimize_thresholds(self, g, p_full, r):
        low = g[r].min()  # self.min_val
        high = g[r].max()  # self.max_val
        iterations_high = 0
        iterations_low = 0
        eps_rm = np.floor((low + high) / 2.0)
        constr = self.constraint(g, p_full, r, (eps_rm, self.max_val))
        print("Initial constraint", constr)
        while constr >= self.alpha and (high - low) > self.tolerance:
            # misclassifications grow with larger eps- --> a violation means we must reduce eps-
            if constr > self.alpha:
                high = eps_rm
            else:
                low = eps_rm
            eps_rm = np.floor((low + high) / 2.0)
            constr = self.constraint(g, p_full, r, (eps_rm, self.max_val))
            iterations_high += 1

        low = eps_rm
        high = g[r].max()  # self.max_val
        eps_rp = np.ceil((low + high) / 2.0)
        constr = self.constraint(g, p_full, r, (eps_rm, eps_rp))
        while constr >= self.alpha and (high - low) > self.tolerance:
            if constr > self.alpha:
                low = eps_rp
            else:
                high = eps_rp
            eps_rp = np.ceil((low + high) / 2.0)
            constr = self.constraint(g, p_full, r, (eps_rm, eps_rp))
            iterations_low += 1

        return eps_rm, eps_rp

    def constraint(self, g, p_full, r, eps_r):
        n_full = ~p_full
        n = p_full.shape[0]
        eps = self.eps[:r]
        eps.append(eps_r)
        sel_idx_1 = np.zeros((n,), dtype=bool)
        sel_idx_2 = np.zeros((n,), dtype=bool)
        for t in range(r + 1):
            p_t = g[t] > eps[t][1]
            n_t = g[t] < eps[t][0]
            c_prev = self.get_uncertain(g, t - 1, eps)
            sel_idx_1 = sel_idx_1 | (c_prev & p_t & n_full)
            sel_idx_2 = sel_idx_2 | (c_prev & n_t & p_full)
        constr = (1 / n) * (sel_idx_1.sum() + sel_idx_2.sum())
        return constr

    def j(self, g, r, eps_r):
        eps = self.eps[:r]
        eps.append(eps_r)
        c_prev = self.get_uncertain(g, r - 1, eps)
        c_curr = self.get_uncertain(g, r, eps)
        additional = c_prev & ~c_curr
        j = self.cost[self.pi[r]] * c_prev.sum() / (additional.sum() + 1e-06)
        return j

    @staticmethod
    def get_uncertain(g, r, eps):
        sel_idx = np.ones((g[0].shape[0],), dtype=bool)
        for t in range(r + 1):
            ut = (eps[t][0] <= g[t]) & (g[t] <= eps[t][1])
            sel_idx = sel_idx & ut
        return sel_idx

    # compute the raw predictions of all weak learners
    def raw_predictions(self, x):
        probs = self.base_model.predict_probs(x, bitwidth=self.leaves_bits)
        if len(probs.shape) > 2:
            probs = probs.reshape((probs.shape[0], probs.shape[1]))
        n_branches = self.base_model.predict_complexity(x).mean(-1)  # (Estimators, 1)
        pred = {
            "branches": n_branches,
            "init": np.zeros((probs.shape[-1])),
            "classifiers": probs,
        }

        return pred

    def g(self, pred, r):
        g = []
        pred_init = pred["init"].copy()
        pred_trees = pred["classifiers"]
        for t in range(r + 1):
            pred_t = pred_init.copy()
            for i in range(t + 1):
                pred_t += pred_trees[self.pi[i]]
            g_t = pred_t.reshape(
                -1
            )  # self.base_model.loss_._raw_prediction_to_proba(pred_t)
            g.append(g_t)
        return g

    def _swap(self, r, k):
        tmp = self.pi[r]
        self.pi[r] = self.pi[k]
        self.pi[k] = tmp
