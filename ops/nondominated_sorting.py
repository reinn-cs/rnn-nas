class NonDominatedSorting:
    """
    Implementation of the nondominated sorting used by the NAS search delegator.
    """

    def __init__(self, population_fitness, evaluating_objectives):
        self.population_fitness = population_fitness
        self.evaluating_objectives = evaluating_objectives

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        print('NDS deleted.')

    def sort(self, architecture_keys):
        """
        Fast non-dominated sort as introduced by: Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and
        elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182–197.
        https://doi.org/10.1109/4235.996017

        :param architecture_keys:
        :return:
        """
        dominating_set = {}
        n_values = {}
        fronts = {1: []}

        for p_key in architecture_keys:
            dominating_set[p_key] = set()
            n_values[p_key] = 0

            for q_key in architecture_keys:
                if p_key == q_key:
                    continue

                if self.does_p_dominate_q(p_key, q_key):
                    dominating_set[p_key].add(q_key)
                elif self.does_p_dominate_q(q_key, p_key):
                    n_values[p_key] += 1

            if n_values[p_key] == 0:
                if 1 not in fronts.keys():
                    fronts[1] = []
                if p_key not in fronts[1]:
                    fronts[1].append(p_key)

        i = 1
        while i in fronts.keys() and len(fronts[i]) > 0:
            h = []

            for p_key in fronts[i]:
                for q_key in dominating_set[p_key]:
                    n_values[q_key] -= 1

                    if n_values[q_key] == 0 and q_key not in h:
                        h.append(q_key)

            i += 1
            if len(h) > 0:
                fronts[i] = h

        return fronts

    def does_p_dominate_q(self, p, q):
        p_fitness = self.population_fitness[p]
        q_fitness = self.population_fitness[q]

        p_score = 0
        q_score = 0

        def test_domination(p_val, q_val):
            """
            return 1, 0 if p_val dominates q_val
            return 0, 1 if q_val dominates p_val
            return 0,0 if neither dominates

            :param p_val:
            :param q_val:
            :return:
            """

            if p_val < 10.0e+10 and q_val >= 10.0e+10:
                return 1, 0
            elif p_val >= 10.0e+10 and q_val < 10.0e+10:
                return 0, 1

            if p_val < q_val:
                return 1, 0
            elif q_val < p_val:
                return 0, 1
            else:
                return 0, 0

        for _attr in self.evaluating_objectives:
            val_p, val_q = test_domination(getattr(p_fitness, _attr), getattr(q_fitness, _attr))

            p_score += val_p
            q_score += val_q
            if val_q == 1:
                # This means that q dominates p in at least one objective.
                # We don't have to look further and simply return False - which means p does not dominate q.
                return False

        # p must have at least one objective value that is better compared to q.
        return p_score > q_score
