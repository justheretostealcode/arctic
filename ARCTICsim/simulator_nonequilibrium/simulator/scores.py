import numpy as np
import ot


class FunctionalScore:
    def __init__(self, settings):
        self.settings = settings

    def __call__(self, dataON, dataOFF):
        settings = self.settings
        median_on = np.median(dataON)
        median_off = np.median(dataOFF)
        # Determine the sign of the exponent based on the median differences
        sign = 1 if np.median(dataON) > np.median(dataOFF) else -1
        score = 0
        DIST = settings["functional_score"]
        if ("kl" == DIST):  # Minimum of Kullback-Leibler Divergence
            mu_1 = np.mean(dataON)
            var_1 = np.var(dataON)
            mu_2 = np.mean(dataOFF)
            var_2 = np.var(dataOFF)

            score = min([0.5 * np.log(var_2 / var_1) + ((mu_1 - mu_2) ** 2 + var_1 - var_2) / (2 * var_2),
                         0.5 * np.log(var_1 / var_2) + ((mu_2 - mu_1) ** 2 + var_2 - var_1) / (2 * var_1)])
        elif ("ws" == DIST):  # Wasserstein Distance
            # score = sign * st.wasserstein_distance(dataON, dataOFF)
            score = sign * ot.wasserstein_1d(dataON, dataOFF, p=settings["wasserstein_p"])
        elif ("cello" == DIST):  # To prefer as Cello Score
            score = min(dataON) / max(dataOFF)
        elif ("ws-cello" == DIST):
            score = sign * ot.wasserstein_1d(dataON, dataOFF, p=settings["wasserstein_p"])
            score = sign * score / np.median(dataOFF) + 1
        elif ("ws-cello-m" == DIST):
            score = sign * ot.wasserstein_1d(dataON, dataOFF, p=settings["wasserstein_p"])
            score = score / np.mean(dataOFF) + 1
        elif ("ws-log" == DIST):
            score = sign * ot.wasserstein_1d(np.log(dataON), np.log(dataOFF), p=settings["wasserstein_p"])
        elif ("ws-exp" == DIST):
            score = sign * ot.wasserstein_1d(np.exp(dataON), np.exp(dataOFF), p=settings["wasserstein_p"])
        elif ("ws-log-exp" == DIST):  # To prefer as WS Score
            score = sign * ot.wasserstein_1d(np.log(dataON), np.log(dataOFF), p=settings["wasserstein_p"])
            score = np.exp(score)
        elif ("ws-log-exp-asym" == DIST):
            if len(dataON) == 1:
                score = np.log(median_on) - np.log(median_off)
            else:
                score = (0.5 * (np.log(median_on) - np.log(median_off))
                         + (np.sum(np.log(dataON[dataON < median_on]))
                            - np.sum(np.log(dataOFF[dataOFF > median_off]))) / len(dataON))
            score = np.exp(score)

        elif "mean" == DIST:
            score = np.mean(dataON) / np.mean(dataOFF)
        elif "median" == DIST:
            score = np.median(dataON) / np.median(dataOFF)
        else:
            score = -9999  # (np.mean(dataON) - np.mean(dataOFF)) ** 2    # Error Code if no appropriate score is used
            raise Exception(f"Score {DIST} not supported")
        return score


class EnergyScore:

    def __init__(self, settings):
        self.settings = settings

        comprehension_func = None
        if "max" == settings["energy_score"]:
            comprehension_func = np.max
        elif "avg" == settings["energy_score"]:
            comprehension_func = np.mean
        elif "sum" == settings["energy_score"]:
            comprehension_func = np.sum
        else:
            raise Exception(f"Score {settings['energy_score']} not supported")

        self.comprehension_func = comprehension_func

    def __call__(self, energy_rates):
        score = self.comprehension_func(energy_rates)

        return score
