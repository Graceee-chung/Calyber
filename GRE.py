#!/usr/bin/env python
# coding: utf-8

# # Student Policies
# Now that you know how model cost, conversion, demand, and optimize price and matching, we can move on the final part of this game, **building your policies!**. Below we have set up skeleton code for you to build your policies. Make when you finish to run the last code cell to upload your policy.

# In[2]:


# Load in all your libraries and packages here!
# NOTE: you are allowed to use packages loaded below ONLY. If you want to use any other packages, please contact the TA.
import pandas as pd
import numpy as np
import itertools
import math
import random
import copy
import time
import pandas as pd
import haversine
import unittest
import pickle
import collections

from utils import * # this imports all helper functions in utils.py

# TODO: Replace the string "TeamName" below with your own team's name in camel case. For e.g. "AwesomeTeam"
# DO NOT RENAME THE FILE, the file name should be "{TEAM_NAME}_Policies.py"
TEAM_NAME = "GRE"


# # Pricing Policy

# Feel free to add more functions to `StudentPricingPolicy` but **DO NOT** modify the initalization and static methods. Your code will only be graded based on the output of your pricing function.

# In[3]:


class StudentPricingPolicy:
    """Pricing policy that uses current queue geometry more than historical route frequency."""

    BASE_PRICE = 0.725
    PRICE_FLOOR = 0.62
    PRICE_CEILING = 0.81
    PAIR_CACHE = {}

    # DO NOT MODIFY
    def __init__(self, c = 0.70):
        self.c = c

    # DO NOT MODIFY
    @staticmethod
    def get_name():
        return TEAM_NAME

    @staticmethod
    def _clip(value, lower=0.0, upper=1.0):
        return max(lower, min(upper, value))

    @staticmethod
    def _safe_ratio(numerator, denominator):
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    @staticmethod
    def _rider_signature(rider):
        return (
            round(float(rider.pickup_lat), 6),
            round(float(rider.pickup_lon), 6),
            round(float(rider.dropoff_lat), 6),
            round(float(rider.dropoff_lon), 6),
            int(rider.pickup_area),
            int(rider.dropoff_area),
            round(float(rider.arrival_time), 3),
        )

    @classmethod
    def _pair_metrics(cls, rider_i, rider_j):
        sig_i = cls._rider_signature(rider_i)
        sig_j = cls._rider_signature(rider_j)
        cache_key = (sig_i, sig_j) if sig_i <= sig_j else (sig_j, sig_i)
        cached = cls.PAIR_CACHE.get(cache_key)
        if cached is not None:
            return cached

        origin_i = (rider_i.pickup_lat, rider_i.pickup_lon)
        dest_i = (rider_i.dropoff_lat, rider_i.dropoff_lon)
        origin_j = (rider_j.pickup_lat, rider_j.pickup_lon)
        dest_j = (rider_j.dropoff_lat, rider_j.dropoff_lon)

        trip_length, shared_length, i_solo_length, j_solo_length, _ = populate_shared_ride_lengths(
            origin_i,
            dest_i,
            origin_j,
            dest_j,
        )

        total_length = rider_i.solo_length + rider_j.solo_length
        min_length = min(rider_i.solo_length, rider_j.solo_length)
        solo_ratio_i = cls._safe_ratio(i_solo_length, rider_i.solo_length)
        solo_ratio_j = cls._safe_ratio(j_solo_length, rider_j.solo_length)

        metrics = {
            "trip_ratio": cls._safe_ratio(trip_length, total_length),
            "shared_ratio": cls._safe_ratio(shared_length, min_length),
            "solo_ratio_max": max(solo_ratio_i, solo_ratio_j),
            "absolute_savings": max(0.0, total_length - trip_length),
            "savings_ratio": max(0.0, 1.0 - cls._safe_ratio(trip_length, total_length)),
            "same_pickup": int(rider_i.pickup_area == rider_j.pickup_area),
            "same_dropoff": int(rider_i.dropoff_area == rider_j.dropoff_area),
        }

        if len(cls.PAIR_CACHE) > 25000:
            cls.PAIR_CACHE.clear()
        cls.PAIR_CACHE[cache_key] = metrics
        return metrics

    @classmethod
    def _shareability_snapshot(cls, state, rider):
        same_route = 0
        same_pickup = 0
        same_dropoff = 0
        viable_count = 0
        strong_count = 0
        best_value = 0.0

        for waiting_rider in state:
            if int(waiting_rider.pickup_area) == int(rider.pickup_area):
                same_pickup += 1
            if int(waiting_rider.dropoff_area) == int(rider.dropoff_area):
                same_dropoff += 1
            if int(waiting_rider.pickup_area) == int(rider.pickup_area) and int(waiting_rider.dropoff_area) == int(rider.dropoff_area):
                same_route += 1

            metrics = cls._pair_metrics(rider, waiting_rider)
            if metrics["shared_ratio"] <= 0.0:
                continue
            if metrics["savings_ratio"] < 0.10 and metrics["absolute_savings"] < 0.25:
                continue
            if metrics["solo_ratio_max"] > 0.80 and metrics["absolute_savings"] < 0.45:
                continue

            value = (
                1.80 * metrics["savings_ratio"]
                + 0.80 * min(metrics["shared_ratio"], 1.0)
                + 0.45 * max(0.0, 1.0 - metrics["solo_ratio_max"])
                + 0.12 * metrics["absolute_savings"]
                + 0.05 * metrics["same_pickup"]
                + 0.05 * metrics["same_dropoff"]
            )
            viable_count += 1
            if value >= 0.95:
                strong_count += 1
            best_value = max(best_value, value)

        return {
            "same_route": same_route,
            "same_pickup": same_pickup,
            "same_dropoff": same_dropoff,
            "viable_count": viable_count,
            "strong_count": strong_count,
            "best_value": best_value,
        }

    @classmethod
    def _baseline_price(cls, rider):
        price = cls.BASE_PRICE

        if rider.solo_length < 1.5:
            price -= 0.040
        elif rider.solo_length < 3.0:
            price -= 0.018
        elif rider.solo_length > 8.5:
            price += 0.032
        elif rider.solo_length > 6.0:
            price += 0.018

        return price

    def pricing_function(self, state, rider):
        """
        Returns the price of the given rider in the given state

        Parameters
        ----------
        state: list
            list of rider(s) (object) waiting in the state
        rider: object
            An incoming rider

        Returns
        -------
        float
            The price of the rider: must be in [0, 1]
        """
        price = self._baseline_price(rider)
        state_size = len(state)

        if state_size == 0:
            price += 0.020
            return float(self._clip(price, self.PRICE_FLOOR, self.PRICE_CEILING))

        snapshot = self._shareability_snapshot(state, rider)

        if snapshot["best_value"] >= 1.10:
            price -= 0.028
        elif snapshot["best_value"] >= 0.88:
            price -= 0.018
        elif snapshot["best_value"] >= 0.68:
            price -= 0.010

        if snapshot["same_route"] > 0:
            price -= 0.010
        elif snapshot["same_pickup"] > 0 and snapshot["same_dropoff"] > 0:
            price -= 0.008

        if snapshot["viable_count"] >= 2:
            price -= 0.008
        if snapshot["strong_count"] >= 2:
            price -= 0.006

        if state_size >= 12 and snapshot["viable_count"] == 0:
            price += 0.008
        if state_size >= 30 and snapshot["viable_count"] == 0:
            price += 0.008
        elif state_size >= 30 and snapshot["viable_count"] >= 3:
            price -= 0.006

        return float(self._clip(price, self.PRICE_FLOOR, self.PRICE_CEILING))


# # Matching Policy

# Again, feel free to add more functions to `StudentMatchingPolicy` but **DO NOT** modify the initalization and static methods. Your code will only be graded based on the output of your matching function.

# In[4]:


class StudentMatchingPolicy:
    # DO NOT MODIFY
    def __init__(self, c = 0.70):
        self.c = c

    # DO NOT MODIFY
    @staticmethod
    def get_name():
        return TEAM_NAME

    @staticmethod
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    @staticmethod
    def get_origin(r):
        return (r.pickup_lat, r.pickup_lon)

    @staticmethod
    def get_destination(r):
        return (r.dropoff_lat, r.dropoff_lon)

    @staticmethod
    def quick_pair_score(r1, r2):
        """
        Cheap compatibility proxy used before exact routing.
        Lower is better.
        """
        pickup_gap = abs(r1.pickup_area - r2.pickup_area)
        dropoff_gap = abs(r1.dropoff_area - r2.dropoff_area)
        time_gap = abs(r1.arrival_time - r2.arrival_time)
        length_gap = abs(r1.solo_length - r2.solo_length)

        return 2.0 * pickup_gap + 2.0 * dropoff_gap + 0.003 * time_gap + 0.2 * length_gap

    def candidate_exact_match_score(self, rider, waiting_rider):
        """
        Exact pair evaluation using the provided helper.
        """
        trip_length, shared_length, i_solo_length, j_solo_length, trip_order = populate_shared_ride_lengths(
            self.get_origin(rider), self.get_destination(rider),
            self.get_origin(waiting_rider), self.get_destination(waiting_rider)
        )

        solo_sum = rider.solo_length + waiting_rider.solo_length
        saving = solo_sum - trip_length
        saving_ratio = saving / max(solo_sum, 1e-6)
        shared_ratio = shared_length / max(trip_length, 1e-6)
        detour_penalty = (i_solo_length + j_solo_length) / max(trip_length, 1e-6)

        score = 2.2 * saving_ratio + 0.35 * shared_ratio + 0.18 * saving - 0.18 * detour_penalty

        return {
            "saving": saving,
            "saving_ratio": saving_ratio,
            "shared_ratio": shared_ratio,
            "detour_penalty": detour_penalty,
            "score": score,
        }

    def matching_function(self, state, rider):
        if len(state) == 0:
            return None

        max_candidates = 12
        base_min_saving_ratio = 0.12
        base_min_absolute_saving = 0.35
        large_queue_threshold = 8

        # Stage 1: cheap shortlist
        candidates = sorted(state, key=lambda w: self.quick_pair_score(rider, w))[:max_candidates]

        min_saving_ratio = base_min_saving_ratio
        min_absolute_saving = base_min_absolute_saving

        # Large queue => match more aggressively
        if len(state) >= large_queue_threshold:
            min_saving_ratio -= 0.02
            min_absolute_saving -= 0.08

        # Late arrival => waiting is less attractive
        if rider.arrival_time > 3300:
            min_saving_ratio -= 0.015
            min_absolute_saving -= 0.05

        min_saving_ratio = max(0.05, min_saving_ratio)
        min_absolute_saving = max(0.12, min_absolute_saving)

        best_candidate = None
        best_score = -1e18

        # Stage 2: exact evaluation
        for w in candidates:
            stats = self.candidate_exact_match_score(rider, w)

            saving = stats["saving"]
            saving_ratio = stats["saving_ratio"]
            score = stats["score"]

            if saving_ratio >= min_saving_ratio and saving >= min_absolute_saving:
                if score > best_score:
                    best_score = score
                    best_candidate = w

        return best_candidate
