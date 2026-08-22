from __future__ import annotations

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl.meta.env_stock_trading.reward_functions import (
    RewardContext,
    calculate_individual_reward_components,
    calculate_reward,
    calculate_stock_weights_after_trade,
)

matplotlib.use("Agg")

# from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat


class StockTradingEnv(gym.Env):
    """
    A stock trading environment for OpenAI gym

    Parameters:
        df (pandas.DataFrame): Dataframe containing data
        hmax (int): Maximum cash to be traded in each trade per asset.
        initial_amount (int): Amount of cash initially available
        buy_cost_pct (float, array): Cost for buying shares, each index corresponds to each asset
        sell_cost_pct (float, array): Cost for selling shares, each index corresponds to each asset
        turbulence_threshold (float): Maximum turbulence allowed in market for purchases to occur. If exceeded, positions are liquidated
        print_verbosity(int): When iterating (step), how often to print stats about state of env
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        reward_type="v0",
        reward_params=None,
        model_name="",
        mode="",
        iteration="",
        terminal_skip_days: int = 0,
        additional_raw_features: list[str] | None = None,
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.reward_type = reward_type
        self.reward_params = reward_params or {}
        self.terminal_skip_days = max(0, int(terminal_skip_days or 0))
        # Default FinRL behavior stops at the last market day.
        # For future-return rewards, training can stop earlier while still
        # keeping the final days in df as labels for reward calculation.
        self.terminal_day = max(1, len(self.df.index.unique()) - 1)
        if self.terminal_skip_days > 0:
            self.terminal_day = max(1, len(self.df.index.unique()) - self.terminal_skip_days)
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.additional_raw_features = list(additional_raw_features or [])
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.reward_components_memory = []
        self.last_episode_reward_components_memory = []
        self.last_episode_reward_component_dates = []
        self.actions_memory = []
        self.agent_actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.last_cash_after_sell = float(self.state[0])
        self.last_cash_after_buy = float(self.state[0])
        self.holdings_memory = [self._get_holdings_snapshot()]
        self.post_trade_weights_memory = []
        self.previous_stock_weights_after_trade = self._calculate_state_stock_weights()
        self.date_memory = [self._get_date()]
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to sell, for simlicity we just add it in techical index
                # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    # update balance
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to buy
                # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):
        self.terminal = self.day >= self.terminal_day
        if self.terminal:
            # VecEnv resets the environment immediately after done=True. Keep a
            # snapshot so REINFORCE can aggregate the just-finished full epoch
            # after env.step() returns without losing the per-stock daily data.
            self.last_episode_reward_components_memory = [
                np.asarray(row, dtype=float).copy()
                for row in self.reward_components_memory
            ]
            self.last_episode_reward_component_dates = list(
                self.date_memory[: len(self.reward_components_memory)]
            )
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[0]
                + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    )
                )
                - self.asset_memory[0]
            )  # initial_amount is only cash part of our initial asset
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_post_trade_weights = self.save_weight_memory()
                df_post_trade_weights.to_csv(
                    "results/post_trade_weights_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                plt.close()

            # Add outputs to logger interface
            # logger.record("environment/portfolio_value", end_total_asset)
            # logger.record("environment/total_reward", tot_reward)
            # logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            # logger.record("environment/total_cost", self.cost)
            # logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, False, {}

        else:
            # Preserve the policy output before hmax scaling, turbulence overrides,
            # and execution constraints. This is the action the agent actually gave.
            agent_actions = np.asarray(actions, dtype=float).copy()

            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            actions = actions.astype(
                int
            )  # convert into integer because we can't by fraction of shares
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            # ==============================
            # Custom Reward Preparation
            # ==============================

            # 記錄交易前的累積交易成本
            cost_before_trade = self.cost

            # 保留原始 action 正負號
            # 因為後面 _sell_stock 後，sell action 會被轉成正數
            raw_actions = actions.copy()

            # 記錄交易發生前的股價
            current_prices_before_trade = np.array(
                self.state[1 : (self.stock_dim + 1)]
            )

            # 最近 n 天高低點，之後可以改成 14 / 20 / 60 做實驗
            n_recent = 14

            # 記錄交易發生的當下是第幾天
            current_day = self.day

            # Calculate the pre-trade weights using today's prices. Reward v6
            # can compare these weights with today's post-trade weights so both
            # sides use the same P(i,t).
            stock_holdings_before_trade = np.asarray(
                self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)],
                dtype=float,
            )
            stock_weights_before_trade = calculate_stock_weights_after_trade(
                stock_prices=current_prices_before_trade,
                stock_holdings=stock_holdings_before_trade,
                cash_after_trade=float(self.state[0]),
            )

            # Original reward v6 method: compare today's post-trade weights
            # with the previous trading day's post-trade weights.
            previous_stock_weights_after_trade = np.asarray(
                self.previous_stock_weights_after_trade,
                dtype=float,
            ).copy()

            # 記錄股票順序
            if self.stock_dim == 1:
                tic_list = [self.data.tic]
            else:
                tic_list = self.data.tic.values

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            self.last_cash_after_sell = float(self.state[0])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            self.last_cash_after_buy = float(self.state[0])

            # Calculate each stock's portfolio weight immediately after the
            # requested trades have actually been executed. Use the current-day
            # prices here, before state advances to the next trading day.
            stock_holdings_after_trade = np.asarray(
                self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)],
                dtype=float,
            )
            stock_weights_after_trade = calculate_stock_weights_after_trade(
                stock_prices=current_prices_before_trade,
                stock_holdings=stock_holdings_after_trade,
                cash_after_trade=self.last_cash_after_buy,
            )
            self.post_trade_weights_memory.append(
                self._get_post_trade_weight_snapshot(
                    trade_date=self._get_date(),
                    tic_list=tic_list,
                    stock_prices=current_prices_before_trade,
                    stock_holdings=stock_holdings_after_trade,
                    stock_weights=stock_weights_after_trade,
                    cash_after_trade=self.last_cash_after_buy,
                )
            )

            self.actions_memory.append(actions)
            self.agent_actions_memory.append(agent_actions)

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())


            # ==============================
            # Reward Function Dispatch
            # ==============================
            reward_context = RewardContext(
                env=self,
                begin_total_asset=begin_total_asset,
                end_total_asset=end_total_asset,
                raw_actions=raw_actions,
                current_prices_before_trade=current_prices_before_trade,
                stock_weights_after_trade=stock_weights_after_trade,
                cost_before_trade=cost_before_trade,
                current_day=current_day,
                tic_list=tic_list,
                previous_stock_weights_after_trade=(
                    previous_stock_weights_after_trade
                ),
                stock_weights_before_trade=stock_weights_before_trade,
                executed_actions=actions.copy(),
            )

            reward_components = calculate_individual_reward_components(
                reward_type=self.reward_type,
                ctx=reward_context,
                params=self.reward_params,
            )
            self.reward_components_memory.append(reward_components.copy())

            self.reward = calculate_reward(
                reward_type=self.reward_type,
                ctx=reward_context,
                params=self.reward_params,
            )
            self.previous_stock_weights_after_trade = (
                stock_weights_after_trade.copy()
            )

            # Store unscaled reward for observation/debugging.
            self.rewards_memory.append(self.reward)

            # Keep FinRL's original scaling flow.
            self.reward = self.reward * self.reward_scaling

            self.state_memory.append(
                self.state
            )  # add current state in state_recorder for each step
            self.holdings_memory.append(self._get_holdings_snapshot())

        return self.state, self.reward, self.terminal, False, {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # initiate state
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.reward_components_memory = []
        self.actions_memory = []
        self.agent_actions_memory = []
        self.state_memory = []
        self.last_cash_after_sell = float(self.state[0])
        self.last_cash_after_buy = float(self.state[0])
        self.holdings_memory = [self._get_holdings_snapshot()]
        self.post_trade_weights_memory = []
        self.previous_stock_weights_after_trade = self._calculate_state_stock_weights()
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + self.num_stock_shares
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                    + sum(
                        (
                            self.data[feature].values.tolist()
                            for feature in self.additional_raw_features
                        ),
                        [],
                    )
                )  # append initial stocks_share to initial state, instead of all zero
            else:
                # for single stock
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                    + sum(
                        ([self.data[feature]] for feature in self.additional_raw_features),
                        [],
                    )
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                    + sum(
                        (
                            self.data[feature].values.tolist()
                            for feature in self.additional_raw_features
                        ),
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                    + sum(
                        ([self.data[feature]] for feature in self.additional_raw_features),
                        [],
                    )
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
                + sum(
                    (
                        self.data[feature].values.tolist()
                        for feature in self.additional_raw_features
                    ),
                    [],
                )
            )

        else:
            # for single stock
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                + sum(
                    ([self.data[feature]] for feature in self.additional_raw_features),
                    [],
                )
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date


    def _get_tic_list(self):
        if len(self.df.tic.unique()) > 1:
            return list(self.data.tic.values)
        return [self.data.tic]

    def _calculate_state_stock_weights(self):
        """Calculate stock weights represented by the current environment state."""
        return calculate_stock_weights_after_trade(
            stock_prices=np.asarray(
                self.state[1 : (self.stock_dim + 1)],
                dtype=float,
            ),
            stock_holdings=np.asarray(
                self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)],
                dtype=float,
            ),
            cash_after_trade=float(self.state[0]),
        )

    def _get_holdings_snapshot(self):
        cash = float(self.state[0])
        cash_after_sell = float(getattr(self, "last_cash_after_sell", cash))
        cash_after_buy = float(getattr(self, "last_cash_after_buy", cash))
        shares = list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
        return [cash, cash_after_sell, cash_after_buy] + [float(x) for x in shares]

    def save_holdings_memory(self):
        columns = ["cash", "cash_after_sell", "cash_after_buy"] + self._get_tic_list()
        df_holdings = pd.DataFrame(self.holdings_memory, columns=columns)
        df_holdings.insert(0, "date", self.date_memory[: len(df_holdings)])
        return df_holdings

    def _get_post_trade_weight_snapshot(
        self,
        trade_date,
        tic_list,
        stock_prices,
        stock_holdings,
        stock_weights,
        cash_after_trade,
    ):
        """Build one row of portfolio weights immediately after trading.

        Weight values are stored as fractions in [0, 1], not percentages.
        The stock columns match the exact weights used by reward v5.
        """
        prices = np.asarray(stock_prices, dtype=float)
        holdings = np.asarray(stock_holdings, dtype=float)
        weights = np.asarray(stock_weights, dtype=float)
        cash = float(cash_after_trade)
        total_asset = cash + float(np.sum(prices * holdings))

        if np.isfinite(total_asset) and total_asset > 1e-8:
            cash_weight = cash / total_asset
        else:
            cash_weight = 0.0

        stock_weight_sum = float(np.sum(weights))
        snapshot = {
            "date": trade_date,
            "cash_after_trade": cash,
            "total_asset_after_trade": total_asset,
            "cash_weight": cash_weight,
            "stock_weight_sum": stock_weight_sum,
            "total_weight": cash_weight + stock_weight_sum,
        }
        snapshot.update(
            {str(tic): float(weight) for tic, weight in zip(tic_list, weights)}
        )
        return snapshot

    def save_weight_memory(self):
        """Return each executed trading day's post-trade portfolio weights."""
        metadata_columns = [
            "date",
            "cash_after_trade",
            "total_asset_after_trade",
            "cash_weight",
            "stock_weight_sum",
            "total_weight",
        ]
        columns = metadata_columns + self._get_tic_list()
        return pd.DataFrame(self.post_trade_weights_memory).reindex(columns=columns)

    def save_reward_components_memory(self):
        """Return the per-stock contribution to each unscaled step reward."""
        date_list = self.date_memory[: len(self.reward_components_memory)]
        df_components = pd.DataFrame(
            self.reward_components_memory,
            columns=self._get_tic_list(),
        )
        df_components.insert(0, "date", date_list)
        return df_components

    def save_last_episode_reward_components_memory(self):
        """Return the completed episode contribution snapshot kept across VecEnv reset."""
        df_components = pd.DataFrame(
            self.last_episode_reward_components_memory,
            columns=self._get_tic_list(),
        )
        df_components.insert(
            0,
            "date",
            self.last_episode_reward_component_dates,
        )
        return df_components

    def save_reward_memory(self):
        # rewards_memory stores unscaled rewards before reward_scaling.
        date_list = self.date_memory[1 : 1 + len(self.rewards_memory)]
        return pd.DataFrame(
            {
                "date": date_list,
                "step_reward_unscaled": self.rewards_memory,
                "cumulative_reward_unscaled": np.cumsum(self.rewards_memory),
            }
        )

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        # print(df_states)
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def save_agent_action_memory(self):
        """Return the original policy actions in the [-1, 1] action space."""
        if len(self.df.tic.unique()) > 1:
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list, columns=["date"])

            df_actions = pd.DataFrame(self.agent_actions_memory)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
        else:
            date_list = self.date_memory[:-1]
            df_actions = pd.DataFrame(
                {"date": date_list, "actions": self.agent_actions_memory}
            )
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
