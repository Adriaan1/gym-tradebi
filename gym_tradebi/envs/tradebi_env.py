"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd

from gym_tradebi.dataloader.json import Dataloader

import json
import datetime
import time

import logging

from .utils import Trade, TradeType, Portfolio

log = logging.getLogger(__name__)
log.info('%s logger started.',__name__)

class TradebiEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(3)
        Num	Action
        0	Hold
        1	Long
        2   Short
        
        Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value between ±0.05
    Episode Termination:
        Pole Angle is more than ±12°
        Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    

    def __init__(self):

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        """ high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None """

        source = Dataloader('EUR_USD_H1_history.json')

        self.df = source.getData()

        self.portfolio = Portfolio()

        self.running_step = 0

        #self.orders = []
        
        self.lot = 1
        self.capital = 10000
        self.precision = 0.00001

        high = np.array([])
        low = np.array([])

        for column_name in list(self.df.columns.values):
            high = np.append(high,np.array([self.df[column_name].max()]))
            low = np.append(low,np.array([self.df[column_name].min()]))

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.current_index = 0

    def init_dataloader(self):

        source = Dataloader('EUR_USD_H1_history.json')

        self.df = source.getData()

        high = np.array([])
        low = np.array([])

        for cname in list(self.df.columns.values):
            high = np.append(high,np.array([self.df[cname].max()]))
            low = np.append(low,np.array([self.df[cname].min()]))

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.state = None
        self.currentIndex = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        self.state = self.df.values[self.current_index]

        current_value = self.state[0]
        trade_time = self.df.index[self.current_index]

        #print("currentValue: " + str(current_value))

        reward = 0

        # if action == 0 calculate capital
        # if action == 1 no order/close the sell order, create LONG order. calculate capital and reward
        # if action == 2 no order/close the sell order, create LONG order. calculate capital and reward
        
        if(action == 1):

            if (len(self.portfolio.trades) == 0): 
                self.portfolio.trades.append(Trade(TradeType.LONG,self.lot,current_value,self.precision, trade_time))
            else:
                existingOrder = self.portfolio.trades[0]

                if(existingOrder.trade_type == TradeType.SHORT):
                    reward = existingOrder.close(current_value)
                    self.portfolio.trades.append(Trade(TradeType.LONG,self.lot,current_value, self.precision, trade_time))

        elif(action == 2):

            # if there is no order
            if (len(self.portfolio.trades) == 0): 
                self.portfolio.trades.append(Trade(TradeType.SHORT,self.lot,current_value,self.precision, trade_time))
            else:
                existingOrder = self.portfolio.trades[0]

                if(existingOrder.trade_type == TradeType.LONG):
                    reward = existingOrder.close(current_value)
                    self.portfolio.trades.append(Trade(TradeType.SHORT,self.lot,current_value,self.precision, trade_time))

        self.capital += reward
        self.portfolio.total_reward += reward 

        done = self.current_index == len(self.df.index) - 1 or \
               self.capital <= 0

        self.current_index += 1

        if(done):
            self.running_step += 1
            print("done: " + str(done))
            print("step: " + str(self.running_step))
            #print("state: ")
            #print(*self.state)
            #print("action: " + str(action))
            print("total_reward: " + str(self.portfolio.total_reward))
            print("capital: " + str(self.capital))
            #print("reward: " + str(reward))
            
            

        """ state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x  = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else: # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x  = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x,x_dot,theta,theta_dot) """
        
        """ done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0 """

        

        return np.array(self.state), self.portfolio.total_reward, done, {}

    def reset(self):
        #self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        #self.steps_beyond_done = None
        self.current_index = 0
        self.state = self.df.values[self.current_index]
        self.capital = 10000
        self.portfolio.reset()

        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None