""" Contains the Episodes for Navigation. """
import random
import torch
import time
import sys
from constants import GOAL_SUCCESS_REWARD, STEP_PENALTY, BASIC_ACTIONS, LOCATE_REWARD, WRONG_PENALTY, PROGRESS_REWARD
from environment import Environment
from utils.net_util import gpuify
import math


class Episode:
    """ Episode for Navigation. """
    def __init__(self, args, gpu_id, rank, strict_done=False):
        super(Episode, self).__init__()

        self._env = None

        self.gpu_id = gpu_id
        self.strict_done = strict_done
        self.task_data = None
        self.glove_embedding = None

        self.seed = args.seed + rank
        random.seed(self.seed)

        with open('./datasets/objects/int_objects.txt') as f:
            int_objects = [s.strip() for s in f.readlines()]
        with open('./datasets/objects/rec_objects.txt') as f:
            rec_objects = [s.strip() for s in f.readlines()]
        
        self.objects = int_objects + rec_objects

        self.actions_list = [{'action':a} for a in BASIC_ACTIONS]
        self.actions_taken = []

        self.locate_tomato = 0
        self.locate_bowl = 0

    @property
    def environment(self):
        return self._env

    def state_for_agent(self):
        return self.environment.current_frame

    def step(self, action_as_int):
        action = self.actions_list[action_as_int]
        self.actions_taken.append(action)
        return self.action_step(action)

    def action_step(self, action):
        self.environment.step(action)
        reward, terminal, action_was_successful = self.judge(action)

        return reward, terminal, action_was_successful

    def slow_replay(self, delay=0.2):
        # Reset the episode
        self._env.reset(self.cur_scene, change_seed = False)
        
        for action in self.actions_taken:
            self.action_step(action)
            time.sleep(delay)
    
    def judge(self, action):
        """ Judge the last event. """
        # immediate reward
        reward = STEP_PENALTY 
        done = False
        action_was_successful = self.environment.last_action_success

        # if action['action'] == 'Done':
        #     done = True
        #     objects = self._env.last_event.metadata['objects']
        #     visible_objects = [o['objectType'] for o in objects if o['visible']]
        #     if self.target in visible_objects:
        #         reward += GOAL_SUCCESS_REWARD
        #         self.success = True

        # agent_pos= self._env.last_event.metadata['agent']['position']
        # (ag_x, ag_y, ag_z) = agent_pos['x'], agent_pos['y'], agent_pos['z']
        #
        # objects = self._env.last_event.metadata['objects']
        # tomato_pos = [o['position'] for o in objects if o['objectType'] == 'Tomato'][0]
        # (tomato_x, tomato_y, tomato_z) = (tomato_pos['x'], tomato_pos['y'], tomato_pos['z'])
        # bowl_pos = [o['position'] for o in objects if o['objectType'] == 'Bowl'][0]
        # (bowl_x, bowl_y, bowl_z) = (bowl_pos['x'], bowl_pos['y'], bowl_pos['z'])
        #
        # agent_tomato = math.fabs(ag_x - tomato_x) + math.fabs(ag_y - tomato_y) + math.fabs(ag_z - tomato_z)
        # agent_bowl = math.fabs(ag_x - bowl_x) + math.fabs(ag_y - bowl_y) + math.fabs(ag_z - bowl_z)
        #
        # if not self.tomato and not self.bowl:
        #     if agent_tomato < agent_bowl:
        #         if agent_tomato < self.last_tomato_distance:
        #             reward += PROGRESS_REWARD
        #     else:
        #         if agent_bowl < self.last_bowl_distance:
        #             reward += PROGRESS_REWARD
        #
        # elif not self.tomato:
        #     if agent_tomato < self.last_tomato_distance:
        #         reward += PROGRESS_REWARD
        #
        # elif not self.bowl:
        #     if agent_bowl < self.last_bowl_distance:
        #         reward += PROGRESS_REWARD
        #
        #
        # self.last_tomato_distance = agent_tomato
        # self.last_bowl_distance = agent_bowl


        if action['action'] == 'LocateTomato':
            self.locate_tomato += 1
            objects = self._env.last_event.metadata['objects']
            visible_objects = [o['objectType'] for o in objects if o['visible']]
            if self.target[0] in visible_objects:
                self.tomato = True
                if self.locate_tomato == 1:
                    reward += LOCATE_REWARD
            else:
                reward += WRONG_PENALTY

        if action['action'] == 'LocateBowl':
            self.locate_bowl += 1
            objects = self._env.last_event.metadata['objects']
            visible_objects = [o['objectType'] for o in objects if o['visible']]
            if self.target[1] in visible_objects:
                self.bowl = True
                if self.locate_bowl == 1:
                    reward += LOCATE_REWARD
            else:
                reward += WRONG_PENALTY

        # if action['action'] == 'LocateBoth':
        #     self.locate_tomato += 1
        #     self.locate_bowl += 1
        #     objects = self._env.last_event.metadata['objects']
        #     visible_objects = [o['objectType'] for o in objects if o['visible']]
        #
        #     if self.target[0] in visible_objects and self.target[1] in visible_objects:
        #         self.tomato = True
        #         self.bowl = True
        #         if self.locate_tomato == 1:
        #             reward += LOCATE_REWARD
        #         if self.locate_bowl == 1:
        #             reward += LOCATE_REWARD
            # else:
            #     reward += WRONG_PENALTY

        if self.tomato and self.bowl:
            self.success = True
            #reward += GOAL_SUCCESS_REWARD

        if self.locate_tomato > 0 and self.locate_bowl > 0:
            #self.done_time += 1
            self.done = True

        # if self.done_time > 2:
        #     self.done = True

        return reward, done, action_was_successful

    def new_episode(self, args, scene):
        
        if self._env is None:
            if args.arch == 'osx':
                local_executable_path = './datasets/builds/thor-local-OSXIntel64.app/Contents/MacOS/thor-local-OSXIntel64'
            else:
                local_executable_path = './datasets/builds/thor-local-Linux64'
            
            self._env = Environment(
                    grid_size=args.grid_size,
                    fov=args.fov,
                    local_executable_path=local_executable_path,
                    randomize_objects=args.randomize_objects,
                    seed=self.seed)
            self._env.start(scene, self.gpu_id)
        else:
            self._env.reset(scene)

        # For now, single target.
        #self.target = 'Tomato'

        self.target = ('Tomato', 'Bowl')
        self.tomato = False
        self.bowl = False
        self.locate_tomato = 0
        self.locate_bowl = 0
        self.success = False
        self.cur_scene = scene
        self.actions_taken = []

#        self.done_time = 0

        self.last_tomato_distance = float('inf')
        self.last_bowl_distance = float('inf')
        
        return True
