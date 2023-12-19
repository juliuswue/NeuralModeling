import numpy as np
from scipy import stats # for gaussian noise
from environment import Environment

class DynaAgent(Environment):

    def __init__(self, alpha, gamma, epsilon, e_greedy = False):

        '''
        Initialise the agent class instance
        Input arguments:
            alpha   -- learning rate \in (0, 1]
            gamma   -- discount factor \in (0, 1)
            epsilon -- controls the influence of the exploration bonus
        '''

        self.alpha   = alpha
        self.gamma   = gamma 
        self.epsilon = epsilon

        self.e_greedy = e_greedy

        return None

    def init_env(self, **env_config):

        '''
        Initialise the environment
        Input arguments:
            **env_config -- dictionary with environment parameters
        '''

        Environment.__init__(self, **env_config)

        return None

    def _init_q_values(self):

        '''
        Initialise the Q-value table
        '''

        self.Q = np.zeros((self.num_states, self.num_actions))

        return None

    def _init_experience_buffer(self):

        '''
        Initialise the experience buffer
        '''

        self.experience_buffer = np.zeros((self.num_states*self.num_actions, 4), dtype=int)
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.experience_buffer[s*self.num_actions+a] = [s, a, 0, s]

        return None

    def _init_history(self):

        '''
        Initialise the history
        '''

        self.history = np.empty((0, 4), dtype=int)

        return None
    
    def _init_action_count(self):

        '''
        Initialise the action count
        '''

        self.action_count = np.zeros((self.num_states, self.num_actions), dtype=int)

        return None

    def _update_experience_buffer(self, s, a, r, s1):

        '''
        Update the experience buffer (world model)
        Input arguments:
            s  -- initial state
            a  -- chosen action
            r  -- received reward
            s1 -- next state
        '''

        # complete the code

        self.experience_buffer[s * self.num_actions + a] = [s, a, r, s1]

        return None

    def _update_qvals(self, s, a, r, s1, bonus=False):

        '''
        Update the Q-value table
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
            bonus -- True / False whether to use exploration bonus or not
        '''

        # complete the code

        if(bonus):
            self.Q[s,a] += self.alpha * (r + self.gamma * np.max(self.Q[s1,:]) - self.Q[s,a] + self.epsilon * np.sqrt(self.action_count[s,a]))
        else:
            self.Q[s,a] += self.alpha * (r + self.gamma * np.max(self.Q[s1,:]) - self.Q[s,a])

        return None

    def _update_action_count(self, s, a):

        '''
        Update the action count
        Input arguments:
            Input arguments:
            s  -- initial state
            a  -- chosen action
        '''

        # complete the code

        self.action_count += 1
        self.action_count[s,a] = 0

        return None

    def _update_history(self, s, a, r, s1):

        '''
        Update the history
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
        '''

        self.history = np.vstack((self.history, np.array([s, a, r, s1])))

        return None

    def _policy(self, s):

        '''
        Agent's policy 
        Input arguments:
            s -- state
        Output:
            a -- index of action to be chosen
        '''

        # complete the code

        # in case of an epsilion greedy agent (for additional task)
        if(self.e_greedy and np.random.rand() < 0.05):
            return np.random.randint(0,4)

        # normal policy
        q_vals = self.Q[s,:] + self.epsilon * np.sqrt(self.action_count[s,:])
        max_ids = np.where(q_vals== np.max(q_vals))[0]
        np.random.shuffle(max_ids)
        a = max_ids[0]

        return a

    def _plan(self, num_planning_updates):

        '''
        Planning computations
        Input arguments:
            num_planning_updates -- number of planning updates to execute
        '''

        # complete the code

        ids_memory = np.random.randint(0, len(self.experience_buffer), num_planning_updates)
        for i in range(num_planning_updates):
            # get random past experience
            id_memory = ids_memory[i]
            # get the state from past experiencee
            s, a, r, s1 = self.experience_buffer[id_memory, :]
            # learning
            self._update_qvals(s, a, r, s1, bonus=True)

        return None

    def get_performace(self):

        '''
        Returns cumulative reward collected prior to each move
        '''

        return np.cumsum(self.history[:, 2])

    def simulate(self, num_trials, reset_agent=True, num_planning_updates=None):

        '''
        Main simulation function
        Input arguments:
            num_trials           -- number of trials (i.e., moves) to simulate
            reset_agent          -- whether to reset all knowledge and begin at the start state
            num_planning_updates -- number of planning updates to execute after every move
        '''

        if reset_agent:
            self._init_q_values()
            self._init_experience_buffer()
            self._init_action_count()
            self._init_history()

            self.s = self.start_state

        for _ in range(num_trials):

            # choose action
            a  = self._policy(self.s)
            # get new state
            s1 = np.random.choice(np.arange(self.num_states), p=self.T[self.s, a, :])
            # receive reward
            r  = self.R[self.s, a]
            # learning
            self._update_qvals(self.s, a, r, s1, bonus=False)
            # update world model 
            self._update_experience_buffer(self.s, a, r, s1)
            # reset action count
            self._update_action_count(self.s, a)
            # update history
            self._update_history(self.s, a, r, s1)
            # plan
            if num_planning_updates is not None:
                self._plan(num_planning_updates)

            if s1 == self.goal_state:
                self.s = self.start_state
            else:
                self.s = s1

        return None
    
class TwoStepAgent:

    def __init__(self, alpha1, alpha2, beta1, beta2, lam, w, p):

        '''
        Initialise the agent class instance
        Input arguments:
            alpha1 -- learning rate for the first stage \in (0, 1]
            alpha2 -- learning rate for the second stage \in (0, 1]
            beta1  -- inverse temperature for the first stage
            beta2  -- inverse temperature for the second stage
            lam    -- eligibility trace parameter
            w      -- mixing weight for MF vs MB \in [0, 1] 
            p      -- perseveration strength
        '''

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1  = beta1
        self.beta2  = beta2
        self.lam    = lam
        self.w      = w
        self.p      = p

        return None
        
    def _init_history(self):

        '''
        Initialise history to later compute stay probabilities
        '''

        self.history = np.empty((0, 3), dtype=int)

        return None
    
    def _update_history(self, a, s1, r1):

        '''
        Update history
        Input arguments:
            a  -- first stage action
            s1 -- second stage state
            r1 -- second stage reward
        '''

        self.history = np.vstack((self.history, [a, s1, r1]))

        return None
    
    def get_stay_probabilities(self):

        '''
        Calculate stay probabilities
        '''

        common_r      = 0
        num_common_r  = 0
        common_nr     = 0
        num_common_nr = 0
        rare_r        = 0
        num_rare_r    = 0
        rare_nr       = 0
        num_rare_nr   = 0

        num_trials = self.history.shape[0]
        for idx_trial in range(num_trials-1):
            a, s1, r1 = self.history[idx_trial, :]
            a_next    = self.history[idx_trial+1, 0]

            # common
            if (a == 0 and s1 == 1) or (a == 1 and s1 == 2):
                # rewarded
                if r1 == 1:
                    if a == a_next:
                        common_r += 1
                    num_common_r += 1
                else:
                    if a == a_next:
                        common_nr += 1
                    num_common_nr += 1
            else:
                if r1 == 1:
                    if a == a_next:
                        rare_r += 1
                    num_rare_r += 1
                else:
                    if a == a_next:
                        rare_nr += 1
                    num_rare_nr += 1

        return np.array([common_r/num_common_r, rare_r/num_rare_r, common_nr/num_common_nr, rare_nr/num_rare_nr])

    def simulate(self, num_trials):

        '''
        Main simulation function
        Input arguments:
            num_trials -- number of trials to simulate
        '''

        # complete the code

        # init agent
        self._init_history()
        
        # 3 states with 2 actions
        q_mf = np.zeros((3,2))
        q_mb = np.zeros((3,2))
        # reward probs for the two states with two actions each - starting with middle value of 0.5
        reward_probs = np.ones((2,2)) * 0.5

        # counter for transitions probabiliy
        p_transitions = np.zeros((2,2))
        n_0_1_plus_1_2 = 0
        n_0_2_plus_1_1 = 0

        rep = np.zeros(2)

        for _ in range(num_trials):
            # 1st stage
            # calculate the combined q values
            q_net = self.w * q_mb[0,:] + (1 - self.w) * q_mf[0,:]
            # calculate action probabilities using softmax
            p_s1 = np.exp(self.beta1 * (q_net + self.p * rep)) / np.sum(np.exp(self.beta1 * (q_net+ self.p * rep)))
            # get action 
            a1 = np.random.choice([0, 1], p=p_s1)
            # decide if we go to the rare or common state
            common = False if np.random.rand() < 0.3 else True

            # 2nd stage
            if((a1 == 0 and common) or (a1 == 1 and not common)):
                id_stage_2 = 1
            else:
                id_stage_2 = 2

            # update transition learning
            if((a1 == 0 and id_stage_2 == 1) or (a1 == 1 and id_stage_2 == 2)):
                n_0_1_plus_1_2 += 1
            else:
                n_0_2_plus_1_1 += 1
            # update learned transitions
            if(n_0_1_plus_1_2 >= n_0_2_plus_1_1):
                p_transitions[0,:] = [0.7, 0.3]
                p_transitions[1,:] = [0.3, 0.7]
            else:
                p_transitions[0,:] = [0.3, 0.7]
                p_transitions[1,:] = [0.7, 0.3]

            # get q values for that stage
            q_stage2 = q_mf[id_stage_2,:]
            # calculate action probabilities using softmax
            p_s2 = np.exp(self.beta2 * q_stage2) / np.sum(np.exp(self.beta2 * q_stage2))
            # get action 
            a2 = np.random.choice([0, 1], p=p_s2)
            # get reward
            r = 0
            if(np.random.rand() <= reward_probs[id_stage_2 - 1, a2]):
                r = 1
            # update q value stage 1
            q_mf[0, a1] += self.alpha1 * (np.max(q_mf[id_stage_2,:]) - q_mf[0, a1])
            # update q value stage 2
            q_mf[id_stage_2,a2] += self.alpha2 * (r - q_mf[id_stage_2,a2])
            # additional stage-skipping update of the first-stage
            q_mf[0, a1] += self.alpha1 * self.lam * (r - q_mf[id_stage_2,a2])
            # update model based q values 
            q_mb[0,0] = p_transitions[0, 0] * np.max(q_mf[1,:]) + p_transitions[0, 1] * np.max(q_mf[2,:])
            q_mb[0,1] = p_transitions[1, 0] * np.max(q_mf[1,:]) + p_transitions[1, 1] * np.max(q_mf[2,:])

            # update history
            self._update_history(a1, id_stage_2, r)

            # update reward probabilities with gaussian walk
            for i in range(2):
                for j in range(2):
                    reward_probs[i,j] += np.random.normal(0,0.025)
                    if(reward_probs[i,j] > 0.75):
                        reward_probs[i,j] = 1.5 - reward_probs[i,j]
                    elif(reward_probs[i,j] < 0.25):
                        reward_probs[i,j] = 0.5 - reward_probs[i,j]

            # update rep
            rep = np.zeros(2)
            rep[a1] = 1

        return None