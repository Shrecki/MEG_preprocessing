def print_step(step_name):
    print("Starting {}".format(step_name))

def print_success():
    print("Done and saved.")


def noskip_step(step, step_curr, skip_list):
    return step == step_curr and step_curr not in skip_list

def step_wrapper(f_name, func):
    def wrapped_func(*args, **kwargs):
        print_step(f_name)
        return func(*args, **kwargs)
    return wrapped_func

class StateMachine:    
    def __init__(self, state_seq):
        assert state_seq is not None
        self.setState(0)
        self._state_seq = state_seq
        self._nstates = len(self._state_seq)
        assert isinstance(self._nstates, int)
        assert isinstance(self._state, int)
        assert self._state_seq is not None
        
    def getStateName(self):
        return self._state_seq[self._state]
    
    def setState(self,state):
        self._state = state
    
    def transition(self):
        ret = None
        if self._state < (self._nstates - 1):
            self.setState(self._state + 1)
            ret = self.getStateName()
        return ret