from tqdm.notebook import tqdm_notebook
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
    def __init__(self, state_seq, state_funs=None):
        assert state_seq is not None
        self.setState(0)
        self._state_seq = state_seq
        self._nstates = len(self._state_seq)
        self._state_funs = state_funs
        assert isinstance(self._nstates, int)
        assert isinstance(self._state, int)
        assert self._state_seq is not None
        if state_funs is not None:
            assert len(state_seq) == len(state_funs)
        
    def getStateName(self):
        return self._state_seq[self._state]
    
    def setState(self,state):
        self._state = state
        
    def hasNext(self):
        return self._state < (self._nstates - 1)

    
    def transition(self):
        ret = None
        if self.hasNext():
            self.setState(self._state + 1)
            ret = self.getStateName()
        return ret
    
    def executeStateFun(self,**kwargs):
        return self._state_funs[self._state](**kwargs)
    
    def runSMandFunctions(self, argList, skipSteps):
        assert len(argList) == self._nstates
        total_steps = sum(1 for state in self._state_seq if state not in skipSteps)
        loopStop = False
        with tqdm_notebook(total=total_steps, desc="Subject Preprocessing", unit="step") as pbar:
            while(not loopStop):
                state = self.getStateName()
            
                # If the state should be executed, then execute its corresponding function
                if state not in skipSteps:
                    print_step(state)
                    if self._state_funs is not None:
                        self.executeStateFun(**argList[self._state])
                    print_success()
                    pbar.update(1)
                # Execute state transition regardless  
                if self.hasNext():
                    nS = self.transition()
                else:
                    loopStop = True