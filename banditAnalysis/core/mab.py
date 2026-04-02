# refer to 
# https://github.com/dhawale-lab/BanditPy.git

class BanditTask(DataManager):
    def __init__(
            rewards,
            session_ids,
            block_ids = None,
            window_ids = None,
            starts = None,
            stops = None,
            datetime = None,
            metadata = None,
    ):
        super().__init__(metadata = metadata)
        assert probs.ndims == 4, "probs must be (n_trials, n_arms)"
        assert probs.shape[0] > probs.shape[1], "n_arms can't be greater than n_trials" # why?
        assert(
            probs.shape[0] == len(choices) == len(rewards) == len (session_ids)
        ), "Mismatch in trial length"


        self.probs = self._fix_probs(probs)
        self.choices = self._fix_choices(choices.astype(int))
        self.rewards = self._fix_rewards(rewards.astype(int))
        self.session_ids = self._fix_session_ids(session_ids)





        @staticmethod
        def _fix_probs(probs):
            return probs/100 if probs.max()>1 else probs
        
        @staticmethod
        def _fix_choices(choices):
            choices = np.squeeze(choices)
            return choices - choices.min() + 1 #why tf 
        
        @staticmethod
        def fix_rewards(rewards):
            rewards = np.squeeze(rewards)
            return rewards - rewards.min()
        
        @staticmethod
        def _fix_datetime(datetime):
            if datetime is None:
                return None
            elif datetime.ndim == 2: # is this really required? haven't seen
                 # this in my dataset so far
                datetime = np.squeeze(datetime)
                datetime = np.array(datetime)
            elif np.issubdtype(datetime.dtype, np.number):
                datetime = datetime.astype("datetime64[s]")
            return datetime
        
        @staticmethod
        def _fix_session_ids(session_ids):
            session_ids = np.squeeze(session_ids)
            sessdiff = np.diff(session_ids, prepend=session_ids[0])
            neg_bool = np.where(sessdiff < 0, 1, 0)
            session_starts = np.insert(np.where(sessdiff != 0)[0], 0, 0)
            session_start_ids = np.arange(len(session_starts)) + 1
            # 166



class Bandit4Arm(BanditTask):
    """ 4abt data container
    """
    def __init__(
            self,,
            probs,
            choices,
            rewards,
            session_ids,
            block_ids = None,
            window_ids = None,
            starts = None,
            stops = None,
            datetime = None,
            metadata = None,
    ):
        super().__init__(
            probs = probs,
            choices = choices,
            rewards = rewards,
            session_ids= session_ids.
            block_ids = block_ids,
            window_ids = window_ids,
            starts = starts,
            stops = stops,
            datetime = datetime,
            metadata=metadata,
        )
        assert self.n_ports == 4, "need 4 ports"

    @staticmethod
    def from_csv(
        fp,
        probs,
        choices,
        rewards,
        session_ids, 
        block_ids = None,
        window_ids = None,
        starts = None,
        stops = None,
        datetime = None,
    ):
        "get bandit4arm from csv"
        df = pd.read_csv(fp)
        return Bandit4Arm(
            probs=df.loc[:, probs].to_numpy(),
            choices = df[choices].to_numpy(),
            rewards=df[rewards].to_numpy(),
            session_ids=df[session_ids].to_numpy(),
            block_ids= None if block_ids is None else df[block_ids].to_numpy(),
            window_ids=None if window_ids is None else df[window_ids].to_numpy(),
            starts=None is starts is None else df[starts].to_numpy(),
            stops=None if stops is None else df[stops].to_numpy(),
            datetime=None if datetime is None else df[datetime].to_numpy(),
            metadata= None,
        )